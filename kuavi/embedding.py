"""Embedding utilities: caption embedding, smoothing, quality checks, coarse levels, prediction."""

from __future__ import annotations

import logging
from collections.abc import Callable

import numpy as np

logger = logging.getLogger(__name__)


def embed_captions(
    segments: list[dict],
    encode_texts_fn: Callable,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Encode segment captions and action briefs into embedding matrices."""
    captions = [seg.get("caption", "") for seg in segments]
    actions = [
        ""
        if (b := seg.get("annotation", {}).get("action", {}).get("brief", "").strip())
        in ("", "N/A")
        else b
        for seg in segments
    ]

    embeddings = None
    if any(captions):
        embeddings = encode_texts_fn(captions)

    action_embeddings = None
    if any(actions):
        action_embeddings = encode_texts_fn(actions)

    return embeddings, action_embeddings


def smooth_embeddings(embs: np.ndarray, window: int = 3) -> np.ndarray:
    """Apply centered moving average smoothing to embedding rows."""
    if embs.shape[0] < window:
        return embs

    w = window // 2
    n = embs.shape[0]
    smoothed = np.empty_like(embs)
    for i in range(n):
        lo = max(0, i - w)
        hi = min(n, i + w + 1)
        smoothed[i] = embs[lo:hi].mean(axis=0)

    norms = np.linalg.norm(smoothed, axis=1, keepdims=True)
    smoothed = smoothed / np.maximum(norms, 1e-10)
    return smoothed


def check_embedding_quality(
    embeddings: np.ndarray,
    label: str = "caption",
) -> dict:
    """Compute embedding quality metrics (uniformity, pairwise similarity)."""
    if embeddings is None or embeddings.shape[0] < 2:
        return {}

    n = embeddings.shape[0]
    rng = np.random.default_rng(42)
    num_pairs = min(500, n * (n - 1) // 2)
    pairs_i = rng.integers(0, n, size=num_pairs)
    pairs_j = rng.integers(0, n - 1, size=num_pairs)
    pairs_j = np.where(pairs_j >= pairs_i, pairs_j + 1, pairs_j)

    ei = embeddings[pairs_i]
    ej = embeddings[pairs_j]

    sq_dists = np.sum((ei - ej) ** 2, axis=1)
    uniformity = float(np.log(np.mean(np.exp(-2.0 * sq_dists))))

    dot_products = np.sum(ei * ej, axis=1)
    mean_pairwise_similarity = float(np.mean(dot_products))

    is_degenerate = mean_pairwise_similarity > 0.99
    if is_degenerate:
        logger.warning(
            "Embedding quality check (%s): DEGENERATE — mean pairwise similarity %.4f > 0.99",
            label,
            mean_pairwise_similarity,
        )
    else:
        logger.info(
            "Embedding quality check (%s): OK — mean pairwise similarity %.4f",
            label,
            mean_pairwise_similarity,
        )

    return {
        "uniformity": uniformity,
        "mean_pairwise_similarity": mean_pairwise_similarity,
        "is_degenerate": is_degenerate,
    }


def build_coarse_level(
    segments: list[dict],
    embeddings: np.ndarray,
    target_duration: float = 30.0,
) -> tuple[list[dict], np.ndarray | None]:
    """Merge fine segments into ~30s coarse chunks."""
    if not segments or embeddings is None or len(embeddings) == 0:
        return [], None

    coarse_segs: list[dict] = []
    coarse_embs: list[np.ndarray] = []

    group_start = 0
    group_duration = 0.0

    for i, seg in enumerate(segments):
        seg_dur = seg["end_time"] - seg["start_time"]
        group_duration += seg_dur

        is_last = i == len(segments) - 1
        if group_duration >= target_duration or is_last:
            group_segs = segments[group_start : i + 1]
            merged_caption = " ".join(s.get("caption", "") for s in group_segs if s.get("caption"))
            coarse_segs.append(
                {
                    "start_time": group_segs[0]["start_time"],
                    "end_time": group_segs[-1]["end_time"],
                    "caption": merged_caption,
                }
            )

            group_emb = embeddings[group_start : i + 1].mean(axis=0)
            norm = np.linalg.norm(group_emb)
            if norm > 1e-10:
                group_emb = group_emb / norm
            coarse_embs.append(group_emb)

            group_start = i + 1
            group_duration = 0.0

    if not coarse_embs:
        return [], None

    return coarse_segs, np.stack(coarse_embs)


def predict_future_embedding(
    scene_predictor,
    scene_torch_device: str,
    context_features: np.ndarray,
    n_future_tokens: int = 16,
) -> np.ndarray | None:
    """Predict future frame representation using V-JEPA 2 predictor.

    The predictor uses a context/target masking scheme: context_mask selects
    patches from the encoder output that serve as context, and target_mask
    selects positions to predict. Both are index tensors of shape
    ``[batch, num_selected]`` with int64 patch indices.

    We use the first ``num_patches - n_future_tokens`` patches as context
    and the last ``n_future_tokens`` patches as target (to predict).

    Args:
        scene_predictor: V-JEPA 2 predictor module.
        scene_torch_device: Device string for the scene model.
        context_features: (num_patches, D) spatial feature map from a segment.
        n_future_tokens: Number of future token positions to predict.

    Returns:
        (n_future_tokens, D) predicted feature map, or None if predictor unavailable.
    """
    if scene_predictor is None:
        return None

    import torch

    num_patches = context_features.shape[0]
    if n_future_tokens >= num_patches:
        n_future_tokens = max(1, num_patches // 4)

    n_context = num_patches - n_future_tokens

    # Convert to tensor and add batch dimension: (1, num_patches, D)
    encoder_hidden_states = (
        torch.from_numpy(context_features).unsqueeze(0).to(scene_torch_device, dtype=torch.float16)
    )

    # Index masks: context = first n_context patches, target = last n_future_tokens
    context_mask = [torch.arange(n_context, dtype=torch.int64).unsqueeze(0).to(scene_torch_device)]
    target_mask = [
        torch.arange(n_context, num_patches, dtype=torch.int64).unsqueeze(0).to(scene_torch_device)
    ]

    with torch.no_grad():
        try:
            output = scene_predictor(encoder_hidden_states, context_mask, target_mask)
            return output.last_hidden_state.squeeze(0).cpu().float().numpy()
        except Exception as e:
            logger.warning("Predictor forward pass failed: %s", e)
            return None
