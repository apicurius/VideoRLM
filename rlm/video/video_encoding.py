"""Neural encoding: SigLIP2 frames/text, V-JEPA 2 clips."""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def encode_frames(
    model,
    image_processor,
    torch_device: str,
    frames: list[np.ndarray],
    temporal_window: int = 1,
    stride: int | None = None,
) -> np.ndarray:
    """Encode a batch of BGR frames into an (N, D) embedding matrix.

    Args:
        model: SigLIP2 model instance.
        image_processor: SigLIP2 image processor.
        torch_device: Torch device string.
        frames: List of BGR numpy arrays.
        temporal_window: Number of consecutive frames to average into one
            embedding. 1 means no grouping (backward-compatible default).
        stride: Sliding window step size. When not None and less than
            temporal_window, overlapping windows are used and each frame
            accumulates the mean embedding of every window it belongs to.
            None preserves the original non-overlapping behavior.
    """
    import torch
    from PIL import Image

    images = [Image.fromarray(f[:, :, ::-1]) for f in frames]  # BGR → RGB

    # Process in batches of 32 to avoid OOM
    all_embs = []
    batch_size = 32
    for i in range(0, len(images), batch_size):
        batch = images[i : i + batch_size]
        inputs = image_processor(images=batch, return_tensors="pt").to(torch_device)
        with torch.no_grad():
            out = model.get_image_features(**inputs)
            emb = out.pooler_output if hasattr(out, "pooler_output") else out
            emb = emb / emb.norm(p=2, dim=-1, keepdim=True)
        all_embs.append(emb.cpu().numpy())

    all_embs_arr = np.concatenate(all_embs, axis=0)  # (N, D)

    # Overlapping sliding window: accumulate window means per frame
    if stride is not None and stride < temporal_window and len(all_embs_arr) >= temporal_window:
        n = len(all_embs_arr)
        accum = np.zeros_like(all_embs_arr)
        counts = np.zeros(n, dtype=np.float32)
        for start in range(0, n - temporal_window + 1, stride):
            window_mean = all_embs_arr[start : start + temporal_window].mean(axis=0)
            for k in range(start, min(start + temporal_window, n)):
                accum[k] += window_mean
                counts[k] += 1
        counts = np.maximum(counts, 1)
        result = accum / counts[:, None]
        norms = np.linalg.norm(result, axis=1, keepdims=True)
        result = result / np.maximum(norms, 1e-10)
        return result

    if temporal_window > 1 and len(all_embs_arr) >= temporal_window:
        n = len(all_embs_arr)
        n_groups = n // temporal_window
        grouped = all_embs_arr[: n_groups * temporal_window].reshape(
            n_groups, temporal_window, -1
        )
        averaged = grouped.mean(axis=1)
        norms = np.linalg.norm(averaged, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        averaged = averaged / norms
        remainder = all_embs_arr[n_groups * temporal_window :]
        if len(remainder) > 0:
            rem_avg = remainder.mean(axis=0, keepdims=True)
            rem_avg = rem_avg / np.maximum(
                np.linalg.norm(rem_avg, axis=1, keepdims=True), 1e-10
            )
            averaged = np.concatenate([averaged, rem_avg], axis=0)
        return averaged

    return all_embs_arr


def encode_texts(
    texts: list[str],
    *,
    text_embedding_model_name: str | None,
    text_model,
    text_model_type: str | None,
    text_tokenizer,
    siglip_model,
    siglip_tokenizer,
    siglip_device: str,
) -> np.ndarray:
    """Encode a list of text strings into an (N, D) embedding matrix.

    When a separate ``text_embedding_model`` is configured, uses that model
    instead of SigLIP2's text encoder for richer semantic representations.
    """
    if text_model is not None and text_embedding_model_name is not None:
        if text_model_type == "sentence_transformers":
            emb = text_model.encode(texts, normalize_embeddings=True)
            return np.asarray(emb)
        else:
            import torch

            inputs = text_tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            with torch.no_grad():
                out = text_model(**inputs)
                emb = out.last_hidden_state[:, 0, :]
                emb = emb / emb.norm(p=2, dim=-1, keepdim=True)
            return emb.cpu().numpy()

    return encode_texts_siglip(siglip_model, siglip_tokenizer, siglip_device, texts)


def encode_texts_siglip(
    model,
    tokenizer,
    torch_device: str,
    texts: list[str],
) -> np.ndarray:
    """Encode texts using SigLIP2's text encoder (ignoring text_embedding_model)."""
    import torch

    inputs = tokenizer(
        texts,
        padding="max_length",
        max_length=64,
        return_tensors="pt",
    ).to(torch_device)
    with torch.no_grad():
        out = model.get_text_features(**inputs)
        emb = out.pooler_output if hasattr(out, "pooler_output") else out
        emb = emb / emb.norm(p=2, dim=-1, keepdim=True)
    return emb.cpu().numpy()


def encode_clips_vjepa(
    scene_model,
    clips: list[list[np.ndarray]],
    scene_processor,
    scene_torch_device: str,
    scene_embed_dim: int = 1024,
    return_full: bool = False,
) -> np.ndarray | tuple[np.ndarray, list[np.ndarray]]:
    """Embed video clips using V-JEPA 2.

    Each clip is a list of BGR frames. Returns an ``(N_clips, 1024)``
    embedding matrix where each row is the mean-pooled patch token
    representation for one clip.

    Clips are grouped by frame count before batching because the video
    processor requires all clips in a batch to have the same temporal
    length (e.g. the remainder clip may be shorter than the rest).

    Args:
        scene_model: V-JEPA 2 model instance.
        clips: List of clips, each a list of frames (H, W, C) BGR.
        scene_processor: V-JEPA 2 video processor.
        scene_torch_device: Torch device string.
        scene_embed_dim: Embedding dimension for batch size selection.
        return_full: When False (default), returns pooled embeddings array
            of shape (N_clips, D). When True, returns a tuple
            ``(pooled_embs, feature_maps)`` where ``feature_maps`` is a
            list of arrays of shape ``(num_patches, D)``, one per clip.
    """
    from itertools import groupby

    import torch

    # Group clips by frame count, preserving original order via index
    indexed_clips = list(enumerate(clips))
    indexed_clips.sort(key=lambda x: len(x[1]))

    result_embs = [None] * len(clips)
    result_maps: list[np.ndarray | None] | None = [None] * len(clips) if return_full else None
    batch_size = 2 if scene_embed_dim >= 1280 else 4

    for _frame_count, group in groupby(indexed_clips, key=lambda x: len(x[1])):
        group_items = list(group)
        for i in range(0, len(group_items), batch_size):
            batch_items = group_items[i : i + batch_size]
            rgb_clips = []
            for _idx, clip in batch_items:
                rgb_frames = [f[:, :, ::-1] for f in clip]  # BGR → RGB
                rgb_clips.append(rgb_frames)

            inputs = scene_processor(rgb_clips, return_tensors="pt").to(
                scene_torch_device
            )
            with torch.no_grad():
                outputs = scene_model(**inputs)
                patch_tokens = outputs.last_hidden_state
                clip_embs = patch_tokens.mean(dim=1)  # (batch, 1024)
                clip_embs = clip_embs / clip_embs.norm(p=2, dim=-1, keepdim=True)
            embs_np = clip_embs.cpu().float().numpy()
            if return_full:
                maps_np = patch_tokens.cpu().float().numpy()
            for j, (orig_idx, _clip) in enumerate(batch_items):
                result_embs[orig_idx] = embs_np[j]
                if return_full:
                    result_maps[orig_idx] = maps_np[j]

    pooled = np.stack(result_embs)
    if return_full:
        return pooled, result_maps
    return pooled


def group_frames_into_clips(
    frames: list[np.ndarray],
    timestamps: list[float],
    clip_size: int,
) -> tuple[list[list[np.ndarray]], list[float]]:
    """Group frames into clips with midpoint timestamps.

    Args:
        frames: All video frames.
        timestamps: Per-frame timestamps.
        clip_size: Number of frames per clip.

    Returns:
        Tuple of (clips, clip_timestamps) where each clip is a list of
        frames and clip_timestamps are the midpoint timestamps.
    """
    clips: list[list[np.ndarray]] = []
    clip_timestamps: list[float] = []
    for i in range(0, len(frames), clip_size):
        clip = frames[i : i + clip_size]
        clips.append(clip)
        mid = min(i + len(clip) // 2, len(frames) - 1)
        clip_timestamps.append(timestamps[mid])
    return clips, clip_timestamps


def encode_frames_overlapping_vjepa(
    encode_clips_fn,
    frames: list[np.ndarray],
    timestamps: list[float],
    clip_size: int,
    stride: int,
    scene_embed_dim: int = 1024,
    store_feature_maps: bool = False,
) -> tuple[np.ndarray, list[float]] | tuple[np.ndarray, list[float], list[np.ndarray]]:
    """Encode frames with overlapping V-JEPA 2 windows, producing per-frame averaged embeddings.

    Each frame appears in up to clip_size/stride windows. The final per-frame
    embedding is the L2-normalized average of all window embeddings containing
    that frame.

    Args:
        encode_clips_fn: Callable that encodes a list of clip frame lists.
            Signature: (clips, return_full=False) -> embeddings or (embeddings, feature_maps).
        frames: BGR numpy arrays.
        timestamps: Per-frame timestamps in seconds.
        clip_size: Number of frames per V-JEPA 2 window (default 64).
        stride: Window stride in frames (default 8).
        scene_embed_dim: Embedding dimension (for empty-frame fallback).
        store_feature_maps: If True, also return per-window spatial feature maps.

    Returns:
        Tuple of (per_frame_embeddings, timestamps) — or
        (per_frame_embeddings, timestamps, per_window_feature_maps) when
        store_feature_maps is True.
    """
    n = len(frames)
    if n == 0:
        if store_feature_maps:
            return np.empty((0, scene_embed_dim), dtype=np.float32), [], []
        return np.empty((0, scene_embed_dim), dtype=np.float32), []

    # Build overlapping windows
    windows = []
    window_frame_ranges = []  # (start_idx, end_idx) for each window
    for start in range(0, n, stride):
        end = min(start + clip_size, n)
        if end - start < 2:  # skip tiny windows
            continue
        windows.append(frames[start:end])
        window_frame_ranges.append((start, end))

    if not windows:
        # Fallback: single window with all frames
        windows = [frames]
        window_frame_ranges = [(0, n)]

    # Encode all windows via the provided clip encoder
    feature_maps: list[np.ndarray] | None = None
    if store_feature_maps:
        clip_embeddings, feature_maps = encode_clips_fn(windows, return_full=True)
    else:
        clip_embeddings = encode_clips_fn(windows)  # (N_windows, D)

    # Per-frame averaging: accumulate embeddings for each frame
    D = clip_embeddings.shape[1]
    frame_emb_sum = np.zeros((n, D), dtype=np.float64)
    frame_emb_count = np.zeros(n, dtype=np.float64)

    for w_idx, (start, end) in enumerate(window_frame_ranges):
        for f_idx in range(start, end):
            frame_emb_sum[f_idx] += clip_embeddings[w_idx]
            frame_emb_count[f_idx] += 1.0

    # Average and L2-normalize
    mask = frame_emb_count > 0
    per_frame = np.zeros((n, D), dtype=np.float32)
    per_frame[mask] = (frame_emb_sum[mask] / frame_emb_count[mask, np.newaxis]).astype(
        np.float32
    )

    norms = np.linalg.norm(per_frame, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    per_frame = per_frame / norms

    if store_feature_maps and feature_maps is not None:
        return per_frame, timestamps, feature_maps
    return per_frame, timestamps
