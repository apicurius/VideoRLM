"""Deduplication strategies: pre-caption, adjacent, global, and semantic."""

from __future__ import annotations

import logging
from collections.abc import Callable

import numpy as np

logger = logging.getLogger(__name__)


def pre_caption_dedup(
    segments: list[dict],
    encode_frames_fn: Callable,
    threshold: float = 0.90,
) -> None:
    """Identify visually near-duplicate segments before captioning.

    Computes mean visual embeddings per segment, then marks duplicates
    with ``_skip_caption = True`` and ``_caption_source = <rep index>``.
    Only representative segments will be captioned; their results are
    propagated to duplicates afterward by the caller.
    """
    if len(segments) < 2:
        return

    seg_embeddings = []
    valid_indices = []
    for i, seg in enumerate(segments):
        frames = seg.get("_frames", [])
        real_frames = [f for f in frames if not isinstance(f, str)]
        if not real_frames:
            seg_embeddings.append(None)
            continue
        try:
            embs = encode_frames_fn(real_frames)  # (N, D)
            mean_emb = embs.mean(axis=0)
            norm = np.linalg.norm(mean_emb)
            if norm > 1e-10:
                mean_emb = mean_emb / norm
            seg_embeddings.append(mean_emb)
            valid_indices.append(i)
        except Exception:
            logger.warning("Failed to encode frames for segment %d", i, exc_info=True)
            seg_embeddings.append(None)

    if len(valid_indices) < 2:
        return

    # Build embedding matrix for valid segments
    valid_embs = np.stack([seg_embeddings[i] for i in valid_indices])  # (M, D)

    # Pairwise cosine similarity
    from sklearn.metrics.pairwise import cosine_similarity

    sim_matrix = cosine_similarity(valid_embs)  # (M, M)

    # Greedy clustering: for each segment, attach to the first earlier
    # representative with similarity > threshold
    representatives: dict[int, int] = {}  # valid_idx -> representative valid_idx
    for vi in range(len(valid_indices)):
        seg_idx = valid_indices[vi]
        found_rep = False
        for rep_vi, rep_seg_idx in representatives.items():
            if sim_matrix[vi, rep_vi] > threshold:
                segments[seg_idx]["_skip_caption"] = True
                segments[seg_idx]["_caption_source"] = rep_seg_idx
                found_rep = True
                break
        if not found_rep:
            representatives[vi] = seg_idx

    skipped = sum(1 for s in segments if s.get("_skip_caption"))
    if skipped:
        logger.info(
            "Pre-caption dedup: %d/%d segments skipped (threshold=%.2f)",
            skipped,
            len(segments),
            threshold,
        )


def semantic_deduplicate(
    segment_infos: list[dict],
    embeddings: np.ndarray | None,
    action_embeddings: np.ndarray | None = None,
    n_clusters: int | None = None,
    similarity_threshold: float = 0.92,
) -> np.ndarray | None:
    """Semantic deduplication via k-means clustering.

    Clusters segments by embedding similarity and marks near-duplicates
    within each cluster. Also stores cluster_id on each segment for
    downstream cluster-aware search diversity.

    Args:
        segment_infos: List of segment dicts (modified in place).
        embeddings: (N, D) caption embeddings. If None, skips dedup.
        action_embeddings: Optional (N, D) action embeddings for combined clustering.
        n_clusters: Number of clusters. If None, auto-computed as
            max(2, len(segments) // 5).
        similarity_threshold: Cosine similarity above which segments in the
            same cluster are considered duplicates (default 0.92).

    Returns:
        cluster_labels array of shape (N,) or None if skipped.
    """
    if embeddings is None or len(embeddings) < 3:
        return None

    from sklearn.cluster import KMeans
    from sklearn.metrics.pairwise import cosine_similarity

    n = len(embeddings)
    if n_clusters is None:
        n_clusters = max(2, n // 5)
    n_clusters = min(n_clusters, n)

    # Combine embeddings if action embeddings available
    if action_embeddings is not None and len(action_embeddings) == n:
        combined = np.concatenate([embeddings, action_embeddings], axis=1)
        norms = np.linalg.norm(combined, axis=1, keepdims=True)
        combined = combined / np.maximum(norms, 1e-10)
    else:
        combined = embeddings

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(combined)

    # Assign cluster_id to each segment
    for i, seg in enumerate(segment_infos):
        seg["cluster_id"] = int(labels[i])

    # Within each cluster, mark duplicates (keep highest-quality representative)
    clusters: dict[int, list[int]] = {}
    for i, label in enumerate(labels):
        clusters.setdefault(int(label), []).append(i)

    dedup_count = 0
    for cluster_indices in clusters.values():
        if len(cluster_indices) < 2:
            continue

        cluster_embs = embeddings[cluster_indices]
        sim_matrix = cosine_similarity(cluster_embs)

        # Sort by quality score (descending), keep best as representative
        scored = [
            (idx, segment_infos[idx].get("quality_score", 0.5)) for idx in cluster_indices
        ]
        scored.sort(key=lambda x: x[1], reverse=True)

        kept: set[int] = set()
        for idx, _score in scored:
            is_dup = False
            local_pos = cluster_indices.index(idx)
            for kept_idx in kept:
                kept_local = cluster_indices.index(kept_idx)
                if sim_matrix[local_pos, kept_local] > similarity_threshold:
                    is_dup = True
                    segment_infos[idx]["is_semantic_duplicate"] = True
                    segment_infos[idx]["_semantic_dup_of"] = kept_idx
                    dedup_count += 1
                    break
            if not is_dup:
                kept.add(idx)

    if dedup_count > 0:
        logger.info(
            "Semantic dedup: %d/%d segments marked as duplicates"
            " (threshold=%.2f, clusters=%d)",
            dedup_count,
            n,
            similarity_threshold,
            n_clusters,
        )

    return labels


def deduplicate_segments(
    segments: list[dict],
    encode_texts_fn: Callable,
    threshold: float = 0.95,
) -> None:
    """Mark near-duplicate adjacent segments.

    Computes cosine similarity between adjacent segment captions.
    If similarity > threshold, marks the shorter segment as duplicate.
    """
    if len(segments) < 2:
        return

    captions = [seg.get("caption", "") for seg in segments]
    if not any(captions):
        return

    try:
        embs = encode_texts_fn(captions)
    except AttributeError:
        return

    for i in range(len(segments) - 1):
        if not captions[i] or not captions[i + 1]:
            continue
        sim = float(np.dot(embs[i], embs[i + 1]))
        if sim > threshold:
            dur_i = segments[i]["end_time"] - segments[i]["start_time"]
            dur_j = segments[i + 1]["end_time"] - segments[i + 1]["start_time"]
            shorter = i if dur_i <= dur_j else i + 1
            segments[shorter]["is_duplicate"] = True


def global_deduplicate(
    segments: list[dict],
    encode_texts_fn: Callable,
    threshold: float = 0.90,
) -> None:
    """Mark globally duplicate segments (non-adjacent) by caption similarity.

    For every pair (i, j) where j > i and abs(i - j) > 1 (adjacent pairs
    are already handled by ``deduplicate_segments``), if cosine similarity
    of their caption embeddings exceeds *threshold*, the shorter segment is
    marked ``is_duplicate = True``.
    """
    if len(segments) < 3:
        return

    captions = [seg.get("caption", "") for seg in segments]
    non_empty = [i for i, c in enumerate(captions) if c]
    if len(non_empty) < 2:
        return

    try:
        all_embs = encode_texts_fn(captions)
    except AttributeError:
        return

    from sklearn.metrics.pairwise import cosine_similarity

    # Build matrix for non-empty caption segments
    valid_embs = all_embs[non_empty]
    sim_matrix = cosine_similarity(valid_embs)  # (M, M)

    for vi in range(len(non_empty)):
        i = non_empty[vi]
        if segments[i].get("is_duplicate"):
            continue
        for vj in range(vi + 1, len(non_empty)):
            j = non_empty[vj]
            if abs(i - j) <= 1:
                continue  # skip adjacent — already handled
            if segments[j].get("is_duplicate"):
                continue
            if sim_matrix[vi, vj] > threshold:
                dur_i = segments[i]["end_time"] - segments[i]["start_time"]
                dur_j = segments[j]["end_time"] - segments[j]["start_time"]
                shorter = i if dur_i <= dur_j else j
                segments[shorter]["is_duplicate"] = True

    n_marked = sum(1 for s in segments if s.get("is_duplicate"))
    if n_marked:
        logger.info(
            "Global dedup: %d/%d segments marked as duplicate (threshold=%.2f)",
            n_marked,
            len(segments),
            threshold,
        )
