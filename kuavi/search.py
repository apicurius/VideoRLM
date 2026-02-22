"""Search tool functions for KUAVi video indices.

Each ``make_*`` function accepts a :class:`~kuavi.indexer.VideoIndex`
and returns a dict ``{"tool": callable, "description": str}``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from kuavi.indexer import VideoIndex


def _align_query_dim(query_emb: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Align query embedding dimension to match the target matrix's column dimension.

    V-JEPA 2 temporal embeddings are 1024-d while SigLIP2 text queries are 768-d.
    Zero-pad the query to match so cosine_similarity doesn't raise a ValueError.
    """
    d_q = query_emb.shape[1]
    d_m = matrix.shape[1]
    if d_q == d_m:
        return query_emb
    if d_q < d_m:
        # Zero-pad query and re-normalize to unit length
        padded = np.zeros((1, d_m), dtype=query_emb.dtype)
        padded[0, :d_q] = query_emb[0]
        norm = np.linalg.norm(padded)
        return padded / norm if norm > 0 else padded
    # d_q > d_m: truncate (shouldn't happen in practice)
    import logging

    logging.getLogger(__name__).warning(
        "Query dim %d > matrix dim %d; truncating query embedding.", d_q, d_m
    )
    return query_emb[:, :d_m]


def _mmr_rerank(
    query_emb: np.ndarray,
    candidate_embs: np.ndarray,
    candidate_indices: np.ndarray,
    scores: np.ndarray,
    top_k: int = 5,
    lambda_param: float = 0.7,
) -> list[int]:
    """Max-Marginal Relevance reranking for diverse search results."""
    if len(candidate_indices) <= top_k:
        return list(candidate_indices[np.argsort(scores)[::-1]])

    selected = []
    remaining = list(range(len(candidate_indices)))

    first = int(np.argmax(scores))
    selected.append(first)
    remaining.remove(first)

    for _ in range(top_k - 1):
        if not remaining:
            break

        best_score = -np.inf
        best_idx = remaining[0]

        for idx in remaining:
            relevance = scores[idx]

            if candidate_embs is not None and len(selected) > 0:
                selected_embs = candidate_embs[selected]
                sim_to_selected = np.dot(candidate_embs[idx], selected_embs.T).max()
            else:
                sim_to_selected = 0.0

            mmr_score = lambda_param * relevance - (1 - lambda_param) * sim_to_selected
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx

        selected.append(best_idx)
        remaining.remove(best_idx)

    return [int(candidate_indices[i]) for i in selected]


def _round_robin_from_clusters(
    clusters: dict[int, list[int]],
    scores: np.ndarray,
    top_k: int,
) -> list[int]:
    """Round-robin selection from clusters sorted by best score per cluster."""
    for label in clusters:
        clusters[label].sort(key=lambda idx: scores[idx], reverse=True)
    cluster_keys = sorted(
        clusters.keys(), key=lambda k: scores[clusters[k][0]], reverse=True
    )
    cluster_ptrs = {k: 0 for k in cluster_keys}
    top_indices: list[int] = []
    while len(top_indices) < top_k:
        added_any = False
        for label in cluster_keys:
            if len(top_indices) >= top_k:
                break
            ptr = cluster_ptrs[label]
            if ptr < len(clusters[label]):
                top_indices.append(clusters[label][ptr])
                cluster_ptrs[label] = ptr + 1
                added_any = True
        if not added_any:
            break
    return top_indices


def make_search_video(index: VideoIndex) -> dict[str, Any]:
    """Semantic search over video segment embeddings."""
    from sklearn.metrics.pairwise import cosine_similarity

    def _search_matrix(
        query_emb: np.ndarray,
        matrix: np.ndarray,
    ) -> np.ndarray:
        return cosine_similarity(query_emb, matrix)[0]

    def search_video(
        query: str,
        top_k: int = 5,
        field: str = "summary",
        exclude_non_action: bool = True,
        diverse: bool = True,
        cluster_diverse: bool = False,
        level: int = 0,
    ) -> list[dict[str, Any]]:
        """Search video segments by semantic similarity to *query*."""
        # Hierarchy level search
        if (
            level > 0
            and hasattr(index, "segment_hierarchy")
            and hasattr(index, "hierarchy_embeddings")
            and index.segment_hierarchy
            and len(index.segment_hierarchy) >= level
            and index.hierarchy_embeddings
            and len(index.hierarchy_embeddings) >= level
        ):
            h_segments = index.segment_hierarchy[level - 1]
            h_emb = index.hierarchy_embeddings[level - 1]
            if h_emb is not None and len(h_emb) > 0:
                query_emb = index.embed_fn(query)
                query_emb = np.asarray(query_emb).reshape(1, -1)
                scores = _search_matrix(query_emb, h_emb)
                top_indices = list(np.argsort(scores)[::-1][:top_k])
                results = []
                for idx in top_indices:
                    seg = h_segments[idx]
                    results.append(
                        {
                            "start_time": seg["start_time"],
                            "end_time": seg["end_time"],
                            "score": round(float(scores[idx]), 4),
                            "caption": seg.get("caption", ""),
                            "annotation": seg.get("annotation", {}),
                        }
                    )
                return results

        # Temporal field (V-JEPA 2 embeddings)
        if field == "temporal":
            if (
                hasattr(index, "temporal_embeddings")
                and index.temporal_embeddings is not None
                and len(index.temporal_embeddings) > 0
            ):
                # V-JEPA is vision-only; use SigLIP2 text encoder for query.
                # Align dims: SigLIP2 produces 768-d but V-JEPA 2 embeddings are 1024-d.
                visual_fn = getattr(index, "visual_embed_fn", None) or index.embed_fn
                query_emb = visual_fn(query)
                query_emb = np.asarray(query_emb).reshape(1, -1)
                query_emb = _align_query_dim(query_emb, index.temporal_embeddings)
                scores = _search_matrix(query_emb, index.temporal_embeddings)

                scores = scores.copy()
                for i, seg in enumerate(index.segments):
                    if seg.get("is_duplicate") or seg.get("is_semantic_duplicate"):
                        scores[i] = -np.inf

                top_indices = list(np.argsort(scores)[::-1][:top_k])
                results = []
                for idx in top_indices:
                    seg = index.segments[idx]
                    results.append(
                        {
                            "start_time": seg["start_time"],
                            "end_time": seg["end_time"],
                            "score": round(float(scores[idx]), 4),
                            "caption": seg.get("caption", ""),
                            "annotation": seg.get("annotation", {}),
                            "quality_score": seg.get("caption_quality_score"),
                        }
                    )
                return results
            import logging
            logging.getLogger(__name__).warning(
                "No temporal embeddings available — field='temporal' falling back to 'summary'. "
                "Temporal search requires V-JEPA 2 embeddings from indexing."
            )
            field = "summary"

        # Visual field
        if field == "visual":
            if (
                hasattr(index, "frame_embeddings")
                and index.frame_embeddings is not None
                and len(index.frame_embeddings) > 0
            ):
                visual_fn = getattr(index, "visual_embed_fn", None) or index.embed_fn
                query_emb = visual_fn(query)
                query_emb = np.asarray(query_emb).reshape(1, -1)
                scores = _search_matrix(query_emb, index.frame_embeddings)

                scores = scores.copy()
                for i, seg in enumerate(index.segments):
                    if seg.get("is_duplicate") or seg.get("is_semantic_duplicate"):
                        scores[i] = -np.inf

                top_indices = list(np.argsort(scores)[::-1][:top_k])
                results = []
                for idx in top_indices:
                    seg = index.segments[idx]
                    results.append(
                        {
                            "start_time": seg["start_time"],
                            "end_time": seg["end_time"],
                            "score": round(float(scores[idx]), 4),
                            "caption": seg.get("caption", ""),
                            "annotation": seg.get("annotation", {}),
                            "quality_score": seg.get("caption_quality_score"),
                        }
                    )
                return results
            import logging
            logging.getLogger(__name__).warning(
                "No frame embeddings available — field='visual' falling back to 'summary'. "
                "Visual search requires SigLIP2 frame embeddings from indexing."
            )
            field = "summary"

        summary_emb = index.embeddings
        action_emb = (
            index.action_embeddings if index.action_embeddings is not None else index.embeddings
        )
        temporal_emb = getattr(index, "temporal_embeddings", None)
        frame_emb = getattr(index, "frame_embeddings", None)

        if field == "summary":
            matrices = [summary_emb]
        elif field == "action":
            matrices = [action_emb]
        elif field == "all":
            # Weighted composite: summary 0.4, action 0.2, visual 0.2, temporal 0.2
            query_emb = index.embed_fn(query)
            query_emb = np.asarray(query_emb).reshape(1, -1)

            # SigLIP2 query for visual/temporal (different embedding space)
            visual_fn = getattr(index, "visual_embed_fn", None) or index.embed_fn
            query_emb_visual = visual_fn(query)
            query_emb_visual = np.asarray(query_emb_visual).reshape(1, -1)

            weighted_scores = np.zeros(len(index.segments))
            total_weight = 0.0

            if summary_emb is not None and len(summary_emb) > 0:
                weighted_scores += 0.4 * _search_matrix(query_emb, summary_emb)
                total_weight += 0.4
            if action_emb is not None and len(action_emb) > 0:
                weighted_scores += 0.2 * _search_matrix(query_emb, action_emb)
                total_weight += 0.2
            if frame_emb is not None and len(frame_emb) > 0:
                weighted_scores += 0.2 * _search_matrix(query_emb_visual, frame_emb)
                total_weight += 0.2
            if temporal_emb is not None and len(temporal_emb) > 0:
                # Align dims: SigLIP2 visual query (768-d) vs V-JEPA 2 temporal (1024-d).
                query_emb_temporal = _align_query_dim(query_emb_visual, temporal_emb)
                weighted_scores += 0.2 * _search_matrix(query_emb_temporal, temporal_emb)
                total_weight += 0.2

            if total_weight > 0:
                scores = weighted_scores / total_weight
            else:
                return []

            # Skip directly to filtering (bypass the max-over-matrices logic below)
            scores = scores.copy()
            for i, seg in enumerate(index.segments):
                if seg.get("is_duplicate"):
                    scores[i] = -np.inf

            if cluster_diverse and top_k > 2:
                valid_mask = scores > -np.inf
                valid_indices_arr = np.where(valid_mask)[0]
                n_valid = len(valid_indices_arr)

                if n_valid <= top_k:
                    top_indices = list(
                        valid_indices_arr[np.argsort(scores[valid_indices_arr])[::-1]]
                    )
                elif all("cluster_id" in index.segments[vi] for vi in valid_indices_arr):
                    # Use pre-computed cluster_ids from _semantic_deduplicate
                    clusters: dict[int, list[int]] = {}
                    for vi in valid_indices_arr:
                        cid = index.segments[vi].get("cluster_id", 0)
                        clusters.setdefault(int(cid), []).append(int(vi))
                    top_indices = _round_robin_from_clusters(clusters, scores, top_k)
                else:
                    from sklearn.cluster import KMeans

                    active_matrix = summary_emb if summary_emb is not None else frame_emb
                    k = min(top_k, n_valid)
                    valid_embs = active_matrix[valid_indices_arr]
                    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                    labels = kmeans.fit_predict(valid_embs)

                    clusters = {}
                    for vi, label in zip(valid_indices_arr, labels, strict=False):
                        clusters.setdefault(int(label), []).append(int(vi))
                    top_indices = _round_robin_from_clusters(clusters, scores, top_k)
            elif diverse and top_k > 1:
                n_candidates = min(top_k * 3, len(scores))
                candidate_indices = np.argsort(scores)[::-1][:n_candidates]
                candidate_scores = scores[candidate_indices]
                active_matrix = summary_emb if summary_emb is not None else frame_emb
                candidate_embs = (
                    active_matrix[candidate_indices] if active_matrix is not None else None
                )
                top_indices = _mmr_rerank(
                    query_emb,
                    candidate_embs,
                    candidate_indices,
                    candidate_scores,
                    top_k=top_k,
                    lambda_param=0.7,
                )
            else:
                top_indices = list(np.argsort(scores)[::-1][:top_k])

            results = []
            for idx in top_indices:
                seg = index.segments[idx]
                results.append(
                    {
                        "start_time": seg["start_time"],
                        "end_time": seg["end_time"],
                        "score": round(float(scores[idx]), 4),
                        "caption": seg.get("caption", ""),
                        "annotation": seg.get("annotation", {}),
                        "quality_score": seg.get("caption_quality_score"),
                    }
                )
            return results
        else:
            matrices = [summary_emb]

        matrices = [m for m in matrices if m is not None and len(m) > 0]
        if not matrices:
            return []

        query_emb = index.embed_fn(query)
        query_emb = np.asarray(query_emb).reshape(1, -1)

        all_scores = np.stack([_search_matrix(query_emb, m) for m in matrices])
        scores = np.max(all_scores, axis=0)

        if field == "action" and exclude_non_action:
            scores = scores.copy()
            for i, seg in enumerate(index.segments):
                if seg.get("is_non_action"):
                    scores[i] = -np.inf

        scores = scores.copy()
        for i, seg in enumerate(index.segments):
            if seg.get("is_duplicate") or seg.get("is_semantic_duplicate"):
                scores[i] = -np.inf

        if cluster_diverse and top_k > 2:
            valid_mask = scores > -np.inf
            valid_indices = np.where(valid_mask)[0]
            n_valid = len(valid_indices)

            if n_valid <= top_k:
                top_indices = list(valid_indices[np.argsort(scores[valid_indices])[::-1]])
            elif all("cluster_id" in index.segments[vi] for vi in valid_indices):
                # Use pre-computed cluster_ids from _semantic_deduplicate
                clusters: dict[int, list[int]] = {}
                for vi in valid_indices:
                    cid = index.segments[vi].get("cluster_id", 0)
                    clusters.setdefault(int(cid), []).append(int(vi))
                top_indices = _round_robin_from_clusters(clusters, scores, top_k)
            else:
                from sklearn.cluster import KMeans

                active_matrix = matrices[0]
                k = min(top_k, n_valid)
                valid_embs = active_matrix[valid_indices]
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(valid_embs)

                clusters = {}
                for vi, label in zip(valid_indices, labels, strict=False):
                    clusters.setdefault(int(label), []).append(int(vi))
                top_indices = _round_robin_from_clusters(clusters, scores, top_k)
        elif diverse and top_k > 1:
            n_candidates = min(top_k * 3, len(scores))
            candidate_indices = np.argsort(scores)[::-1][:n_candidates]
            candidate_scores = scores[candidate_indices]

            active_matrix = matrices[0]
            candidate_embs = active_matrix[candidate_indices] if active_matrix is not None else None

            top_indices = _mmr_rerank(
                query_emb,
                candidate_embs,
                candidate_indices,
                candidate_scores,
                top_k=top_k,
                lambda_param=0.7,
            )
        else:
            top_indices = list(np.argsort(scores)[::-1][:top_k])

        results = []
        for idx in top_indices:
            seg = index.segments[idx]
            results.append(
                {
                    "start_time": seg["start_time"],
                    "end_time": seg["end_time"],
                    "score": round(float(scores[idx]), 4),
                    "caption": seg.get("caption", ""),
                    "annotation": seg.get("annotation", {}),
                    "quality_score": seg.get("caption_quality_score"),
                }
            )
        return results

    return {
        "tool": search_video,
        "description": (
            "Semantic search over pre-indexed video segments. "
            "Parameters: query (str), top_k (int, default 5), "
            'field (str, default "summary" — can be "summary", "action", "visual", "temporal", or "all"), '
            "exclude_non_action (bool, default True — filters non-action segments when field is action), "
            "diverse (bool, default True — MMR reranking for varied results), "
            "cluster_diverse (bool, default False — KMeans clustering alternative to MMR), "
            "level (int, default 0 — higher levels search coarser hierarchy). "
            "Returns list of dicts with start_time, end_time, score, caption, and annotation."
        ),
    }


def make_search_transcript(index: VideoIndex) -> dict[str, Any]:
    """Keyword search over ASR transcript entries."""

    def search_transcript(query: str) -> list[dict[str, Any]]:
        """Search the video transcript for lines containing *query*."""
        if not index.transcript:
            return []

        query_lower = query.lower()
        results = []
        for i, entry in enumerate(index.transcript):
            if query_lower in entry["text"].lower():
                context_entries = index.transcript[max(0, i - 1) : i + 2]
                context = " ".join(e["text"] for e in context_entries)
                results.append(
                    {
                        "start_time": entry["start_time"],
                        "end_time": entry["end_time"],
                        "text": entry["text"],
                        "context": context,
                    }
                )
        return results

    return {
        "tool": search_transcript,
        "description": (
            "Search spoken words in the video transcript (ASR). "
            "Parameters: query (str — keyword or phrase, case-insensitive). "
            "Returns list of dicts with start_time, end_time, text, and surrounding context."
        ),
    }


def make_get_transcript(index: VideoIndex) -> dict[str, Any]:
    """Retrieve transcript text for a time range."""

    def get_transcript(start_time: float, end_time: float) -> str:
        """Return transcript text for a specific time range."""
        if not index.transcript:
            return ""

        lines = []
        for entry in index.transcript:
            if entry["end_time"] >= start_time and entry["start_time"] <= end_time:
                lines.append(f"[{entry['start_time']:.1f}s] {entry['text']}")
        return "\n".join(lines)

    return {
        "tool": get_transcript,
        "description": (
            "Get the spoken transcript for a specific time range of the video. "
            "Parameters: start_time (float, seconds), end_time (float, seconds). "
            "Returns concatenated transcript text as a string."
        ),
    }


def make_discriminative_vqa(index: VideoIndex) -> dict[str, Any]:
    """Embedding-based multiple-choice VQA without LLM generation."""
    from sklearn.metrics.pairwise import cosine_similarity

    def discriminative_vqa(
        question: str,
        candidates: list[str],
        time_range: tuple[float, float] | None = None,
    ) -> list[dict[str, Any]]:
        """Answer a multiple-choice question by embedding matching."""
        if not candidates or index.embeddings is None or index.embed_fn is None:
            return []

        candidate_embs = []
        for c in candidates:
            emb = index.embed_fn(f"{question} {c}")
            candidate_embs.append(emb)
        candidate_embs = np.stack(candidate_embs)

        seg_embs = index.embeddings
        seg_mask = np.ones(len(index.segments), dtype=bool)
        if time_range is not None:
            for i, seg in enumerate(index.segments):
                if seg["end_time"] < time_range[0] or seg["start_time"] > time_range[1]:
                    seg_mask[i] = False

        active_embs = seg_embs[seg_mask]
        if len(active_embs) == 0:
            return []

        sims = cosine_similarity(candidate_embs, active_embs)
        max_sims = sims.max(axis=1)
        best_seg_indices = sims.argmax(axis=1)

        active_indices = np.where(seg_mask)[0]

        results = []
        for i, candidate in enumerate(candidates):
            orig_idx = int(active_indices[best_seg_indices[i]])
            seg = index.segments[orig_idx]
            results.append(
                {
                    "answer": candidate,
                    "confidence": round(float(max_sims[i]), 4),
                    "best_segment": {
                        "start_time": seg["start_time"],
                        "end_time": seg["end_time"],
                        "caption": seg.get("caption", ""),
                    },
                }
            )

        results.sort(key=lambda x: x["confidence"], reverse=True)
        return results

    return {
        "tool": discriminative_vqa,
        "description": (
            "Answer a multiple-choice question about the video by embedding matching. "
            "Parameters: question (str), candidates (list of answer strings), "
            "time_range (optional tuple of start/end seconds). "
            "Returns sorted list of dicts with answer, confidence score, and best matching segment. "
            "Faster than LLM generation for closed-form questions."
        ),
    }


def make_get_scene_list(index: VideoIndex) -> dict[str, Any]:
    """Return scene boundaries with captions."""

    def get_scene_list() -> list[dict[str, Any]]:
        """List all detected scene boundaries with descriptions."""
        scenes = []
        for i, seg in enumerate(index.segments):
            scenes.append(
                {
                    "scene_index": i,
                    "start_time": seg["start_time"],
                    "end_time": seg["end_time"],
                    "caption": seg.get("caption", ""),
                    "annotation": seg.get("annotation", {}),
                }
            )
        return scenes

    return {
        "tool": get_scene_list,
        "description": (
            "List all detected scene boundaries. Takes no parameters. "
            "Returns list of dicts with scene_index, start_time, end_time, "
            "caption, and annotation. Use this to understand the video structure."
        ),
    }
