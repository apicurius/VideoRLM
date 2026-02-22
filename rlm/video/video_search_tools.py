"""Factory functions that create REPL-injectable search tool closures.

Each ``make_*`` function accepts a :class:`~rlm.video.video_indexer.VideoIndex`
and returns a dict ``{"tool": callable, "description": str}`` compatible with
the custom-tools interface used by :class:`~rlm.video.video_rlm.VideoRLM`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from rlm.video.video_indexer import VideoIndex


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
    """Max-Marginal Relevance reranking for diverse search results.

    Balances relevance to query (lambda_param) with diversity among
    selected results (1 - lambda_param).

    Args:
        query_emb: (1, D) query embedding.
        candidate_embs: (N, D) candidate embeddings.
        candidate_indices: Original indices of candidates.
        scores: Relevance scores for candidates.
        top_k: Number of results to return.
        lambda_param: Trade-off between relevance (1.0) and diversity (0.0).

    Returns:
        List of original indices selected by MMR.
    """
    if len(candidate_indices) <= top_k:
        return list(candidate_indices[np.argsort(scores)[::-1]])

    selected = []
    remaining = list(range(len(candidate_indices)))

    # Start with the most relevant candidate
    first = int(np.argmax(scores))
    selected.append(first)
    remaining.remove(first)

    for _ in range(top_k - 1):
        if not remaining:
            break

        best_score = -np.inf
        best_idx = remaining[0]

        for idx in remaining:
            # Relevance to query
            relevance = scores[idx]

            # Max similarity to already selected items
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


def make_search_video(index: VideoIndex) -> dict[str, Any]:
    """Semantic search over video segment embeddings.

    Embeds the query text with the same model used during indexing and returns
    the *top_k* most similar segments by cosine similarity.  Supports
    field-targeted search over summary or action embeddings.
    """
    from sklearn.metrics.pairwise import cosine_similarity

    def _search_matrix(
        query_emb: np.ndarray,
        matrix: np.ndarray,
    ) -> np.ndarray:
        """Return per-segment cosine similarity scores."""
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
        """Search video segments by semantic similarity to *query*.

        Args:
            query: Natural-language description of what to find.
            top_k: Number of results to return.
            field: Embedding field to search against.

                - ``"summary"`` (default): search summary.brief embeddings.
                - ``"action"``: search action.brief embeddings.
                - ``"visual"``: search frame embeddings directly (bypasses caption quality).
                - ``"all"``: search both and merge by max score.
            exclude_non_action: When True and field is ``"action"``, segments
                marked ``is_non_action`` (action.brief is empty or ``"N/A"``)
                are excluded from results. Default True.
            diverse: When True, use Max-Marginal Relevance (MMR) to return
                diverse results. Default True.
            cluster_diverse: When True and top_k > 2, use KMeans clustering
                on segment embeddings and round-robin pick from each cluster
                to ensure global semantic diversity. This runs instead of MMR
                when enabled. Default False.
            level: Hierarchy level to search. 0 (default) searches the
                primary fine-grained segments. Higher levels search
                progressively coarser segments from the hierarchy when
                available.

        Returns:
            List of dicts with ``start_time``, ``end_time``, ``score``,
            ``caption``, and ``annotation`` for each matching segment,
            sorted by descending score.
        """
        # Hierarchy level search: redirect to coarser level segments
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
                    if seg.get("is_duplicate"):
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

        # Visual field: search frame embeddings using SigLIP2 text encoder
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

                # Suppress duplicates
                scores = scores.copy()
                for i, seg in enumerate(index.segments):
                    if seg.get("is_duplicate"):
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
            # Fallback to summary embeddings if no frame embeddings
            import logging
            logging.getLogger(__name__).warning(
                "No frame embeddings available — field='visual' falling back to 'summary'. "
                "Visual search requires SigLIP2 frame embeddings from indexing."
            )
            field = "summary"

        # Resolve which embedding matrices to use
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
                from sklearn.cluster import KMeans

                active_matrix = summary_emb if summary_emb is not None else frame_emb
                valid_mask = scores > -np.inf
                valid_indices_arr = np.where(valid_mask)[0]
                n_valid = len(valid_indices_arr)

                if n_valid <= top_k:
                    top_indices = list(
                        valid_indices_arr[np.argsort(scores[valid_indices_arr])[::-1]]
                    )
                else:
                    k = min(top_k, n_valid)
                    valid_embs = active_matrix[valid_indices_arr]
                    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                    labels = kmeans.fit_predict(valid_embs)

                    clusters: dict[int, list[int]] = {}
                    for vi, label in zip(valid_indices_arr, labels, strict=False):
                        clusters.setdefault(int(label), []).append(int(vi))
                    for label in clusters:
                        clusters[label].sort(key=lambda idx: scores[idx], reverse=True)
                    top_indices: list[int] = []
                    cluster_keys = sorted(
                        clusters.keys(), key=lambda k: scores[clusters[k][0]], reverse=True
                    )
                    cluster_ptrs = {k: 0 for k in cluster_keys}
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

        # Filter out None matrices
        matrices = [m for m in matrices if m is not None and len(m) > 0]
        if not matrices:
            return []

        query_emb = index.embed_fn(query)
        query_emb = np.asarray(query_emb).reshape(1, -1)

        # Compute scores across all requested matrices, take max per segment
        all_scores = np.stack([_search_matrix(query_emb, m) for m in matrices])
        scores = np.max(all_scores, axis=0)

        # Suppress non-action segments when searching by action field
        if field == "action" and exclude_non_action:
            scores = scores.copy()
            for i, seg in enumerate(index.segments):
                if seg.get("is_non_action"):
                    scores[i] = -np.inf

        # Suppress near-duplicate segments
        scores = scores.copy()
        for i, seg in enumerate(index.segments):
            if seg.get("is_duplicate"):
                scores[i] = -np.inf

        if cluster_diverse and top_k > 2:
            # Cluster-aware diverse selection using KMeans
            from sklearn.cluster import KMeans

            active_matrix = matrices[0]
            # Only cluster over valid (non-suppressed) segments
            valid_mask = scores > -np.inf
            valid_indices = np.where(valid_mask)[0]
            n_valid = len(valid_indices)

            if n_valid <= top_k:
                top_indices = list(valid_indices[np.argsort(scores[valid_indices])[::-1]])
            else:
                k = min(top_k, n_valid)
                valid_embs = active_matrix[valid_indices]
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(valid_embs)

                # Group valid indices by cluster label
                clusters: dict[int, list[int]] = {}
                for vi, label in zip(valid_indices, labels, strict=False):
                    clusters.setdefault(int(label), []).append(int(vi))

                # Sort each cluster's members by descending score
                for label in clusters:
                    clusters[label].sort(key=lambda idx: scores[idx], reverse=True)

                # Round-robin pick from clusters
                top_indices: list[int] = []
                cluster_keys = sorted(
                    clusters.keys(), key=lambda k: scores[clusters[k][0]], reverse=True
                )
                cluster_ptrs = {k: 0 for k in cluster_keys}
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
        elif diverse and top_k > 1:
            # Use MMR for diverse results — gather 3x candidates then rerank
            n_candidates = min(top_k * 3, len(scores))
            candidate_indices = np.argsort(scores)[::-1][:n_candidates]
            candidate_scores = scores[candidate_indices]

            # Use primary matrix embeddings for diversity computation
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
        """Search the video transcript for lines containing *query*.

        Args:
            query: Keyword or phrase to search for (case-insensitive).

        Returns:
            List of dicts with ``start_time``, ``end_time``, ``text``, and
            surrounding ``context`` for each match.
        """
        if not index.transcript:
            return []

        query_lower = query.lower()
        results = []
        for i, entry in enumerate(index.transcript):
            if query_lower in entry["text"].lower():
                # Build context from surrounding entries
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
        """Return transcript text for a specific time range.

        Args:
            start_time: Start of the range in seconds.
            end_time: End of the range in seconds.

        Returns:
            Concatenated transcript text for the given time range.
        """
        if not index.transcript:
            return ""

        lines = []
        for entry in index.transcript:
            # Include entries that overlap with the requested range
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

        # Encode each candidate as "question candidate"
        candidate_embs = []
        for c in candidates:
            emb = index.embed_fn(f"{question} {c}")
            candidate_embs.append(emb)
        candidate_embs = np.stack(candidate_embs)  # (C, D)

        # Get segment embeddings, optionally filtered by time range
        seg_embs = index.embeddings  # (S, D)
        seg_mask = np.ones(len(index.segments), dtype=bool)
        if time_range is not None:
            for i, seg in enumerate(index.segments):
                if seg["end_time"] < time_range[0] or seg["start_time"] > time_range[1]:
                    seg_mask[i] = False

        active_embs = seg_embs[seg_mask]
        if len(active_embs) == 0:
            return []

        # For each candidate, compute max cosine similarity against active segments
        sims = cosine_similarity(candidate_embs, active_embs)  # (C, S')
        max_sims = sims.max(axis=1)  # (C,)
        best_seg_indices = sims.argmax(axis=1)  # (C,)

        # Map back to original segment indices
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
        """List all detected scene boundaries with descriptions.

        Returns:
            List of dicts with ``scene_index``, ``start_time``, ``end_time``,
            ``caption``, and ``annotation`` for each scene.
        """
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
