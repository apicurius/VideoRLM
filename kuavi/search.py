"""Search tool functions for KUAVi video indices.

Each ``make_*`` function accepts a :class:`~kuavi.indexer.VideoIndex`
and returns a dict ``{"tool": callable, "description": str}``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

try:
    from sklearn.metrics.pairwise import cosine_similarity as _sklearn_cosine_similarity
except ModuleNotFoundError:
    _sklearn_cosine_similarity = None

if TYPE_CHECKING:
    from kuavi.indexer import VideoIndex


def _align_query_dim(query_emb: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Align query embedding dimension to match the target matrix's column dimension.

    V-JEPA 2 temporal embeddings are 1024-d while LanguageBind text queries are 768-d.
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


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity with sklearn when available and numpy fallback otherwise."""
    if _sklearn_cosine_similarity is not None:
        return _sklearn_cosine_similarity(a, b)

    a_arr = np.asarray(a, dtype=np.float32)
    b_arr = np.asarray(b, dtype=np.float32)
    a_norm = np.linalg.norm(a_arr, axis=1, keepdims=True)
    b_norm = np.linalg.norm(b_arr, axis=1, keepdims=True)
    a_safe = a_arr / np.clip(a_norm, 1e-12, None)
    b_safe = b_arr / np.clip(b_norm, 1e-12, None)
    return a_safe @ b_safe.T


def _cluster_labels(matrix: np.ndarray, n_clusters: int) -> np.ndarray:
    """KMeans labels with deterministic fallback when sklearn is unavailable."""
    if len(matrix) == 0:
        return np.array([], dtype=int)
    try:
        from sklearn.cluster import KMeans

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        return kmeans.fit_predict(matrix)
    except ModuleNotFoundError:
        return np.arange(len(matrix), dtype=int) % max(1, n_clusters)


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


def search_by_embedding(query: str, index: VideoIndex, top_k: int = 5) -> list[dict[str, Any]]:
    """Search languagebind embeddings for a given query."""
    import logging
    import numpy as np
    from pathlib import Path
    import json
    import hashlib
    import os
    from kuavi.indexer import get_embedder

    logger = logging.getLogger(__name__)

    video_path = getattr(index, "video_path", None)
    if not video_path:
        return []

    p = Path(video_path).resolve()
    try:
        stat = os.stat(p)
        raw = f"{p}|{stat.st_size}|{stat.st_mtime}"
        cache_key = hashlib.md5(raw.encode()).hexdigest()
    except (FileNotFoundError, OSError):
        return []

    sidecar = Path(video_path).with_suffix(".kuavi") / cache_key / "languagebind_embeddings.json"
    if not sidecar.exists():
        return []

    emb_data = json.loads(sidecar.read_text())

    embedder = get_embedder()
    query_emb = np.array(embedder.embed_query(query))

    scores: list[float] = []
    for entry in emb_data:
        sims = []
        if entry.get("video_emb") is not None:
            sims.append(embedder.similarity(query_emb, entry["video_emb"]))
        if entry.get("audio_emb") is not None:
            sims.append(embedder.similarity(query_emb, entry["audio_emb"]))
        if entry.get("text_emb") is not None:
            sims.append(embedder.similarity(query_emb, entry["text_emb"]))
        scores.append(float(max(sims)) if sims else 0.0)

    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

    results = []
    for idx in top_indices:
        entry = emb_data[idx]
        res = {
            "start_time": entry["start"],
            "end_time": entry["end"],
            "score": round(scores[idx], 4),
            "caption": "",
            "annotation": {},
        }
        for seg in index.segments:
            if abs(seg.get("start_time", -1) - res["start_time"]) < 0.01:
                res["caption"] = seg.get("caption", "")
                res["annotation"] = seg.get("annotation", {})
                break
        results.append(res)

    return results


def make_search_video(index: VideoIndex) -> dict[str, Any]:
    """Search for specific content in the video using multimodal embeddings."""

    def search_video(
        query: str,
        top_k: int = 5,
        field: str = "summary",
        diverse: bool = True,
        cluster_diverse: bool = False,
        exclude_non_action: bool = True,
        level: int = 0,
    ) -> list[dict[str, Any]]:
        """Search the video for visual or audio events using LanguageBind."""
        # The unified embedding search uses video, audio, and text embeddings simultaneously.
        return search_by_embedding(query, index, top_k=top_k)

    return {
        "tool": search_video,
        "description": (
            "Search for specific visual or audio events in the video using multimodal embeddings. "
            "Works for complex descriptions (e.g. 'a person playing guitar'). "
            "Parameters: query (str), top_k (int, default 5). "
            "Returns list of matching segments with timestamps, scores, and captions."
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
                hit: dict[str, Any] = {
                    "start_time": entry["start_time"],
                    "end_time": entry["end_time"],
                    "text": entry["text"],
                    "context": context,
                }
                # Narrow to word-level timestamps for the matched span
                words = entry.get("words")
                if words:
                    matched = [
                        w for w in words if query_lower in w["text"].lower()
                    ]
                    if matched:
                        hit["word_start_time"] = matched[0]["start_time"]
                        hit["word_end_time"] = matched[-1]["end_time"]
                        hit["matched_words"] = matched
                results.append(hit)
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
                # Use word-level timestamps for tighter time display when available
                words = entry.get("words")
                if words:
                    relevant = [
                        w
                        for w in words
                        if w["end_time"] >= start_time and w["start_time"] <= end_time
                    ]
                    if relevant:
                        text = " ".join(w["text"] for w in relevant)
                        lines.append(f"[{relevant[0]['start_time']:.2f}s] {text}")
                        continue
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

        sims = _cosine_similarity(candidate_embs, active_embs)
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


def make_anticipate_action(index: VideoIndex) -> dict[str, Any]:
    """Predict what happens next after a given time point."""
    def anticipate_action(
        time_point: float,
        top_k: int = 3,
        candidates: list[str] | None = None,
    ) -> dict[str, Any]:
        """Predict what happens after the given time point.

        Args:
            time_point: Time in seconds to predict from.
            top_k: Number of candidate future segments to return.
            candidates: Optional list of action descriptions to rank.

        Returns:
            Dict with 'predicted_segments' (nearest future segments by embedding)
            and optionally 'candidate_ranking' (if candidates provided).
        """
        # Check if anticipation is available via predictor
        predict_fn = getattr(index, "_predict_fn", None)
        if predict_fn is None:
            # Fallback: use temporal proximity + embedding similarity
            # Find the segment at/just before time_point
            context_seg_idx = None
            for i, seg in enumerate(index.segments):
                if seg["start_time"] <= time_point <= seg["end_time"]:
                    context_seg_idx = i
                    break
                if seg["end_time"] <= time_point:
                    context_seg_idx = i

            if context_seg_idx is None:
                return {"error": "No segment found at the given time point", "predicted_segments": []}

            # Use embedding similarity to find what typically follows similar content
            if index.embeddings is None:
                return {"error": "No embeddings available", "predicted_segments": []}

            context_emb = index.embeddings[context_seg_idx].reshape(1, -1)

            # Only consider segments AFTER the context segment
            future_mask = np.array([
                seg["start_time"] > time_point for seg in index.segments
            ])

            if not future_mask.any():
                return {"predicted_segments": [], "note": "No future segments available"}

            scores = _cosine_similarity(context_emb, index.embeddings)[0]
            scores[~future_mask] = -np.inf
            scores[context_seg_idx] = -np.inf  # exclude self

            top_indices = np.argsort(scores)[::-1][:top_k]

            predicted = []
            for idx in top_indices:
                if scores[idx] <= -np.inf:
                    continue
                seg = index.segments[idx]
                predicted.append({
                    "start_time": seg["start_time"],
                    "end_time": seg["end_time"],
                    "score": round(float(scores[idx]), 4),
                    "caption": seg.get("caption", ""),
                    "annotation": seg.get("annotation", {}),
                })

            result: dict[str, Any] = {
                "context_segment": {
                    "start_time": index.segments[context_seg_idx]["start_time"],
                    "end_time": index.segments[context_seg_idx]["end_time"],
                    "caption": index.segments[context_seg_idx].get("caption", ""),
                },
                "predicted_segments": predicted,
                "method": "embedding_similarity",
                "note": "V-JEPA 2 predictor not available; using embedding similarity fallback",
            }

            # Rank candidates if provided
            if candidates and index.embed_fn is not None:
                candidate_embs = np.stack([
                    np.asarray(index.embed_fn(c)).flatten() for c in candidates
                ])
                # Score candidates against the mean of predicted segment embeddings
                if predicted:
                    pred_indices = [
                        i for i in top_indices if scores[i] > -np.inf
                    ][:top_k]
                    pred_embs = index.embeddings[pred_indices]
                    mean_pred = pred_embs.mean(axis=0).reshape(1, -1)
                    cand_scores = _cosine_similarity(candidate_embs, mean_pred).flatten()
                    ranking = sorted(
                        zip(candidates, cand_scores, strict=False),
                        key=lambda x: x[1], reverse=True,
                    )
                    result["candidate_ranking"] = [
                        {"action": a, "confidence": round(float(s), 4)}
                        for a, s in ranking
                    ]

            return result

        # Full predictor path (when V-JEPA 2 predictor is available)
        predicted_emb = predict_fn(time_point)
        if predicted_emb is None:
            return {"error": "Prediction failed", "predicted_segments": [], "method": "vjepa2_predictor"}

        # Find nearest segments to predicted embedding
        predicted_emb = predicted_emb.reshape(1, -1)
        if index.temporal_embeddings is not None:
            scores = _cosine_similarity(predicted_emb, index.temporal_embeddings)[0]
        elif index.embeddings is not None:
            scores = _cosine_similarity(predicted_emb, index.embeddings)[0]
        else:
            return {"error": "No embeddings available", "predicted_segments": []}

        top_indices = np.argsort(scores)[::-1][:top_k]
        predicted_segs = []
        for idx in top_indices:
            seg = index.segments[idx]
            predicted_segs.append({
                "start_time": seg["start_time"],
                "end_time": seg["end_time"],
                "score": round(float(scores[idx]), 4),
                "caption": seg.get("caption", ""),
            })

        return {
            "predicted_segments": predicted_segs,
            "method": "vjepa2_predictor",
        }

    return {
        "tool": anticipate_action,
        "description": (
            "Predict what happens next after a given time point in the video. "
            "Parameters: time_point (float, seconds), top_k (int, default 3), "
            "candidates (optional list of action descriptions to rank). "
            "Returns predicted future segments and optional candidate ranking."
        ),
    }


def make_predict_future(index: VideoIndex) -> dict[str, Any]:
    """Predict likely future content after a given time range (world model)."""
    def predict_future(
        start_time: float,
        end_time: float,
        n_future_tokens: int = 16,
    ) -> dict[str, Any]:
        """Predict what content is likely to follow a given time range.

        Args:
            start_time: Start of context window (seconds).
            end_time: End of context window (seconds).
            n_future_tokens: Number of future tokens for predictor (default 16).

        Returns:
            Dict with 'predicted_segments', 'method', and 'context'.
            method is 'vjepa2_predictor' or 'temporal_continuation'.
        """
        # Find segments overlapping the context window
        context_segs = [
            (i, seg)
            for i, seg in enumerate(index.segments)
            if seg["end_time"] > start_time and seg["start_time"] < end_time
        ]

        if not context_segs:
            return {
                "error": "No segments found in the given time range",
                "predicted_segments": [],
            }

        # Use the last segment in the range as primary context
        ctx_idx, ctx_seg = context_segs[-1]

        # Predictor path: use stored feature maps + _predict_future_fn
        predict_future_fn = getattr(index, "_predict_future_fn", None)
        if (
            predict_future_fn is not None
            and index.temporal_feature_maps is not None
            and ctx_idx < len(index.temporal_feature_maps)
        ):
            feature_map = index.temporal_feature_maps[ctx_idx]
            predicted_features = predict_future_fn(feature_map, n_future_tokens)
            if predicted_features is not None:
                # Mean-pool predicted features → single embedding
                predicted_emb = predicted_features.mean(axis=0).reshape(1, -1)
                emb_matrix = (
                    index.temporal_embeddings
                    if index.temporal_embeddings is not None
                    else index.embeddings
                )
                if emb_matrix is None:
                    return {"error": "No embeddings available", "predicted_segments": []}

                future_mask = np.array([
                    seg["start_time"] >= end_time for seg in index.segments
                ])
                if not future_mask.any():
                    return {
                        "predicted_segments": [],
                        "method": "vjepa2_predictor",
                        "context": {
                            "start_time": start_time,
                            "end_time": end_time,
                            "caption": ctx_seg.get("caption", ""),
                        },
                        "note": "No future segments available",
                    }

                scores = _cosine_similarity(predicted_emb, emb_matrix)[0]
                scores[~future_mask] = -np.inf

                top_indices = np.argsort(scores)[::-1][:5]
                predicted = []
                for idx in top_indices:
                    if scores[idx] <= -np.inf:
                        continue
                    seg = index.segments[idx]
                    predicted.append({
                        "start_time": seg["start_time"],
                        "end_time": seg["end_time"],
                        "score": round(float(scores[idx]), 4),
                        "caption": seg.get("caption", ""),
                        "annotation": seg.get("annotation", {}),
                    })

                return {
                    "predicted_segments": predicted,
                    "method": "vjepa2_predictor",
                    "context": {
                        "start_time": start_time,
                        "end_time": end_time,
                        "caption": ctx_seg.get("caption", ""),
                    },
                }

        # Fallback: temporal continuation heuristic
        # Weight future segments by cosine similarity AND temporal proximity
        emb_matrix = (
            index.temporal_embeddings
            if index.temporal_embeddings is not None
            else index.embeddings
        )
        if emb_matrix is None:
            return {"error": "No embeddings available", "predicted_segments": []}

        # Mean-pool all context segment embeddings
        ctx_indices = [i for i, _ in context_segs]
        ctx_emb = emb_matrix[ctx_indices].mean(axis=0).reshape(1, -1)

        future_segs = [
            (i, seg)
            for i, seg in enumerate(index.segments)
            if seg["start_time"] >= end_time
        ]
        if not future_segs:
            return {
                "predicted_segments": [],
                "method": "temporal_continuation",
                "context": {
                    "start_time": start_time,
                    "end_time": end_time,
                    "caption": ctx_seg.get("caption", ""),
                },
                "note": "No future segments available",
            }

        sim_scores = _cosine_similarity(ctx_emb, emb_matrix)[0]

        predicted = []
        for i, seg in future_segs:
            # Temporal proximity: exponential decay (30s half-life)
            temporal_distance = seg["start_time"] - end_time
            temporal_weight = float(np.exp(-temporal_distance / 30.0))
            combined_score = float(sim_scores[i]) * 0.5 + temporal_weight * 0.5
            predicted.append({
                "start_time": seg["start_time"],
                "end_time": seg["end_time"],
                "score": round(combined_score, 4),
                "caption": seg.get("caption", ""),
                "annotation": seg.get("annotation", {}),
            })

        predicted.sort(key=lambda x: x["score"], reverse=True)
        predicted = predicted[:5]

        return {
            "predicted_segments": predicted,
            "method": "temporal_continuation",
            "context": {
                "start_time": start_time,
                "end_time": end_time,
                "caption": ctx_seg.get("caption", ""),
            },
        }

    return {
        "tool": predict_future,
        "description": (
            "Predict what content is likely to follow a given time range (world model). "
            "Parameters: start_time (float, seconds), end_time (float, seconds), "
            "n_future_tokens (int, default 16, for V-JEPA 2 predictor). "
            "Returns predicted_segments (sorted by score), method "
            "('vjepa2_predictor' or 'temporal_continuation'), and context info."
        ),
    }


def make_verify_coherence(index: VideoIndex) -> dict[str, Any]:
    """Verify temporal coherence and detect anomalies in a video segment."""
    def verify_coherence(
        start_time: float,
        end_time: float,
        threshold: float = 0.3,
    ) -> dict[str, Any]:
        """Measure temporal coherence between consecutive segments and flag anomalies.

        Args:
            start_time: Start of analysis window (seconds).
            end_time: End of analysis window (seconds).
            threshold: Coherence score below which a transition is anomalous (default 0.3).

        Returns:
            Dict with overall_score, segment_scores, anomalies, and method.
            method is 'vjepa2_predictor' or 'pairwise_similarity'.
        """
        range_segs = [
            (i, seg)
            for i, seg in enumerate(index.segments)
            if seg["end_time"] > start_time and seg["start_time"] < end_time
        ]

        if len(range_segs) < 2:
            overall = 1.0 if len(range_segs) == 1 else 0.0
            return {
                "overall_score": overall,
                "segment_scores": [],
                "anomalies": [],
                "method": "pairwise_similarity",
                "note": "Not enough segments to compute coherence (need at least 2)",
            }

        emb_matrix = (
            index.temporal_embeddings
            if index.temporal_embeddings is not None
            else index.embeddings
        )
        if emb_matrix is None:
            return {"error": "No embeddings available"}

        predict_future_fn = getattr(index, "_predict_future_fn", None)
        method = "pairwise_similarity"
        segment_scores = []

        for k in range(len(range_segs) - 1):
            curr_idx, curr_seg = range_segs[k]
            next_idx, next_seg = range_segs[k + 1]

            score: float
            if (
                predict_future_fn is not None
                and index.temporal_feature_maps is not None
                and curr_idx < len(index.temporal_feature_maps)
            ):
                method = "vjepa2_predictor"
                feature_map = index.temporal_feature_maps[curr_idx]
                predicted_features = predict_future_fn(feature_map, 16)
                if predicted_features is not None:
                    predicted_emb = predicted_features.mean(axis=0).reshape(1, -1)
                    actual_emb = emb_matrix[next_idx].reshape(1, -1)
                    score = float(_cosine_similarity(predicted_emb, actual_emb)[0, 0])
                else:
                    curr_emb = emb_matrix[curr_idx].reshape(1, -1)
                    next_emb = emb_matrix[next_idx].reshape(1, -1)
                    score = float(_cosine_similarity(curr_emb, next_emb)[0, 0])
            else:
                curr_emb = emb_matrix[curr_idx].reshape(1, -1)
                next_emb = emb_matrix[next_idx].reshape(1, -1)
                score = float(_cosine_similarity(curr_emb, next_emb)[0, 0])

            is_anomalous = score < threshold
            segment_scores.append({
                "start": curr_seg["start_time"],
                "end": next_seg["end_time"],
                "score": round(score, 4),
                "is_anomalous": is_anomalous,
            })

        overall_score = float(np.mean([s["score"] for s in segment_scores]))
        anomalies = [
            {
                "start": s["start"],
                "end": s["end"],
                "score": s["score"],
                "description": (
                    f"Unexpected transition at t={s['start']:.1f}s-{s['end']:.1f}s "
                    f"(coherence: {s['score']:.3f})"
                ),
            }
            for s in segment_scores
            if s["is_anomalous"]
        ]

        return {
            "overall_score": round(overall_score, 4),
            "segment_scores": segment_scores,
            "anomalies": anomalies,
            "method": method,
        }

    return {
        "tool": verify_coherence,
        "description": (
            "Verify temporal coherence between consecutive segments and detect anomalies. "
            "Parameters: start_time (float, seconds), end_time (float, seconds), "
            "threshold (float, default 0.3 — scores below this are flagged as anomalous). "
            "Returns overall_score, segment_scores [{start, end, score, is_anomalous}], "
            "anomalies [{start, end, score, description}], and method."
        ),
    }


def make_classify_segment(index: VideoIndex) -> dict[str, Any]:
    """Classify video segments using attentive probes on V-JEPA 2 features."""

    def classify_segment(
        start_time: float | None = None,
        end_time: float | None = None,
        segment_index: int | None = None,
        task: str = "k400",
        top_k: int = 5,
    ) -> dict[str, Any]:
        """Classify a video segment using a pre-trained attentive probe.

        Args:
            start_time: Start time of segment (used with end_time).
            end_time: End time of segment (used with start_time).
            segment_index: Direct segment index (alternative to time range).
            task: Classification task name (e.g., "k400", "ssv2").
            top_k: Number of top predictions.

        Returns:
            Dict with predictions and segment info.
        """
        # Check if we have feature maps
        if index.temporal_feature_maps is None:
            return {
                "error": (
                    "No temporal feature maps available. "
                    "Re-index with store_feature_maps=True to enable classification."
                )
            }

        # Resolve segment index
        if segment_index is not None:
            if segment_index < 0 or segment_index >= len(index.segments):
                return {"error": f"Invalid segment_index {segment_index}"}
            seg_idx = segment_index
        elif start_time is not None and end_time is not None:
            # Find best matching segment by overlap
            seg_idx = None
            best_overlap = 0.0
            for i, seg in enumerate(index.segments):
                overlap = min(seg["end_time"], end_time) - max(seg["start_time"], start_time)
                if overlap > best_overlap:
                    best_overlap = overlap
                    seg_idx = i
            if seg_idx is None:
                return {"error": "No segment found in the given time range"}
        else:
            return {"error": "Provide either segment_index or (start_time, end_time)"}

        # Get feature map for segment
        features = index.temporal_feature_maps[seg_idx]  # (num_patches, D)

        # Get or create probe registry (cached on index object)
        registry = getattr(index, "_probe_registry", None)
        if registry is None:
            from kuavi.probes import ProbeRegistry

            registry = ProbeRegistry.from_configs()
            index._probe_registry = registry  # type: ignore[attr-defined]

        probe = registry.get(task)
        if probe is None:
            available = registry.available_tasks()
            return {"error": f"Unknown task '{task}'. Available: {available}"}

        # Classify
        predictions = probe.classify(features, top_k=top_k)

        seg = index.segments[seg_idx]
        return {
            "segment": {
                "index": seg_idx,
                "start_time": seg["start_time"],
                "end_time": seg["end_time"],
                "caption": seg.get("caption", ""),
            },
            "task": task,
            "task_description": probe.config.description,
            "predictions": predictions,
            "weights_loaded": probe._model is not None and probe.config.weights_path is not None,
        }

    return {
        "tool": classify_segment,
        "description": (
            "Classify a video segment using attentive probes on V-JEPA 2 features. "
            "Parameters: start_time (float), end_time (float), OR segment_index (int); "
            "task (str, default 'k400' — options: ssv2, k400, diving48, jester, coin, imagenet); "
            "top_k (int, default 5). "
            "Requires store_feature_maps=True during indexing."
        ),
    }
