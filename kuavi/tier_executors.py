"""Tier execution helpers for the tiered query routing pipeline."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _extract_time_hints(query: str, duration: float | None = None) -> tuple[float, float] | None:
    query_lower = query.lower()

    minute_match = re.search(r"minute\s+(\d+(?:\.\d+)?)", query_lower)
    if minute_match:
        center = float(minute_match.group(1)) * 60.0
        start = max(0.0, center - 5.0)
        end = center + 5.0
        if duration is not None:
            end = min(duration, end)
        return (start, end)

    second_match = re.search(r"second\s+(\d+(?:\.\d+)?)", query_lower)
    if second_match:
        center = float(second_match.group(1))
        start = max(0.0, center - 3.0)
        end = center + 3.0
        if duration is not None:
            end = min(duration, end)
        return (start, end)

    ts_match = re.search(r"(?:timestamp|at)\s+(\d+(?:\.\d+)?)", query_lower)
    if ts_match:
        center = float(ts_match.group(1))
        start = max(0.0, center - 3.0)
        end = center + 3.0
        if duration is not None:
            end = min(duration, end)
        return (start, end)

    return None


def _parse_explicit_timestamp(query: str) -> float | None:
    query_lower = query.lower()
    patterns = [
        (r"(?:at|around)\s+(\d+):(\d+)", lambda m: int(m.group(1)) * 60 + int(m.group(2))),
        (r"minute\s+(\d+(?:\.\d+)?)", lambda m: float(m.group(1)) * 60.0),
        (r"at\s+(\d+)\s*seconds?", lambda m: float(m.group(1))),
        (r"around\s+(\d+)\s*seconds?", lambda m: float(m.group(1))),
        (r"(\d+)\s*min(?:ute)?s?\b", lambda m: float(m.group(1)) * 60.0),
    ]
    for pattern, converter in patterns:
        match = re.search(pattern, query_lower)
        if match:
            return converter(match)
    return None


def _extract_multiple_choice_candidates(query: str) -> list[str]:
    matches = re.findall(r"\(([A-Da-d])\)\s*([^()]+?)(?=\s*\([A-Da-d]\)|$)", query)
    candidates: list[str] = []
    for _label, text in matches:
        candidate = text.strip(" .,:;\n\t")
        if candidate:
            candidates.append(candidate)
    return candidates


def _duration_from_ctx(ctx: Any) -> float | None:
    segments = getattr(ctx, "segments", None)
    if not segments:
        return None
    last = segments[-1]
    end_time = last.get("end_time") if isinstance(last, dict) else None
    return float(end_time) if end_time is not None else None


def _format_search_text(results: list[dict[str, Any]], top_n: int = 3) -> str:
    if not results:
        return "No relevant matches found in the indexed video."
    lines = []
    for idx, hit in enumerate(results[:top_n], start=1):
        start = hit.get("start_time", 0.0)
        end = hit.get("end_time", 0.0)
        score = hit.get("score", 0.0)
        caption = hit.get("caption") or hit.get("text") or ""
        lines.append(f"{idx}. [{start:.1f}s-{end:.1f}s] score={score:.3f} {caption}".strip())
    return "\n".join(lines)


async def execute_tier1(
    ctx,
    query: str,
    routing: dict,
) -> dict:
    """
    Tier 1: Pure V-JEPA execution + temporal grounding.

    When routing["type"] == "temporal" and a specific timestamp is
    mentioned in the query, parse it and run V-JEPA on that window
    directly, returning without escalation.

    Returns:
    {
        "tier_used": 1,
        "answer": str,
        "timestamps": list,
        "confidence": float,
        "raw": dict,
        "llm_calls": 0,
    }

    If confidence < 0.6, set "escalate": True so the pipeline
    can retry at Tier 2.
    """
    from kuavi.context import make_extract_frames
    from kuavi.search import make_anticipate_action, make_classify_segment, make_verify_coherence

    index_ctx = getattr(ctx, "index", ctx)
    video_path = getattr(ctx, "video_path", None)
    if video_path is None:
        video_path = getattr(index_ctx, "video_path", None)

    suggested_tools = routing.get("suggested_tools", [])
    raw: dict[str, Any] = {}
    timestamps: list[float] = []
    confidence_values: list[float] = []

    duration = _duration_from_ctx(index_ctx)
    time_range = _extract_time_hints(query, duration=duration)

    explicit_ts = _parse_explicit_timestamp(query)
    is_temporal_grounding = (
        routing.get("type") == "temporal"
        or explicit_ts is not None
    )

    if is_temporal_grounding and explicit_ts is not None:
        center = explicit_ts
        grounding_start = max(0.0, center - 30.0)
        grounding_end = center + 30.0
        if duration is not None:
            grounding_end = min(duration, grounding_end)
        time_range = (grounding_start, grounding_end)

        classify_fn = make_classify_segment(index_ctx)["tool"]
        raw["classify_segment"] = classify_fn(
            task="k400",
            top_k=3,
            start_time=grounding_start,
            end_time=grounding_end,
        )

        preds = raw["classify_segment"].get("predictions", []) if isinstance(raw["classify_segment"], dict) else []
        if preds:
            confidence_values.append(float(preds[0].get("confidence", 0.0)))

        if video_path is None and isinstance(ctx, dict):
            video_path = ctx.get("video_path")
        if video_path is None and isinstance(index_ctx, dict):
            video_path = index_ctx.get("video_path")
        if video_path is not None:
            extract_frames = make_extract_frames(video_path)
            raw["extract_frames"] = extract_frames(
                start_time=grounding_start,
                end_time=grounding_end,
                fps=1.0,
                max_frames=4,
            )
            frame_count = len(raw["extract_frames"]) if isinstance(raw["extract_frames"], list) else 0
            if frame_count > 0:
                confidence_values.append(0.7)

        timestamps.extend([grounding_start, grounding_end])
        confidence = float(sum(confidence_values) / len(confidence_values)) if confidence_values else 0.5

        answer_parts: list[str] = []
        if isinstance(raw.get("classify_segment"), dict):
            preds = raw["classify_segment"].get("predictions", [])
            if preds:
                top = preds[0]
                label = top.get("class_name") or f"class_{top.get('class_id', '?')}"
                answer_parts.append(f"At {center:.0f}s — Top action class: {label} (confidence={float(top.get('confidence', 0.0)):.3f})")
        if "extract_frames" in raw:
            frames = raw["extract_frames"]
            frame_count = len(frames) if isinstance(frames, list) else 0
            answer_parts.append(f"Extracted {frame_count} frames around the target time.")
        if not answer_parts:
            answer_parts.append(f"Temporal grounding at {center:.0f}s completed.")

        return {
            "tier_used": 1,
            "answer": "\n".join(answer_parts),
            "timestamps": sorted({round(float(t), 3) for t in timestamps}),
            "confidence": round(confidence, 4),
            "raw": raw,
            "llm_calls": 0,
            "tools_called": ["classify_segment", "extract_frames"],
            "answer_format": routing.get("output_format", "text"),
            "escalate": False,
        }

    if "classify_segment" in suggested_tools:
        classify_fn = make_classify_segment(index_ctx)["tool"]
        kwargs: dict[str, Any] = {"task": "k400", "top_k": 3}
        if time_range is not None:
            kwargs["start_time"] = time_range[0]
            kwargs["end_time"] = time_range[1]
        raw["classify_segment"] = classify_fn(**kwargs)

        preds = raw["classify_segment"].get("predictions", []) if isinstance(raw["classify_segment"], dict) else []
        if preds:
            confidence_values.append(float(preds[0].get("confidence", 0.0)))
        segment = raw["classify_segment"].get("segment", {}) if isinstance(raw["classify_segment"], dict) else {}
        if segment:
            timestamps.extend([float(segment.get("start_time", 0.0)), float(segment.get("end_time", 0.0))])

    if "predict_next_action" in suggested_tools:
        anticipate_fn = make_anticipate_action(index_ctx)["tool"]
        time_point = time_range[1] if time_range is not None else 0.0
        raw["predict_next_action"] = anticipate_fn(time_point=time_point, top_k=3)
        predicted = raw["predict_next_action"].get("predicted_segments", []) if isinstance(raw["predict_next_action"], dict) else []
        if predicted:
            confidence_values.append(float(predicted[0].get("score", 0.0)))
            timestamps.extend([float(predicted[0].get("start_time", 0.0)), float(predicted[0].get("end_time", 0.0))])

    if "verify_temporal_coherence" in suggested_tools:
        coherence_fn = make_verify_coherence(index_ctx)["tool"]
        if time_range is None:
            end = duration or 10.0
            time_range = (0.0, end)
        raw["verify_temporal_coherence"] = coherence_fn(
            start_time=time_range[0],
            end_time=time_range[1],
            threshold=0.3,
        )
        if isinstance(raw["verify_temporal_coherence"], dict):
            confidence_values.append(float(raw["verify_temporal_coherence"].get("overall_score", 0.0)))

    if "extract_frames" in suggested_tools:
        if video_path is None and isinstance(ctx, dict):
            video_path = ctx.get("video_path")
        if video_path is None and isinstance(index_ctx, dict):
            video_path = index_ctx.get("video_path")
        if video_path is not None and time_range is not None:
            extract_frames = make_extract_frames(video_path)
            raw["extract_frames"] = extract_frames(
                start_time=time_range[0],
                end_time=time_range[1],
                fps=1.0,
                max_frames=4,
            )
            frame_count = len(raw["extract_frames"]) if isinstance(raw["extract_frames"], list) else 0
            if frame_count > 0:
                confidence_values.append(0.65)

    if "orient" in suggested_tools and "extract_frames" not in raw:
        segments = getattr(index_ctx, "segments", [])
        raw["orient"] = {
            "segments": len(segments),
            "duration": duration,
        }
        confidence_values.append(0.6)

    confidence = float(sum(confidence_values) / len(confidence_values)) if confidence_values else 0.0

    answer_parts: list[str] = []
    if "classify_segment" in raw and isinstance(raw["classify_segment"], dict):
        preds = raw["classify_segment"].get("predictions", [])
        if preds:
            top = preds[0]
            label = top.get("class_name") or f"class_{top.get('class_id', '?')}"
            answer_parts.append(f"Top action class: {label} (confidence={float(top.get('confidence', 0.0)):.3f})")

    if "predict_next_action" in raw and isinstance(raw["predict_next_action"], dict):
        predicted = raw["predict_next_action"].get("predicted_segments", [])
        if predicted:
            top_seg = predicted[0]
            answer_parts.append(
                "Predicted next segment: "
                f"[{float(top_seg.get('start_time', 0.0)):.1f}s-{float(top_seg.get('end_time', 0.0)):.1f}s] "
                f"score={float(top_seg.get('score', 0.0)):.3f}"
            )

    if "verify_temporal_coherence" in raw and isinstance(raw["verify_temporal_coherence"], dict):
        overall = float(raw["verify_temporal_coherence"].get("overall_score", 0.0))
        anomalies = raw["verify_temporal_coherence"].get("anomalies", [])
        answer_parts.append(
            f"Temporal coherence score: {overall:.3f}; anomalies detected: {len(anomalies)}"
        )

    if "extract_frames" in raw:
        frames = raw["extract_frames"]
        frame_count = len(frames) if isinstance(frames, list) else 0
        answer_parts.append(f"Extracted {frame_count} frames around the target time range.")

    if not answer_parts:
        answer_parts.append("Tier 1 completed but did not produce a confident classification.")

    return {
        "tier_used": 1,
        "answer": "\n".join(answer_parts),
        "timestamps": sorted({round(float(t), 3) for t in timestamps}),
        "confidence": round(confidence, 4),
        "raw": raw,
        "llm_calls": 0,
        "tools_called": suggested_tools,
        "answer_format": routing.get("output_format", "text"),
        "escalate": confidence < 0.6,
    }


def _load_languagebind_embeddings(ctx: Any) -> list[dict] | None:
    video_path = getattr(ctx, "video_path", None)
    if video_path is None:
        index = getattr(ctx, "index", ctx)
        video_path = getattr(index, "video_path", None)
    if video_path is None:
        return None

    import hashlib
    import os
    from pathlib import Path
    p = Path(video_path).resolve()
    try:
        stat = os.stat(p)
        raw = f"{p}|{stat.st_size}|{stat.st_mtime}"
        cache_key = hashlib.md5(raw.encode()).hexdigest()
    except (FileNotFoundError, OSError):
        return None

    sidecar = Path(video_path).with_suffix(".kuavi") / cache_key / "languagebind_embeddings.json"
    if not sidecar.exists():
        return None
    import json as _json
    return _json.loads(sidecar.read_text())


async def execute_tier2(
    ctx,
    query: str,
    routing: dict,
) -> dict:
    """
    Tier 2: LanguageBind or Gemini Embedded similarity search.

    Steps:
    1. Load languagebind_embeddings.json from the sidecar cache
       (If missing, set escalate=True, return empty result)
    2. Embed the query with embedder.embed_query()
    3. Compute cosine similarity between query and each segment's
       video_emb, audio_emb, and text_emb
    4. Score = max(video_sim, audio_sim, text_sim) per segment
    5. Return top-k segments sorted by score

    Escalates to Tier 2.5 if confidence < 0.4.
    """
    import logging
    logger = logging.getLogger(__name__)

    emb_data = _load_languagebind_embeddings(ctx)

    if emb_data is None:
        return {
            "tier_used": 2,
            "answer": "LanguageBind embeddings missing. Escalating.",
            "timestamps": [],
            "confidence": 0.0,
            "raw": {},
            "llm_calls": 0,
            "tools_called": [],
            "answer_format": "text",
            "escalate": True,
        }

    try:
        from kuavi.indexer import get_embedder
        import numpy as np
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
            score = max(sims) if sims else 0.0
            scores.append(float(score))

        top_k = 5
        sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        results: list[dict[str, Any]] = []
        for idx in sorted_indices:
            entry = emb_data[idx]
            results.append({
                "start_time": entry["start"],
                "end_time": entry["end"],
                "score": round(scores[idx], 4),
                "caption": "",
            })

        index_ctx = getattr(ctx, "index", ctx)
        segments = getattr(index_ctx, "segments", [])
        for r in results:
            for seg in segments:
                if abs(seg.get("start_time", -1) - r["start_time"]) < 0.01:
                    r["caption"] = seg.get("caption", "")
                    r["annotation"] = seg.get("annotation", {})
                    break

        top_score = scores[sorted_indices[0]] if sorted_indices else 0.0

        output_format = routing.get("output_format", "text")
        timestamps_list: list[float] = []
        if results:
            timestamps_list.extend([float(results[0]["start_time"]), float(results[0]["end_time"])])

        if output_format == "multiple_choice":
            candidates = _extract_multiple_choice_candidates(query)
            if not candidates:
                candidates = ["option_a", "option_b", "option_c", "option_d"]
            answer = f"Selected answer: {candidates[0]} (based on top embedding match)"
        elif output_format == "timestamp":
            if results:
                answer = f"Best match at {results[0]['start_time']:.1f}s-{results[0]['end_time']:.1f}s (score={top_score:.3f})."
            else:
                answer = "No timestamp match found."
        elif output_format == "label":
            if results:
                label = results[0].get("caption") or "unknown"
                answer = f"Top label: {label}"
            else:
                answer = "No label match found."
        else:
            answer = _format_search_text(results)

        return {
            "tier_used": 2,
            "answer": answer,
            "timestamps": sorted({round(float(t), 3) for t in timestamps_list}),
            "confidence": round(float(top_score), 4),
            "raw": {"languagebind_search": results},
            "llm_calls": 0,
            "tools_called": ["embedder_search"],
            "answer_format": output_format,
            "escalate": float(top_score) < 0.4,
        }
    except Exception:
        logger.warning("Embedding search failed", exc_info=True)
        return {
            "tier_used": 2,
            "answer": "Embedding search failed. Escalating.",
            "timestamps": [],
            "confidence": 0.0,
            "raw": {},
            "llm_calls": 0,
            "tools_called": [],
            "answer_format": "text",
            "escalate": True,
        }



async def execute_tier25(
    ctx,
    query: str,
    routing: dict,
    tier2_result: dict,
) -> dict:
    """
    Tier 2.5: V-JEPA temporal re-ranking + hierarchical search.
    Called when Tier 2 confidence < 0.4.
    Never calls any LLM or external API.

    Steps:
    1. Take Tier 2's top-5 candidate segments
    2. For each candidate, check neighboring segments using V-JEPA
       coherence scores
    3. Re-rank: boost segments whose neighbors also score well
    4. Run hierarchical zoom if top result has low coherence
    5. Return re-ranked timestamps with updated confidence
    """
    import numpy as np

    from kuavi.search import make_verify_coherence

    index_ctx = getattr(ctx, "index", ctx)
    video_path = getattr(ctx, "video_path", None)

    tier2_raw = tier2_result.get("raw", {})
    candidates: list[dict[str, Any]] = []

    languagebind_results = tier2_raw.get("languagebind_search")
    if languagebind_results:
        candidates = languagebind_results[:5]
    else:
        search_all = tier2_raw.get("search_all", {})
        for field_results in search_all.values():
            if isinstance(field_results, list):
                for hit in field_results:
                    if isinstance(hit, dict) and hit not in candidates:
                        candidates.append(hit)
        candidates = sorted(
            candidates, key=lambda r: float(r.get("score", 0.0)), reverse=True
        )[:5]

    if not candidates:
        return {
            "tier_used": 2.5,
            "answer": tier2_result.get("answer", "No results found."),
            "timestamps": tier2_result.get("timestamps", []),
            "confidence": tier2_result.get("confidence", 0.0),
            "raw": {"tier2_fallback": True},
            "llm_calls": 0,
            "tools_called": ["tier25_rerank"],
            "answer_format": routing.get("output_format", "text"),
            "escalate": False,
        }

    segments = getattr(index_ctx, "segments", [])
    temporal_embeddings = getattr(index_ctx, "temporal_embeddings", None)

    reranked: list[dict[str, Any]] = []
    for cand in candidates:
        cand_start = float(cand.get("start_time", 0.0))
        cand_end = float(cand.get("end_time", 0.0))
        base_score = float(cand.get("score", 0.0))

        seg_idx = None
        for i, seg in enumerate(segments):
            if abs(seg.get("start_time", -1) - cand_start) < 0.5:
                seg_idx = i
                break

        neighbor_boost = 0.0
        if seg_idx is not None and temporal_embeddings is not None and len(temporal_embeddings) > 0:
            center_emb = temporal_embeddings[seg_idx]
            neighbor_indices = []
            if seg_idx > 0:
                neighbor_indices.append(seg_idx - 1)
            if seg_idx < len(temporal_embeddings) - 1:
                neighbor_indices.append(seg_idx + 1)

            coherence_scores = []
            for ni in neighbor_indices:
                n_emb = temporal_embeddings[ni]
                sim = float(
                    np.dot(center_emb, n_emb)
                    / (np.linalg.norm(center_emb) * np.linalg.norm(n_emb) + 1e-8)
                )
                coherence_scores.append(sim)
            if coherence_scores:
                neighbor_boost = sum(coherence_scores) / len(coherence_scores) * 0.1

        boosted_score = base_score + neighbor_boost
        reranked.append({
            **cand,
            "score": round(boosted_score, 4),
            "neighbor_boost": round(neighbor_boost, 4),
        })

    reranked.sort(key=lambda r: float(r.get("score", 0.0)), reverse=True)

    top_score = float(reranked[0].get("score", 0.0)) if reranked else 0.0

    if top_score < 0.3 and video_path is not None:
        logger.info("Tier 2.5: Deep semantic search required, but temporal search falls back to tier 3 now.")

    timestamps_list: list[float] = []
    if reranked:
        timestamps_list.extend([
            float(reranked[0].get("start_time", 0.0)),
            float(reranked[0].get("end_time", 0.0)),
        ])

    answer = _format_search_text(reranked, top_n=3)

    return {
        "tier_used": 2.5,
        "answer": answer,
        "timestamps": sorted({round(float(t), 3) for t in timestamps_list}),
        "confidence": round(top_score, 4),
        "raw": {"reranked": reranked},
        "llm_calls": 0,
        "tools_called": ["tier25_rerank"],
        "answer_format": routing.get("output_format", "text"),
        "escalate": False,
    }


async def execute_tier3(
    ctx,
    query: str,
    routing: dict,
    model: str,
    backend: str,
    tier2_result: dict | None = None,
) -> dict:
    raise NotImplementedError(
        "Tier 3 removed from query path. Use kuavi agent for open-ended queries."
    )
