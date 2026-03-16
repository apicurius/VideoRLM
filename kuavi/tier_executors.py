"""Tier execution helpers for the 3-tier query routing pipeline."""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any


def _extract_time_hints(query: str, duration: float | None = None) -> tuple[float, float] | None:
    """Extract an approximate time range from a query string."""
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


def _extract_multiple_choice_candidates(query: str) -> list[str]:
    """Extract (A)/(B)/(C)/(D) style candidates from a query."""
    matches = re.findall(r"\(([A-Da-d])\)\s*([^()]+?)(?=\s*\([A-Da-d]\)|$)", query)
    candidates: list[str] = []
    for _label, text in matches:
        candidate = text.strip(" .,:;\n\t")
        if candidate:
            candidates.append(candidate)
    return candidates


def _duration_from_ctx(ctx: Any) -> float | None:
    """Best-effort duration extraction from index context."""
    segments = getattr(ctx, "segments", None)
    if not segments:
        return None
    last = segments[-1]
    end_time = last.get("end_time") if isinstance(last, dict) else None
    return float(end_time) if end_time is not None else None


def _format_search_text(results: list[dict[str, Any]], top_n: int = 3) -> str:
    """Format top search hits as plain structured text without LLM."""
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
    Tier 1: Pure V-JEPA execution.
    Calls classify_segment, predict_next_action, verify_temporal_coherence,
    or extract_frames depending on routing["suggested_tools"].

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
        # ctx may be a dict-like wrapper with video_path
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


async def execute_tier2(
    ctx,
    query: str,
    routing: dict,
) -> dict:
    """
    Tier 2: Embedding search + optional discriminative VQA.
    No LLM generation — only retrieval and multiple-choice selection.

    Steps:
    1. Call search_all(query) or search_transcript(query)
    2. If output_format == "multiple_choice": call discriminative_vqa
       on top result segments
    3. If output_format == "timestamp": return top result timestamps
    4. If output_format == "label": return top classification label
    5. If output_format == "text": format top 3 search results as
       a structured text answer WITHOUT calling an LLM

    Returns same shape as execute_tier1 but with llm_calls: 0.
    If top search result score < 0.5, set "escalate": True.
    """
    from kuavi.search import make_discriminative_vqa, make_search_transcript, make_search_video

    output_format = routing.get("output_format", "text")
    suggested_tools = routing.get("suggested_tools", [])

    search_video = make_search_video(ctx)["tool"]
    search_transcript = make_search_transcript(ctx)["tool"]
    vqa = make_discriminative_vqa(ctx)["tool"]

    raw: dict[str, Any] = {}
    timestamps: list[float] = []

    if "search_transcript" in suggested_tools and "search_all" not in suggested_tools:
        transcript_results = search_transcript(query=query)
        raw["search_transcript"] = transcript_results
        top_results = transcript_results
        top_score = 0.7 if transcript_results else 0.0
    else:
        # Structured "search_all" without LLM: query across key fields.
        fields = ["summary", "action", "visual"]
        raw["search_all"] = {}
        merged: list[dict[str, Any]] = []
        for field in fields:
            field_results = search_video(query=query, top_k=5, field=field)
            raw["search_all"][field] = field_results
            for hit in field_results:
                if isinstance(hit, dict):
                    merged.append({**hit, "field": field})

        # Add transcript retrieval in parallel path
        transcript_results = search_transcript(query=query)
        raw["search_all"]["transcript"] = transcript_results

        merged.sort(key=lambda r: float(r.get("score", 0.0)), reverse=True)
        top_results = merged
        top_score = float(top_results[0].get("score", 0.0)) if top_results else 0.0

    answer = ""
    if output_format == "multiple_choice":
        candidates = _extract_multiple_choice_candidates(query)
        if not candidates:
            # fallback candidates for "which of the following" style without explicit labels
            candidates = ["option_a", "option_b", "option_c", "option_d"]

        time_range = None
        if top_results and isinstance(top_results[0], dict):
            start = top_results[0].get("start_time")
            end = top_results[0].get("end_time")
            if start is not None and end is not None:
                time_range = (float(start), float(end))
                timestamps.extend([float(start), float(end)])

        raw["discriminative_vqa"] = vqa(question=query, candidates=candidates, time_range=time_range)
        ranked = raw["discriminative_vqa"]
        if ranked:
            top = ranked[0]
            answer = f"Selected answer: {top.get('answer')} (confidence={float(top.get('confidence', 0.0)):.3f})"
            confidence = float(top.get("confidence", top_score))
        else:
            answer = "Could not rank candidates from retrieved segments."
            confidence = top_score

    elif output_format == "timestamp":
        if top_results:
            top = top_results[0]
            start = float(top.get("start_time", 0.0))
            end = float(top.get("end_time", 0.0))
            timestamps.extend([start, end])
            answer = f"Best match at {start:.1f}s-{end:.1f}s (score={float(top.get('score', top_score)):.3f})."
        else:
            answer = "No timestamp match found."
        confidence = top_score

    elif output_format == "label":
        if top_results:
            top = top_results[0]
            label = top.get("caption") or top.get("annotation", {}).get("summary", {}).get("brief") or "unknown"
            answer = f"Top label: {label}"
            start = top.get("start_time")
            end = top.get("end_time")
            if start is not None and end is not None:
                timestamps.extend([float(start), float(end)])
        else:
            answer = "No label match found."
        confidence = top_score

    else:
        answer = _format_search_text(top_results)
        if top_results:
            first = top_results[0]
            start = first.get("start_time")
            end = first.get("end_time")
            if start is not None and end is not None:
                timestamps.extend([float(start), float(end)])
        confidence = top_score

    return {
        "tier_used": 2,
        "answer": answer,
        "timestamps": sorted({round(float(t), 3) for t in timestamps}),
        "confidence": round(float(confidence), 4),
        "raw": raw,
        "llm_calls": 0,
        "tools_called": suggested_tools or ["search_all"],
        "answer_format": output_format,
        "escalate": float(confidence) < 0.5,
    }


async def execute_tier3(
    ctx,
    query: str,
    routing: dict,
    model: str,
    backend: str,
    tier2_result: dict | None = None,
) -> dict:
    """
    Tier 3: Full LLM agent. Last resort only.

    If tier2_result is provided (escalation path), prepend its
    search results to the LLM context so it doesn't start blind.
    This reduces token usage even in the LLM path.

    Returns same shape but with llm_calls: int (actual count).
    """
    from kuavi.agent_runner import run_agent

    video_path = getattr(ctx, "video_path", None)
    if video_path is None and isinstance(ctx, dict):
        video_path = ctx.get("video_path")
    index = getattr(ctx, "index", None)
    if index is None and not isinstance(ctx, dict):
        index = ctx
    if isinstance(ctx, dict) and index is None:
        index = ctx.get("index")

    if video_path is None or index is None:
        return {
            "tier_used": 3,
            "answer": "Tier 3 execution failed: missing video context.",
            "timestamps": [],
            "confidence": 0.0,
            "raw": {"error": "missing_context"},
            "llm_calls": 0,
            "tools_called": ["full_agent"],
            "answer_format": "text",
            "escalate": False,
        }

    prompt = query
    if tier2_result is not None:
        tier2_context = {
            "tier2_answer": tier2_result.get("answer"),
            "tier2_timestamps": tier2_result.get("timestamps", []),
            "tier2_confidence": tier2_result.get("confidence", 0.0),
            "tier2_raw": tier2_result.get("raw", {}),
        }
        prompt = (
            "Use these retrieval results as grounding context before further reasoning:\n"
            f"{json.dumps(tier2_context, default=str)[:4000]}\n\n"
            f"User query: {query}"
        )

    result_answer = ""
    llm_calls = 0
    raw_events: list[dict[str, Any]] = []

    def _run_sync() -> tuple[str, int, list[dict[str, Any]]]:
        answer = ""
        calls = 0
        events: list[dict[str, Any]] = []
        for event in run_agent(
            video_path=video_path,
            question=prompt,
            model=model,
            backend=backend,
            index=index,
        ):
            events.append(event)
            if event.get("type") == "iteration":
                calls += 1
            if event.get("type") == "result":
                answer = event.get("answer", "")
        return answer, calls, events

    result_answer, llm_calls, raw_events = await asyncio.to_thread(_run_sync)

    return {
        "tier_used": 3,
        "answer": result_answer,
        "timestamps": [],
        "confidence": 1.0 if result_answer else 0.0,
        "raw": {"events": raw_events},
        "llm_calls": llm_calls,
        "tools_called": ["full_agent"],
        "answer_format": routing.get("output_format", "text"),
        "escalate": False,
    }
