"""Tiered query pipeline: route -> execute -> escalate."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from kuavi.query_router import QueryRouter
from kuavi.tier_executors import execute_tier1, execute_tier2, execute_tier25


@dataclass
class TieredContext:
    index: Any
    video_path: str


async def load_or_build_index(
    video_path: str,
    index_mode: str,
    asr_model: str,
    force_reindex: bool = False,
):
    from kuavi.indexer import VideoIndexer
    from kuavi.loader import VideoLoader

    def _build_sync() -> TieredContext:
        loader = VideoLoader(fps=0.5)
        loaded = loader.load(video_path)

        indexer = VideoIndexer(
            scene_model="facebook/vjepa2-vitl-fpc64-256",
        )
        index = indexer.index_video(
            loaded,
            asr_model=asr_model,
            mode="full" if index_mode == "captioned" else "fast",
            force_reindex=force_reindex,
        )
        return TieredContext(index=index, video_path=video_path)

    return await asyncio.to_thread(_build_sync)


TIER_ORDER = [1, 2, 2.5]


async def run_tiered_pipeline(
    video_path: str,
    query: str,
    model: str,
    backend: str,
    index_mode: str = "fast",
    asr_model: str = "faster-whisper/base",
    force_reindex: bool = False,
    max_tier: float = 2.5,
) -> AsyncIterator[dict]:
    """
    Main pipeline. Yields SSE-compatible event dicts.

    Tier progression: 1 -> 2 -> 2.5

    Tier 1: V-JEPA probes + temporal grounding ($0)
    Tier 2: Gemini Embedding 2 similarity search ($0)
    Tier 2.5: V-JEPA temporal re-ranking + hierarchical search ($0)

    No LLM calls are made in any tier.
    """
    max_tier = max(1.0, min(2.5, float(max_tier)))

    router = QueryRouter()
    routing = router.classify(query)

    yield {
        "type": "routing",
        "tier": routing["tier"],
        "reason": routing["reason"],
        "tools": routing["suggested_tools"],
    }

    yield {"type": "step", "id": "index", "status": "running"}
    ctx = await load_or_build_index(
        video_path,
        index_mode,
        asr_model,
        force_reindex=force_reindex,
    )
    yield {"type": "step", "id": "index", "status": "done"}

    start_tier = min(routing["tier"], max_tier)
    allowed_tiers = [t for t in TIER_ORDER if start_tier <= t <= max_tier]

    tier2_result = None
    result: dict[str, Any] | None = None

    for current_tier in allowed_tiers:
        tier_id = f"tier_{current_tier}" if current_tier != 2.5 else "tier_2_5"
        yield {"type": "step", "id": tier_id, "status": "running"}

        if current_tier == 1:
            result = await execute_tier1(ctx, query, routing)
        elif current_tier == 2:
            result = await execute_tier2(ctx.index, query, routing)
            tier2_result = result
        elif current_tier == 2.5:
            if tier2_result is None:
                result = await execute_tier2(ctx.index, query, routing)
                tier2_result = result
            result = await execute_tier25(
                ctx,
                query,
                routing,
                tier2_result=tier2_result,
            )

        yield {
            "type": "step",
            "id": tier_id,
            "status": "done",
        }

        if result.get("escalate") and current_tier < max_tier:
            next_tiers = [t for t in TIER_ORDER if t > current_tier and t <= max_tier]
            if next_tiers:
                yield {
                    "type": "escalation",
                    "from_tier": current_tier,
                    "to_tier": next_tiers[0],
                    "reason": f"confidence too low: {result.get('confidence', '?')}",
                }
                continue

        yield {
            "type": "result",
            "answer": result["answer"],
            "timestamps": result.get("timestamps", []),
            "llm_calls": result.get("llm_calls", 0),
        }

        cost = {
            "type": "cost",
            "tier_used": current_tier,
            "llm_calls": result.get("llm_calls", 0),
            "estimated_usd": _estimate_cost(result),
        }
        yield cost

        _append_query_trace(
            query=query,
            routing=routing,
            tier_executed=current_tier,
            result=result,
            estimated_cost_usd=cost["estimated_usd"],
        )
        break


def _estimate_cost(result: dict) -> float:
    llm_calls = result.get("llm_calls", 0)
    if llm_calls == 0:
        return 0.0
    tokens = result.get("tokens_used", 10000)
    return round((tokens / 1_000_000) * 3.0, 6)


def _append_query_trace(
    query: str,
    routing: dict,
    tier_executed: float,
    result: dict,
    estimated_cost_usd: float,
) -> None:
    trace = {
        "query": query,
        "tier_classified": routing.get("tier"),
        "tier_executed": tier_executed,
        "escalated": tier_executed != routing.get("tier"),
        "escalation_reason": (
            f"confidence {result.get('confidence')} < threshold 0.6"
            if tier_executed != routing.get("tier")
            else ""
        ),
        "llm_calls": result.get("llm_calls", 0),
        "estimated_cost_usd": estimated_cost_usd,
        "tools_called": result.get("tools_called", routing.get("suggested_tools", [])),
        "answer_format": result.get("answer_format", routing.get("output_format", "text")),
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
    }

    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    out_file = log_dir / "kuavi_tiered_queries.jsonl"
    with open(out_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(trace, default=str) + "\n")

    try:
        from kuavi.mcp_server import log_tiered_query_trace

        log_tiered_query_trace(trace)
    except Exception:
        pass
