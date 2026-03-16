from __future__ import annotations

import asyncio

from kuavi.query_router import QueryRouter
from kuavi.tiered_pipeline import run_tiered_pipeline


def test_tier1_classifications() -> None:
    router = QueryRouter()
    assert router.classify("what action is happening at minute 2")["tier"] == 1
    assert router.classify("what sport is this")["tier"] == 1
    assert router.classify("predict the next action")["tier"] == 1
    assert router.classify("is this motion natural")["tier"] == 1


def test_tier2_classifications() -> None:
    router = QueryRouter()
    assert router.classify("find the scene where someone falls")["tier"] == 2
    assert router.classify("when does the music start")["tier"] == 2
    assert router.classify("what is said at the beginning")["tier"] == 2
    assert router.classify("(A) running (B) walking (C) sitting")["tier"] == 2


def test_tier3_classifications() -> None:
    router = QueryRouter()
    assert router.classify("summarize the entire video")["tier"] == 3
    assert router.classify("why did the person leave")["tier"] == 3
    assert router.classify("explain what happened")["tier"] == 3


def test_output_format_inference() -> None:
    router = QueryRouter()
    assert router.classify("at what timestamp does X happen")["output_format"] == "timestamp"
    assert router.classify("which of the following")["output_format"] == "multiple_choice"
    assert router.classify("what action is this")["output_format"] == "label"


def test_pipeline_escalates_to_tier2(monkeypatch) -> None:
    async def fake_load_or_build_index(video_path: str, index_mode: str, asr_model: str):
        return type("Ctx", (), {"index": object(), "video_path": video_path})()

    async def fake_execute_tier1(ctx, query: str, routing: dict) -> dict:
        return {
            "tier_used": 1,
            "answer": "tier1 low confidence",
            "timestamps": [],
            "confidence": 0.4,
            "raw": {},
            "llm_calls": 0,
            "escalate": True,
        }

    async def fake_execute_tier2(ctx, query: str, routing: dict) -> dict:
        return {
            "tier_used": 2,
            "answer": "tier2 answer",
            "timestamps": [12.0, 15.0],
            "confidence": 0.9,
            "raw": {},
            "llm_calls": 0,
            "escalate": False,
        }

    monkeypatch.setattr("kuavi.tiered_pipeline.load_or_build_index", fake_load_or_build_index)
    monkeypatch.setattr("kuavi.tiered_pipeline.execute_tier1", fake_execute_tier1)
    monkeypatch.setattr("kuavi.tiered_pipeline.execute_tier2", fake_execute_tier2)

    async def _collect() -> list[dict]:
        events: list[dict] = []
        async for event in run_tiered_pipeline(
            video_path="dummy.mp4",
            query="what action is happening at minute 2",
            model="dummy-model",
            backend="dummy-backend",
            max_tier=2,
        ):
            events.append(event)
        return events

    events = asyncio.run(_collect())
    escalation_events = [e for e in events if e.get("type") == "escalation"]
    assert escalation_events
    assert escalation_events[0]["from_tier"] == 1
    assert escalation_events[0]["to_tier"] == 2
