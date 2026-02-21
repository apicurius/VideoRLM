"""Tests for kuavi_analyze_shards MCP tool."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from kuavi.mcp_server import _state, kuavi_analyze_shards


def _make_segments(ranges: list[tuple[float, float]]) -> list[dict]:
    """Create minimal segment dicts for testing."""
    return [
        {
            "segment_index": i,
            "start_time": start,
            "end_time": end,
            "annotation": {"summary": {"brief": f"segment {i} content"}},
        }
        for i, (start, end) in enumerate(ranges)
    ]


def _setup_index(segments: list[dict], video_id: str = "test") -> None:
    """Set up _state with a mock index containing given segments."""
    mock_index = SimpleNamespace(segments=segments)
    _state["videos"] = {video_id: {"index": mock_index, "video_path": "test.mp4"}}
    _state["active_video"] = video_id


class TestAnalyzeShards:
    def setup_method(self):
        _state["videos"] = {}
        _state["active_video"] = None
        _state["stats"] = {
            "tool_calls": 0,
            "frames_extracted": 0,
            "searches_performed": 0,
            "session_start": None,
        }

    def test_no_index_error(self):
        result = kuavi_analyze_shards("test question")
        assert "error" in result
        assert "No video indexed" in result["error"]

    def test_no_segments_error(self):
        _setup_index([])
        result = kuavi_analyze_shards("test question")
        assert "error" in result
        assert "No segments" in result["error"]

    @patch("kuavi.mcp_server._call_llm", return_value="shard answer")
    def test_shard_splitting(self, mock_llm):
        # Segments spanning 0-90s, shard_duration=30 → 3 shards
        segments = _make_segments([(0, 10), (15, 25), (30, 45), (50, 60), (60, 80), (85, 90)])
        _setup_index(segments)

        result = kuavi_analyze_shards("what happens?", shard_duration=30.0)

        assert result["shard_count"] == 3
        assert len(result["results"]) == 3
        # LLM called once per shard
        assert mock_llm.call_count == 3

    @patch("kuavi.mcp_server._call_llm", return_value="answer")
    def test_result_structure(self, mock_llm):
        segments = _make_segments([(0, 10), (10, 20)])
        _setup_index(segments)

        result = kuavi_analyze_shards("describe the video", shard_duration=30.0)

        assert result["question"] == "describe the video"
        assert result["shard_count"] == 1
        assert len(result["results"]) == 1

        shard_result = result["results"][0]
        assert shard_result["shard_index"] == 0
        assert "start_time" in shard_result
        assert "end_time" in shard_result
        assert shard_result["answer"] == "answer"

    @patch("kuavi.mcp_server._call_llm", return_value="limited answer")
    def test_max_shards_limit(self, mock_llm):
        # Create many segments across a wide range
        segments = _make_segments([(i * 10, i * 10 + 8) for i in range(20)])
        _setup_index(segments)

        result = kuavi_analyze_shards("question", shard_duration=10.0, max_shards=2)

        assert result["shard_count"] == 2
        assert len(result["results"]) == 2
        assert mock_llm.call_count == 2

    @patch("kuavi.mcp_server._call_llm")
    def test_shard_error_handling(self, mock_llm):
        # First call succeeds, second raises
        mock_llm.side_effect = [
            "good answer",
            RuntimeError("API error"),
            "another good answer",
        ]
        segments = _make_segments([(0, 10), (30, 40), (60, 70)])
        _setup_index(segments)

        result = kuavi_analyze_shards("question", shard_duration=30.0)

        assert result["shard_count"] == 3
        # Check that successful shards have answers
        answers = {r["shard_index"]: r for r in result["results"]}
        assert answers[0]["answer"] == "good answer"
        assert "error" in answers[1]
        assert "API error" in answers[1]["error"]
        assert answers[2]["answer"] == "another good answer"

    @patch("kuavi.mcp_server._call_llm", return_value="ok")
    def test_parallel_execution(self, mock_llm):
        segments = _make_segments([(0, 10), (30, 40), (60, 70), (90, 100)])
        _setup_index(segments)

        result = kuavi_analyze_shards("question", shard_duration=30.0)

        assert result["shard_count"] == 4
        assert mock_llm.call_count == 4
        # Results should be ordered by shard_index
        for i, r in enumerate(result["results"]):
            assert r["shard_index"] == i

    @patch("kuavi.mcp_server._call_llm", return_value="ok")
    def test_tool_call_tracked(self, mock_llm):
        segments = _make_segments([(0, 10)])
        _setup_index(segments)

        kuavi_analyze_shards("question")

        assert _state["stats"]["tool_calls"] == 1

    @patch("kuavi.mcp_server._call_llm", return_value="vid2 answer")
    def test_specific_video_id(self, mock_llm):
        # Set up two videos, query the non-active one
        segments1 = _make_segments([(0, 10)])
        segments2 = _make_segments([(0, 20), (20, 40)])
        mock_index1 = SimpleNamespace(segments=segments1)
        mock_index2 = SimpleNamespace(segments=segments2)
        _state["videos"] = {
            "vid1": {"index": mock_index1, "video_path": "v1.mp4"},
            "vid2": {"index": mock_index2, "video_path": "v2.mp4"},
        }
        _state["active_video"] = "vid1"

        result = kuavi_analyze_shards("question", video_id="vid2", shard_duration=50.0)

        assert result["shard_count"] == 1
        # Should have processed vid2's 2 segments, not vid1's 1
        prompt_arg = mock_llm.call_args[0][0]
        assert "segment 0" in prompt_arg.lower() or "Segment 0" in prompt_arg
        assert "Segment 1" in prompt_arg
