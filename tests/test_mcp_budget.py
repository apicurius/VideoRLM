"""Tests for the KUAVi MCP budget enforcement mechanism."""

from __future__ import annotations

import time

import pytest

from kuavi.mcp_server import (
    _check_budget_gate,
    _state,
    kuavi_get_session_stats,
    kuavi_search_transcript,
    kuavi_search_video,
    kuavi_set_budget,
)


def _reset_state() -> None:
    """Reset stats, budget, and video state to defaults."""
    _state["stats"] = {
        "tool_calls": 0,
        "frames_extracted": 0,
        "searches_performed": 0,
        "session_start": None,
    }
    _state["budget"] = {
        "max_tool_calls": 50,
        "warn_tool_calls": 35,
        "max_elapsed_seconds": 300,
        "warn_elapsed_seconds": 200,
        "exceeded": False,
    }
    _state["videos"] = {}
    _state["active_video"] = None
    _state["eval_namespace"] = None


@pytest.fixture(autouse=True)
def _clean_state():
    """Reset state before and after every test."""
    _reset_state()
    yield
    _reset_state()


class TestCheckBudgetGate:
    def setup_method(self):
        _reset_state()

    def test_within_budget_returns_none(self):
        gate, warning = _check_budget_gate()
        assert gate is None
        assert warning is None

    def test_exceeded_flag_returns_error(self):
        _state["budget"]["exceeded"] = True
        _state["stats"]["session_start"] = time.time()
        gate, warning = _check_budget_gate()
        assert gate is not None
        assert "BUDGET EXCEEDED" in gate["error"]
        assert warning is None

    def test_tool_calls_hard_limit(self):
        _state["stats"]["tool_calls"] = 50
        _state["stats"]["session_start"] = time.time()
        gate, warning = _check_budget_gate()
        assert gate is not None
        assert "BUDGET EXCEEDED" in gate["error"]
        assert _state["budget"]["exceeded"] is True

    def test_time_hard_limit(self):
        _state["stats"]["session_start"] = time.time() - 301
        gate, warning = _check_budget_gate()
        assert gate is not None
        assert "BUDGET EXCEEDED" in gate["error"]
        assert _state["budget"]["exceeded"] is True

    def test_warning_zone_tool_calls(self):
        _state["stats"]["tool_calls"] = 36
        _state["stats"]["session_start"] = time.time()
        gate, warning = _check_budget_gate()
        assert gate is None
        assert warning is not None
        assert "Approaching" in warning
        assert "36/50" in warning

    def test_warning_zone_time(self):
        _state["stats"]["session_start"] = time.time() - 201
        _state["stats"]["tool_calls"] = 5
        gate, warning = _check_budget_gate()
        assert gate is None
        assert warning is not None
        assert "Approaching" in warning

    def test_no_warning_below_threshold(self):
        _state["stats"]["tool_calls"] = 10
        _state["stats"]["session_start"] = time.time()
        gate, warning = _check_budget_gate()
        assert gate is None
        assert warning is None


class TestSetBudget:
    def setup_method(self):
        _reset_state()

    def test_set_budget_updates_state(self):
        result = kuavi_set_budget(max_tool_calls=10, warn_tool_calls=7)
        assert _state["budget"]["max_tool_calls"] == 10
        assert _state["budget"]["warn_tool_calls"] == 7
        assert result["status"] == "budget_configured"
        assert result["max_tool_calls"] == 10

    def test_set_budget_resets_exceeded(self):
        _state["budget"]["exceeded"] = True
        kuavi_set_budget(max_tool_calls=100)
        assert _state["budget"]["exceeded"] is False

    def test_set_budget_returns_remaining(self):
        _state["stats"]["tool_calls"] = 5
        result = kuavi_set_budget(max_tool_calls=20)
        assert result["remaining_tool_calls"] == 15
        assert result["current_tool_calls"] == 5

    def test_set_budget_time_limits(self):
        kuavi_set_budget(
            max_elapsed_seconds=600, warn_elapsed_seconds=400
        )
        assert _state["budget"]["max_elapsed_seconds"] == 600
        assert _state["budget"]["warn_elapsed_seconds"] == 400


class TestBudgetBlocksTools:
    def setup_method(self):
        _reset_state()
        _state["videos"] = {}
        _state["active_video"] = None

    def test_search_video_blocked_when_exceeded(self):
        _state["budget"]["exceeded"] = True
        _state["stats"]["session_start"] = time.time()
        # Even without an index, budget should block before the index check
        # But search_video checks index first, so set up a minimal index
        from types import SimpleNamespace

        mock_index = SimpleNamespace(segments=[], transcript=[], scene_boundaries=[])
        _state["videos"]["test"] = {
            "index": mock_index,
            "video_path": "test.mp4",
            "indexer": None,
            "loaded_video": None,
        }
        _state["active_video"] = "test"

        result = kuavi_search_video("test query")
        assert isinstance(result, list)
        assert len(result) == 1
        assert "BUDGET EXCEEDED" in str(result[0])

    def test_search_transcript_blocked_when_exceeded(self):
        _state["budget"]["exceeded"] = True
        _state["stats"]["session_start"] = time.time()
        from types import SimpleNamespace

        mock_index = SimpleNamespace(segments=[], transcript=[], scene_boundaries=[])
        _state["videos"]["test"] = {
            "index": mock_index,
            "video_path": "test.mp4",
            "indexer": None,
            "loaded_video": None,
        }
        _state["active_video"] = "test"

        result = kuavi_search_transcript("test query")
        assert isinstance(result, list)
        assert "BUDGET EXCEEDED" in str(result[0])

    def test_budget_blocks_after_reaching_limit(self):
        """Set a low budget, exhaust it with tool calls, then verify blocking."""
        kuavi_set_budget(max_tool_calls=2, warn_tool_calls=1)
        from types import SimpleNamespace

        mock_index = SimpleNamespace(segments=[], transcript=[], scene_boundaries=[])
        _state["videos"]["test"] = {
            "index": mock_index,
            "video_path": "test.mp4",
            "indexer": None,
            "loaded_video": None,
        }
        _state["active_video"] = "test"

        # The set_budget call does not count toward the budget.
        # Force tool_calls to the limit.
        _state["stats"]["tool_calls"] = 2
        _state["stats"]["session_start"] = time.time()

        # Now the next gated tool should be blocked
        gate, _ = _check_budget_gate()
        assert gate is not None
        assert "BUDGET EXCEEDED" in gate["error"]


class TestStatsIncludesBudget:
    def setup_method(self):
        _reset_state()

    def test_stats_always_work_when_exceeded(self):
        _state["budget"]["exceeded"] = True
        _state["stats"]["session_start"] = time.time()
        result = kuavi_get_session_stats()
        # Stats should NOT be blocked
        assert "tool_calls" in result
        assert "budget" in result

    def test_budget_info_in_stats(self):
        result = kuavi_get_session_stats()
        assert "budget" in result
        budget_info = result["budget"]
        assert "max_tool_calls" in budget_info
        assert "remaining_tool_calls" in budget_info
        assert "exceeded" in budget_info
        assert budget_info["exceeded"] is False

    def test_remaining_tool_calls_accurate(self):
        _state["stats"]["tool_calls"] = 10
        result = kuavi_get_session_stats()
        # get_session_stats increments tool_calls by 1, so 10+1=11, remaining=50-11=39
        assert result["budget"]["remaining_tool_calls"] == 39

    def test_remaining_after_set_budget(self):
        _state["stats"]["tool_calls"] = 5
        kuavi_set_budget(max_tool_calls=20)
        result = kuavi_get_session_stats()
        # tool_calls is now 5 + 1 (from stats call itself) = 6
        # But set_budget doesn't track, so 5 + 1 (stats) = 6
        remaining = result["budget"]["remaining_tool_calls"]
        assert remaining == 20 - result["tool_calls"]


class TestBudgetWarningInResults:
    def setup_method(self):
        _reset_state()

    def test_warning_zone_detected(self):
        _state["stats"]["tool_calls"] = 36
        _state["stats"]["session_start"] = time.time()
        gate, warning = _check_budget_gate()
        assert gate is None
        assert warning is not None
        assert "wrapping up" in warning.lower()

    def test_exceeded_sets_flag_permanently(self):
        _state["stats"]["tool_calls"] = 50
        _state["stats"]["session_start"] = time.time()
        gate1, _ = _check_budget_gate()
        assert gate1 is not None
        assert _state["budget"]["exceeded"] is True

        # Subsequent calls also blocked
        gate2, _ = _check_budget_gate()
        assert gate2 is not None

    def test_set_budget_clears_exceeded(self):
        _state["budget"]["exceeded"] = True
        kuavi_set_budget(max_tool_calls=100)
        assert _state["budget"]["exceeded"] is False
        gate, _ = _check_budget_gate()
        assert gate is None
