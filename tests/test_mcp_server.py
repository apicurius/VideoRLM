"""Tests for the KUAVi MCP server tool functions."""

from __future__ import annotations

import pytest

from kuavi.mcp_server import (
    _get_active_index,
    _get_active_video_path,
    _get_video_entry,
    _state,
    kuavi_get_index_info,
    kuavi_get_session_stats,
)


class TestMcpServerState:
    """Test MCP server state management."""

    def test_initial_state_has_multi_video_structure(self):
        assert isinstance(_state["videos"], dict)
        assert _state["active_video"] is None
        assert _state["eval_namespace"] is None
        assert isinstance(_state["stats"], dict)
        assert _state["stats"]["tool_calls"] >= 0
        assert _state["stats"]["frames_extracted"] >= 0
        assert _state["stats"]["searches_performed"] >= 0

    def test_get_index_info_without_index(self):
        result = kuavi_get_index_info()
        assert "error" in result

    def test_get_video_entry_returns_none_when_empty(self):
        assert _get_video_entry() is None
        assert _get_video_entry("nonexistent") is None

    def test_get_active_index_returns_none_when_empty(self):
        assert _get_active_index() is None

    def test_get_active_video_path_returns_none_when_empty(self):
        assert _get_active_video_path() is None

    def test_get_session_stats_initial(self):
        result = kuavi_get_session_stats()
        assert isinstance(result, dict)
        assert "tool_calls" in result
        assert "frames_extracted" in result
        assert "searches_performed" in result
        assert "elapsed_seconds" in result
        assert "videos_loaded" in result
        assert result["videos_loaded"] == len(_state["videos"])


class TestMcpServerImports:
    """Test that MCP server can be imported."""

    def test_import_mcp_server(self):
        from kuavi.mcp_server import mcp

        assert mcp is not None
        assert mcp.name == "kuavi"

    def test_import_all_tools(self):
        from kuavi.mcp_server import (
            kuavi_blend_frames,
            kuavi_crop_frame,
            kuavi_diff_frames,
            kuavi_discriminative_vqa,
            kuavi_extract_frames,
            kuavi_frame_info,
            kuavi_get_index_info,
            kuavi_get_scene_list,
            kuavi_get_session_stats,
            kuavi_get_transcript,
            kuavi_index_video,
            kuavi_load_index,
            kuavi_search_transcript,
            kuavi_search_video,
            kuavi_threshold_frame,
            kuavi_zoom_frames,
        )

        # Original 8 tools
        assert callable(kuavi_index_video)
        assert callable(kuavi_search_video)
        assert callable(kuavi_search_transcript)
        assert callable(kuavi_get_transcript)
        assert callable(kuavi_get_scene_list)
        assert callable(kuavi_discriminative_vqa)
        assert callable(kuavi_extract_frames)
        assert callable(kuavi_get_index_info)

        # New tools (state refactor)
        assert callable(kuavi_get_session_stats)
        assert callable(kuavi_zoom_frames)
        assert callable(kuavi_load_index)

        # Pixel tools
        assert callable(kuavi_crop_frame)
        assert callable(kuavi_diff_frames)
        assert callable(kuavi_blend_frames)
        assert callable(kuavi_threshold_frame)
        assert callable(kuavi_frame_info)
