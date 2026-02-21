"""Tests for the kuavi_eval MCP tool."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from kuavi.mcp_server import _state, kuavi_eval


@pytest.fixture(autouse=True)
def _reset_eval_namespace():
    """Reset eval namespace before each test."""
    _state["eval_namespace"] = None
    yield
    _state["eval_namespace"] = None


class TestKuaviEval:
    def test_simple_expression(self):
        result = kuavi_eval("1 + 1")
        assert result["result"] == 2
        assert result["stdout"] == ""

    def test_variable_persistence(self):
        kuavi_eval("x = 42")
        result = kuavi_eval("x")
        assert result["result"] == 42

    def test_statement_execution(self):
        result = kuavi_eval("y = [1, 2, 3]")
        assert result["result"] is None
        result = kuavi_eval("len(y)")
        assert result["result"] == 3

    def test_stdout_capture(self):
        result = kuavi_eval("print('hello')")
        assert "hello" in result["stdout"]

    def test_error_handling(self):
        result = kuavi_eval("1/0")
        assert "error" in result
        assert "ZeroDivisionError" in result["error"]
        assert result["stdout"] == ""

    def test_syntax_error(self):
        result = kuavi_eval("def")
        assert "error" in result
        assert "SyntaxError" in result["error"]

    def test_numpy_available(self):
        result = kuavi_eval("np.array([1, 2, 3]).sum()")
        assert result["result"] == 6

    def test_kuavi_functions_available(self):
        result = kuavi_eval("callable(search_video)")
        assert result["result"] is True

    def test_cv2_available(self):
        result = kuavi_eval("cv2.__name__")
        assert result["result"] == "cv2"

    def test_reset_namespace(self):
        kuavi_eval("myvar = 99")
        result = kuavi_eval("myvar")
        assert result["result"] == 99

        # Reset namespace
        _state["eval_namespace"] = None

        result = kuavi_eval("myvar")
        assert "error" in result
        assert "NameError" in result["error"]

    def test_multiline_code(self):
        result = kuavi_eval("a = 5\nb = 10\nprint(a + b)")
        assert "15" in result["stdout"]

    def test_all_tool_functions_present(self):
        expected = [
            "search_video",
            "search_transcript",
            "get_transcript",
            "get_scene_list",
            "discriminative_vqa",
            "extract_frames",
            "get_index_info",
            "crop_frame",
            "diff_frames",
            "blend_frames",
            "threshold_frame",
            "frame_info",
            "zoom_frames",
            "get_session_stats",
            "llm_query",
            "llm_query_batched",
        ]
        # Trigger namespace initialization
        kuavi_eval("1")
        ns = _state["eval_namespace"]
        for name in expected:
            assert name in ns, f"{name} not found in eval namespace"
            assert callable(ns[name]), f"{name} is not callable"


class TestEvalLlmQuery:
    def test_llm_query_available(self):
        result = kuavi_eval("callable(llm_query)")
        assert result["result"] is True

    def test_llm_query_batched_available(self):
        result = kuavi_eval("callable(llm_query_batched)")
        assert result["result"] is True

    @patch("kuavi.mcp_server._call_llm", return_value="test response")
    def test_llm_query_calls_backend(self, mock_llm):
        result = kuavi_eval('llm_query("hello")')
        assert result["result"] == "test response"
        mock_llm.assert_called_once()

    @patch("kuavi.mcp_server._call_llm", side_effect=lambda p, b, m: f"answer to: {p}")
    def test_llm_query_batched_parallel(self, mock_llm):
        result = kuavi_eval('llm_query_batched(["q1", "q2", "q3"])')
        assert len(result["result"]) == 3
        assert result["result"][0] == "answer to: q1"
        assert result["result"][1] == "answer to: q2"
        assert result["result"][2] == "answer to: q3"

    @patch("kuavi.mcp_server._call_llm", side_effect=Exception("API error"))
    def test_llm_query_batched_handles_errors(self, mock_llm):
        result = kuavi_eval('llm_query_batched(["q1"])')
        assert len(result["result"]) == 1
        assert "ERROR" in result["result"][0]

    def test_llm_query_custom_params(self):
        """Verify llm_query accepts backend and model kwargs."""
        kuavi_eval("import inspect; _params = list(inspect.signature(llm_query).parameters.keys())")
        result = kuavi_eval("_params")
        assert "prompt" in result["result"]
        assert "backend" in result["result"]
        assert "model" in result["result"]
