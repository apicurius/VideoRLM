"""Tests for WI-11: Predictive Video Understanding (World Model).

Covers make_predict_future() and make_verify_coherence() in kuavi/search.py
and their MCP tool wrappers.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from kuavi.indexer import VideoIndex
from kuavi.search import make_predict_future, make_verify_coherence


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_index(
    num_segments: int = 6,
    embed_dim: int = 4,
    with_temporal: bool = False,
    with_feature_maps: bool = False,
) -> VideoIndex:
    """Return a minimal VideoIndex with synthetic embeddings."""
    rng = np.random.default_rng(42)
    segments = [
        {
            "start_time": float(i * 5),
            "end_time": float(i * 5 + 5),
            "caption": f"segment {i}",
            "annotation": {},
        }
        for i in range(num_segments)
    ]
    embeddings = rng.standard_normal((num_segments, embed_dim)).astype(np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.maximum(norms, 1e-10)

    temporal_embeddings = None
    if with_temporal:
        temporal_embeddings = rng.standard_normal((num_segments, embed_dim)).astype(np.float32)
        norms_t = np.linalg.norm(temporal_embeddings, axis=1, keepdims=True)
        temporal_embeddings = temporal_embeddings / np.maximum(norms_t, 1e-10)

    temporal_feature_maps = None
    if with_feature_maps:
        num_patches = 8
        temporal_feature_maps = rng.standard_normal(
            (num_segments, num_patches, embed_dim)
        ).astype(np.float32)

    def embed_fn(text: str) -> np.ndarray:
        seed = sum(ord(c) for c in text) % 100
        v = np.random.default_rng(seed).standard_normal(embed_dim).astype(np.float32)
        return v / (np.linalg.norm(v) + 1e-10)

    return VideoIndex(
        segments=segments,
        embeddings=embeddings,
        temporal_embeddings=temporal_embeddings,
        temporal_feature_maps=temporal_feature_maps,
        embed_fn=embed_fn,
    )


# ---------------------------------------------------------------------------
# make_predict_future — factory structure
# ---------------------------------------------------------------------------


class TestPredictFutureFactory:
    def test_returns_dict_with_tool_and_description(self):
        index = _make_index()
        result = make_predict_future(index)
        assert "tool" in result
        assert "description" in result
        assert callable(result["tool"])

    def test_tool_name_is_predict_future(self):
        index = _make_index()
        result = make_predict_future(index)
        assert result["tool"].__name__ == "predict_future"

    def test_description_contains_key_terms(self):
        index = _make_index()
        result = make_predict_future(index)
        desc = result["description"]
        assert "start_time" in desc
        assert "end_time" in desc
        assert "predicted_segments" in desc


# ---------------------------------------------------------------------------
# make_predict_future — fallback path (no predictor)
# ---------------------------------------------------------------------------


class TestPredictFutureFallback:
    def test_returns_predicted_segments(self):
        index = _make_index(num_segments=6)
        tool = make_predict_future(index)["tool"]
        result = tool(start_time=0.0, end_time=5.0)
        assert "predicted_segments" in result
        assert isinstance(result["predicted_segments"], list)

    def test_method_is_temporal_continuation(self):
        index = _make_index(num_segments=6)
        tool = make_predict_future(index)["tool"]
        result = tool(start_time=0.0, end_time=5.0)
        assert result["method"] == "temporal_continuation"

    def test_returns_context_info(self):
        index = _make_index(num_segments=6)
        tool = make_predict_future(index)["tool"]
        result = tool(start_time=0.0, end_time=5.0)
        assert "context" in result
        assert result["context"]["start_time"] == 0.0
        assert result["context"]["end_time"] == 5.0

    def test_predicted_segments_are_future_only(self):
        index = _make_index(num_segments=6)
        tool = make_predict_future(index)["tool"]
        result = tool(start_time=0.0, end_time=5.0)
        for seg in result["predicted_segments"]:
            assert seg["start_time"] >= 5.0, "Predicted segments must be after context window"

    def test_predicted_segments_sorted_by_score_desc(self):
        index = _make_index(num_segments=8)
        tool = make_predict_future(index)["tool"]
        result = tool(start_time=0.0, end_time=5.0)
        scores = [s["score"] for s in result["predicted_segments"]]
        assert scores == sorted(scores, reverse=True)

    def test_scores_are_floats_in_range(self):
        index = _make_index(num_segments=6)
        tool = make_predict_future(index)["tool"]
        result = tool(start_time=0.0, end_time=5.0)
        for seg in result["predicted_segments"]:
            assert isinstance(seg["score"], float)
            assert -1.0 <= seg["score"] <= 1.0

    def test_no_future_segments(self):
        index = _make_index(num_segments=3)
        tool = make_predict_future(index)["tool"]
        # Context covers all segments
        result = tool(start_time=0.0, end_time=20.0)
        assert result["predicted_segments"] == []
        assert "note" in result or result["method"] in ("temporal_continuation", "vjepa2_predictor")

    def test_empty_range_returns_error(self):
        index = _make_index(num_segments=3)
        tool = make_predict_future(index)["tool"]
        result = tool(start_time=100.0, end_time=200.0)
        assert "error" in result

    def test_uses_temporal_embeddings_when_available(self):
        index = _make_index(num_segments=6, with_temporal=True)
        tool = make_predict_future(index)["tool"]
        result = tool(start_time=0.0, end_time=5.0)
        assert result["method"] == "temporal_continuation"
        assert len(result["predicted_segments"]) > 0

    def test_n_future_tokens_param_accepted(self):
        index = _make_index(num_segments=6)
        tool = make_predict_future(index)["tool"]
        # Should not raise
        result = tool(start_time=0.0, end_time=5.0, n_future_tokens=32)
        assert "predicted_segments" in result

    def test_max_five_predictions_returned(self):
        index = _make_index(num_segments=10)
        tool = make_predict_future(index)["tool"]
        result = tool(start_time=0.0, end_time=5.0)
        assert len(result["predicted_segments"]) <= 5

    def test_segment_fields_present(self):
        index = _make_index(num_segments=6)
        tool = make_predict_future(index)["tool"]
        result = tool(start_time=0.0, end_time=5.0)
        for seg in result["predicted_segments"]:
            assert "start_time" in seg
            assert "end_time" in seg
            assert "score" in seg
            assert "caption" in seg


# ---------------------------------------------------------------------------
# make_predict_future — predictor path (mocked _predict_future_fn)
# ---------------------------------------------------------------------------


class TestPredictFutureWithPredictor:
    def test_uses_predictor_when_available(self):
        index = _make_index(num_segments=6, with_feature_maps=True)
        predicted_features = np.ones((16, 4), dtype=np.float32)
        index._predict_future_fn = lambda fm, n: predicted_features  # type: ignore[attr-defined]

        tool = make_predict_future(index)["tool"]
        result = tool(start_time=0.0, end_time=5.0)
        assert result["method"] == "vjepa2_predictor"

    def test_predictor_returns_future_segments(self):
        index = _make_index(num_segments=6, with_feature_maps=True)
        predicted_features = np.ones((16, 4), dtype=np.float32)
        index._predict_future_fn = lambda fm, n: predicted_features  # type: ignore[attr-defined]

        tool = make_predict_future(index)["tool"]
        result = tool(start_time=0.0, end_time=5.0)
        assert "predicted_segments" in result

    def test_predictor_none_return_falls_back(self):
        index = _make_index(num_segments=6, with_feature_maps=True)
        index._predict_future_fn = lambda fm, n: None  # type: ignore[attr-defined]

        tool = make_predict_future(index)["tool"]
        result = tool(start_time=0.0, end_time=5.0)
        # Falls back to temporal_continuation
        assert result["method"] == "temporal_continuation"

    def test_no_future_segs_with_predictor(self):
        index = _make_index(num_segments=2, with_feature_maps=True)
        predicted_features = np.ones((16, 4), dtype=np.float32)
        index._predict_future_fn = lambda fm, n: predicted_features  # type: ignore[attr-defined]

        tool = make_predict_future(index)["tool"]
        result = tool(start_time=0.0, end_time=10.0)  # covers all segments
        assert result["predicted_segments"] == []


# ---------------------------------------------------------------------------
# make_verify_coherence — factory structure
# ---------------------------------------------------------------------------


class TestVerifyCoherenceFactory:
    def test_returns_dict_with_tool_and_description(self):
        index = _make_index()
        result = make_verify_coherence(index)
        assert "tool" in result
        assert "description" in result
        assert callable(result["tool"])

    def test_tool_name_is_verify_coherence(self):
        index = _make_index()
        result = make_verify_coherence(index)
        assert result["tool"].__name__ == "verify_coherence"


# ---------------------------------------------------------------------------
# make_verify_coherence — basic behavior
# ---------------------------------------------------------------------------


class TestVerifyCoherenceFallback:
    def test_returns_required_fields(self):
        index = _make_index(num_segments=6)
        tool = make_verify_coherence(index)["tool"]
        result = tool(start_time=0.0, end_time=30.0)
        assert "overall_score" in result
        assert "segment_scores" in result
        assert "anomalies" in result
        assert "method" in result

    def test_method_is_pairwise_similarity(self):
        index = _make_index(num_segments=6)
        tool = make_verify_coherence(index)["tool"]
        result = tool(start_time=0.0, end_time=30.0)
        assert result["method"] == "pairwise_similarity"

    def test_overall_score_is_float(self):
        index = _make_index(num_segments=6)
        tool = make_verify_coherence(index)["tool"]
        result = tool(start_time=0.0, end_time=30.0)
        assert isinstance(result["overall_score"], float)

    def test_segment_scores_have_required_fields(self):
        index = _make_index(num_segments=6)
        tool = make_verify_coherence(index)["tool"]
        result = tool(start_time=0.0, end_time=30.0)
        for s in result["segment_scores"]:
            assert "start" in s
            assert "end" in s
            assert "score" in s
            assert "is_anomalous" in s

    def test_anomaly_count_matches_is_anomalous(self):
        index = _make_index(num_segments=6)
        tool = make_verify_coherence(index)["tool"]
        result = tool(start_time=0.0, end_time=30.0, threshold=0.0)
        anomaly_count = sum(1 for s in result["segment_scores"] if s["is_anomalous"])
        assert len(result["anomalies"]) == anomaly_count

    def test_single_segment_returns_note(self):
        index = _make_index(num_segments=3)
        tool = make_verify_coherence(index)["tool"]
        result = tool(start_time=0.0, end_time=5.0)  # only one segment
        assert "note" in result
        assert result["segment_scores"] == []

    def test_empty_range_returns_note(self):
        index = _make_index(num_segments=3)
        tool = make_verify_coherence(index)["tool"]
        result = tool(start_time=100.0, end_time=200.0)
        assert "note" in result

    def test_high_threshold_flags_all_as_anomalous(self):
        index = _make_index(num_segments=6)
        tool = make_verify_coherence(index)["tool"]
        result = tool(start_time=0.0, end_time=30.0, threshold=1.0)
        # All cosine similarities <= 1.0, so all should be anomalous
        for s in result["segment_scores"]:
            assert s["is_anomalous"] is True

    def test_zero_threshold_flags_none_as_anomalous(self):
        index = _make_index(num_segments=6)
        tool = make_verify_coherence(index)["tool"]
        result = tool(start_time=0.0, end_time=30.0, threshold=-1.0)
        # All cosine similarities >= -1.0
        for s in result["segment_scores"]:
            assert s["is_anomalous"] is False

    def test_uses_temporal_embeddings_when_available(self):
        index = _make_index(num_segments=6, with_temporal=True)
        tool = make_verify_coherence(index)["tool"]
        result = tool(start_time=0.0, end_time=30.0)
        assert "overall_score" in result

    def test_anomalies_have_description(self):
        index = _make_index(num_segments=6)
        tool = make_verify_coherence(index)["tool"]
        result = tool(start_time=0.0, end_time=30.0, threshold=1.0)
        for a in result["anomalies"]:
            assert "description" in a
            assert isinstance(a["description"], str)

    def test_overall_score_is_mean_of_segment_scores(self):
        index = _make_index(num_segments=6)
        tool = make_verify_coherence(index)["tool"]
        result = tool(start_time=0.0, end_time=30.0)
        expected = round(
            float(np.mean([s["score"] for s in result["segment_scores"]])), 4
        )
        assert abs(result["overall_score"] - expected) < 1e-4


# ---------------------------------------------------------------------------
# make_verify_coherence — predictor path
# ---------------------------------------------------------------------------


class TestVerifyCoherenceWithPredictor:
    def test_uses_predictor_method_when_available(self):
        index = _make_index(num_segments=6, with_feature_maps=True)
        predicted_features = np.ones((16, 4), dtype=np.float32)
        index._predict_future_fn = lambda fm, n: predicted_features  # type: ignore[attr-defined]

        tool = make_verify_coherence(index)["tool"]
        result = tool(start_time=0.0, end_time=30.0)
        assert result["method"] == "vjepa2_predictor"

    def test_predictor_none_falls_back_to_pairwise(self):
        index = _make_index(num_segments=6, with_feature_maps=True)
        index._predict_future_fn = lambda fm, n: None  # type: ignore[attr-defined]

        tool = make_verify_coherence(index)["tool"]
        result = tool(start_time=0.0, end_time=30.0)
        # Falls back for each pair when predicted_features is None
        assert "overall_score" in result


# ---------------------------------------------------------------------------
# MCP tool wiring
# ---------------------------------------------------------------------------


class TestMcpToolWiring:
    """Verify that MCP tools call the search factories correctly."""

    def _make_mock_entry(self, index: VideoIndex) -> dict:
        return {"index": index, "tools": {}}

    def test_kuavi_predict_future_wires_factory(self):
        """kuavi_predict_future should instantiate make_predict_future on first call."""
        from kuavi.mcp_server import kuavi_predict_future

        index = _make_index(num_segments=6)
        mock_entry = self._make_mock_entry(index)

        with (
            patch("kuavi.mcp_server._get_video_entry", return_value=mock_entry),
            patch("kuavi.mcp_server._track_tool_call"),
            patch("kuavi.mcp_server._check_budget_gate", return_value=(None, None)),
            patch("kuavi.mcp_server._track_response_tokens"),
        ):
            result = kuavi_predict_future(start_time=0.0, end_time=5.0)

        assert "predicted_segments" in result
        assert "predict_future" in mock_entry["tools"]

    def test_kuavi_verify_coherence_wires_factory(self):
        """kuavi_verify_coherence should instantiate make_verify_coherence on first call."""
        from kuavi.mcp_server import kuavi_verify_coherence

        index = _make_index(num_segments=6)
        mock_entry = self._make_mock_entry(index)

        with (
            patch("kuavi.mcp_server._get_video_entry", return_value=mock_entry),
            patch("kuavi.mcp_server._track_tool_call"),
            patch("kuavi.mcp_server._check_budget_gate", return_value=(None, None)),
            patch("kuavi.mcp_server._track_response_tokens"),
        ):
            result = kuavi_verify_coherence(start_time=0.0, end_time=30.0)

        assert "overall_score" in result
        assert "verify_coherence" in mock_entry["tools"]

    def test_kuavi_predict_future_no_video(self):
        from kuavi.mcp_server import kuavi_predict_future

        with patch("kuavi.mcp_server._get_video_entry", return_value=None):
            result = kuavi_predict_future(start_time=0.0, end_time=5.0)
        assert "error" in result

    def test_kuavi_verify_coherence_no_video(self):
        from kuavi.mcp_server import kuavi_verify_coherence

        with patch("kuavi.mcp_server._get_video_entry", return_value=None):
            result = kuavi_verify_coherence(start_time=0.0, end_time=30.0)
        assert "error" in result

    def test_kuavi_predict_future_budget_gate(self):
        from kuavi.mcp_server import kuavi_predict_future

        index = _make_index()
        mock_entry = self._make_mock_entry(index)
        gate_response = {"error": "Budget exceeded"}

        with (
            patch("kuavi.mcp_server._get_video_entry", return_value=mock_entry),
            patch("kuavi.mcp_server._track_tool_call"),
            patch("kuavi.mcp_server._check_budget_gate", return_value=(gate_response, None)),
        ):
            result = kuavi_predict_future(start_time=0.0, end_time=5.0)
        assert result == gate_response

    def test_kuavi_verify_coherence_budget_gate(self):
        from kuavi.mcp_server import kuavi_verify_coherence

        index = _make_index()
        mock_entry = self._make_mock_entry(index)
        gate_response = {"error": "Budget exceeded"}

        with (
            patch("kuavi.mcp_server._get_video_entry", return_value=mock_entry),
            patch("kuavi.mcp_server._track_tool_call"),
            patch("kuavi.mcp_server._check_budget_gate", return_value=(gate_response, None)),
        ):
            result = kuavi_verify_coherence(start_time=0.0, end_time=30.0)
        assert result == gate_response
