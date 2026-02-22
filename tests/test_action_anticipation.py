"""Tests for WI-9: Action Anticipation via V-JEPA 2 Predictor."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from kuavi.indexer import VideoIndex, VideoIndexer
from kuavi.search import make_anticipate_action


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_index(num_segments: int = 5, embed_dim: int = 4) -> VideoIndex:
    """Return a minimal VideoIndex with synthetic embeddings."""
    rng = np.random.default_rng(0)
    segments = [
        {
            "start_time": float(i * 2),
            "end_time": float(i * 2 + 2),
            "caption": f"segment {i}",
            "annotation": {},
        }
        for i in range(num_segments)
    ]
    embeddings = rng.standard_normal((num_segments, embed_dim)).astype(np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.maximum(norms, 1e-10)

    def embed_fn(text: str) -> np.ndarray:
        seed = sum(ord(c) for c in text) % 100
        v = np.random.default_rng(seed).standard_normal(embed_dim).astype(np.float32)
        return v / (np.linalg.norm(v) + 1e-10)

    return VideoIndex(
        segments=segments,
        embeddings=embeddings,
        embed_fn=embed_fn,
    )


# ---------------------------------------------------------------------------
# VideoIndexer unit tests
# ---------------------------------------------------------------------------


class TestScenePredictorInit:
    """_scene_predictor is initialized to None in __init__."""

    def test_scene_predictor_none_on_init(self):
        indexer = VideoIndexer()
        assert indexer._scene_predictor is None


class TestEnsureSceneModelPredictor:
    """_ensure_scene_model() attempts predictor loading."""

    def test_predictor_loaded_when_present(self):
        import sys

        indexer = VideoIndexer(scene_model="facebook/vjepa2-vitl-fpc64-256")

        mock_predictor = MagicMock()
        mock_model = MagicMock()
        mock_model.predictor = mock_predictor
        mock_model.eval.return_value = mock_model
        mock_model.to.return_value = mock_model

        mock_torch = MagicMock()
        mock_torch.backends.mps.is_available.return_value = False
        mock_torch.cuda.is_available.return_value = False
        mock_torch.float16 = "float16"

        mock_transformers = MagicMock()
        mock_transformers.AutoVideoProcessor.from_pretrained.return_value = MagicMock()
        mock_transformers.AutoModel.from_pretrained.return_value = mock_model

        indexer._scene_model = None
        with patch.dict(sys.modules, {"torch": mock_torch, "transformers": mock_transformers}):
            indexer._ensure_scene_model()

        assert indexer._scene_predictor is mock_predictor

    def test_predictor_none_when_absent(self):
        import sys

        indexer = VideoIndexer(scene_model="facebook/vjepa2-vitl-fpc64-256")

        mock_model = MagicMock()
        # explicitly delete 'predictor' attribute so getattr returns None
        del mock_model.predictor
        mock_model.eval.return_value = mock_model
        mock_model.to.return_value = mock_model

        mock_torch = MagicMock()
        mock_torch.backends.mps.is_available.return_value = False
        mock_torch.cuda.is_available.return_value = False
        mock_torch.float16 = "float16"

        mock_transformers = MagicMock()
        mock_transformers.AutoVideoProcessor.from_pretrained.return_value = MagicMock()
        mock_transformers.AutoModel.from_pretrained.return_value = mock_model

        indexer._scene_model = None
        with patch.dict(sys.modules, {"torch": mock_torch, "transformers": mock_transformers}):
            indexer._ensure_scene_model()

        assert indexer._scene_predictor is None


class TestPredictFutureEmbedding:
    """_predict_future_embedding() unit tests."""

    def test_returns_none_when_predictor_none(self):
        indexer = VideoIndexer()
        assert indexer._scene_predictor is None
        features = np.zeros((8, 1024), dtype=np.float32)
        result = indexer._predict_future_embedding(features)
        assert result is None

    def test_returns_correct_shape_when_predictor_mocked(self):
        import sys

        indexer = VideoIndexer()
        num_patches = 32
        n_future = 8
        D = 1024
        expected_output = np.ones((n_future, D), dtype=np.float32)

        # The predictor returns an object with .last_hidden_state
        mock_output = MagicMock()
        mock_output.last_hidden_state.squeeze.return_value.cpu.return_value.float.return_value.numpy.return_value = (
            expected_output
        )
        mock_predictor = MagicMock(return_value=mock_output)
        indexer._scene_predictor = mock_predictor
        indexer._scene_torch_device = "cpu"

        mock_tensor = MagicMock()
        mock_arange_result = MagicMock()
        mock_arange_result.unsqueeze.return_value = mock_arange_result
        mock_arange_result.to.return_value = mock_arange_result

        mock_torch = MagicMock()
        mock_torch.from_numpy.return_value = mock_tensor
        mock_tensor.unsqueeze.return_value = mock_tensor
        mock_tensor.to.return_value = mock_tensor
        mock_torch.arange.return_value = mock_arange_result
        mock_torch.no_grad.return_value.__enter__ = lambda s: None
        mock_torch.no_grad.return_value.__exit__ = lambda s, *a: False
        mock_torch.float16 = "float16"
        mock_torch.int64 = "int64"

        with patch.dict(sys.modules, {"torch": mock_torch}):
            result = indexer._predict_future_embedding(
                np.zeros((num_patches, D), dtype=np.float32), n_future_tokens=n_future
            )

        assert result is not None
        np.testing.assert_array_equal(result, expected_output)

    def test_returns_none_on_predictor_exception(self):
        import sys

        indexer = VideoIndexer()
        indexer._scene_predictor = MagicMock(side_effect=RuntimeError("forward failed"))
        indexer._scene_torch_device = "cpu"

        mock_tensor = MagicMock()
        mock_arange_result = MagicMock()
        mock_arange_result.unsqueeze.return_value = mock_arange_result
        mock_arange_result.to.return_value = mock_arange_result

        mock_torch = MagicMock()
        mock_torch.from_numpy.return_value = mock_tensor
        mock_tensor.unsqueeze.return_value = mock_tensor
        mock_tensor.to.return_value = mock_tensor
        mock_torch.arange.return_value = mock_arange_result
        mock_torch.no_grad.return_value.__enter__ = lambda s: None
        mock_torch.no_grad.return_value.__exit__ = lambda s, *a: False
        mock_torch.float16 = "float16"
        mock_torch.int64 = "int64"

        with patch.dict(sys.modules, {"torch": mock_torch}):
            result = indexer._predict_future_embedding(
                np.zeros((32, 1024), dtype=np.float32)
            )
        assert result is None


# ---------------------------------------------------------------------------
# make_anticipate_action() factory tests
# ---------------------------------------------------------------------------


class TestMakeAnticipateAction:
    """make_anticipate_action() factory structure."""

    def test_returns_tool_dict(self):
        index = _make_index()
        result = make_anticipate_action(index)
        assert isinstance(result, dict)
        assert "tool" in result
        assert "description" in result
        assert callable(result["tool"])

    def test_description_is_string(self):
        index = _make_index()
        result = make_anticipate_action(index)
        assert isinstance(result["description"], str)
        assert len(result["description"]) > 0


class TestAnticipateActionFallback:
    """anticipate_action() with embedding similarity fallback (no predictor)."""

    def test_returns_predicted_segments(self):
        index = _make_index(num_segments=5)
        tool = make_anticipate_action(index)["tool"]
        result = tool(time_point=2.0, top_k=2)
        assert "predicted_segments" in result
        assert isinstance(result["predicted_segments"], list)

    def test_filters_to_future_segments_only(self):
        index = _make_index(num_segments=5)
        tool = make_anticipate_action(index)["tool"]
        time_point = 4.0  # segments 0-1 start before, segments 2+ start after
        result = tool(time_point=time_point, top_k=5)
        for seg in result["predicted_segments"]:
            assert seg["start_time"] > time_point, (
                f"Segment at {seg['start_time']} should be after {time_point}"
            )

    def test_no_future_segments_returns_empty(self):
        index = _make_index(num_segments=3)
        tool = make_anticipate_action(index)["tool"]
        # time_point beyond all segments
        result = tool(time_point=999.0, top_k=3)
        assert result["predicted_segments"] == []

    def test_invalid_time_point_returns_error(self):
        # time_point before any segment (no segments cover t=-10)
        # An empty list of segments
        index = VideoIndex(segments=[], embeddings=None, embed_fn=None)
        tool = make_anticipate_action(index)["tool"]
        result = tool(time_point=5.0)
        assert "error" in result

    def test_no_embeddings_returns_error(self):
        index = _make_index(num_segments=3)
        index.embeddings = None  # remove embeddings
        tool = make_anticipate_action(index)["tool"]
        result = tool(time_point=2.0)
        assert "error" in result

    def test_with_candidates_returns_ranking(self):
        index = _make_index(num_segments=5)
        tool = make_anticipate_action(index)["tool"]
        candidates = ["person walking", "person running", "person sitting"]
        result = tool(time_point=2.0, top_k=3, candidates=candidates)
        assert "candidate_ranking" in result
        ranking = result["candidate_ranking"]
        assert len(ranking) == len(candidates)
        for entry in ranking:
            assert "action" in entry
            assert "confidence" in entry
            assert entry["action"] in candidates

    def test_candidate_ranking_sorted_by_confidence(self):
        index = _make_index(num_segments=5)
        tool = make_anticipate_action(index)["tool"]
        candidates = ["action A", "action B", "action C"]
        result = tool(time_point=2.0, candidates=candidates)
        ranking = result.get("candidate_ranking", [])
        if len(ranking) > 1:
            confidences = [r["confidence"] for r in ranking]
            assert confidences == sorted(confidences, reverse=True)

    def test_method_field_is_embedding_similarity(self):
        index = _make_index(num_segments=5)
        tool = make_anticipate_action(index)["tool"]
        result = tool(time_point=2.0)
        assert result.get("method") == "embedding_similarity"

    def test_context_segment_included_in_result(self):
        index = _make_index(num_segments=5)
        tool = make_anticipate_action(index)["tool"]
        result = tool(time_point=2.0)
        if "predicted_segments" in result and result["predicted_segments"]:
            assert "context_segment" in result


class TestAnticipateActionWithPredictor:
    """anticipate_action() when _predict_fn is set on index."""

    def test_uses_predict_fn_when_available(self):
        index = _make_index(num_segments=5)
        predicted_emb = np.ones(4, dtype=np.float32)
        index._predict_fn = lambda t: predicted_emb  # type: ignore[attr-defined]

        tool = make_anticipate_action(index)["tool"]
        result = tool(time_point=2.0, top_k=2)
        assert result.get("method") == "vjepa2_predictor"
        assert "predicted_segments" in result

    def test_predict_fn_failure_returns_error(self):
        index = _make_index(num_segments=5)
        index._predict_fn = lambda t: None  # type: ignore[attr-defined]

        tool = make_anticipate_action(index)["tool"]
        result = tool(time_point=2.0)
        assert "error" in result


# ---------------------------------------------------------------------------
# MCP tool signature test
# ---------------------------------------------------------------------------


class TestMcpToolSignature:
    """kuavi_anticipate_action MCP tool signature is correct."""

    def test_tool_registered(self):
        import inspect

        import kuavi.mcp_server as srv

        assert hasattr(srv, "kuavi_anticipate_action"), (
            "kuavi_anticipate_action must be defined in mcp_server"
        )
        fn = srv.kuavi_anticipate_action
        sig = inspect.signature(fn)
        params = list(sig.parameters)
        assert "time_point" in params
        assert "top_k" in params
        assert "candidates" in params
        assert "video_id" in params

    def test_tool_returns_error_when_no_video(self):
        import kuavi.mcp_server as srv

        # Reset active video
        original_active = srv._state["active_video"]
        original_videos = dict(srv._state["videos"])
        srv._state["active_video"] = None
        srv._state["videos"] = {}

        try:
            result = srv.kuavi_anticipate_action(time_point=5.0)
        finally:
            srv._state["active_video"] = original_active
            srv._state["videos"] = original_videos

        assert "error" in result
