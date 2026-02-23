"""Integration tests for predictor wiring, feature maps, and overlapping V-JEPA.

Verifies end-to-end that:
1. _predict_fn / _predict_future_fn are attached to VideoIndex after index_video
2. store_feature_maps=True works with overlapping_vjepa=True
3. make_anticipate_action / make_predict_future / make_verify_coherence use
   the attached predictor closures (method == "vjepa2_predictor")
4. classify_segment returns predictions when probes are available
"""

from __future__ import annotations

from contextlib import ExitStack
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from kuavi.indexer import VideoIndex, VideoIndexer
from kuavi.search import (
    make_anticipate_action,
    make_predict_future,
    make_verify_coherence,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

D = 8
NUM_PATCHES = 4
NUM_SEGMENTS = 6


def _make_loaded_video(num_frames=24, fps=2.0):
    """Create a mock LoadedVideo."""
    from kuavi.loader import VideoMetadata

    rng = np.random.default_rng(42)
    frames = [rng.integers(0, 255, (16, 16, 3), dtype=np.uint8) for _ in range(num_frames)]
    lv = MagicMock()
    lv.frames = frames
    lv.segments = []
    lv.metadata = VideoMetadata(
        path="/fake/video.mp4",
        total_frames=num_frames,
        original_fps=fps,
        duration=num_frames / fps,
        width=16,
        height=16,
        extraction_fps=fps,
        extracted_frame_count=num_frames,
    )
    return lv


def _fake_vjepa_clips(windows, D=D, return_full=False, **kw):
    """Return deterministic L2-normalized embeddings and optional feature maps."""
    n = len(windows)
    rng = np.random.default_rng(99)
    embs = rng.standard_normal((n, D)).astype(np.float32)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    embs = embs / np.maximum(norms, 1e-10)

    if return_full:
        fmaps = [
            rng.standard_normal((NUM_PATCHES, D)).astype(np.float32) for _ in range(n)
        ]
        return embs, fmaps
    return embs


def _fake_predict_future(context_features, n_future_tokens=16):
    """Mock predictor: return random predicted features."""
    rng = np.random.default_rng(7)
    return rng.standard_normal((n_future_tokens, D)).astype(np.float32)


def _make_indexer_with_predictor():
    """Create a VideoIndexer with a mock scene model + predictor attached."""
    indexer = VideoIndexer(
        scene_model="facebook/vjepa2-vitl-fpc64-256",
        scene_clip_size=4,
        scene_stride=2,
    )
    indexer._scene_embed_dim = D

    # Simulate that _ensure_scene_model was called and found a predictor
    indexer._scene_predictor = MagicMock()
    indexer._scene_torch_device = "cpu"

    # Wire the real _predict_future_embedding to use our fake
    indexer._predict_future_embedding = _fake_predict_future

    return indexer


def _patch_indexer_for_index_video(indexer):
    """Return an ExitStack that mocks all heavy dependencies."""
    stack = ExitStack()
    stack.enter_context(patch.object(indexer, "_ensure_model"))
    stack.enter_context(patch.object(indexer, "_ensure_scene_model"))
    stack.enter_context(patch.object(indexer, "_get_transcript", return_value=[]))
    stack.enter_context(
        patch.object(
            indexer,
            "_embed_captions",
            return_value=(
                np.eye(NUM_SEGMENTS, D, dtype=np.float32)[:, :D],
                np.eye(NUM_SEGMENTS, D, dtype=np.float32)[:, :D],
            ),
        )
    )

    def fake_encode_frames(frames, **kw):
        n = len(frames)
        embs = np.random.default_rng(0).standard_normal((n, D)).astype(np.float32)
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        return embs / np.maximum(norms, 1e-10)

    stack.enter_context(patch.object(indexer, "_encode_frames", side_effect=fake_encode_frames))
    stack.enter_context(patch.object(indexer, "_pre_caption_dedup"))
    stack.enter_context(patch.object(indexer, "_selective_decode"))
    stack.enter_context(patch.object(indexer, "_action_first_pass"))
    stack.enter_context(patch.object(indexer, "_build_coarse_level", return_value=([], None)))
    stack.enter_context(
        patch("kuavi.indexer.detect_scenes", return_value=[(i * 2.0, (i + 1) * 2.0) for i in range(NUM_SEGMENTS)])
    )
    stack.enter_context(
        patch("kuavi.indexer.detect_scenes_hierarchical", return_value={"levels": [
            [(i * 2.0, (i + 1) * 2.0) for i in range(NUM_SEGMENTS)]
        ]})
    )
    stack.enter_context(
        patch("kuavi.scene_detection.detect_scenes_perframe", return_value=[
            (i * 2.0, (i + 1) * 2.0) for i in range(NUM_SEGMENTS)
        ])
    )
    # Provide a query encoder
    indexer._encode_query = lambda t: np.ones(D, dtype=np.float32) / np.sqrt(D)
    indexer._encode_query_siglip = lambda t: np.ones(D, dtype=np.float32) / np.sqrt(D)
    return stack


# ---------------------------------------------------------------------------
# Fix 1: _predict_fn / _predict_future_fn attached after index_video
# ---------------------------------------------------------------------------


class TestPredictorAttachment:
    """Verify _predict_fn and _predict_future_fn are set on VideoIndex."""

    def test_predict_fn_attached_when_predictor_available(self):
        indexer = _make_indexer_with_predictor()
        lv = _make_loaded_video()

        with _patch_indexer_for_index_video(indexer):
            with patch.object(indexer, "_encode_clips_vjepa", side_effect=_fake_vjepa_clips):
                index = indexer.index_video(lv, mode="fast", store_feature_maps=True)

        assert hasattr(index, "_predict_fn"), "_predict_fn must be attached"
        assert index._predict_fn is not None
        assert callable(index._predict_fn)

    def test_predict_future_fn_attached_when_predictor_available(self):
        indexer = _make_indexer_with_predictor()
        lv = _make_loaded_video()

        with _patch_indexer_for_index_video(indexer):
            with patch.object(indexer, "_encode_clips_vjepa", side_effect=_fake_vjepa_clips):
                index = indexer.index_video(lv, mode="fast", store_feature_maps=True)

        assert hasattr(index, "_predict_future_fn"), "_predict_future_fn must be attached"
        assert index._predict_future_fn is not None
        assert callable(index._predict_future_fn)

    def test_predict_fn_not_attached_when_no_predictor(self):
        indexer = VideoIndexer(
            scene_model="facebook/vjepa2-vitl-fpc64-256",
            scene_clip_size=4,
            scene_stride=2,
        )
        indexer._scene_embed_dim = D
        indexer._scene_predictor = None  # No predictor
        lv = _make_loaded_video()

        with _patch_indexer_for_index_video(indexer):
            with patch.object(indexer, "_encode_clips_vjepa", side_effect=_fake_vjepa_clips):
                index = indexer.index_video(lv, mode="fast")

        assert getattr(index, "_predict_fn", None) is None
        assert getattr(index, "_predict_future_fn", None) is None

    def test_predict_fn_returns_embedding(self):
        """_predict_fn(time_point) must return a 1-D embedding vector."""
        indexer = _make_indexer_with_predictor()
        lv = _make_loaded_video()

        with _patch_indexer_for_index_video(indexer):
            with patch.object(indexer, "_encode_clips_vjepa", side_effect=_fake_vjepa_clips):
                index = indexer.index_video(lv, mode="fast", store_feature_maps=True)

        result = index._predict_fn(2.0)
        assert result is not None, "_predict_fn should return an embedding when feature maps exist"
        assert isinstance(result, np.ndarray)
        assert result.ndim == 1
        assert result.shape[0] == D

    def test_predict_future_fn_returns_features(self):
        """_predict_future_fn(feature_map, n) must return (n, D) array."""
        indexer = _make_indexer_with_predictor()
        lv = _make_loaded_video()

        with _patch_indexer_for_index_video(indexer):
            with patch.object(indexer, "_encode_clips_vjepa", side_effect=_fake_vjepa_clips):
                index = indexer.index_video(lv, mode="fast", store_feature_maps=True)

        feature_map = index.temporal_feature_maps[0]
        result = index._predict_future_fn(feature_map, 16)
        assert result is not None
        assert isinstance(result, np.ndarray)
        assert result.shape == (16, D)

    def test_predict_fn_returns_none_for_invalid_time(self):
        """_predict_fn should return None if time_point doesn't match any segment."""
        indexer = _make_indexer_with_predictor()
        lv = _make_loaded_video()

        with _patch_indexer_for_index_video(indexer):
            with patch.object(indexer, "_encode_clips_vjepa", side_effect=_fake_vjepa_clips):
                index = indexer.index_video(lv, mode="fast", store_feature_maps=True)

        # Time way beyond any segment
        result = index._predict_fn(-999.0)
        assert result is None


# ---------------------------------------------------------------------------
# Fix 2: store_feature_maps + overlapping_vjepa
# ---------------------------------------------------------------------------


class TestOverlappingVjepaFeatureMaps:
    """Verify store_feature_maps=True works with overlapping_vjepa=True."""

    def test_feature_maps_stored_with_overlapping(self):
        indexer = _make_indexer_with_predictor()
        lv = _make_loaded_video()

        with _patch_indexer_for_index_video(indexer):
            with patch.object(indexer, "_encode_clips_vjepa", side_effect=_fake_vjepa_clips):
                index = indexer.index_video(
                    lv, mode="fast", store_feature_maps=True, overlapping_vjepa=True,
                )

        assert index.temporal_feature_maps is not None, (
            "temporal_feature_maps must not be None with store_feature_maps=True + overlapping_vjepa=True"
        )
        assert isinstance(index.temporal_feature_maps, (np.ndarray, list))

    def test_feature_maps_not_stored_when_false(self):
        indexer = _make_indexer_with_predictor()
        lv = _make_loaded_video()

        with _patch_indexer_for_index_video(indexer):
            with patch.object(indexer, "_encode_clips_vjepa", side_effect=_fake_vjepa_clips):
                index = indexer.index_video(
                    lv, mode="fast", store_feature_maps=False, overlapping_vjepa=True,
                )

        assert index.temporal_feature_maps is None

    def test_feature_maps_stored_non_overlapping(self):
        """store_feature_maps=True with overlapping_vjepa=False also works."""
        indexer = _make_indexer_with_predictor()
        lv = _make_loaded_video()

        with _patch_indexer_for_index_video(indexer):
            with patch.object(indexer, "_encode_clips_vjepa", side_effect=_fake_vjepa_clips):
                index = indexer.index_video(
                    lv, mode="fast", store_feature_maps=True, overlapping_vjepa=False,
                )

        assert index.temporal_feature_maps is not None

    def test_encode_overlapping_with_feature_maps_returns_3_tuple(self):
        """_encode_frames_overlapping_vjepa returns 3-tuple when store_feature_maps=True."""
        indexer = _make_indexer_with_predictor()
        indexer._encode_clips_vjepa = _fake_vjepa_clips

        rng = np.random.default_rng(42)
        frames = [rng.integers(0, 255, (16, 16, 3), dtype=np.uint8) for _ in range(20)]
        timestamps = [i * 0.5 for i in range(20)]

        result = indexer._encode_frames_overlapping_vjepa(
            frames, timestamps, clip_size=4, stride=2, store_feature_maps=True,
        )
        assert len(result) == 3, "Should return (embeddings, timestamps, feature_maps)"
        embs, ts, fmaps = result
        assert isinstance(embs, np.ndarray)
        assert len(ts) == len(frames)
        assert isinstance(fmaps, list)
        assert all(isinstance(fm, np.ndarray) for fm in fmaps)

    def test_encode_overlapping_without_feature_maps_returns_2_tuple(self):
        """_encode_frames_overlapping_vjepa returns 2-tuple when store_feature_maps=False."""
        indexer = _make_indexer_with_predictor()
        indexer._encode_clips_vjepa = _fake_vjepa_clips

        rng = np.random.default_rng(42)
        frames = [rng.integers(0, 255, (16, 16, 3), dtype=np.uint8) for _ in range(20)]
        timestamps = [i * 0.5 for i in range(20)]

        result = indexer._encode_frames_overlapping_vjepa(
            frames, timestamps, clip_size=4, stride=2, store_feature_maps=False,
        )
        assert len(result) == 2, "Should return (embeddings, timestamps)"


# ---------------------------------------------------------------------------
# Fix 1 + Fix 2 end-to-end: search tools use predictor
# ---------------------------------------------------------------------------


class TestSearchToolsUsePredictor:
    """Verify make_anticipate_action / make_predict_future / make_verify_coherence
    use the attached predictor closures and report method == 'vjepa2_predictor'."""

    def _make_index_with_predictor(self):
        indexer = _make_indexer_with_predictor()
        lv = _make_loaded_video()

        with _patch_indexer_for_index_video(indexer):
            with patch.object(indexer, "_encode_clips_vjepa", side_effect=_fake_vjepa_clips):
                return indexer.index_video(lv, mode="fast", store_feature_maps=True)

    def test_anticipate_action_uses_predictor(self):
        index = self._make_index_with_predictor()
        assert index._predict_fn is not None

        tool = make_anticipate_action(index)["tool"]
        result = tool(time_point=2.0, top_k=3)

        assert result.get("method") == "vjepa2_predictor", (
            f"Expected vjepa2_predictor, got {result.get('method')}"
        )
        assert "predicted_segments" in result

    def test_predict_future_uses_predictor(self):
        index = self._make_index_with_predictor()
        assert index._predict_future_fn is not None
        assert index.temporal_feature_maps is not None

        tool = make_predict_future(index)["tool"]
        result = tool(start_time=0.0, end_time=2.0)

        assert result.get("method") == "vjepa2_predictor", (
            f"Expected vjepa2_predictor, got {result.get('method')}"
        )
        assert "predicted_segments" in result

    def test_verify_coherence_uses_predictor(self):
        index = self._make_index_with_predictor()
        assert index._predict_future_fn is not None
        assert index.temporal_feature_maps is not None

        tool = make_verify_coherence(index)["tool"]
        result = tool(start_time=0.0, end_time=12.0)

        assert result.get("method") == "vjepa2_predictor", (
            f"Expected vjepa2_predictor, got {result.get('method')}"
        )
        assert "overall_score" in result
        assert "segment_scores" in result
        assert "anomalies" in result

    def test_anticipate_action_with_candidates(self):
        """Predictor path returns predictions; candidate ranking is fallback-only."""
        index = self._make_index_with_predictor()
        tool = make_anticipate_action(index)["tool"]
        candidates = ["walking", "running", "sitting"]
        result = tool(time_point=2.0, top_k=3, candidates=candidates)

        assert result.get("method") == "vjepa2_predictor"
        assert "predicted_segments" in result


# ---------------------------------------------------------------------------
# Fix 2: _encode_frames_overlapping_vjepa edge cases with feature maps
# ---------------------------------------------------------------------------


class TestOverlappingFeatureMapEdgeCases:
    def test_empty_frames_returns_empty_feature_maps(self):
        indexer = _make_indexer_with_predictor()
        indexer._encode_clips_vjepa = _fake_vjepa_clips

        result = indexer._encode_frames_overlapping_vjepa(
            [], [], clip_size=4, stride=2, store_feature_maps=True,
        )
        assert len(result) == 3
        embs, ts, fmaps = result
        assert embs.shape[0] == 0
        assert ts == []
        assert fmaps == []

    def test_single_frame_with_feature_maps(self):
        indexer = _make_indexer_with_predictor()
        indexer._encode_clips_vjepa = _fake_vjepa_clips

        frames = [np.zeros((16, 16, 3), dtype=np.uint8)]
        timestamps = [0.0]

        result = indexer._encode_frames_overlapping_vjepa(
            frames, timestamps, clip_size=4, stride=2, store_feature_maps=True,
        )
        assert len(result) == 3
        embs, ts, fmaps = result
        assert embs.shape[0] == 1
        assert len(fmaps) >= 1


# ---------------------------------------------------------------------------
# Fix 3: ASR graceful fallback (import-level test only)
# ---------------------------------------------------------------------------


class TestASRGracefulFallback:
    """Verify _run_asr handles missing qwen_asr gracefully."""

    def test_missing_qwen_asr_returns_empty(self):
        indexer = VideoIndexer()
        with patch.dict("sys.modules", {"qwen_asr": None}):
            result = indexer._run_asr("/fake/video.mp4", "Qwen/Qwen3-ASR-0.6B")
        assert result == []

    def test_ensure_asr_model_imports_only_qwen3asrmodel(self):
        """_ensure_asr_model should import Qwen3ASRModel, not Qwen3ForcedAligner."""
        import inspect
        source = inspect.getsource(VideoIndexer._ensure_asr_model)
        assert "from qwen_asr import Qwen3ASRModel" in source
        lines = source.split("\n")
        top_import_lines = [
            l for l in lines
            if "from qwen_asr import" in l and "Qwen3ForcedAligner" in l
            and "# noqa" not in l
        ]
        assert len(top_import_lines) == 0, (
            "Qwen3ForcedAligner should not be imported at top level of _ensure_asr_model"
        )

    def test_uses_correct_aligner_model_name(self):
        """_ensure_asr_model should reference Qwen/Qwen3-ForcedAligner-0.6B, not the old 404 name."""
        import inspect
        source = inspect.getsource(VideoIndexer._ensure_asr_model)
        assert "Qwen/Qwen3-ForcedAligner-0.6B" in source
        assert "Qwen/Qwen3-ASR-ForcedAligner" not in source


# ---------------------------------------------------------------------------
# Fix 4: VideoRLM passes store_feature_maps
# ---------------------------------------------------------------------------


class TestVideoRLMStoreFeatureMaps:
    """Verify VideoRLM constructor accepts and passes store_feature_maps."""

    def test_constructor_accepts_store_feature_maps(self):
        from rlm.video.video_rlm import VideoRLM

        rlm = VideoRLM(store_feature_maps=True, enable_search=False)
        assert rlm.store_feature_maps is True

    def test_constructor_defaults_false(self):
        from rlm.video.video_rlm import VideoRLM

        rlm = VideoRLM(enable_search=False)
        assert rlm.store_feature_maps is False


# ---------------------------------------------------------------------------
# RLM mirror: same predictor wiring in video_indexer.py
# ---------------------------------------------------------------------------


class TestRLMMirrorPredictorWiring:
    """Verify the RLM mirror file also attaches predictor closures."""

    def test_rlm_indexer_has_predict_future_embedding(self):
        from rlm.video.video_indexer import VideoIndexer as RLMVideoIndexer

        indexer = RLMVideoIndexer()
        assert hasattr(indexer, "_predict_future_embedding")

    def test_rlm_indexer_scene_predictor_init_none(self):
        from rlm.video.video_indexer import VideoIndexer as RLMVideoIndexer

        indexer = RLMVideoIndexer()
        assert indexer._scene_predictor is None

    def test_rlm_asr_uses_correct_aligner_name(self):
        """RLM mirror _ensure_asr_model should reference Qwen/Qwen3-ForcedAligner-0.6B."""
        import inspect

        from rlm.video.video_indexer import VideoIndexer as RLMVideoIndexer

        source = inspect.getsource(RLMVideoIndexer._ensure_asr_model)
        assert "Qwen/Qwen3-ForcedAligner-0.6B" in source
        assert "Qwen/Qwen3-ASR-ForcedAligner" not in source
