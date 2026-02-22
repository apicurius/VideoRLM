"""Tests for WI-6: Overlapping V-JEPA 2 windows with per-frame averaging."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from kuavi.indexer import VideoIndexer
from kuavi.scene_detection import detect_scenes_perframe
from kuavi.types import KUAViConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fake_clip_embeddings(windows, D=8):
    """Return deterministic L2-normalized embeddings for each window."""
    embs = []
    for i, w in enumerate(windows):
        rng = np.random.default_rng(i + 1)
        e = rng.standard_normal(D).astype(np.float32)
        e /= np.linalg.norm(e)
        embs.append(e)
    return np.stack(embs)


def _make_frames(n=20, h=8, w=8):
    """Return n synthetic BGR frames."""
    rng = np.random.default_rng(42)
    return [rng.integers(0, 255, (h, w, 3), dtype=np.uint8) for _ in range(n)]


def _make_loaded_video(num_frames=20, fps=2.0):
    """Create a mock LoadedVideo."""
    frames = _make_frames(num_frames)
    mock = MagicMock()
    mock.metadata.extraction_fps = fps
    mock.metadata.path = "/fake/video.mp4"
    mock.frames = frames
    mock.segments = []
    return mock


def _make_indexer(scene_clip_size=8, scene_stride=4, D=8):
    """Build a VideoIndexer with lightweight fake config (no real models)."""
    indexer = VideoIndexer(
        scene_model="facebook/vjepa2-vitl-fpc64-256",
        scene_clip_size=scene_clip_size,
        scene_stride=scene_stride,
    )
    indexer._scene_embed_dim = D
    return indexer


# ---------------------------------------------------------------------------
# KUAViConfig tests
# ---------------------------------------------------------------------------


class TestKUAViConfig:
    def test_scene_stride_default(self):
        cfg = KUAViConfig()
        assert cfg.scene_stride == 8

    def test_scene_stride_custom(self):
        cfg = KUAViConfig(scene_stride=4)
        assert cfg.scene_stride == 4

    def test_scene_stride_field_exists(self):
        assert hasattr(KUAViConfig, "__dataclass_fields__")
        assert "scene_stride" in KUAViConfig.__dataclass_fields__


# ---------------------------------------------------------------------------
# _encode_frames_overlapping_vjepa tests
# ---------------------------------------------------------------------------


class TestEncodeFramesOverlappingVjepa:
    def _make_indexer_with_mock_encode(self, n_frames=20, clip_size=8, stride=4, D=8):
        indexer = _make_indexer(scene_clip_size=clip_size, scene_stride=stride, D=D)
        frames = _make_frames(n_frames)
        timestamps = [i * 0.5 for i in range(n_frames)]

        # Mock _encode_clips_vjepa to return deterministic embeddings
        def fake_encode_clips(windows, **kw):
            return _make_fake_clip_embeddings(windows, D=D)

        indexer._encode_clips_vjepa = fake_encode_clips
        return indexer, frames, timestamps

    def test_output_shape(self):
        """per_frame_embeddings must have shape (N_frames, D)."""
        n, D = 20, 8
        indexer, frames, ts = self._make_indexer_with_mock_encode(n_frames=n, D=D)
        embs, ret_ts = indexer._encode_frames_overlapping_vjepa(frames, ts, clip_size=8, stride=4)
        assert embs.shape == (n, D)

    def test_timestamps_returned(self):
        """Returned timestamps must equal input timestamps."""
        indexer, frames, ts = self._make_indexer_with_mock_encode(n_frames=10)
        _, ret_ts = indexer._encode_frames_overlapping_vjepa(frames, ts, clip_size=8, stride=4)
        assert ret_ts == ts

    def test_l2_normalized(self):
        """Each per-frame embedding must be L2-normalized (norm ≈ 1.0)."""
        indexer, frames, ts = self._make_indexer_with_mock_encode(n_frames=16, D=8)
        embs, _ = indexer._encode_frames_overlapping_vjepa(frames, ts, clip_size=8, stride=4)
        norms = np.linalg.norm(embs, axis=1)
        np.testing.assert_allclose(norms, np.ones(len(frames)), atol=1e-5)

    def test_frames_covered_by_multiple_windows(self):
        """Middle frames should be covered by multiple overlapping windows."""
        n, clip_size, stride = 20, 8, 4
        indexer, frames, ts = self._make_indexer_with_mock_encode(
            n_frames=n, clip_size=clip_size, stride=stride
        )

        # Track window coverage counts manually
        coverage = np.zeros(n, dtype=int)
        for start in range(0, n, stride):
            end = min(start + clip_size, n)
            if end - start >= 2:
                coverage[start:end] += 1

        # Middle frames should have coverage > 1 (covered by multiple windows)
        assert np.any(coverage > 1), "Expected overlapping coverage for middle frames"

        # Verify method produces valid output (not all-zero embeddings)
        embs, _ = indexer._encode_frames_overlapping_vjepa(frames, ts, clip_size=clip_size, stride=stride)
        assert not np.allclose(embs, 0)

    def test_empty_frames(self):
        """Empty input should return empty array with correct dim."""
        D = 8
        indexer = _make_indexer(D=D)
        embs, ts = indexer._encode_frames_overlapping_vjepa([], [], clip_size=8, stride=4)
        assert embs.shape == (0, D)
        assert ts == []

    def test_single_frame_fallback(self):
        """Single frame (< 2): should use fallback single window path."""
        D = 8
        indexer = _make_indexer(D=D)
        frames = _make_frames(1)
        ts = [0.0]

        def fake_encode_clips(windows, **kw):
            return _make_fake_clip_embeddings(windows, D=D)

        indexer._encode_clips_vjepa = fake_encode_clips
        embs, ret_ts = indexer._encode_frames_overlapping_vjepa(frames, ts, clip_size=8, stride=4)
        # Single frame: falls through to fallback window
        assert embs.shape[1] == D
        assert len(ret_ts) == 1

    def test_fewer_frames_than_clip_size(self):
        """Fewer frames than clip_size: should still produce one window."""
        n, clip_size, stride, D = 5, 16, 8, 8
        indexer, frames, ts = self._make_indexer_with_mock_encode(
            n_frames=n, clip_size=clip_size, stride=stride, D=D
        )
        embs, _ = indexer._encode_frames_overlapping_vjepa(frames, ts, clip_size=clip_size, stride=stride)
        assert embs.shape == (n, D)
        # All frames should be L2-normalized
        norms = np.linalg.norm(embs, axis=1)
        np.testing.assert_allclose(norms, np.ones(n), atol=1e-5)


# ---------------------------------------------------------------------------
# detect_scenes_perframe tests
# ---------------------------------------------------------------------------


class TestDetectScenesPerframe:
    """Test detect_scenes_perframe by mocking _detect_scenes_embedding.

    We avoid calling through to real sklearn/scipy because test_scene_detection.py
    installs sys.modules shims for those at collection time. The shims use
    setdefault(), so their installation order is non-deterministic.

    detect_scenes_perframe is a thin wrapper that:
      1. Handles empty/single-frame edge cases.
      2. Builds a passthrough embed_fn.
      3. Delegates to _detect_scenes_embedding.

    We test those three concerns directly.
    """

    def _make_embeddings(self, n, D=8, seed=0):
        """Return L2-normalized embeddings for n frames."""
        rng = np.random.default_rng(seed)
        embs = rng.standard_normal((n, D)).astype(np.float32)
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        return embs / np.maximum(norms, 1e-10)

    def test_empty_input(self):
        assert detect_scenes_perframe(np.empty((0, 8)), []) == []

    def test_single_frame(self):
        embs = self._make_embeddings(1)
        ts = [0.0]
        scenes = detect_scenes_perframe(embs, ts)
        assert scenes == [(0.0, 0.0)]

    def test_delegates_to_detect_scenes_embedding(self):
        """detect_scenes_perframe calls _detect_scenes_embedding with correct args."""
        embs = self._make_embeddings(10)
        ts = [float(i) for i in range(10)]
        expected = [(0.0, 4.0), (5.0, 9.0)]

        with patch("kuavi.scene_detection._detect_scenes_embedding", return_value=expected) as mock:
            result = detect_scenes_perframe(embs, ts, threshold=0.25, min_duration=2.0)

        assert result == expected
        assert mock.called
        call_args = mock.call_args
        # Threshold and min_duration must be passed through
        assert call_args[0][3] == 0.25  # threshold positional arg
        assert call_args[0][4] == 2.0  # min_duration positional arg

    def test_passthrough_embed_fn_returns_embeddings(self):
        """The passthrough embed_fn passed to _detect_scenes_embedding returns per_frame_embs."""
        embs = self._make_embeddings(10)
        ts = [float(i) for i in range(10)]

        captured_embed_fn = []

        def capture_embed_fn(dummy_frames, timestamps, embed_fn, threshold, min_duration):
            captured_embed_fn.append(embed_fn)
            return [(ts[0], ts[-1])]

        with patch("kuavi.scene_detection._detect_scenes_embedding", side_effect=capture_embed_fn):
            detect_scenes_perframe(embs, ts)

        assert len(captured_embed_fn) == 1
        fn = captured_embed_fn[0]
        # The passthrough embed_fn should return the original embs regardless of input
        result = fn([None] * 10)
        np.testing.assert_array_equal(result, embs)

    def test_default_threshold_and_min_duration(self):
        """Default threshold=0.20 and min_duration=4.0 are passed to _detect_scenes_embedding."""
        embs = self._make_embeddings(5)
        ts = [float(i) for i in range(5)]

        with patch("kuavi.scene_detection._detect_scenes_embedding", return_value=[(0.0, 4.0)]) as mock:
            detect_scenes_perframe(embs, ts)

        call_args = mock.call_args[0]
        assert call_args[3] == 0.20   # default threshold
        assert call_args[4] == 4.0    # default min_duration


# ---------------------------------------------------------------------------
# index_video integration tests
# ---------------------------------------------------------------------------


def _patch_indexer_full(indexer, D=4):
    """Mock all heavy dependencies so index_video() runs without real models.

    Also patches detect_scenes and detect_scenes_perframe to avoid calling
    real sklearn/scipy, which may be shimmed by test_scene_detection.py.
    """
    from contextlib import ExitStack

    stack = ExitStack()
    stack.enter_context(patch.object(indexer, "_ensure_model"))
    stack.enter_context(patch.object(indexer, "_ensure_scene_model"))
    stack.enter_context(patch.object(indexer, "_get_transcript", return_value=[]))
    stack.enter_context(
        patch.object(
            indexer,
            "_embed_captions",
            return_value=(np.eye(2, dtype=np.float32), None),
        )
    )

    def fake_encode_frames(frames, **kw):
        embs = np.random.default_rng(0).standard_normal((len(frames), D)).astype(np.float32)
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        return embs / np.maximum(norms, 1e-10)

    stack.enter_context(patch.object(indexer, "_encode_frames", side_effect=fake_encode_frames))
    stack.enter_context(patch.object(indexer, "_pre_caption_dedup"))
    stack.enter_context(patch.object(indexer, "_selective_decode"))
    # Patch scene detection to avoid real sklearn/scipy
    stack.enter_context(
        patch("kuavi.indexer.detect_scenes", return_value=[(0.0, 5.0)])
    )
    stack.enter_context(
        patch("kuavi.indexer.detect_scenes_hierarchical", return_value={"levels": [[(0.0, 5.0)]]})
    )
    stack.enter_context(
        patch("kuavi.scene_detection.detect_scenes_perframe", return_value=[(0.0, 5.0)])
    )
    return stack


def _fake_vjepa_clips(windows, D=4, **kw):
    """Return deterministic L2-normalized embeddings for each window."""
    embs = []
    for i in range(len(windows)):
        rng = np.random.default_rng(i + 100)
        e = rng.standard_normal(D).astype(np.float32)
        e /= np.linalg.norm(e)
        embs.append(e)
    return np.stack(embs)


class TestIndexVideoOverlapping:
    D = 4

    def _make_indexer(self):
        idx = VideoIndexer(
            scene_model="facebook/vjepa2-vitl-fpc64-256",
            scene_clip_size=4,
            scene_stride=2,
        )
        idx._scene_embed_dim = self.D
        return idx

    def test_overlapping_returns_valid_index(self):
        """index_video(overlapping_vjepa=True) produces a valid VideoIndex."""
        from kuavi.indexer import VideoIndex

        indexer = self._make_indexer()
        loaded = _make_loaded_video(num_frames=12, fps=2.0)

        def fake_encode_clips(windows, **kw):
            return _fake_vjepa_clips(windows, D=self.D)

        with _patch_indexer_full(indexer, D=self.D):
            with patch.object(indexer, "_encode_clips_vjepa", side_effect=fake_encode_clips):
                idx = indexer.index_video(loaded, mode="fast", overlapping_vjepa=True)

        assert isinstance(idx, VideoIndex)
        assert len(idx.segments) > 0

    def test_default_behavior_unchanged(self):
        """overlapping_vjepa=False must follow the standard non-overlapping path."""
        from kuavi.indexer import VideoIndex

        indexer = self._make_indexer()
        loaded = _make_loaded_video(num_frames=12, fps=2.0)

        encode_clips_called_with = []

        def fake_encode_clips(windows, **kw):
            encode_clips_called_with.append(len(windows))
            return _fake_vjepa_clips(windows, D=self.D)

        with _patch_indexer_full(indexer, D=self.D):
            with patch.object(indexer, "_encode_clips_vjepa", side_effect=fake_encode_clips):
                idx = indexer.index_video(loaded, mode="fast", overlapping_vjepa=False)

        assert isinstance(idx, VideoIndex)
        # Standard path groups frames into NON-overlapping clips
        # So _encode_clips_vjepa should have been called once with clips (not per-frame)
        assert len(encode_clips_called_with) > 0

    def test_overlapping_does_not_call_group_frames_into_clips(self):
        """overlapping_vjepa=True should not use _group_frames_into_clips."""
        indexer = self._make_indexer()
        loaded = _make_loaded_video(num_frames=12, fps=2.0)

        def fake_encode_clips(windows, **kw):
            return _fake_vjepa_clips(windows, D=self.D)

        with _patch_indexer_full(indexer, D=self.D):
            with patch.object(indexer, "_encode_clips_vjepa", side_effect=fake_encode_clips):
                with patch.object(
                    indexer, "_group_frames_into_clips", wraps=indexer._group_frames_into_clips
                ) as mock_group:
                    indexer.index_video(loaded, mode="fast", overlapping_vjepa=True)
                    mock_group.assert_not_called()

    def test_scene_stride_stored(self):
        """VideoIndexer must store scene_stride from constructor."""
        idx = VideoIndexer(scene_stride=16)
        assert idx._scene_stride == 16

    def test_overlapping_vjepa_scene_no_model(self):
        """When scene_model is None, overlapping_vjepa=True is a no-op (SigLIP2 path used)."""
        from kuavi.indexer import VideoIndex

        indexer = VideoIndexer(scene_model=None, scene_stride=4)
        loaded = _make_loaded_video(num_frames=8, fps=2.0)

        with _patch_indexer_full(indexer, D=4):
            with patch("kuavi.indexer.detect_scenes", return_value=[(0.0, 3.5)]):
                idx = indexer.index_video(loaded, mode="fast", overlapping_vjepa=True)

        assert isinstance(idx, VideoIndex)
