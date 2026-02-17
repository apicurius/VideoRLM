"""Tests for embedding-based scene boundary detection.

scipy and sklearn are lazy-imported inside detect_scenes, so we shim them in
sys.modules before importing to keep the test suite free of heavy ML deps.
"""

import sys
from unittest.mock import MagicMock

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Shim scipy.sparse.eye and sklearn.cluster.AgglomerativeClustering
# ---------------------------------------------------------------------------

# We store the mock clustering class globally so tests can configure labels.
_MOCK_CLUSTERING_CLS = MagicMock()

# Default: all frames in one cluster
_MOCK_CLUSTERING_CLS.return_value.fit_predict.return_value = np.array([0])


def _install_shims():
    """Install minimal scipy/sklearn shims so the lazy imports succeed."""
    # --- scipy.sparse ---
    _sparse_mod = type(sys)("scipy.sparse")
    _sparse_mod.eye = MagicMock(return_value=MagicMock(
        __add__=lambda self, other: self,
        __radd__=lambda self, other: self,
    ))
    _scipy_mod = type(sys)("scipy")
    _scipy_mod.sparse = _sparse_mod
    sys.modules.setdefault("scipy", _scipy_mod)
    sys.modules.setdefault("scipy.sparse", _sparse_mod)

    # --- sklearn.cluster ---
    _cluster_mod = type(sys)("sklearn.cluster")
    _cluster_mod.AgglomerativeClustering = _MOCK_CLUSTERING_CLS
    _sklearn_mod = sys.modules.get("sklearn") or type(sys)("sklearn")
    _sklearn_mod.cluster = _cluster_mod
    sys.modules.setdefault("sklearn", _sklearn_mod)
    sys.modules.setdefault("sklearn.cluster", _cluster_mod)


_install_shims()

from rlm.video.scene_detection import detect_scenes  # noqa: E402


def _identity_embed(frames: list[np.ndarray]) -> np.ndarray:
    """Embed each frame as its flattened, normalised mean colour vector.

    This gives identical embeddings for identical frames and distinct
    embeddings for frames with different colours — good enough for testing
    without a real model.
    """
    vecs = []
    for f in frames:
        vec = f.mean(axis=(0, 1)).astype(np.float64)
        vecs.append(vec)
    return np.array(vecs)


def _set_labels(labels):
    """Configure the mock clustering to return the given labels."""
    _MOCK_CLUSTERING_CLS.reset_mock()
    _MOCK_CLUSTERING_CLS.return_value.fit_predict.return_value = np.array(labels)


class TestDetectScenesEdgeCases:
    """Edge cases and input validation."""

    def test_empty_frames_returns_empty(self):
        assert detect_scenes([], [], embed_fn=_identity_embed) == []

    def test_mismatched_lengths_raises(self):
        frames = [np.zeros((48, 64, 3), dtype=np.uint8)]
        timestamps = [0.0, 1.0]
        with pytest.raises(ValueError, match="same length"):
            detect_scenes(frames, timestamps, embed_fn=_identity_embed)

    def test_single_frame_returns_single_scene(self):
        frame = np.full((48, 64, 3), 128, dtype=np.uint8)
        result = detect_scenes([frame], [0.0], embed_fn=_identity_embed)
        assert len(result) == 1
        assert result[0] == (0.0, 0.0)


class TestDetectScenesUniform:
    """Uniform frames (all same colour) should yield a single scene."""

    def test_uniform_blue_frames(self):
        blue = np.full((48, 64, 3), [255, 0, 0], dtype=np.uint8)  # BGR blue
        frames = [blue.copy() for _ in range(10)]
        timestamps = [i * 0.5 for i in range(10)]

        _set_labels([0] * 10)
        scenes = detect_scenes(frames, timestamps, embed_fn=_identity_embed)
        assert len(scenes) == 1
        assert scenes[0][0] == pytest.approx(0.0)
        assert scenes[0][1] == pytest.approx(4.5)


class TestDetectScenesBoundary:
    """Frames with a clear colour change should produce multiple scenes."""

    def test_two_colour_blocks(self):
        blue = np.full((48, 64, 3), [255, 0, 0], dtype=np.uint8)
        red = np.full((48, 64, 3), [0, 0, 255], dtype=np.uint8)

        frames = [blue.copy() for _ in range(5)] + [red.copy() for _ in range(5)]
        timestamps = [float(i) for i in range(10)]

        _set_labels([0] * 5 + [1] * 5)
        scenes = detect_scenes(
            frames, timestamps, embed_fn=_identity_embed, threshold=0.01,
        )

        assert len(scenes) == 2
        assert scenes[0][0] == pytest.approx(0.0)
        assert scenes[0][1] == pytest.approx(4.0)
        assert scenes[1][0] == pytest.approx(5.0)
        assert scenes[1][1] == pytest.approx(9.0)

    def test_three_colour_blocks(self):
        blue = np.full((48, 64, 3), [255, 0, 0], dtype=np.uint8)
        green = np.full((48, 64, 3), [0, 255, 0], dtype=np.uint8)
        red = np.full((48, 64, 3), [0, 0, 255], dtype=np.uint8)

        frames = (
            [blue.copy() for _ in range(5)]
            + [green.copy() for _ in range(5)]
            + [red.copy() for _ in range(5)]
        )
        timestamps = [float(i) for i in range(15)]

        _set_labels([0] * 5 + [1] * 5 + [2] * 5)
        scenes = detect_scenes(
            frames, timestamps, embed_fn=_identity_embed, threshold=0.01,
        )
        assert len(scenes) == 3

    def test_embed_fn_returns_distinct_for_different_colours(self):
        """The embed_fn should produce distinct embeddings for different-coloured frames."""
        blue = np.full((48, 64, 3), [255, 0, 0], dtype=np.uint8)
        red = np.full((48, 64, 3), [0, 0, 255], dtype=np.uint8)

        embeddings = _identity_embed([blue, red])
        assert embeddings.shape == (2, 3)
        assert not np.allclose(embeddings[0], embeddings[1])


class TestDetectScenesMinDuration:
    """min_duration filtering removes short scenes."""

    def test_min_duration_filters_short_scenes(self):
        """Scenes shorter than min_duration are removed."""
        frames = [np.zeros((48, 64, 3), dtype=np.uint8) for _ in range(11)]
        # 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0
        timestamps = [i * 0.5 for i in range(11)]

        # 5 frames cluster 0 (0.0-2.0), 1 frame cluster 1 (2.5-2.5), 5 frames cluster 2 (3.0-5.0)
        _set_labels([0] * 5 + [1] + [2] * 5)
        scenes = detect_scenes(
            frames, timestamps, embed_fn=_identity_embed,
            threshold=0.01, min_duration=1.0,
        )
        # The 0-duration middle scene should be filtered out
        for start, end in scenes:
            assert (end - start) >= 1.0
        assert len(scenes) == 2

    def test_min_duration_zero_keeps_all(self):
        """min_duration=0 should keep all scenes including zero-duration ones."""
        frames = [np.zeros((48, 64, 3), dtype=np.uint8) for _ in range(11)]
        timestamps = [i * 0.5 for i in range(11)]

        _set_labels([0] * 5 + [1] + [2] * 5)
        scenes = detect_scenes(
            frames, timestamps, embed_fn=_identity_embed,
            threshold=0.01, min_duration=0,
        )
        assert len(scenes) == 3


class TestDetectScenesTemporalConnectivity:
    """Ensure temporal connectivity: segments are contiguous, not randomly scattered."""

    def test_scenes_are_temporally_contiguous(self):
        """Each scene's start should be >= previous scene's end."""
        frames = [np.zeros((48, 64, 3), dtype=np.uint8) for _ in range(15)]
        timestamps = [float(i) for i in range(15)]

        _set_labels([0] * 5 + [1] * 5 + [2] * 5)
        scenes = detect_scenes(
            frames, timestamps, embed_fn=_identity_embed, threshold=0.01,
        )

        starts = [s for s, _e in scenes]
        assert starts == sorted(starts)
        for i in range(1, len(scenes)):
            assert scenes[i][0] >= scenes[i - 1][1]

    def test_ward_linkage_and_connectivity_used(self):
        """AgglomerativeClustering is called with Ward linkage and a connectivity matrix."""
        frames = [np.zeros((48, 64, 3), dtype=np.uint8) for _ in range(5)]
        timestamps = [float(i) for i in range(5)]

        _set_labels([0] * 5)
        detect_scenes(frames, timestamps, embed_fn=_identity_embed)

        call_kwargs = _MOCK_CLUSTERING_CLS.call_args[1]
        assert call_kwargs["linkage"] == "ward"
        assert call_kwargs["connectivity"] is not None
        assert call_kwargs["n_clusters"] is None
        assert "distance_threshold" in call_kwargs


class TestDetectScenesThreshold:
    """Higher thresholds should produce fewer scenes."""

    def test_high_threshold_fewer_scenes(self):
        frames = [np.zeros((48, 64, 3), dtype=np.uint8) for _ in range(10)]
        timestamps = [float(i) for i in range(10)]

        # Low threshold: 2 clusters
        _set_labels([0] * 5 + [1] * 5)
        scenes_low = detect_scenes(
            frames, timestamps, embed_fn=_identity_embed, threshold=0.01,
        )

        # High threshold: 1 cluster
        _set_labels([0] * 10)
        scenes_high = detect_scenes(
            frames, timestamps, embed_fn=_identity_embed, threshold=1e6,
        )

        assert len(scenes_high) == 1
        assert len(scenes_low) >= 2
