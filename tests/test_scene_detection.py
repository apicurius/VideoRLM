"""Tests for embedding-based scene boundary detection."""

import numpy as np
import pytest

from rlm.video.scene_detection import detect_scenes


def _identity_embed(frames: list[np.ndarray]) -> np.ndarray:
    """Embed each frame as its flattened, normalised mean colour vector.

    This gives identical embeddings for identical frames and distinct
    embeddings for frames with different colours — good enough for testing
    without a real model.
    """
    vecs = []
    for f in frames:
        # Mean colour per channel → 3-dim vector
        vec = f.mean(axis=(0, 1)).astype(np.float64)
        vecs.append(vec)
    return np.array(vecs)


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

        scenes = detect_scenes(frames, timestamps, embed_fn=_identity_embed, threshold=0.01)

        assert len(scenes) >= 2
        assert scenes[0][0] == pytest.approx(0.0)
        boundary_starts = [s for s, _e in scenes]
        assert any(abs(b - 5.0) <= 1.0 for b in boundary_starts)

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

        scenes = detect_scenes(frames, timestamps, embed_fn=_identity_embed, threshold=0.01)
        assert len(scenes) >= 3


class TestDetectScenesThreshold:
    """Higher thresholds should produce fewer scenes."""

    def test_high_threshold_fewer_scenes(self):
        blue = np.full((48, 64, 3), [255, 0, 0], dtype=np.uint8)
        red = np.full((48, 64, 3), [0, 0, 255], dtype=np.uint8)

        frames = [blue.copy() for _ in range(5)] + [red.copy() for _ in range(5)]
        timestamps = [float(i) for i in range(10)]

        scenes_low = detect_scenes(
            frames, timestamps, embed_fn=_identity_embed, threshold=0.01
        )
        scenes_high = detect_scenes(
            frames, timestamps, embed_fn=_identity_embed, threshold=1e6
        )

        assert len(scenes_high) == 1
        assert len(scenes_low) >= 2
