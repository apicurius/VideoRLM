"""Tests for VideoRLM._make_extract_frames and the extract_frames closure."""

import base64

import cv2
import numpy as np
import pytest

from rlm.video.video_rlm import VideoRLM


def _make_synthetic_video(
    path: str,
    num_frames: int = 30,
    fps: float = 30.0,
    width: int = 64,
    height: int = 48,
) -> str:
    """Write a small synthetic video file using OpenCV."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
    for i in range(num_frames):
        frame = np.full((height, width, 3), fill_value=i % 256, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path


class TestExtractFramesReturnsTaggedDicts:
    """extract_frames should return dicts with __image__, data, mime_type keys."""

    def test_returns_list_of_tagged_image_dicts(self, tmp_path):
        video_path = _make_synthetic_video(str(tmp_path / "test.mp4"), num_frames=30, fps=30.0)
        extract_frames = VideoRLM._make_extract_frames(video_path)

        result = extract_frames(start_time=0.0, end_time=0.5, fps=2.0, max_frames=10)

        assert isinstance(result, list)
        assert len(result) > 0
        for frame_dict in result:
            assert isinstance(frame_dict, dict)
            assert frame_dict["__image__"] is True
            assert "data" in frame_dict
            assert "mime_type" in frame_dict
            # data should be valid base64
            decoded = base64.b64decode(frame_dict["data"])
            assert len(decoded) > 0

    def test_default_mime_type_is_jpeg(self, tmp_path):
        video_path = _make_synthetic_video(str(tmp_path / "test.mp4"), num_frames=30, fps=30.0)
        extract_frames = VideoRLM._make_extract_frames(video_path, image_format=".jpg")

        result = extract_frames(start_time=0.0, end_time=0.5)
        for frame_dict in result:
            assert frame_dict["mime_type"] == "image/jpeg"

    def test_png_format(self, tmp_path):
        video_path = _make_synthetic_video(str(tmp_path / "test.mp4"), num_frames=30, fps=30.0)
        extract_frames = VideoRLM._make_extract_frames(video_path, image_format=".png")

        result = extract_frames(start_time=0.0, end_time=0.5)
        for frame_dict in result:
            assert frame_dict["mime_type"] == "image/png"


class TestExtractFramesMaxFrames:
    """extract_frames should respect the max_frames limit."""

    def test_max_frames_limits_output(self, tmp_path):
        video_path = _make_synthetic_video(str(tmp_path / "test.mp4"), num_frames=90, fps=30.0)
        extract_frames = VideoRLM._make_extract_frames(video_path)

        # Request high fps over a large range — should be capped by max_frames
        result = extract_frames(start_time=0.0, end_time=3.0, fps=30.0, max_frames=5)
        assert len(result) == 5

    def test_returns_fewer_when_range_has_fewer_frames(self, tmp_path):
        video_path = _make_synthetic_video(str(tmp_path / "test.mp4"), num_frames=30, fps=30.0)
        extract_frames = VideoRLM._make_extract_frames(video_path)

        # Very short range at low fps — should return fewer than max_frames
        result = extract_frames(start_time=0.0, end_time=0.5, fps=2.0, max_frames=100)
        assert len(result) > 0
        assert len(result) <= 100


class TestExtractFramesEdgeCases:
    """Edge cases for extract_frames."""

    def test_start_greater_than_end_raises(self, tmp_path):
        video_path = _make_synthetic_video(str(tmp_path / "test.mp4"), num_frames=30, fps=30.0)
        extract_frames = VideoRLM._make_extract_frames(video_path)

        with pytest.raises(ValueError, match="end_time.*must be greater than start_time"):
            extract_frames(start_time=2.0, end_time=1.0)

    def test_start_equals_end_raises(self, tmp_path):
        video_path = _make_synthetic_video(str(tmp_path / "test.mp4"), num_frames=30, fps=30.0)
        extract_frames = VideoRLM._make_extract_frames(video_path)

        with pytest.raises(ValueError, match="end_time.*must be greater than start_time"):
            extract_frames(start_time=1.0, end_time=1.0)

    def test_time_range_beyond_video_returns_empty_or_partial(self, tmp_path):
        # 30 frames at 30fps = 1 second video
        video_path = _make_synthetic_video(str(tmp_path / "test.mp4"), num_frames=30, fps=30.0)
        extract_frames = VideoRLM._make_extract_frames(video_path)

        # Entirely beyond video duration — should be clamped and return empty
        result = extract_frames(start_time=5.0, end_time=10.0)
        assert isinstance(result, list)
        assert len(result) == 0

    def test_partial_overlap_returns_frames(self, tmp_path):
        # 30 frames at 30fps = 1 second video
        video_path = _make_synthetic_video(str(tmp_path / "test.mp4"), num_frames=30, fps=30.0)
        extract_frames = VideoRLM._make_extract_frames(video_path)

        # Range partially overlaps with video
        result = extract_frames(start_time=0.5, end_time=5.0, fps=2.0)
        assert isinstance(result, list)
        assert len(result) > 0

    def test_resize_applies_to_extracted_frames(self, tmp_path):
        video_path = _make_synthetic_video(
            str(tmp_path / "test.mp4"), num_frames=30, fps=30.0, width=64, height=48
        )
        extract_frames = VideoRLM._make_extract_frames(video_path)

        result = extract_frames(start_time=0.0, end_time=0.5, fps=2.0, resize=(32, 24))
        assert len(result) > 0
        # Decode one frame and check dimensions
        decoded = base64.b64decode(result[0]["data"])
        arr = np.frombuffer(decoded, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        assert img is not None
        assert img.shape[1] == 32  # width
        assert img.shape[0] == 24  # height

    def test_invalid_video_path_raises(self):
        extract_frames = VideoRLM._make_extract_frames("/nonexistent/video.mp4")

        with pytest.raises(ValueError, match="Cannot open video file"):
            extract_frames(start_time=0.0, end_time=1.0)
