"""Tests for video loading and frame extraction."""

import cv2
import numpy as np
import pytest

from kuavi.loader import LoadedVideo, VideoLoader, VideoMetadata, VideoSegment


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
        # Each frame has a unique colour so tests can distinguish them
        frame = np.full((height, width, 3), fill_value=i % 256, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path


class TestVideoSegment:
    """Tests for the VideoSegment dataclass."""

    def test_duration(self):
        seg = VideoSegment(
            segment_index=0,
            start_time=1.0,
            end_time=3.5,
            start_frame=0,
            end_frame=5,
            frames=[np.zeros((2, 2, 3), dtype=np.uint8)],
            fps=1.0,
        )
        assert seg.duration == pytest.approx(2.5)

    def test_frame_count(self):
        frames = [np.zeros((2, 2, 3), dtype=np.uint8) for _ in range(4)]
        seg = VideoSegment(
            segment_index=0,
            start_time=0,
            end_time=4,
            start_frame=0,
            end_frame=4,
            frames=frames,
            fps=1.0,
        )
        assert seg.frame_count == 4


class TestVideoLoaderLoad:
    """Tests for VideoLoader.load()."""

    def test_load_basic(self, tmp_path):
        video_path = str(tmp_path / "test.mp4")
        _make_synthetic_video(video_path, num_frames=30, fps=30.0, width=64, height=48)

        loader = VideoLoader(fps=1.0)
        result = loader.load(video_path)

        assert isinstance(result, LoadedVideo)
        assert isinstance(result.metadata, VideoMetadata)
        assert result.metadata.original_fps == pytest.approx(30.0, abs=1.0)
        assert result.metadata.width == 64
        assert result.metadata.height == 48
        assert result.metadata.duration > 0
        assert len(result.frames) > 0

    def test_load_native_fps(self, tmp_path):
        video_path = str(tmp_path / "test.mp4")
        _make_synthetic_video(video_path, num_frames=30, fps=30.0)

        loader = VideoLoader(fps=None)
        result = loader.load(video_path)

        # With native fps, should extract close to all frames
        assert result.metadata.extraction_fps == pytest.approx(30.0, abs=1.0)
        assert len(result.frames) >= 25  # allow some tolerance

    def test_load_max_frames(self, tmp_path):
        video_path = str(tmp_path / "test.mp4")
        _make_synthetic_video(video_path, num_frames=60, fps=30.0)

        loader = VideoLoader(fps=None, max_frames=5)
        result = loader.load(video_path)

        assert len(result.frames) == 5

    def test_load_with_resize(self, tmp_path):
        video_path = str(tmp_path / "test.mp4")
        _make_synthetic_video(video_path, num_frames=10, fps=10.0, width=64, height=48)

        loader = VideoLoader(fps=1.0, resize=(32, 24))
        result = loader.load(video_path)

        for frame in result.frames:
            assert frame.shape[1] == 32  # width
            assert frame.shape[0] == 24  # height

    def test_load_missing_file(self):
        loader = VideoLoader()
        with pytest.raises(FileNotFoundError):
            loader.load("/nonexistent/video.mp4")

    def test_load_invalid_file(self, tmp_path):
        bad_path = tmp_path / "bad.mp4"
        bad_path.write_text("not a video")

        loader = VideoLoader()
        with pytest.raises(ValueError):
            loader.load(str(bad_path))


class TestVideoLoaderSegment:
    """Tests for VideoLoader.load_and_segment()."""

    def test_segment_by_num_segments(self, tmp_path):
        video_path = str(tmp_path / "test.mp4")
        _make_synthetic_video(video_path, num_frames=60, fps=30.0)

        loader = VideoLoader(fps=1.0)
        result = loader.load_and_segment(video_path, num_segments=3)

        assert len(result.segments) == 3
        for seg in result.segments:
            assert isinstance(seg, VideoSegment)
            assert seg.start_time < seg.end_time

    def test_segment_by_duration(self, tmp_path):
        video_path = str(tmp_path / "test.mp4")
        _make_synthetic_video(video_path, num_frames=60, fps=30.0)

        loader = VideoLoader(fps=1.0)
        result = loader.load_and_segment(video_path, segment_duration=0.5)

        assert len(result.segments) >= 2
        # First segment should start at 0
        assert result.segments[0].start_time == pytest.approx(0.0)

    def test_segment_requires_exactly_one_param(self, tmp_path):
        video_path = str(tmp_path / "test.mp4")
        _make_synthetic_video(video_path, num_frames=10, fps=10.0)

        loader = VideoLoader(fps=1.0)

        # Neither provided
        with pytest.raises(ValueError):
            loader.load_and_segment(video_path)

        # Both provided
        with pytest.raises(ValueError):
            loader.load_and_segment(video_path, segment_duration=1.0, num_segments=2)

    def test_segment_indices_are_sequential(self, tmp_path):
        video_path = str(tmp_path / "test.mp4")
        _make_synthetic_video(video_path, num_frames=60, fps=30.0)

        loader = VideoLoader(fps=1.0)
        result = loader.load_and_segment(video_path, num_segments=4)

        for i, seg in enumerate(result.segments):
            assert seg.segment_index == i

    def test_segments_cover_full_duration(self, tmp_path):
        video_path = str(tmp_path / "test.mp4")
        _make_synthetic_video(video_path, num_frames=60, fps=30.0)

        loader = VideoLoader(fps=1.0)
        result = loader.load_and_segment(video_path, num_segments=3)

        # First segment starts at 0
        assert result.segments[0].start_time == pytest.approx(0.0)
        # Last segment ends at or near the video duration
        assert result.segments[-1].end_time == pytest.approx(result.metadata.duration, abs=0.5)
