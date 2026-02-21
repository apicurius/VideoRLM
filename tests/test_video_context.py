"""Tests for video context creation and frame encoding."""

import base64

import cv2
import numpy as np
import pytest

from kuavi.context import VideoContext, _encode_frame
from kuavi.loader import LoadedVideo, VideoMetadata, VideoSegment


def _make_frame(width: int = 8, height: int = 8, value: int = 128) -> np.ndarray:
    """Create a small synthetic BGR frame."""
    return np.full((height, width, 3), fill_value=value, dtype=np.uint8)


def _make_metadata(**overrides) -> VideoMetadata:
    defaults = dict(
        path="/tmp/test.mp4",
        total_frames=30,
        original_fps=30.0,
        duration=1.0,
        width=8,
        height=8,
        extraction_fps=1.0,
        extracted_frame_count=1,
    )
    defaults.update(overrides)
    return VideoMetadata(**defaults)


def _make_segment(
    index: int = 0,
    num_frames: int = 3,
    start_time: float = 0.0,
    end_time: float = 1.0,
) -> VideoSegment:
    frames = [_make_frame(value=(index * 50 + i) % 256) for i in range(num_frames)]
    return VideoSegment(
        segment_index=index,
        start_time=start_time,
        end_time=end_time,
        start_frame=0,
        end_frame=num_frames,
        frames=frames,
        fps=1.0,
    )


class TestEncodeFrame:
    """Tests for the _encode_frame helper."""

    def test_encode_jpg(self):
        frame = _make_frame()
        result = _encode_frame(frame, format=".jpg")

        # Should be a tagged image dict
        assert result["__image__"] is True
        assert result["mime_type"] == "image/jpeg"

        # Should be valid base64
        decoded = base64.b64decode(result["data"])
        assert len(decoded) > 0

        # Should decode back to an image
        arr = np.frombuffer(decoded, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        assert img is not None
        assert img.shape[0] == 8
        assert img.shape[1] == 8

    def test_encode_png(self):
        frame = _make_frame()
        result = _encode_frame(frame, format=".png")

        assert result["__image__"] is True
        assert result["mime_type"] == "image/png"

        decoded = base64.b64decode(result["data"])
        arr = np.frombuffer(decoded, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        assert img is not None

    def test_encode_produces_different_output_for_different_frames(self):
        frame_a = _make_frame(value=50)
        frame_b = _make_frame(value=200)

        encoded_a = _encode_frame(frame_a, format=".png")
        encoded_b = _encode_frame(frame_b, format=".png")

        assert encoded_a["data"] != encoded_b["data"]


class TestVideoContextInit:
    """Tests for VideoContext initialisation and validation."""

    def test_default_format(self):
        ctx = VideoContext()
        assert ctx.format == ".jpg"
        assert ctx.quality == 85

    def test_invalid_format(self):
        with pytest.raises(ValueError, match="Unsupported format"):
            VideoContext(format=".bmp")


class TestBuildContext:
    """Tests for VideoContext.build_context()."""

    def test_flat_context_without_segments(self):
        frames = [_make_frame(value=i * 50) for i in range(3)]
        video = LoadedVideo(
            metadata=_make_metadata(extracted_frame_count=3),
            frames=frames,
        )
        ctx = VideoContext()
        result = ctx.build_context(video)

        assert result["type"] == "video"
        assert "metadata" in result
        assert result["metadata"]["duration"] == 1.0
        assert result["num_frames"] == 3
        assert len(result["frames"]) == 3
        # Each frame should be a tagged image dict
        for f in result["frames"]:
            assert isinstance(f, dict)
            assert f["__image__"] is True
            base64.b64decode(f["data"])  # should not raise

    def test_segmented_context(self):
        frames = [_make_frame(value=i * 30) for i in range(6)]
        segments = [
            _make_segment(index=0, num_frames=3, start_time=0.0, end_time=0.5),
            _make_segment(index=1, num_frames=3, start_time=0.5, end_time=1.0),
        ]
        video = LoadedVideo(
            metadata=_make_metadata(extracted_frame_count=6),
            frames=frames,
            segments=segments,
        )
        ctx = VideoContext()
        result = ctx.build_context(video)

        assert result["type"] == "video"
        assert result["num_segments"] == 2
        assert len(result["segments"]) == 2
        for seg in result["segments"]:
            assert "segment_index" in seg
            assert "start_time" in seg
            assert "end_time" in seg
            assert "frames" in seg
            assert len(seg["frames"]) > 0

    def test_max_frames_per_segment_flat(self):
        frames = [_make_frame(value=i * 10) for i in range(10)]
        video = LoadedVideo(
            metadata=_make_metadata(extracted_frame_count=10),
            frames=frames,
        )
        ctx = VideoContext(max_frames_per_segment=3)
        result = ctx.build_context(video)

        assert result["num_frames"] == 3

    def test_max_frames_per_segment_segmented(self):
        segments = [_make_segment(index=i, num_frames=5) for i in range(2)]
        frames = [_make_frame() for _ in range(10)]
        video = LoadedVideo(
            metadata=_make_metadata(extracted_frame_count=10),
            frames=frames,
            segments=segments,
        )
        ctx = VideoContext(max_frames_per_segment=2)
        result = ctx.build_context(video)

        for seg in result["segments"]:
            assert seg["frame_count"] <= 2

    def test_metadata_fields(self):
        video = LoadedVideo(
            metadata=_make_metadata(
                path="/videos/clip.mp4",
                duration=5.5,
                original_fps=24.0,
                extraction_fps=2.0,
                width=640,
                height=480,
                extracted_frame_count=11,
            ),
            frames=[_make_frame()],
        )
        ctx = VideoContext()
        result = ctx.build_context(video)
        meta = result["metadata"]

        assert meta["path"] == "/videos/clip.mp4"
        assert meta["duration"] == 5.5
        assert meta["original_fps"] == 24.0
        assert meta["extraction_fps"] == 2.0
        assert meta["width"] == 640
        assert meta["height"] == 480
        assert meta["total_extracted_frames"] == 11


class TestBuildContextForSegment:
    """Tests for VideoContext.build_context_for_segment()."""

    def test_single_segment_context(self):
        seg = _make_segment(index=2, num_frames=4, start_time=2.0, end_time=4.0)
        ctx = VideoContext()
        result = ctx.build_context_for_segment(seg)

        assert result["segment_index"] == 2
        assert result["start_time"] == 2.0
        assert result["end_time"] == 4.0
        assert result["duration"] == 2.0
        assert result["frame_count"] == 4
        assert len(result["frames"]) == 4

    def test_segment_context_with_frame_limit(self):
        seg = _make_segment(index=0, num_frames=10)
        ctx = VideoContext(max_frames_per_segment=3)
        result = ctx.build_context_for_segment(seg)

        assert result["frame_count"] == 3
        assert len(result["frames"]) == 3


class TestSubsample:
    """Tests for the _subsample static method."""

    def test_no_subsample_when_within_limit(self):
        frames = [_make_frame(value=i) for i in range(3)]
        result = VideoContext._subsample(frames, max_count=5)
        assert len(result) == 3

    def test_subsample_to_exact_count(self):
        frames = [_make_frame(value=i) for i in range(10)]
        result = VideoContext._subsample(frames, max_count=3)
        assert len(result) == 3

    def test_subsample_preserves_order(self):
        # Use pixel value as a unique identifier
        frames = [_make_frame(value=i * 20) for i in range(10)]
        result = VideoContext._subsample(frames, max_count=4)
        values = [f[0, 0, 0] for f in result]
        assert values == sorted(values)
