"""Convert video frames and segments into structured context."""

from __future__ import annotations

import base64
from typing import Any

import cv2
import numpy as np

from kuavi.loader import LoadedVideo, VideoSegment


def _encode_frame(frame: np.ndarray, format: str = ".jpg", quality: int = 85) -> dict[str, Any]:
    """Encode a single frame to a tagged image dict for multimodal APIs.

    Args:
        frame: BGR numpy array from OpenCV.
        format: Image format (".jpg" or ".png").
        quality: JPEG quality (1-100). Ignored for PNG.

    Returns:
        Dict with ``__image__``, ``data`` (base64), and ``mime_type`` keys.
    """
    params: list[int] = []
    if format == ".jpg":
        params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    elif format == ".png":
        params = [cv2.IMWRITE_PNG_COMPRESSION, 3]

    success, buffer = cv2.imencode(format, frame, params)
    if not success:
        raise ValueError(f"Failed to encode frame to {format}")

    mime = "image/jpeg" if format == ".jpg" else "image/png"
    return {
        "__image__": True,
        "data": base64.b64encode(buffer.tobytes()).decode("utf-8"),
        "mime_type": mime,
    }


def _decode_frame(image_dict: dict[str, Any]) -> np.ndarray:
    """Decode a tagged image dict back to a BGR numpy array.

    Args:
        image_dict: Dict with ``data`` (base64 string) and optionally ``mime_type``.

    Returns:
        BGR numpy array suitable for OpenCV operations.
    """
    raw = base64.b64decode(image_dict["data"])
    buf = np.frombuffer(raw, dtype=np.uint8)
    frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Failed to decode image from base64 data")
    return frame


def _segment_to_dict(segment: VideoSegment, format: str, quality: int) -> dict[str, Any]:
    """Convert a VideoSegment to a serializable dict with encoded frames."""
    return {
        "segment_index": segment.segment_index,
        "start_time": round(segment.start_time, 2),
        "end_time": round(segment.end_time, 2),
        "duration": round(segment.duration, 2),
        "frame_count": segment.frame_count,
        "frames": [_encode_frame(f, format=format, quality=quality) for f in segment.frames],
    }


class VideoContext:
    """Convert video data into structured context payloads.

    Args:
        format: Image encoding format (".jpg" or ".png").
        quality: JPEG quality (1-100). Ignored for PNG.
        max_frames_per_segment: Cap frames per segment to limit context size.
            None means no limit.
    """

    def __init__(
        self,
        format: str = ".jpg",
        quality: int = 85,
        max_frames_per_segment: int | None = None,
    ):
        if format not in (".jpg", ".png"):
            raise ValueError(f"Unsupported format: {format}. Use '.jpg' or '.png'.")
        self.format = format
        self.quality = quality
        self.max_frames_per_segment = max_frames_per_segment

    def build_context(self, loaded_video: LoadedVideo) -> dict[str, Any]:
        """Build a structured context from a loaded video.

        If the video has segments, produces a segmented context with per-segment
        frames. Otherwise, produces a flat context with all frames.

        Args:
            loaded_video: A LoadedVideo from VideoLoader.

        Returns:
            Dict with video metadata and frames/segments.
        """
        meta = loaded_video.metadata
        context: dict[str, Any] = {
            "type": "video",
            "metadata": {
                "path": meta.path,
                "duration": round(meta.duration, 2),
                "original_fps": round(meta.original_fps, 2),
                "extraction_fps": round(meta.extraction_fps, 2),
                "width": meta.width,
                "height": meta.height,
                "total_extracted_frames": meta.extracted_frame_count,
            },
        }

        if loaded_video.segments:
            context["segments"] = [
                self._build_segment_context(seg) for seg in loaded_video.segments
            ]
            context["num_segments"] = len(loaded_video.segments)
        else:
            frames = loaded_video.frames
            if self.max_frames_per_segment is not None:
                frames = self._subsample(frames, self.max_frames_per_segment)
            context["frames"] = [
                _encode_frame(f, format=self.format, quality=self.quality) for f in frames
            ]
            context["num_frames"] = len(context["frames"])

        return context

    def build_context_for_segment(self, segment: VideoSegment) -> dict[str, Any]:
        """Build context for a single video segment.

        Args:
            segment: A VideoSegment from LoadedVideo.

        Returns:
            Dict with segment metadata and encoded frames.
        """
        seg = segment
        if (
            self.max_frames_per_segment is not None
            and seg.frame_count > self.max_frames_per_segment
        ):
            # Subsample frames within the segment
            frames = self._subsample(seg.frames, self.max_frames_per_segment)
        else:
            frames = seg.frames

        return {
            "segment_index": seg.segment_index,
            "start_time": round(seg.start_time, 2),
            "end_time": round(seg.end_time, 2),
            "duration": round(seg.duration, 2),
            "frame_count": len(frames),
            "frames": [_encode_frame(f, format=self.format, quality=self.quality) for f in frames],
        }

    def _build_segment_context(self, segment: VideoSegment) -> dict[str, Any]:
        """Build context dict for a segment, applying frame limits."""
        frames = segment.frames
        if self.max_frames_per_segment is not None and len(frames) > self.max_frames_per_segment:
            frames = self._subsample(frames, self.max_frames_per_segment)

        return _segment_to_dict(
            VideoSegment(
                segment_index=segment.segment_index,
                start_time=segment.start_time,
                end_time=segment.end_time,
                start_frame=segment.start_frame,
                end_frame=segment.end_frame,
                frames=frames,
                fps=segment.fps,
            ),
            format=self.format,
            quality=self.quality,
        )

    @staticmethod
    def _subsample(frames: list[np.ndarray], max_count: int) -> list[np.ndarray]:
        """Uniformly subsample frames to at most max_count."""
        if len(frames) <= max_count:
            return frames
        step = len(frames) / max_count
        return [frames[int(i * step)] for i in range(max_count)]


def make_extract_frames(
    video_path: str,
    *,
    image_format: str = ".jpg",
    image_quality: int = 85,
):
    """Create a closure that extracts frames from a time range.

    Args:
        video_path: Absolute path to the video file.
        image_format: Encoding format for frames.
        image_quality: JPEG quality (1-100).

    Returns:
        A callable with signature
        ``(start_time, end_time, fps=2.0, resize=(720,540), max_frames=10)``
        that returns a list of tagged image dicts.
    """

    def extract_frames(
        start_time: float,
        end_time: float,
        fps: float = 2.0,
        resize: tuple[int, int] = (720, 540),
        max_frames: int = 10,
    ) -> list[dict[str, Any]]:
        """Extract frames from *start_time* to *end_time* (seconds)."""
        if end_time <= start_time:
            raise ValueError(
                f"end_time ({end_time}) must be greater than start_time ({start_time})"
            )

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")

        try:
            original_fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / original_fps if original_fps > 0 else 0.0

            start_time = max(0.0, start_time)
            end_time = min(end_time, duration)
            if end_time <= start_time:
                return []

            interval = 1.0 / fps
            times = []
            t = start_time
            while t < end_time:
                times.append(t)
                t += interval

            if len(times) > max_frames:
                step = len(times) / max_frames
                times = [times[int(i * step)] for i in range(max_frames)]

            frames: list[dict[str, Any]] = []
            for t in times:
                frame_idx = int(t * original_fps)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    continue
                if resize is not None:
                    frame = cv2.resize(frame, resize)
                frames.append(_encode_frame(frame, format=image_format, quality=image_quality))

            return frames
        finally:
            cap.release()

    return extract_frames
