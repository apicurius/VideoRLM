"""Convert video frames and segments into RLM-compatible context."""

from __future__ import annotations

import base64
from typing import Any

import cv2
import numpy as np

from rlm.video.video_loader import LoadedVideo, VideoSegment


def _blend_frames(frames: list, blend_size: int = 4) -> list:
    """Blend consecutive frames into composites for context compression.

    Args:
        frames: List of BGR numpy arrays.
        blend_size: Number of consecutive frames to blend into one composite.
            1 means no blending (pass-through).

    Returns:
        List of blended frames (length ≈ len(frames) / blend_size).
    """
    if blend_size <= 1:
        return frames
    blended = []
    for i in range(0, len(frames), blend_size):
        group = frames[i : i + blend_size]
        if len(group) == 1:
            blended.append(group[0])
        else:
            composite = np.mean(np.stack(group), axis=0).astype(np.uint8)
            blended.append(composite)
    return blended


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
    """Convert video data into RLM-compatible context payloads.

    The context produced by this class becomes the `context` variable in the
    RLM REPL environment, allowing the LLM to reason over video frames.

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
        """Build an RLM-compatible context from a loaded video.

        If the video has segments, produces a segmented context with per-segment
        frames. Otherwise, produces a flat context with all frames.

        Args:
            loaded_video: A LoadedVideo from VideoLoader.

        Returns:
            Dict suitable for use as RLM context_payload. Structure:
            {
                "type": "video",
                "metadata": {...},
                "segments": [{...}, ...],  # if segmented
                "frames": [...],           # if not segmented
            }
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

        Useful when processing segments individually (e.g., one per llm_query call).

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
