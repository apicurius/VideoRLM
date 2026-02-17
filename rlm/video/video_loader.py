"""Video loading and frame extraction for RLM Long Video Understanding."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np


@dataclass
class VideoSegment:
    """A temporal segment of a video with its extracted frames."""

    segment_index: int
    start_time: float
    end_time: float
    start_frame: int
    end_frame: int
    frames: list[np.ndarray]
    fps: float

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    @property
    def frame_count(self) -> int:
        return len(self.frames)


@dataclass
class VideoMetadata:
    """Metadata about a loaded video."""

    path: str
    total_frames: int
    original_fps: float
    duration: float
    width: int
    height: int
    extraction_fps: float
    extracted_frame_count: int


@dataclass
class LoadedVideo:
    """Result of loading a video: metadata, frames, and optional segments."""

    metadata: VideoMetadata
    frames: list[np.ndarray]
    segments: list[VideoSegment] = field(default_factory=list)


class VideoLoader:
    """Load video files and extract frames at configurable FPS.

    Supports splitting videos into temporal segments for recursive processing.

    Args:
        fps: Frames per second to extract. If None, uses the video's native FPS.
        max_frames: Maximum total frames to extract. None means no limit.
        resize: Optional (width, height) to resize extracted frames.
    """

    def __init__(
        self,
        fps: float | None = 1.0,
        max_frames: int | None = None,
        resize: tuple[int, int] | None = None,
    ):
        self.fps = fps
        self.max_frames = max_frames
        self.resize = resize

    def load(self, video_path: str | Path) -> LoadedVideo:
        """Load a video file and extract frames.

        Args:
            video_path: Path to the video file.

        Returns:
            LoadedVideo with metadata and extracted frames.

        Raises:
            FileNotFoundError: If the video file does not exist.
            ValueError: If the video cannot be opened or has no frames.
        """
        path = Path(video_path)
        if not path.exists():
            raise FileNotFoundError(f"Video file not found: {path}")

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if total_frames <= 0 or original_fps <= 0:
            cap.release()
            raise ValueError(f"Invalid video: {total_frames} frames, {original_fps} FPS")

        duration = total_frames / original_fps
        extraction_fps = self.fps if self.fps is not None else original_fps

        # Calculate which frames to extract based on target FPS
        frame_interval = original_fps / extraction_fps
        frame_indices = []
        idx = 0.0
        while idx < total_frames:
            frame_indices.append(int(idx))
            idx += frame_interval

        # Apply max_frames limit
        if self.max_frames is not None and len(frame_indices) > self.max_frames:
            # Uniformly sample max_frames from the available indices
            step = len(frame_indices) / self.max_frames
            frame_indices = [frame_indices[int(i * step)] for i in range(self.max_frames)]

        frames = self._extract_frames(cap, frame_indices)
        cap.release()

        metadata = VideoMetadata(
            path=str(path),
            total_frames=total_frames,
            original_fps=original_fps,
            duration=duration,
            width=width,
            height=height,
            extraction_fps=extraction_fps,
            extracted_frame_count=len(frames),
        )

        return LoadedVideo(metadata=metadata, frames=frames)

    def load_and_segment(
        self,
        video_path: str | Path,
        segment_duration: float | None = None,
        num_segments: int | None = None,
    ) -> LoadedVideo:
        """Load a video and split it into temporal segments.

        Exactly one of segment_duration or num_segments must be provided.

        Args:
            video_path: Path to the video file.
            segment_duration: Duration of each segment in seconds.
            num_segments: Number of equal segments to split into.

        Returns:
            LoadedVideo with metadata, all frames, and segment information.

        Raises:
            ValueError: If neither or both of segment_duration/num_segments are provided.
        """
        if (segment_duration is None) == (num_segments is None):
            raise ValueError("Exactly one of segment_duration or num_segments must be provided.")

        loaded = self.load(video_path)
        duration = loaded.metadata.duration
        extraction_fps = loaded.metadata.extraction_fps

        if num_segments is not None:
            seg_dur = duration / num_segments
        else:
            seg_dur = segment_duration
            num_segments = max(1, math.ceil(duration / seg_dur))

        segments: list[VideoSegment] = []
        for i in range(num_segments):
            start_time = i * seg_dur
            end_time = min((i + 1) * seg_dur, duration)

            # Map time boundaries to frame indices in the extracted frames list
            start_frame = int(start_time * extraction_fps)
            end_frame = min(int(end_time * extraction_fps), len(loaded.frames))

            # Clamp to valid range
            start_frame = min(start_frame, len(loaded.frames))
            end_frame = min(end_frame, len(loaded.frames))

            segment_frames = loaded.frames[start_frame:end_frame]

            segments.append(
                VideoSegment(
                    segment_index=i,
                    start_time=start_time,
                    end_time=end_time,
                    start_frame=start_frame,
                    end_frame=end_frame,
                    frames=segment_frames,
                    fps=extraction_fps,
                )
            )

        loaded.segments = segments
        return loaded

    def _extract_frames(self, cap: cv2.VideoCapture, frame_indices: list[int]) -> list[np.ndarray]:
        """Extract specific frames from an opened video capture."""
        frames: list[np.ndarray] = []
        current_idx = 0

        for target_idx in frame_indices:
            # Seek if we need to jump forward significantly
            if target_idx - current_idx > 10:
                cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
                current_idx = target_idx

            # Read frames until we reach the target
            while current_idx <= target_idx:
                ret, frame = cap.read()
                if not ret:
                    return frames
                if current_idx == target_idx:
                    if self.resize is not None:
                        frame = cv2.resize(frame, self.resize)
                    frames.append(frame)
                current_idx += 1

        return frames
