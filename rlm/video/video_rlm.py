"""Convenience wrapper for video analysis with RLM."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import cv2

from rlm.core.rlm import RLM
from rlm.core.types import ClientBackend, EnvironmentType, RLMChatCompletion
from rlm.logger import RLMLogger
from rlm.video.video_context import VideoContext, _encode_frame
from rlm.video.video_loader import LoadedVideo, VideoLoader
from rlm.video.video_prompts import VIDEO_SYSTEM_PROMPT


class VideoRLM:
    """High-level interface for video understanding with RLM.

    Wraps the full pipeline: load video -> extract frames -> build context -> run RLM.

    Args:
        backend: LLM backend to use (e.g. "openai", "anthropic").
        backend_kwargs: Arguments passed to the backend client.
        fps: Frames per second to extract. Defaults to 1.0.
        max_frames: Maximum total frames to extract. None means no limit.
        resize: Optional (width, height) to resize frames.
        num_segments: Number of temporal segments. None for no segmentation.
        segment_duration: Duration per segment in seconds. Alternative to num_segments.
        image_format: Encoding format for frames (".jpg" or ".png").
        image_quality: JPEG quality (1-100).
        max_frames_per_segment: Cap frames per segment in context.
        max_depth: Maximum RLM recursion depth.
        max_iterations: Maximum REPL iterations per completion.
        environment: RLM environment type.
        environment_kwargs: Extra kwargs for the environment.
        other_backends: Additional backends for sub-LLM calls.
        other_backend_kwargs: Kwargs for other backends.
        logger: Optional RLM logger.
        verbose: Print verbose output.
        custom_tools: Custom tools available in the REPL.
        custom_sub_tools: Custom tools for sub-agents.
        refine_fn: Optional function ``(draft, context) -> str`` for
            Self-Refine of segment annotations.  Passed through to
            :meth:`VideoIndexer.index_video`.
    """

    def __init__(
        self,
        backend: ClientBackend = "openai",
        backend_kwargs: dict[str, Any] | None = None,
        *,
        fps: float | None = 1.0,
        max_frames: int | None = None,
        resize: tuple[int, int] | None = None,
        num_segments: int | None = None,
        segment_duration: float | None = None,
        image_format: str = ".jpg",
        image_quality: int = 85,
        max_frames_per_segment: int | None = None,
        max_depth: int = 1,
        max_iterations: int = 30,
        environment: EnvironmentType = "local",
        environment_kwargs: dict[str, Any] | None = None,
        other_backends: list[ClientBackend] | None = None,
        other_backend_kwargs: list[dict[str, Any]] | None = None,
        logger: RLMLogger | None = None,
        verbose: bool = False,
        custom_tools: dict[str, Any] | None = None,
        custom_sub_tools: dict[str, Any] | None = None,
        enable_search: bool = True,
        embedding_model: str | None = None,
        whisper_model: str = "base",
        transcript_path: str | None = None,
        refine_fn: Callable | None = None,
    ):
        self.loader = VideoLoader(fps=fps, max_frames=max_frames, resize=resize)
        self.context_builder = VideoContext(
            format=image_format,
            quality=image_quality,
            max_frames_per_segment=max_frames_per_segment,
        )

        self.num_segments = num_segments
        self.segment_duration = segment_duration
        self.enable_search = enable_search
        self.embedding_model = embedding_model
        self.whisper_model = whisper_model
        self.transcript_path = transcript_path
        self.refine_fn = refine_fn

        self.rlm = RLM(
            backend=backend,
            backend_kwargs=backend_kwargs,
            environment=environment,
            environment_kwargs=environment_kwargs or {},
            max_depth=max_depth,
            max_iterations=max_iterations,
            custom_system_prompt=VIDEO_SYSTEM_PROMPT,
            other_backends=other_backends,
            other_backend_kwargs=other_backend_kwargs,
            logger=logger,
            verbose=verbose,
            custom_tools=custom_tools,
            custom_sub_tools=custom_sub_tools,
        )

    def completion(
        self,
        video_path: str | Path,
        prompt: str | None = None,
    ) -> RLMChatCompletion:
        """Analyze a video with RLM.

        Loads the video, builds context, and runs an RLM completion.

        Args:
            video_path: Path to the video file.
            prompt: Optional question or instruction about the video. Passed as
                root_prompt so the LLM sees it alongside each iteration.

        Returns:
            RLMChatCompletion with the analysis result.
        """
        loaded_video = self._load_video(video_path)
        context = self.context_builder.build_context(loaded_video)

        # Inject extract_frames tool so the LLM can zoom into time ranges
        extract_fn = self._make_extract_frames(
            str(Path(video_path).resolve()),
            image_format=self.context_builder.format,
            image_quality=self.context_builder.quality,
        )
        extra_tools: dict[str, Any] = {
            "extract_frames": {
                "tool": extract_fn,
                "description": (
                    "extract_frames(start_time, end_time, fps=2.0, "
                    "resize=(720, 540), max_frames=10) -> list[image_dict]. "
                    "Extract high-resolution frames from a specific time range "
                    "of the video. Returns a list of image dicts that can be "
                    "passed to llm_query for detailed visual analysis."
                ),
            },
        }
        # Conditionally build search index and add search tools
        if self.enable_search:
            from rlm.video.video_indexer import VideoIndexer
            from rlm.video.video_search_tools import (
                make_get_scene_list,
                make_get_transcript,
                make_search_transcript,
                make_search_video,
            )

            indexer = VideoIndexer(
                embedding_model=self.embedding_model or "google/siglip2-base-patch16-256",
            )
            video_index = indexer.index_video(
                loaded_video,
                refine_fn=self.refine_fn,
                whisper_model=self.whisper_model,
                transcript_path=self.transcript_path,
            )

            for factory in [
                make_search_video,
                make_search_transcript,
                make_get_transcript,
                make_get_scene_list,
            ]:
                result = factory(video_index)
                tool_name = result["tool"].__name__
                extra_tools[tool_name] = result

        # Merge with any user-supplied custom tools
        prev_tools = self.rlm.custom_tools
        merged = {**(prev_tools or {}), **extra_tools}
        self.rlm.custom_tools = merged
        try:
            return self.rlm.completion(context, root_prompt=prompt)
        finally:
            self.rlm.custom_tools = prev_tools

    @staticmethod
    def _make_extract_frames(
        video_path: str,
        *,
        image_format: str = ".jpg",
        image_quality: int = 85,
    ):
        """Create a closure that extracts frames from a time range.

        The returned function is injected into the REPL as a custom tool so the
        LLM can dynamically "zoom in" on specific parts of the video.

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
            """Extract frames from *start_time* to *end_time* (seconds).

            Args:
                start_time: Start of the time range in seconds.
                end_time: End of the time range in seconds.
                fps: Frames per second to extract within the range.
                resize: (width, height) to resize each frame.
                max_frames: Maximum number of frames to return.

            Returns:
                List of tagged image dicts (``__image__``, ``data``, ``mime_type``).
            """
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

                # Clamp time range to video duration
                start_time = max(0.0, start_time)
                end_time = min(end_time, duration)
                if end_time <= start_time:
                    return []

                # Calculate frame indices to extract
                interval = 1.0 / fps
                times = []
                t = start_time
                while t < end_time:
                    times.append(t)
                    t += interval

                # Subsample if exceeding max_frames
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
                    frames.append(
                        _encode_frame(frame, format=image_format, quality=image_quality)
                    )

                return frames
            finally:
                cap.release()

        return extract_frames

    def _load_video(self, video_path: str | Path) -> LoadedVideo:
        """Load and optionally segment a video."""
        if self.num_segments is not None or self.segment_duration is not None:
            return self.loader.load_and_segment(
                video_path,
                num_segments=self.num_segments,
                segment_duration=self.segment_duration,
            )
        return self.loader.load(video_path)
