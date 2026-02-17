"""Convenience wrapper for video analysis with RLM."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rlm.core.rlm import RLM
from rlm.core.types import ClientBackend, EnvironmentType, RLMChatCompletion
from rlm.logger import RLMLogger
from rlm.video.video_context import VideoContext
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
    ):
        self.loader = VideoLoader(fps=fps, max_frames=max_frames, resize=resize)
        self.context_builder = VideoContext(
            format=image_format,
            quality=image_quality,
            max_frames_per_segment=max_frames_per_segment,
        )

        self.num_segments = num_segments
        self.segment_duration = segment_duration

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
        return self.rlm.completion(context, root_prompt=prompt)

    def _load_video(self, video_path: str | Path) -> LoadedVideo:
        """Load and optionally segment a video."""
        if self.num_segments is not None or self.segment_duration is not None:
            return self.loader.load_and_segment(
                video_path,
                num_segments=self.num_segments,
                segment_duration=self.segment_duration,
            )
        return self.loader.load(video_path)
