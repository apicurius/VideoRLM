"""Convenience wrapper for video analysis with RLM."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

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
        token_budget: Maximum total tokens (input + output) before injecting a
            wrap-up signal. None means no budget (default).
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
        caption_fn: Optional function that produces a caption for a list of
            frames (segment-level captioning).  Passed through to
            :meth:`VideoIndexer.index_video`.
        frame_caption_fn: Optional function that captions a single keyframe
            (Tree-of-Captions leaf level).  Passed through to
            :meth:`VideoIndexer.index_video`.
        enable_sharding: When True, videos with more segments than
            ``shard_max_segments`` are analyzed via parallel sub-agent calls
            (one per shard) before a final aggregation pass.  Defaults to
            False for backward compatibility.
        shard_max_segments: Maximum number of segments per shard when sharding
            is enabled.  Defaults to 5.
        auto_fps: When True, dynamically compute extraction FPS based on video
            duration so that approximately ``target_frames`` frames are
            extracted.  The ``fps`` parameter is used as a fallback when the
            video duration cannot be determined.  Defaults to False.
        target_frames: Target number of frames to extract when ``auto_fps``
            is enabled.  The computed FPS is clamped to [0.1, 5.0].
            Defaults to 120.
        caption_resize: Optional (width, height) to resize frames before
            passing them to caption_fn.  Reduces VLM cost and standardizes
            input resolution for captioning only (embeddings use original
            resolution).  None means no resizing (default).
        text_embedding_model: Optional HuggingFace model id for a separate text
            encoder (e.g. ``"google/embedding-gemma-300m"``).  When set, text
            queries and captions are encoded with this model instead of SigLIP2's
            built-in text tower.  This can improve search quality when the text
            model has richer semantic representations than the CLIP-style text
            encoder.  None means use SigLIP2 for both vision and text (default).
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
        token_budget: int | None = None,
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
        asr_model: str = "Qwen/Qwen3-ASR-0.6B",
        transcript_path: str | None = None,
        refine_fn: Callable | None = None,
        caption_fn: Callable | None = None,
        frame_caption_fn: Callable | None = None,
        enable_sharding: bool = False,
        shard_max_segments: int = 5,
        auto_fps: bool = False,
        target_frames: int = 120,
        cache_dir: str | None = None,
        caption_resize: tuple[int, int] | None = None,
        text_embedding_model: str | None = None,
        scene_model: str | None = None,
        store_feature_maps: bool = False,
    ):
        self.store_feature_maps = store_feature_maps
        self.scene_model = scene_model
        self.caption_resize = caption_resize
        self.auto_fps = auto_fps
        self.target_frames = target_frames
        self.cache_dir = cache_dir
        self.text_embedding_model = text_embedding_model
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
        self.asr_model = asr_model
        self.transcript_path = transcript_path
        self.refine_fn = refine_fn
        self.caption_fn = caption_fn
        self.frame_caption_fn = frame_caption_fn
        self.enable_sharding = enable_sharding
        self.shard_max_segments = shard_max_segments

        self.rlm = RLM(
            backend=backend,
            backend_kwargs=backend_kwargs,
            environment=environment,
            environment_kwargs=environment_kwargs or {},
            max_depth=max_depth,
            max_iterations=max_iterations,
            token_budget=token_budget,
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

        # Emit supplemental video metadata so the visualizer can show video info
        self._log_video_metadata(video_path, loaded_video)

        # Route to sharded completion for long videos
        if (
            self.enable_sharding
            and loaded_video.segments
            and len(loaded_video.segments) > self.shard_max_segments
        ):
            return self._sharded_completion(loaded_video, video_path, prompt)

        context = self.context_builder.build_context(loaded_video)
        extra_tools = self._build_extra_tools(video_path, loaded_video)

        return self.rlm.completion(context, root_prompt=prompt, extra_tools=extra_tools)

    def _build_extra_tools(
        self,
        video_path: str | Path,
        loaded_video: LoadedVideo,
    ) -> dict[str, Any]:
        """Build the standard extra tools dict injected into the REPL environment.

        Includes extract_frames, pixel tools, and (when enabled) search tools.

        Args:
            video_path: Path to the video file.
            loaded_video: Already-loaded video data.

        Returns:
            Dict mapping tool names to ``{"tool": callable, "description": str}`` dicts.
        """
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
                cache_dir=self.cache_dir,
                caption_resize=self.caption_resize,
                text_embedding_model=self.text_embedding_model,
                scene_model=self.scene_model,
            )
            video_index = indexer.index_video(
                loaded_video,
                caption_fn=self.caption_fn,
                frame_caption_fn=self.frame_caption_fn,
                refine_fn=self.refine_fn,
                asr_model=self.asr_model,
                transcript_path=self.transcript_path,
                store_feature_maps=self.store_feature_maps,
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

            from rlm.video.video_search_tools import (
                make_anticipate_action,
                make_classify_segment,
                make_discriminative_vqa,
                make_predict_future,
                make_verify_coherence,
            )

            for extra_factory in [
                make_discriminative_vqa,
                make_anticipate_action,
                make_classify_segment,
                make_predict_future,
                make_verify_coherence,
            ]:
                extra_result = extra_factory(video_index)
                extra_tools[extra_result["tool"].__name__] = extra_result

        # Add pixel manipulation tools
        extra_tools.update(self._make_pixel_tools())

        return extra_tools

    def _sharded_completion(
        self,
        loaded_video: LoadedVideo,
        video_path: str | Path,
        prompt: str | None = None,
    ) -> RLMChatCompletion:
        """Analyze a long video by sharding segments across sub-agent calls.

        Divides segments into groups (shards), builds pre-encoded multimodal
        prompts for each shard, and injects them as ``shard_prompts`` into the
        REPL so the orchestrator LLM can call
        ``llm_query_batched(shard_prompts)`` to analyze all shards in parallel,
        then aggregate the results.

        This maps to K2.5's Agent Swarm architecture (pp. 4-6) where each
        sub-agent has an isolated context and only returns task-relevant
        summaries back to the orchestrator.

        Args:
            loaded_video: Already-loaded video with segments.
            video_path: Path to the video file (for extract_frames tool).
            prompt: Optional question or instruction about the video.

        Returns:
            RLMChatCompletion with the aggregated analysis result.
        """
        segments = loaded_video.segments

        # Group segments into shards of at most shard_max_segments each
        shards = [
            segments[i : i + self.shard_max_segments]
            for i in range(0, len(segments), self.shard_max_segments)
        ]

        # Build a multimodal prompt for each shard (strings + encoded image dicts)
        shard_prompts: list[list[Any]] = []
        for shard_idx, shard in enumerate(shards):
            parts: list[Any] = [
                f"You are analyzing shard {shard_idx + 1}/{len(shards)} of a video "
                f"({loaded_video.metadata.duration:.1f}s total). "
                f"This shard covers segments "
                f"{shard[0].segment_index}-{shard[-1].segment_index} "
                f"({shard[0].start_time:.1f}s - {shard[-1].end_time:.1f}s).\n"
                f"Question: {prompt}\n"
                f"Analyze these frames and provide a detailed answer for your shard."
            ]
            for seg in shard:
                parts.append(
                    f"\n--- Segment {seg.segment_index} "
                    f"({seg.start_time:.1f}s - {seg.end_time:.1f}s) ---"
                )
                for frame in seg.frames[:3]:
                    parts.append(
                        _encode_frame(
                            frame,
                            format=self.context_builder.format,
                            quality=self.context_builder.quality,
                        )
                    )
            shard_prompts.append(parts)

        context = self.context_builder.build_context(loaded_video)
        extra_tools = self._build_extra_tools(video_path, loaded_video)

        # Expose shard info as a callable tool
        def get_shard_info() -> list[str]:
            """Return a description of each video shard's segment range and timestamps."""
            return [
                f"Shard {i + 1}/{len(shards)}: segments "
                f"{shards[i][0].segment_index}-{shards[i][-1].segment_index} "
                f"({shards[i][0].start_time:.1f}s-{shards[i][-1].end_time:.1f}s)"
                for i in range(len(shards))
            ]

        extra_tools["get_shard_info"] = {
            "tool": get_shard_info,
            "description": (
                "get_shard_info() -> list[str]. "
                "Get a list of video shards with their segment ranges and timestamps. "
                "Use llm_query_batched(shard_prompts) to analyze all shards in parallel."
            ),
        }
        # Expose pre-built multimodal shard prompts as a REPL variable
        extra_tools["shard_prompts"] = {
            "tool": shard_prompts,
            "description": (
                "shard_prompts: list[list]. Pre-built multimodal prompts for each "
                "video shard. Pass to llm_query_batched(shard_prompts) to analyze "
                "all shards in parallel, then aggregate the results."
            ),
        }

        shard_guidance = (
            f"\n\nSHARDED VIDEO ANALYSIS:\n"
            f"This video has been divided into {len(shards)} shards for efficient "
            f"analysis. A `shard_prompts` variable is available containing pre-built "
            f"multimodal prompts for each shard. Use `llm_query_batched(shard_prompts)` "
            f"to analyze all shards in parallel, then aggregate the results to answer "
            f"the question.\n"
            f"Example:\n"
            f"```repl\n"
            f"shard_results = llm_query_batched(shard_prompts)\n"
            f"for i, result in enumerate(shard_results):\n"
            f"    print(f'Shard {{i+1}}: {{result[:200]}}')\n"
            f"```\n"
        )

        original_system_prompt = self.rlm.system_prompt
        self.rlm.system_prompt = VIDEO_SYSTEM_PROMPT + shard_guidance
        try:
            return self.rlm.completion(context, root_prompt=prompt, extra_tools=extra_tools)
        finally:
            self.rlm.system_prompt = original_system_prompt

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
                    frames.append(_encode_frame(frame, format=image_format, quality=image_quality))

                return frames
            finally:
                cap.release()

        return extract_frames

    @staticmethod
    def _make_pixel_tools() -> dict[str, Any]:
        """Create pixel-manipulation tools for code-based visual reasoning."""
        import base64

        import cv2
        import numpy as np

        def _decode_image_dict(image_dict: dict) -> np.ndarray:
            """Convert a tagged image dict back to a BGR numpy array."""
            data = base64.b64decode(image_dict["data"])
            arr = np.frombuffer(data, dtype=np.uint8)
            return cv2.imdecode(arr, cv2.IMREAD_COLOR)

        def _to_image_dict(frame: np.ndarray, mime_type: str = "image/jpeg") -> dict:
            """Convert BGR numpy array to tagged image dict."""
            _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            return {
                "__image__": True,
                "data": base64.b64encode(buffer.tobytes()).decode("utf-8"),
                "mime_type": mime_type,
            }

        def threshold_frame(image_dict: dict, value: int = 128) -> dict:
            """Convert frame to binary mask. Pixels > value become white, else black.
            Useful for object counting via pixel analysis."""
            frame = _decode_image_dict(image_dict)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray, value, 255, cv2.THRESH_BINARY)
            result = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            return _to_image_dict(result)

        def crop_frame(
            image_dict: dict,
            x1_pct: float,
            y1_pct: float,
            x2_pct: float,
            y2_pct: float,
        ) -> dict:
            """Crop a region of interest from a frame using percentage coordinates.
            x1_pct, y1_pct = top-left corner (0.0-1.0), x2_pct, y2_pct = bottom-right."""
            frame = _decode_image_dict(image_dict)
            h, w = frame.shape[:2]
            x1, y1 = int(x1_pct * w), int(y1_pct * h)
            x2, y2 = int(x2_pct * w), int(y2_pct * h)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            cropped = frame[y1:y2, x1:x2]
            return _to_image_dict(cropped)

        def diff_frames(image_dict_a: dict, image_dict_b: dict) -> dict:
            """Compute absolute difference between two frames.
            Bright areas in result = regions that changed. Useful for motion detection."""
            frame_a = _decode_image_dict(image_dict_a)
            frame_b = _decode_image_dict(image_dict_b)
            if frame_a.shape != frame_b.shape:
                frame_b = cv2.resize(frame_b, (frame_a.shape[1], frame_a.shape[0]))
            diff = cv2.absdiff(frame_a, frame_b)
            return _to_image_dict(diff)

        def blend_frames(image_dicts: list[dict]) -> dict:
            """Blend multiple frames into a single composite by averaging pixels.
            Useful for creating motion summary or detecting static elements."""
            frames = [_decode_image_dict(d) for d in image_dicts]
            if not frames:
                raise ValueError("No frames to blend")
            target_shape = frames[0].shape[:2]
            resized = [
                cv2.resize(f, (target_shape[1], target_shape[0]))
                if f.shape[:2] != target_shape
                else f
                for f in frames
            ]
            composite = np.mean(np.stack(resized), axis=0).astype(np.uint8)
            return _to_image_dict(composite)

        def frame_info(image_dict: dict) -> dict:
            """Get frame dimensions and basic statistics (mean brightness, etc.)."""
            frame = _decode_image_dict(image_dict)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            h, w = frame.shape[:2]
            return {
                "width": w,
                "height": h,
                "channels": frame.shape[2] if len(frame.shape) > 2 else 1,
                "mean_brightness": float(gray.mean()),
                "std_brightness": float(gray.std()),
                "min_brightness": int(gray.min()),
                "max_brightness": int(gray.max()),
            }

        return {
            "threshold_frame": {
                "tool": threshold_frame,
                "description": (
                    "threshold_frame(image_dict, value=128) -> image_dict. "
                    "Convert frame to binary mask (pixels > value = white, else black). "
                    "Pass any image dict from extract_frames() or context frames."
                ),
            },
            "crop_frame": {
                "tool": crop_frame,
                "description": (
                    "crop_frame(image_dict, x1_pct, y1_pct, x2_pct, y2_pct) -> image_dict. "
                    "Crop region of interest using percentage coordinates (0.0-1.0). "
                    "Example: crop_frame(frame, 0.0, 0.0, 0.5, 0.5) crops top-left quarter."
                ),
            },
            "diff_frames": {
                "tool": diff_frames,
                "description": (
                    "diff_frames(image_dict_a, image_dict_b) -> image_dict. "
                    "Compute absolute pixel difference between two frames. "
                    "Bright areas = regions that changed. Use for motion detection."
                ),
            },
            "blend_frames": {
                "tool": blend_frames,
                "description": (
                    "blend_frames(image_dicts: list) -> image_dict. "
                    "Average multiple frames into one composite. "
                    "Use for motion summary or detecting static elements."
                ),
            },
            "frame_info": {
                "tool": frame_info,
                "description": (
                    "frame_info(image_dict) -> dict. "
                    "Get frame dimensions and brightness statistics "
                    "(width, height, mean_brightness, std_brightness)."
                ),
            },
        }

    def _log_video_metadata(self, video_path: str | Path, loaded_video: LoadedVideo) -> None:
        """Emit supplemental metadata with video-specific fields.

        Written as a second ``type: "metadata"`` JSONL line so the visualizer
        can merge it with the initial RLM config metadata.
        """
        if self.rlm.logger is None:
            return
        self.rlm.logger.log_supplemental_metadata(
            video_path=str(Path(video_path).resolve()),
            fps=self.loader.fps,
            num_segments=len(loaded_video.segments) if loaded_video.segments else 0,
            max_frames_per_segment=self.context_builder.max_frames_per_segment,
            resize=list(self.context_builder.resize) if getattr(self.context_builder, "resize", None) else None,
        )

    def _load_video(self, video_path: str | Path) -> LoadedVideo:
        """Load and optionally segment a video."""
        original_fps = self.loader.fps
        if self.auto_fps:
            cap = cv2.VideoCapture(str(video_path))
            try:
                video_fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                if video_fps > 0 and frame_count > 0:
                    duration = frame_count / video_fps
                    optimal_fps = self.target_frames / duration
                    self.loader.fps = max(0.1, min(5.0, optimal_fps))
            finally:
                cap.release()

        try:
            if self.num_segments is not None or self.segment_duration is not None:
                return self.loader.load_and_segment(
                    video_path,
                    num_segments=self.num_segments,
                    segment_duration=self.segment_duration,
                )
            return self.loader.load(video_path)
        finally:
            if self.auto_fps:
                self.loader.fps = original_fps
