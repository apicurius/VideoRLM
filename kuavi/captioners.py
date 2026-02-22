"""Pluggable captioning backends for KUAVi video indexing.

Provides Protocol definitions and concrete implementations for captioning and
annotation refinement. Use ``create_captioner()`` to instantiate from a preset.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class CaptionerBackend(Protocol):
    """Protocol for video captioning backends."""

    def caption_frame(self, frames: list[np.ndarray]) -> str:
        """Caption a single keyframe (midpoint frame of a segment)."""
        ...

    def caption_segment(self, frames: list[Any]) -> dict | str:
        """Caption a segment (multiple frames + optional string context tokens)."""
        ...


@runtime_checkable
class AggregatorBackend(Protocol):
    """Protocol for annotation refinement/aggregation backends."""

    def refine(self, draft: str, context: str, effort: str = "high") -> str:
        """Refine a draft annotation given context."""
        ...


class GeminiCaptioner:
    """Captioner using Gemini API (existing behavior)."""

    def __init__(self, api_key: str | None = None):
        self._api_key = api_key
        self._caption_fn = None
        self._frame_caption_fn = None

    def _get_caption_fn(self):
        if self._caption_fn is None:
            from kuavi.captioning import make_gemini_caption_fn

            self._caption_fn = make_gemini_caption_fn(api_key=self._api_key)
        return self._caption_fn

    def _get_frame_caption_fn(self):
        if self._frame_caption_fn is None:
            from kuavi.captioning import make_gemini_frame_caption_fn

            self._frame_caption_fn = make_gemini_frame_caption_fn(api_key=self._api_key)
        return self._frame_caption_fn

    def caption_frame(self, frames: list[np.ndarray]) -> str:
        return self._get_frame_caption_fn()(frames)

    def caption_segment(self, frames: list[Any]) -> dict | str:
        return self._get_caption_fn()(frames)


class GeminiAggregator:
    """Aggregator using Gemini API (existing behavior)."""

    def __init__(self, api_key: str | None = None):
        self._api_key = api_key
        self._refine_fn = None

    def _get_refine_fn(self):
        if self._refine_fn is None:
            from kuavi.captioning import make_gemini_refine_fn

            self._refine_fn = make_gemini_refine_fn(api_key=self._api_key)
        return self._refine_fn

    def refine(self, draft: str, context: str, effort: str = "high") -> str:
        return self._get_refine_fn()(draft, context, effort)


class LocalVLMCaptioner:
    """Captioner using local VLM models (stub — not yet implemented)."""

    def __init__(
        self,
        frame_model: str = "meta-llama/Llama-3.2-Vision-11B",
        segment_model: str = "facebook/Perception-LM-3B",
    ):
        self.frame_model = frame_model
        self.segment_model = segment_model

    def caption_frame(self, frames: list[np.ndarray]) -> str:
        raise NotImplementedError(
            f"Local VLM captioning requires {self.frame_model}. "
            "Install with: pip install transformers accelerate. "
            "This feature will be available in a future release."
        )

    def caption_segment(self, frames: list[Any]) -> dict | str:
        raise NotImplementedError(
            f"Local VLM segment captioning requires {self.segment_model}. "
            "Install with: pip install transformers accelerate. "
            "This feature will be available in a future release."
        )


class LocalLLMAggregator:
    """Aggregator using local LLM models (stub — not yet implemented)."""

    def __init__(self, model: str = "meta-llama/Llama-3.3-8B"):
        self.model = model

    def refine(self, draft: str, context: str, effort: str = "high") -> str:
        raise NotImplementedError(
            f"Local LLM aggregation requires {self.model}. "
            "Install with: pip install transformers accelerate. "
            "This feature will be available in a future release."
        )


# Preset configurations
CAPTION_PRESETS: dict[str, dict] = {
    "api": {
        "description": "Gemini API (current default behavior)",
        "captioner": "gemini",
        "aggregator": "gemini",
        "vram": "0GB (API-based)",
    },
    "local-full": {
        "description": "Full local stack: Llama-3.2-Vision-11B + PLM-3B + Llama-3.3-8B",
        "captioner": "local",
        "aggregator": "local",
        "frame_model": "meta-llama/Llama-3.2-Vision-11B",
        "segment_model": "facebook/Perception-LM-3B",
        "aggregator_model": "meta-llama/Llama-3.3-8B",
        "vram": "~48GB",
    },
    "local-efficient": {
        "description": "Efficient local: PLM-3B for all captioning + Llama-3.3-8B",
        "captioner": "local",
        "aggregator": "local",
        "frame_model": "facebook/Perception-LM-3B",
        "segment_model": "facebook/Perception-LM-3B",
        "aggregator_model": "meta-llama/Llama-3.3-8B",
        "vram": "~16GB",
    },
    "local-minimal": {
        "description": "Minimal local: PLM-3B only, no aggregation",
        "captioner": "local",
        "aggregator": None,
        "frame_model": "facebook/Perception-LM-3B",
        "segment_model": "facebook/Perception-LM-3B",
        "vram": "~8GB",
    },
}


def create_captioner(
    preset: str, api_key: str | None = None
) -> tuple[CaptionerBackend, AggregatorBackend | None]:
    """Create captioner and aggregator from a preset name.

    Args:
        preset: One of the keys in ``CAPTION_PRESETS``.
        api_key: Optional API key for cloud-based backends (e.g. Gemini).

    Returns:
        Tuple of (captioner, aggregator). Aggregator may be None for minimal presets.

    Raises:
        ValueError: If ``preset`` is not a recognized key.
    """
    if preset not in CAPTION_PRESETS:
        raise ValueError(
            f"Unknown caption preset {preset!r}. Valid: {list(CAPTION_PRESETS)}"
        )

    config = CAPTION_PRESETS[preset]

    if config["captioner"] == "gemini":
        captioner: CaptionerBackend = GeminiCaptioner(api_key=api_key)
    else:
        captioner = LocalVLMCaptioner(
            frame_model=config.get("frame_model", "meta-llama/Llama-3.2-Vision-11B"),
            segment_model=config.get("segment_model", "facebook/Perception-LM-3B"),
        )

    aggregator: AggregatorBackend | None = None
    if config.get("aggregator") == "gemini":
        aggregator = GeminiAggregator(api_key=api_key)
    elif config.get("aggregator") == "local":
        aggregator = LocalLLMAggregator(
            model=config.get("aggregator_model", "meta-llama/Llama-3.3-8B")
        )

    return captioner, aggregator
