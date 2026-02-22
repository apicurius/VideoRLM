"""KUAVi configuration types."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

VJEPA2_PRESETS: dict[str, dict] = {
    "fast": {
        "model": "facebook/vjepa2-vitl-fpc64-256",
        "clip_size": 16,
        "resolution": 256,
        "embed_dim": 1024,
    },
    "balanced": {
        "model": "facebook/vjepa2-vith-fpc64-256",
        "clip_size": 32,
        "resolution": 256,
        "embed_dim": 1280,
    },
    "quality": {
        "model": "facebook/vjepa2-vitg-fpc64-384",
        "clip_size": 64,
        "resolution": 384,
        "embed_dim": 1536,
    },
}


@dataclass
class KUAViConfig:
    """Configuration for the KUAVi video analysis pipeline.

    Args:
        embedding_model: HuggingFace model id for vision-language embeddings.
        scene_model: HuggingFace model id for scene detection (V-JEPA 2).
            None disables model-based scene detection.
        text_embedding_model: HuggingFace model id for a separate text encoder.
            None uses the embedding_model's text tower.
        fps: Frames per second to extract from video.
        auto_fps: Dynamically compute FPS based on video duration.
        target_frames: Target frame count when auto_fps is enabled.
        cache_dir: Directory for caching video indices.
        asr_model: Qwen3-ASR model name for speech transcription.
        max_frames_per_segment: Cap frames per segment for memory efficiency.
        resize: Optional (width, height) to resize extracted frames.
        hierarchical: Enable multi-level scene hierarchy.
        scene_clip_size: Number of frames per V-JEPA 2 clip.
        scene_stride: Stride for overlapping V-JEPA 2 windows (used when overlapping_vjepa=True).
    """

    embedding_model: str = "google/siglip2-base-patch16-256"
    scene_model: str | None = "facebook/vjepa2-vitl-fpc64-256"
    text_embedding_model: str | None = None
    fps: float = 1.0
    auto_fps: bool = False
    target_frames: int = 120
    cache_dir: str | Path | None = None
    asr_model: str = "Qwen/Qwen3-ASR-1.7B"
    max_frames_per_segment: int = 32
    resize: tuple[int, int] | None = None
    hierarchical: bool = False
    scene_clip_size: int = 16
    scene_stride: int = 8
    scene_model_preset: str | None = None
    caption_preset: str | None = None
