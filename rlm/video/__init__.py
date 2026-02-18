"""Video processing modules for RLM Long Video Understanding (LVU).

Requires the 'video' optional dependency: pip install rlms[video]
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rlm.video.video_prompts import VIDEO_SYSTEM_PROMPT

if TYPE_CHECKING:
    from rlm.video.video_context import VideoContext
    from rlm.video.video_indexer import VideoIndex, VideoIndexer
    from rlm.video.video_loader import LoadedVideo, VideoLoader, VideoMetadata, VideoSegment
    from rlm.video.video_rlm import VideoRLM


def __getattr__(name: str):
    """Lazy imports for modules that require opencv-python."""
    _cv2_exports = {
        "VideoContext": "rlm.video.video_context",
        "VideoLoader": "rlm.video.video_loader",
        "VideoMetadata": "rlm.video.video_loader",
        "VideoSegment": "rlm.video.video_loader",
        "LoadedVideo": "rlm.video.video_loader",
        "VideoRLM": "rlm.video.video_rlm",
        "VideoIndex": "rlm.video.video_indexer",
        "VideoIndexer": "rlm.video.video_indexer",
    }
    if name in _cv2_exports:
        try:
            import importlib

            module = importlib.import_module(_cv2_exports[name])
            return getattr(module, name)
        except ImportError as e:
            raise ImportError(
                f"Cannot import '{name}': opencv-python is required. "
                f"Install with: pip install rlms[video]"
            ) from e
    raise AttributeError(f"module 'rlm.video' has no attribute {name!r}")


__all__ = [
    "LoadedVideo",
    "VIDEO_SYSTEM_PROMPT",
    "VideoContext",
    "VideoIndex",
    "VideoIndexer",
    "VideoLoader",
    "VideoMetadata",
    "VideoRLM",
    "VideoSegment",
]
