"""KUAVi: Agentic Vision Intelligence for video analysis.

Provides video indexing, semantic search, scene detection, and frame extraction
for use with Claude Code's MCP server, skills, and custom agents.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from kuavi.prompts import VIDEO_ANALYSIS_PROMPT

if TYPE_CHECKING:
    from kuavi.context import VideoContext
    from kuavi.indexer import VideoIndex, VideoIndexer
    from kuavi.loader import LoadedVideo, VideoLoader, VideoMetadata, VideoSegment
    from kuavi.types import KUAViConfig


def __getattr__(name: str):
    """Lazy imports for modules that require opencv-python."""
    _cv2_exports = {
        "VideoContext": "kuavi.context",
        "VideoLoader": "kuavi.loader",
        "VideoMetadata": "kuavi.loader",
        "VideoSegment": "kuavi.loader",
        "LoadedVideo": "kuavi.loader",
        "VideoIndex": "kuavi.indexer",
        "VideoIndexer": "kuavi.indexer",
        "KUAViConfig": "kuavi.types",
    }
    if name in _cv2_exports:
        try:
            import importlib

            module = importlib.import_module(_cv2_exports[name])
            return getattr(module, name)
        except ImportError as e:
            raise ImportError(
                f"Cannot import '{name}': opencv-python is required. "
                f"Install with: pip install kuavi[embeddings]"
            ) from e
    raise AttributeError(f"module 'kuavi' has no attribute {name!r}")


__all__ = [
    "KUAViConfig",
    "LoadedVideo",
    "VIDEO_ANALYSIS_PROMPT",
    "VideoContext",
    "VideoIndex",
    "VideoIndexer",
    "VideoLoader",
    "VideoMetadata",
    "VideoSegment",
]
