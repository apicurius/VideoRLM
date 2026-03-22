import os
import sys
from pathlib import Path

# Add vendored LanguageBind to sys.path
vendor_path = Path(__file__).parent.parent.parent / "vendor" / "LanguageBind"
if str(vendor_path) not in sys.path and vendor_path.exists():
    sys.path.insert(0, str(vendor_path))

from .base_embedder import BaseEmbedder

def create_embedder() -> BaseEmbedder:
    """
    Factory function. Returns the configured embedder.

    Set EMBEDDING_BACKEND env var to switch:
      EMBEDDING_BACKEND=languagebind  (default, free, local)
      EMBEDDING_BACKEND=gemini        (optional, API, higher quality)
    """
    backend = os.environ.get(
        "EMBEDDING_BACKEND", "languagebind"
    ).lower().strip()

    if backend == "gemini":
        from .gemini_embedder import GeminiEmbedder
        return GeminiEmbedder()
    elif backend == "languagebind":
        from .languagebind_embedder import LanguageBindEmbedder
        return LanguageBindEmbedder()
    else:
        raise ValueError(
            f"Unknown EMBEDDING_BACKEND: '{backend}'. "
            "Choose 'languagebind' or 'gemini'."
        )

__all__ = ["BaseEmbedder", "create_embedder"]
