from abc import ABC, abstractmethod

class BaseEmbedder(ABC):
    """
    Shared interface for all embedding backends.
    LanguageBind and Gemini Embedding 2 both implement this.
    Switching backends = one env var, zero code changes.
    """

    @abstractmethod
    def embed_text(self, text: str) -> list[float]:
        """Embed a text document for indexing."""
        ...

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Embed a search query (may use different task type)."""
        ...

    @abstractmethod
    def embed_video_segment(
        self,
        video_path: str,
        start_sec: float,
        end_sec: float,
    ) -> list[float]:
        """Embed a video clip between start_sec and end_sec."""
        ...

    @abstractmethod
    def embed_audio_segment(
        self,
        audio_path: str,
    ) -> list[float]:
        """Embed an audio file or segment."""
        ...

    def similarity(
        self,
        a: list[float],
        b: list[float],
    ) -> float:
        """Cosine similarity. Shared across all backends."""
        import numpy as np
        a, b = np.array(a), np.array(b)
        denom = np.linalg.norm(a) * np.linalg.norm(b) + 1e-8
        return float(np.dot(a, b) / denom)
