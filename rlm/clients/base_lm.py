from abc import ABC, abstractmethod
from typing import Any

from rlm.core.types import ModelUsageSummary, UsageSummary

# Default timeout for LM API calls (in seconds)
DEFAULT_TIMEOUT: float = 300.0


def _prompt_cache_key(prompt: str | dict[str, Any] | list) -> str | None:
    """Compute a stable cache key for a prompt.

    Returns None for prompts that contain non-hashable content (e.g. images),
    so they always bypass the cache.
    """
    import hashlib
    import json

    try:
        serialized = json.dumps(prompt, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode()).hexdigest()
    except (TypeError, ValueError):
        return None


class BaseLM(ABC):
    """
    Base class for all language model routers / clients. When the RLM makes sub-calls, it currently
    does so in a model-agnostic way, so this class provides a base interface for all language models.
    """

    def __init__(self, model_name: str, timeout: float = DEFAULT_TIMEOUT, **kwargs):
        self.model_name = model_name
        self.timeout = timeout
        self.kwargs = kwargs

        # Completion cache: hash(prompt) -> response string.
        # Avoids duplicate API calls for identical prompts within a session.
        # Disable via enable_cache=False in subclass kwargs.
        self._cache_enabled: bool = kwargs.get("enable_cache", True)
        self._cache: dict[str, str] = {}
        self._cache_hits: int = 0

    def cached_completion(self, prompt: str | dict[str, Any]) -> tuple[str | None, str | None]:
        """Check cache for a previous response.  Returns (response, key) or (None, key)."""
        if not self._cache_enabled:
            return None, None
        key = _prompt_cache_key(prompt)
        if key is not None and key in self._cache:
            self._cache_hits += 1
            return self._cache[key], key
        return None, key

    def cache_store(self, key: str | None, response: str) -> None:
        """Store a response in the cache."""
        if self._cache_enabled and key is not None:
            self._cache[key] = response

    @abstractmethod
    def completion(self, prompt: str | dict[str, Any]) -> str:
        raise NotImplementedError

    @abstractmethod
    async def acompletion(self, prompt: str | dict[str, Any]) -> str:
        raise NotImplementedError

    @abstractmethod
    def get_usage_summary(self) -> UsageSummary:
        """Get cost summary for all model calls."""
        raise NotImplementedError

    @abstractmethod
    def get_last_usage(self) -> ModelUsageSummary:
        """Get the last cost summary of the model."""
        raise NotImplementedError
