from typing import Any

from dotenv import load_dotenv

from rlm.clients.base_lm import BaseLM
from rlm.core.types import ClientBackend

load_dotenv()


def get_client(
    backend: ClientBackend,
    backend_kwargs: dict[str, Any],
) -> BaseLM:
    """Route a backend + kwargs to the appropriate :class:`BaseLM` subclass.

    When the ``LITELLM_PROXY_URL`` environment variable is set, **all** backends
    are transparently routed through a single LiteLLM proxy instance.  The
    original *model_name* is forwarded as-is so the proxy can dispatch to the
    correct upstream provider.

    Currently supported backends:
        openai, vllm, portkey, openrouter, vercel, litellm,
        anthropic, gemini, azure_openai
    """
    # --- LiteLLM proxy override -------------------------------------------
    from rlm.utils.litellm_proxy import get_litellm_proxy_url

    proxy_url = get_litellm_proxy_url()
    if proxy_url is not None:
        from rlm.clients.litellm import LiteLLMClient

        # Forward model_name through the proxy; let proxy handle routing.
        return LiteLLMClient(
            model_name=backend_kwargs.get("model_name"),
            api_base=proxy_url,
            api_key=backend_kwargs.get("api_key"),
            **{k: v for k, v in backend_kwargs.items() if k not in ("model_name", "api_key", "base_url", "api_base")},
        )
    # --- end proxy override -----------------------------------------------

    if backend == "openai":
        from rlm.clients.openai import OpenAIClient

        return OpenAIClient(**backend_kwargs)
    elif backend == "vllm":
        from rlm.clients.openai import OpenAIClient

        assert "base_url" in backend_kwargs, (
            "base_url is required to be set to local vLLM server address for vLLM"
        )
        return OpenAIClient(**backend_kwargs)
    elif backend == "portkey":
        from rlm.clients.portkey import PortkeyClient

        return PortkeyClient(**backend_kwargs)
    elif backend == "openrouter":
        from rlm.clients.openai import OpenAIClient

        backend_kwargs.setdefault("base_url", "https://openrouter.ai/api/v1")
        return OpenAIClient(**backend_kwargs)
    elif backend == "vercel":
        from rlm.clients.openai import OpenAIClient

        backend_kwargs.setdefault("base_url", "https://ai-gateway.vercel.sh/v1")
        return OpenAIClient(**backend_kwargs)
    elif backend == "litellm":
        from rlm.clients.litellm import LiteLLMClient

        return LiteLLMClient(**backend_kwargs)
    elif backend == "anthropic":
        from rlm.clients.anthropic import AnthropicClient

        return AnthropicClient(**backend_kwargs)
    elif backend == "gemini":
        from rlm.clients.gemini import GeminiClient

        return GeminiClient(**backend_kwargs)
    elif backend == "azure_openai":
        from rlm.clients.azure_openai import AzureOpenAIClient

        return AzureOpenAIClient(**backend_kwargs)
    else:
        raise ValueError(
            f"Unknown backend: {backend}. Supported backends: ['openai', 'vllm', 'portkey', 'openrouter', 'litellm', 'anthropic', 'azure_openai', 'gemini', 'vercel']"
        )


def create_client(
    backend: ClientBackend,
    backend_kwargs: dict[str, Any],
) -> BaseLM:
    """Backward-compatible alias for legacy call sites."""
    return get_client(backend=backend, backend_kwargs=backend_kwargs)
