import base64
import logging
import os
import time
from collections import defaultdict
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.errors import ServerError

from rlm.clients.base_lm import BaseLM
from rlm.core.types import ModelUsageSummary, UsageSummary

logger = logging.getLogger(__name__)

load_dotenv()

DEFAULT_GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


class GeminiClient(BaseLM):
    """
    LM Client for running models with the Google Gemini API.
    Uses the official google-genai SDK.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str | None = "gemini-2.5-flash",
        thinking_level: str | None = None,
        **kwargs,
    ):
        super().__init__(model_name=model_name, **kwargs)
        self.thinking_level = thinking_level or "LOW"

        if api_key is None:
            api_key = DEFAULT_GEMINI_API_KEY

        if api_key is None:
            raise ValueError(
                "Gemini API key is required. Set GEMINI_API_KEY env var or pass api_key."
            )

        # Configure HTTP options with timeout (app-level retry handles transient errors)
        http_options = types.HttpOptions(
            timeout=int(self.timeout * 1000),  # milliseconds
        )
        self.client = genai.Client(api_key=api_key, http_options=http_options)
        self.model_name = model_name

        # Per-model usage tracking
        self.model_call_counts: dict[str, int] = defaultdict(int)
        self.model_input_tokens: dict[str, int] = defaultdict(int)
        self.model_output_tokens: dict[str, int] = defaultdict(int)
        # Last call tracking
        self.last_prompt_tokens = 0
        self.last_completion_tokens = 0

    # Retry settings for transient server errors (500/504)
    MAX_RETRIES = 8
    RETRY_BASE_DELAY = 2.0  # seconds
    RETRY_MAX_DELAY = 30.0  # seconds

    def _retry_call(self, fn, *args, **kwargs):
        """Retry a Gemini API call on transient 500/504 server errors."""
        last_exc = None
        for attempt in range(self.MAX_RETRIES):
            try:
                return fn(*args, **kwargs)
            except ServerError as e:
                last_exc = e
                status = getattr(e, "status_code", None) or getattr(e, "code", 0)
                if status not in (500, 504):
                    raise
                delay = min(
                    self.RETRY_BASE_DELAY * (2**attempt),
                    self.RETRY_MAX_DELAY,
                )
                logger.warning(
                    "Gemini server error %s on attempt %d/%d, retrying in %.1fs",
                    status,
                    attempt + 1,
                    self.MAX_RETRIES,
                    delay,
                )
                time.sleep(delay)
        raise last_exc

    def _is_gemini3(self, model: str | None = None) -> bool:
        """Check if the model is a Gemini 3 variant."""
        m = model or self.model_name or ""
        return "gemini-3" in m

    def _build_config(
        self, system_instruction: str | None, model: str
    ) -> types.GenerateContentConfig | None:
        """Build GenerateContentConfig, adding thinking_config for Gemini 3."""
        kwargs: dict[str, Any] = {}
        if system_instruction:
            kwargs["system_instruction"] = system_instruction
        if self._is_gemini3(model):
            kwargs["thinking_config"] = types.ThinkingConfig(thinking_level=self.thinking_level)
        return types.GenerateContentConfig(**kwargs) if kwargs else None

    @staticmethod
    def _extract_text(response: types.GenerateContentResponse) -> str:
        """Extract text from response, handling MALFORMED_FUNCTION_CALL gracefully.

        Gemini 3 thinking models may interpret REPL code examples in the system
        prompt as function call declarations and emit ``function_call`` parts
        instead of text.  When that happens we convert the function call back
        into a ``repl`` code block so the RLM pipeline can execute it normally.
        """
        try:
            text = response.text
        except (ValueError, AttributeError):
            text = None

        if text is not None:
            return text

        text_parts: list[str] = []
        if response.candidates:
            for part in (response.candidates[0].content.parts or []):
                if hasattr(part, "thought") and part.thought:
                    continue
                if hasattr(part, "text") and part.text:
                    text_parts.append(part.text)
                elif hasattr(part, "function_call") and part.function_call:
                    fc = part.function_call
                    # Convert function_call back into a repl code block
                    args = dict(fc.args) if fc.args else {}
                    if fc.name == "REPL" and "code" in args:
                        code = args["code"]
                    else:
                        # Reconstruct as a Python call
                        arg_strs = [f"{k}={v!r}" for k, v in args.items()]
                        code = f"{fc.name}({', '.join(arg_strs)})"
                    text_parts.append(f"```repl\n{code}\n```")

        return "\n".join(text_parts) if text_parts else ""

    def completion(self, prompt: str | list[dict[str, Any]], model: str | None = None) -> str:
        cached, cache_key = self.cached_completion(prompt)
        if cached is not None:
            return cached

        contents, system_instruction = self._prepare_contents(prompt)

        model = model or self.model_name
        if not model:
            raise ValueError("Model name is required for Gemini client.")

        config = self._build_config(system_instruction, model)

        response = self._retry_call(
            self.client.models.generate_content,
            model=model,
            contents=contents,
            config=config,
        )

        self._track_cost(response, model)
        content = self._extract_text(response)
        self.cache_store(cache_key, content)
        return content

    async def _async_retry_call(self, fn, *args, **kwargs):
        """Async retry for Gemini API calls on transient 500/504 server errors."""
        import asyncio

        last_exc = None
        for attempt in range(self.MAX_RETRIES):
            try:
                return await fn(*args, **kwargs)
            except ServerError as e:
                last_exc = e
                status = getattr(e, "status_code", None) or getattr(e, "code", 0)
                if status not in (500, 504):
                    raise
                delay = min(
                    self.RETRY_BASE_DELAY * (2**attempt),
                    self.RETRY_MAX_DELAY,
                )
                logger.warning(
                    "Gemini server error %s on attempt %d/%d, retrying in %.1fs",
                    status,
                    attempt + 1,
                    self.MAX_RETRIES,
                    delay,
                )
                await asyncio.sleep(delay)
        raise last_exc

    async def acompletion(
        self, prompt: str | list[dict[str, Any]], model: str | None = None
    ) -> str:
        contents, system_instruction = self._prepare_contents(prompt)

        model = model or self.model_name
        if not model:
            raise ValueError("Model name is required for Gemini client.")

        config = self._build_config(system_instruction, model)

        response = await self._async_retry_call(
            self.client.aio.models.generate_content,
            model=model,
            contents=contents,
            config=config,
        )

        self._track_cost(response, model)
        return self._extract_text(response)

    @staticmethod
    def _to_parts(content: Any) -> list[types.Part]:
        """Convert message content into a list of Gemini ``Part`` objects.

        Handles:
        - Plain strings → ``types.Part(text=...)``
        - Tagged image dicts (``{"__image__": True, "data": ..., "mime_type": ...}``)
          → ``types.Part(inline_data=types.Blob(...))``
        - Strings that look like serialised dicts/lists (from ``str(context)``)
          are kept as text — only explicit tagged dicts become image parts.
        - Lists/dicts are walked recursively so nested frame data is found.
        """
        parts: list[types.Part] = []

        if isinstance(content, str):
            parts.append(types.Part(text=content))
        elif isinstance(content, dict):
            if content.get("__image__"):
                raw = base64.b64decode(content["data"])
                parts.append(
                    types.Part(
                        inline_data=types.Blob(
                            data=raw,
                            mime_type=content.get("mime_type", "image/jpeg"),
                        )
                    )
                )
            else:
                # Recurse into dict values to find nested image dicts
                for v in content.values():
                    parts.extend(GeminiClient._to_parts(v))
        elif isinstance(content, list):
            for item in content:
                parts.extend(GeminiClient._to_parts(item))
        else:
            # Fallback: coerce to string
            parts.append(types.Part(text=str(content)))

        return parts

    def _prepare_contents(
        self, prompt: str | list[dict[str, Any]]
    ) -> tuple[list[types.Content] | str, str | None]:
        """Prepare contents and extract system instruction for Gemini API."""
        system_instruction = None

        if isinstance(prompt, str):
            return prompt, None

        if isinstance(prompt, list) and all(isinstance(item, dict) for item in prompt):
            # Convert OpenAI-style messages to Gemini format
            contents = []
            for msg in prompt:
                role = msg.get("role")
                content = msg.get("content", "")

                if role == "system":
                    # System instruction is always text
                    system_instruction = content if isinstance(content, str) else str(content)
                elif role == "user":
                    parts = self._to_parts(content)
                    contents.append(types.Content(role="user", parts=parts))
                elif role == "assistant":
                    # Gemini uses "model" instead of "assistant"
                    parts = self._to_parts(content)
                    contents.append(types.Content(role="model", parts=parts))
                else:
                    # Default to user role for unknown roles
                    parts = self._to_parts(content)
                    contents.append(types.Content(role="user", parts=parts))

            # Gemini requires conversations to start with a user turn.
            # Merge any leading model messages into the first user message.
            if contents and contents[0].role == "model":
                first_user_idx = next((i for i, c in enumerate(contents) if c.role == "user"), None)
                if first_user_idx is not None:
                    # Prepend model parts as text context into the first user message
                    model_parts = []
                    for c in contents[:first_user_idx]:
                        model_parts.extend(c.parts)
                    contents[first_user_idx].parts = model_parts + contents[first_user_idx].parts
                    contents = contents[first_user_idx:]
                else:
                    # No user message at all — convert model messages to user
                    for c in contents:
                        c.role = "user"

            return contents, system_instruction

        raise ValueError(f"Invalid prompt type: {type(prompt)}")

    def _track_cost(self, response: types.GenerateContentResponse, model: str):
        self.model_call_counts[model] += 1

        # Extract token usage from response
        usage = response.usage_metadata
        if usage:
            input_tokens = usage.prompt_token_count or 0
            output_tokens = usage.candidates_token_count or 0

            self.model_input_tokens[model] += input_tokens
            self.model_output_tokens[model] += output_tokens
            # Track last call for handler to read
            self.last_prompt_tokens = input_tokens
            self.last_completion_tokens = output_tokens
        else:
            self.last_prompt_tokens = 0
            self.last_completion_tokens = 0

    def get_usage_summary(self) -> UsageSummary:
        model_summaries = {}
        for model in self.model_call_counts:
            model_summaries[model] = ModelUsageSummary(
                total_calls=self.model_call_counts[model],
                total_input_tokens=self.model_input_tokens[model],
                total_output_tokens=self.model_output_tokens[model],
            )
        return UsageSummary(model_usage_summaries=model_summaries)

    def get_last_usage(self) -> ModelUsageSummary:
        return ModelUsageSummary(
            total_calls=1,
            total_input_tokens=self.last_prompt_tokens,
            total_output_tokens=self.last_completion_tokens,
        )
