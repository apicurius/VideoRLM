"""Tests for the Gemini client."""

import base64
import os
from unittest.mock import MagicMock, patch

import pytest
from dotenv import load_dotenv
from google.genai import types

from rlm.clients.gemini import GeminiClient
from rlm.core.types import ModelUsageSummary, UsageSummary

load_dotenv()


class TestGeminiClientUnit:
    """Unit tests that don't require API calls."""

    def test_init_with_api_key(self):
        """Test client initialization with explicit API key."""
        with patch("rlm.clients.gemini.genai.Client"):
            client = GeminiClient(api_key="test-key", model_name="gemini-2.5-flash")
            assert client.model_name == "gemini-2.5-flash"

    def test_init_default_model(self):
        """Test client uses default model name."""
        with patch("rlm.clients.gemini.genai.Client"):
            client = GeminiClient(api_key="test-key")
            assert client.model_name == "gemini-2.5-flash"

    def test_init_requires_api_key(self):
        """Test client raises error when no API key provided."""
        with patch.dict(os.environ, {}, clear=True):
            with patch("rlm.clients.gemini.DEFAULT_GEMINI_API_KEY", None):
                with pytest.raises(ValueError, match="Gemini API key is required"):
                    GeminiClient(api_key=None)

    def test_usage_tracking_initialization(self):
        """Test that usage tracking is properly initialized."""
        with patch("rlm.clients.gemini.genai.Client"):
            client = GeminiClient(api_key="test-key")
            assert client.model_call_counts == {}
            assert client.model_input_tokens == {}
            assert client.model_output_tokens == {}
            assert client.last_prompt_tokens == 0
            assert client.last_completion_tokens == 0

    def test_get_usage_summary_empty(self):
        """Test usage summary when no calls have been made."""
        with patch("rlm.clients.gemini.genai.Client"):
            client = GeminiClient(api_key="test-key")
            summary = client.get_usage_summary()
            assert isinstance(summary, UsageSummary)
            assert summary.model_usage_summaries == {}

    def test_get_last_usage(self):
        """Test last usage returns correct format."""
        with patch("rlm.clients.gemini.genai.Client"):
            client = GeminiClient(api_key="test-key")
            client.last_prompt_tokens = 100
            client.last_completion_tokens = 50
            usage = client.get_last_usage()
            assert isinstance(usage, ModelUsageSummary)
            assert usage.total_calls == 1
            assert usage.total_input_tokens == 100
            assert usage.total_output_tokens == 50

    def test_prepare_contents_string(self):
        """Test _prepare_contents with string input."""
        with patch("rlm.clients.gemini.genai.Client"):
            client = GeminiClient(api_key="test-key")
            contents, system = client._prepare_contents("Hello world")
            assert contents == "Hello world"
            assert system is None

    def test_prepare_contents_messages_with_system(self):
        """Test _prepare_contents extracts system message."""
        with patch("rlm.clients.gemini.genai.Client"):
            client = GeminiClient(api_key="test-key")
            messages = [
                {"role": "system", "content": "You are helpful"},
                {"role": "user", "content": "Hello"},
            ]
            contents, system = client._prepare_contents(messages)
            assert system == "You are helpful"
            assert len(contents) == 1
            assert contents[0].role == "user"

    def test_prepare_contents_role_mapping(self):
        """Test _prepare_contents maps assistant to model."""
        with patch("rlm.clients.gemini.genai.Client"):
            client = GeminiClient(api_key="test-key")
            messages = [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there"},
                {"role": "user", "content": "How are you?"},
            ]
            contents, system = client._prepare_contents(messages)
            assert system is None
            assert len(contents) == 3
            assert contents[0].role == "user"
            assert contents[1].role == "model"  # assistant -> model
            assert contents[2].role == "user"

    def test_prepare_contents_invalid_type(self):
        """Test _prepare_contents raises on invalid input."""
        with patch("rlm.clients.gemini.genai.Client"):
            client = GeminiClient(api_key="test-key")
            with pytest.raises(ValueError, match="Invalid prompt type"):
                client._prepare_contents(12345)

    def test_completion_requires_model(self):
        """Test completion raises when no model specified."""
        with patch("rlm.clients.gemini.genai.Client"):
            client = GeminiClient(api_key="test-key", model_name=None)
            with pytest.raises(ValueError, match="Model name is required"):
                client.completion("Hello")

    def test_completion_with_mocked_response(self):
        """Test completion with mocked API response."""
        mock_response = MagicMock()
        mock_response.text = "Hello from Gemini!"
        mock_response.usage_metadata.prompt_token_count = 10
        mock_response.usage_metadata.candidates_token_count = 5

        with patch("rlm.clients.gemini.genai.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client.models.generate_content.return_value = mock_response
            mock_client_class.return_value = mock_client

            client = GeminiClient(api_key="test-key", model_name="gemini-2.5-flash")
            result = client.completion("Hello")

            assert result == "Hello from Gemini!"
            assert client.model_call_counts["gemini-2.5-flash"] == 1
            assert client.model_input_tokens["gemini-2.5-flash"] == 10
            assert client.model_output_tokens["gemini-2.5-flash"] == 5


class TestToPartsMultimodal:
    """Tests for GeminiClient._to_parts() multimodal conversion."""

    def test_string_to_text_part(self):
        """String content produces a single Part(text=...)."""
        parts = GeminiClient._to_parts("hello")
        assert len(parts) == 1
        assert parts[0].text == "hello"

    def test_image_dict_to_inline_data_part(self):
        """Tagged image dict produces Part(inline_data=Blob(...))."""
        raw_bytes = b"\x89PNG\r\n"
        b64 = base64.b64encode(raw_bytes).decode()
        img = {"__image__": True, "data": b64, "mime_type": "image/png"}

        parts = GeminiClient._to_parts(img)
        assert len(parts) == 1
        assert parts[0].inline_data is not None
        assert parts[0].inline_data.mime_type == "image/png"
        assert parts[0].inline_data.data == raw_bytes

    def test_image_dict_default_mime_type(self):
        """Image dict without explicit mime_type defaults to image/jpeg."""
        b64 = base64.b64encode(b"\xff\xd8").decode()
        img = {"__image__": True, "data": b64}

        parts = GeminiClient._to_parts(img)
        assert parts[0].inline_data.mime_type == "image/jpeg"

    def test_mixed_list_produces_multiple_parts(self):
        """A list of text + image dicts yields the correct sequence of parts."""
        b64 = base64.b64encode(b"\x00\x01").decode()
        content = [
            "Describe this image:",
            {"__image__": True, "data": b64, "mime_type": "image/jpeg"},
            "What do you see?",
        ]

        parts = GeminiClient._to_parts(content)
        assert len(parts) == 3
        assert parts[0].text == "Describe this image:"
        assert parts[1].inline_data is not None
        assert parts[2].text == "What do you see?"

    def test_non_image_dict_recurses_values(self):
        """A plain dict (no __image__ key) recurses into its values."""
        b64 = base64.b64encode(b"\x00").decode()
        content = {"frame": {"__image__": True, "data": b64, "mime_type": "image/jpeg"}}

        parts = GeminiClient._to_parts(content)
        assert len(parts) == 1
        assert parts[0].inline_data is not None

    def test_fallback_coerces_to_string(self):
        """Non-str/dict/list content is coerced to a text Part."""
        parts = GeminiClient._to_parts(42)
        assert len(parts) == 1
        assert parts[0].text == "42"

    def test_prepare_contents_multimodal_message(self):
        """_prepare_contents with multimodal user content calls _to_parts."""
        b64 = base64.b64encode(b"\x00").decode()
        messages = [
            {"role": "system", "content": "Analyze the image."},
            {
                "role": "user",
                "content": [
                    "What is in this image?",
                    {"__image__": True, "data": b64, "mime_type": "image/png"},
                ],
            },
        ]

        with patch("rlm.clients.gemini.genai.Client"):
            client = GeminiClient(api_key="test-key")
            contents, system = client._prepare_contents(messages)

        assert system == "Analyze the image."
        assert len(contents) == 1
        assert contents[0].role == "user"
        # The user message should have 2 parts: text + image
        assert len(contents[0].parts) == 2
        assert contents[0].parts[0].text == "What is in this image?"
        assert contents[0].parts[1].inline_data is not None


class TestLlmQueryListPrompt:
    """Test that _llm_query wraps list prompts into a message dict before sending."""

    def test_llm_query_wraps_list_as_message_dict(self):
        """When prompt is a list, _llm_query wraps it as {'role': 'user', 'content': <list>}."""
        from rlm.environments.local_repl import LocalREPL

        repl = LocalREPL()
        repl.lm_handler_address = ("localhost", 9999)

        b64 = base64.b64encode(b"\x00").decode()
        multimodal_prompt = [
            "describe this",
            {"__image__": True, "data": b64, "mime_type": "image/jpeg"},
        ]

        with patch("rlm.environments.local_repl.send_lm_request") as mock_send:
            mock_response = MagicMock()
            mock_response.success = True
            mock_response.chat_completion.response = "I see an image"
            mock_send.return_value = mock_response

            result = repl._llm_query(multimodal_prompt)

        # Verify send_lm_request was called
        mock_send.assert_called_once()
        sent_request = mock_send.call_args[0][1]
        # The prompt should have been wrapped into a list of message dicts
        assert isinstance(sent_request.prompt, list)
        assert len(sent_request.prompt) == 1
        assert sent_request.prompt[0]["role"] == "user"
        assert sent_request.prompt[0]["content"] is multimodal_prompt
        assert result == "I see an image"

        repl.cleanup()

    def test_llm_query_string_prompt_unchanged(self):
        """When prompt is a plain string, it is sent as-is (not wrapped)."""
        from rlm.environments.local_repl import LocalREPL

        repl = LocalREPL()
        repl.lm_handler_address = ("localhost", 9999)

        with patch("rlm.environments.local_repl.send_lm_request") as mock_send:
            mock_response = MagicMock()
            mock_response.success = True
            mock_response.chat_completion.response = "42"
            mock_send.return_value = mock_response

            result = repl._llm_query("What is 6*7?")

        sent_request = mock_send.call_args[0][1]
        assert isinstance(sent_request.prompt, str)
        assert sent_request.prompt == "What is 6*7?"
        assert result == "42"

        repl.cleanup()


class TestGeminiClientIntegration:
    """Integration tests that require a real API key."""

    @pytest.mark.skipif(
        not os.environ.get("GEMINI_API_KEY"),
        reason="GEMINI_API_KEY not set",
    )
    def test_simple_completion(self):
        """Test a simple completion with real API."""
        client = GeminiClient(model_name="gemini-2.5-flash")
        result = client.completion("What is 2+2? Reply with just the number.")
        assert "4" in result

        # Verify usage was tracked
        usage = client.get_usage_summary()
        assert "gemini-2.5-flash" in usage.model_usage_summaries
        assert usage.model_usage_summaries["gemini-2.5-flash"].total_calls == 1

    @pytest.mark.skipif(
        not os.environ.get("GEMINI_API_KEY"),
        reason="GEMINI_API_KEY not set",
    )
    def test_message_list_completion(self):
        """Test completion with message list format."""
        client = GeminiClient(model_name="gemini-2.5-flash")
        messages = [
            {"role": "system", "content": "You are a helpful math tutor."},
            {"role": "user", "content": "What is 5 * 5? Reply with just the number."},
        ]
        result = client.completion(messages)
        assert "25" in result

    @pytest.mark.skipif(
        not os.environ.get("GEMINI_API_KEY"),
        reason="GEMINI_API_KEY not set",
    )
    async def test_async_completion(self):
        """Test async completion."""
        client = GeminiClient(model_name="gemini-2.5-flash")
        result = await client.acompletion("What is 3+3? Reply with just the number.")
        assert "6" in result


if __name__ == "__main__":
    # Run integration tests directly
    test = TestGeminiClientIntegration()
    print("Testing simple completion...")
    test.test_simple_completion()
    print("Testing message list completion...")
    test.test_message_list_completion()
    print("All integration tests passed!")
