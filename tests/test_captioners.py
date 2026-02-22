"""Tests for kuavi.captioners — pluggable captioning backend system."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from kuavi.captioners import (
    CAPTION_PRESETS,
    AggregatorBackend,
    CaptionerBackend,
    GeminiAggregator,
    GeminiCaptioner,
    LocalLLMAggregator,
    LocalVLMCaptioner,
    create_captioner,
)
from kuavi.types import KUAViConfig


class TestProtocolCompliance:
    def test_gemini_captioner_implements_captioner_backend(self):
        assert isinstance(GeminiCaptioner(), CaptionerBackend)

    def test_local_vlm_captioner_implements_captioner_backend(self):
        assert isinstance(LocalVLMCaptioner(), CaptionerBackend)

    def test_gemini_aggregator_implements_aggregator_backend(self):
        assert isinstance(GeminiAggregator(), AggregatorBackend)

    def test_local_llm_aggregator_implements_aggregator_backend(self):
        assert isinstance(LocalLLMAggregator(), AggregatorBackend)


class TestCaptionPresets:
    def test_all_presets_exist(self):
        assert set(CAPTION_PRESETS.keys()) == {
            "api",
            "local-full",
            "local-efficient",
            "local-minimal",
        }

    def test_presets_have_required_fields(self):
        required = {"description", "captioner", "aggregator", "vram"}
        for name, preset in CAPTION_PRESETS.items():
            assert required <= set(preset.keys()), f"Preset {name!r} missing keys"

    def test_api_preset_uses_gemini(self):
        p = CAPTION_PRESETS["api"]
        assert p["captioner"] == "gemini"
        assert p["aggregator"] == "gemini"

    def test_local_full_preset_uses_local(self):
        p = CAPTION_PRESETS["local-full"]
        assert p["captioner"] == "local"
        assert p["aggregator"] == "local"
        assert "frame_model" in p
        assert "segment_model" in p
        assert "aggregator_model" in p

    def test_local_minimal_has_no_aggregator(self):
        p = CAPTION_PRESETS["local-minimal"]
        assert p["aggregator"] is None


class TestCreateCaptionerFactory:
    def test_api_preset_returns_gemini_instances(self):
        captioner, aggregator = create_captioner("api")
        assert isinstance(captioner, GeminiCaptioner)
        assert isinstance(aggregator, GeminiAggregator)

    def test_local_full_returns_local_instances(self):
        captioner, aggregator = create_captioner("local-full")
        assert isinstance(captioner, LocalVLMCaptioner)
        assert isinstance(aggregator, LocalLLMAggregator)

    def test_local_efficient_returns_local_instances(self):
        captioner, aggregator = create_captioner("local-efficient")
        assert isinstance(captioner, LocalVLMCaptioner)
        assert isinstance(aggregator, LocalLLMAggregator)

    def test_local_minimal_returns_none_aggregator(self):
        captioner, aggregator = create_captioner("local-minimal")
        assert isinstance(captioner, LocalVLMCaptioner)
        assert aggregator is None

    def test_invalid_preset_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown caption preset"):
            create_captioner("nonexistent")

    def test_api_key_forwarded_to_gemini_captioner(self):
        captioner, aggregator = create_captioner("api", api_key="test-key")
        assert isinstance(captioner, GeminiCaptioner)
        assert captioner._api_key == "test-key"
        assert isinstance(aggregator, GeminiAggregator)
        assert aggregator._api_key == "test-key"

    def test_local_full_uses_preset_model_names(self):
        captioner, aggregator = create_captioner("local-full")
        assert isinstance(captioner, LocalVLMCaptioner)
        assert captioner.frame_model == "meta-llama/Llama-3.2-Vision-11B"
        assert captioner.segment_model == "facebook/Perception-LM-3B"
        assert isinstance(aggregator, LocalLLMAggregator)
        assert aggregator.model == "meta-llama/Llama-3.3-8B"

    def test_local_efficient_uses_plm_for_both(self):
        captioner, aggregator = create_captioner("local-efficient")
        assert isinstance(captioner, LocalVLMCaptioner)
        assert captioner.frame_model == "facebook/Perception-LM-3B"
        assert captioner.segment_model == "facebook/Perception-LM-3B"


class TestGeminiCaptionerDelegation:
    def test_caption_segment_delegates_to_make_gemini_caption_fn(self):
        mock_fn = MagicMock(return_value={"summary": {"brief": "test"}, "action": {}})
        with patch("kuavi.captioners.GeminiCaptioner._get_caption_fn", return_value=mock_fn):
            captioner = GeminiCaptioner(api_key="key")
            frames = [np.zeros((10, 10, 3), dtype=np.uint8)]
            result = captioner.caption_segment(frames)
            mock_fn.assert_called_once_with(frames)
            assert result == {"summary": {"brief": "test"}, "action": {}}

    def test_caption_frame_delegates_to_make_gemini_frame_caption_fn(self):
        mock_fn = MagicMock(return_value="A frame showing a cat.")
        with patch(
            "kuavi.captioners.GeminiCaptioner._get_frame_caption_fn", return_value=mock_fn
        ):
            captioner = GeminiCaptioner(api_key="key")
            frames = [np.zeros((10, 10, 3), dtype=np.uint8)]
            result = captioner.caption_frame(frames)
            mock_fn.assert_called_once_with(frames)
            assert result == "A frame showing a cat."

    def test_lazy_import_of_caption_fn(self):
        captioner = GeminiCaptioner(api_key="key")
        assert captioner._caption_fn is None

        mock_make = MagicMock(return_value=MagicMock())
        with patch("kuavi.captioners.make_gemini_caption_fn", mock_make, create=True):
            with patch("kuavi.captioning.make_gemini_caption_fn", mock_make):
                fn = captioner._get_caption_fn()
                assert fn is not None


class TestGeminiAggregatorDelegation:
    def test_refine_delegates_to_make_gemini_refine_fn(self):
        mock_fn = MagicMock(return_value="refined annotation")
        with patch("kuavi.captioners.GeminiAggregator._get_refine_fn", return_value=mock_fn):
            aggregator = GeminiAggregator(api_key="key")
            result = aggregator.refine("draft", "context", effort="high")
            mock_fn.assert_called_once_with("draft", "context", "high")
            assert result == "refined annotation"

    def test_lazy_import_of_refine_fn(self):
        aggregator = GeminiAggregator(api_key="key")
        assert aggregator._refine_fn is None


class TestLocalVLMCaptionerNotImplemented:
    def test_caption_frame_raises_not_implemented(self):
        captioner = LocalVLMCaptioner()
        with pytest.raises(NotImplementedError, match="Local VLM captioning requires"):
            captioner.caption_frame([np.zeros((10, 10, 3), dtype=np.uint8)])

    def test_caption_segment_raises_not_implemented(self):
        captioner = LocalVLMCaptioner()
        with pytest.raises(NotImplementedError, match="Local VLM segment captioning requires"):
            captioner.caption_segment([np.zeros((10, 10, 3), dtype=np.uint8)])

    def test_error_message_includes_model_name(self):
        captioner = LocalVLMCaptioner(frame_model="my-custom-model")
        with pytest.raises(NotImplementedError, match="my-custom-model"):
            captioner.caption_frame([])


class TestLocalLLMAggregatorNotImplemented:
    def test_refine_raises_not_implemented(self):
        aggregator = LocalLLMAggregator()
        with pytest.raises(NotImplementedError, match="Local LLM aggregation requires"):
            aggregator.refine("draft", "context")

    def test_error_message_includes_model_name(self):
        aggregator = LocalLLMAggregator(model="my-local-llm")
        with pytest.raises(NotImplementedError, match="my-local-llm"):
            aggregator.refine("draft", "context")


class TestKUAViConfigCaptionPreset:
    def test_default_is_none(self):
        cfg = KUAViConfig()
        assert cfg.caption_preset is None

    def test_can_set_api_preset(self):
        cfg = KUAViConfig(caption_preset="api")
        assert cfg.caption_preset == "api"

    def test_can_set_local_preset(self):
        cfg = KUAViConfig(caption_preset="local-minimal")
        assert cfg.caption_preset == "local-minimal"
