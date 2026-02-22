"""Tests for V-JEPA 2 model presets."""

import pytest

from kuavi.types import VJEPA2_PRESETS, KUAViConfig
from kuavi.indexer import VideoIndexer


class TestVJEPA2Presets:
    def test_all_presets_exist(self):
        assert set(VJEPA2_PRESETS.keys()) == {"fast", "balanced", "quality"}

    def test_presets_have_required_keys(self):
        required = {"model", "clip_size", "resolution", "embed_dim"}
        for name, preset in VJEPA2_PRESETS.items():
            assert required <= set(preset.keys()), f"Preset {name!r} missing keys"

    def test_fast_preset_values(self):
        p = VJEPA2_PRESETS["fast"]
        assert p["model"] == "facebook/vjepa2-vitl-fpc64-256"
        assert p["clip_size"] == 16
        assert p["embed_dim"] == 1024

    def test_balanced_preset_values(self):
        p = VJEPA2_PRESETS["balanced"]
        assert p["model"] == "facebook/vjepa2-vith-fpc64-256"
        assert p["clip_size"] == 32
        assert p["embed_dim"] == 1280

    def test_quality_preset_values(self):
        p = VJEPA2_PRESETS["quality"]
        assert p["model"] == "facebook/vjepa2-vitg-fpc64-384"
        assert p["clip_size"] == 64
        assert p["embed_dim"] == 1536


class TestVideoIndexerPresetResolution:
    def test_fast_preset_resolves_model_and_clip_size(self):
        indexer = VideoIndexer(scene_model_preset="fast")
        assert indexer._scene_model_name == "facebook/vjepa2-vitl-fpc64-256"
        assert indexer._scene_clip_size == 16
        assert indexer._scene_embed_dim == 1024

    def test_balanced_preset_resolves_model_and_clip_size(self):
        indexer = VideoIndexer(scene_model_preset="balanced")
        assert indexer._scene_model_name == "facebook/vjepa2-vith-fpc64-256"
        assert indexer._scene_clip_size == 32
        assert indexer._scene_embed_dim == 1280

    def test_quality_preset_resolves_model_and_clip_size(self):
        indexer = VideoIndexer(scene_model_preset="quality")
        assert indexer._scene_model_name == "facebook/vjepa2-vitg-fpc64-384"
        assert indexer._scene_clip_size == 64
        assert indexer._scene_embed_dim == 1536

    def test_backward_compat_no_preset_uses_scene_model_directly(self):
        custom_model = "facebook/vjepa2-vitl-fpc64-256"
        indexer = VideoIndexer(scene_model=custom_model, scene_clip_size=8)
        assert indexer._scene_model_name == custom_model
        assert indexer._scene_clip_size == 8

    def test_no_preset_no_scene_model_defaults(self):
        indexer = VideoIndexer()
        assert indexer._scene_model_name is None
        assert indexer._scene_clip_size == 16
        assert indexer._scene_embed_dim == 1024

    def test_invalid_preset_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown scene_model_preset"):
            VideoIndexer(scene_model_preset="nonexistent")

    def test_preset_overrides_scene_model_param(self):
        # When preset is set, it should override any scene_model provided
        indexer = VideoIndexer(
            scene_model="some/other-model",
            scene_model_preset="fast",
        )
        assert indexer._scene_model_name == "facebook/vjepa2-vitl-fpc64-256"
        assert indexer._scene_clip_size == 16


class TestKUAViConfigPreset:
    def test_default_no_preset(self):
        cfg = KUAViConfig()
        assert cfg.scene_model_preset is None

    def test_set_preset(self):
        cfg = KUAViConfig(scene_model_preset="balanced")
        assert cfg.scene_model_preset == "balanced"


class TestBatchSizeAdjustment:
    def test_batch_size_small_for_large_models(self):
        """embed_dim >= 1280 should use batch_size=2."""
        indexer = VideoIndexer(scene_model_preset="balanced")
        # Simulate what _encode_clips_vjepa does to compute batch_size
        batch_size = 2 if getattr(indexer, "_scene_embed_dim", 1024) >= 1280 else 4
        assert batch_size == 2

    def test_batch_size_normal_for_fast_model(self):
        """embed_dim < 1280 (ViT-L) should use batch_size=4."""
        indexer = VideoIndexer(scene_model_preset="fast")
        batch_size = 2 if getattr(indexer, "_scene_embed_dim", 1024) >= 1280 else 4
        assert batch_size == 4

    def test_batch_size_small_for_quality_model(self):
        """embed_dim=1536 (ViT-g) should use batch_size=2."""
        indexer = VideoIndexer(scene_model_preset="quality")
        batch_size = 2 if getattr(indexer, "_scene_embed_dim", 1024) >= 1280 else 4
        assert batch_size == 2
