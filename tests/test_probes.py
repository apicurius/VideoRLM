"""Tests for WI-10: Attentive Probe Classification."""

from __future__ import annotations

import inspect
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fake_features(num_patches: int = 8, embed_dim: int = 4) -> np.ndarray:
    """Return a (num_patches, embed_dim) float32 array."""
    rng = np.random.default_rng(42)
    return rng.standard_normal((num_patches, embed_dim)).astype(np.float32)


def _make_video_index(with_feature_maps: bool = True, num_segments: int = 3):
    """Create a minimal VideoIndex for testing."""
    from kuavi.indexer import VideoIndex

    segments = [
        {"start_time": float(i), "end_time": float(i + 1), "caption": f"scene {i}"}
        for i in range(num_segments)
    ]
    feature_maps = None
    if with_feature_maps:
        rng = np.random.default_rng(0)
        feature_maps = rng.standard_normal((num_segments, 8, 4)).astype(np.float32)

    return VideoIndex(segments=segments, temporal_feature_maps=feature_maps)


# ---------------------------------------------------------------------------
# ProbeConfig
# ---------------------------------------------------------------------------


class TestProbeConfig:
    def test_creation_defaults(self):
        from kuavi.probes import ProbeConfig

        cfg = ProbeConfig(name="test", num_classes=10)
        assert cfg.name == "test"
        assert cfg.num_classes == 10
        assert cfg.embed_dim == 1024
        assert cfg.num_heads == 16
        assert cfg.num_layers == 4
        assert cfg.mlp_ratio == 4.0
        assert cfg.class_names == []
        assert cfg.weights_path is None
        assert cfg.description == ""

    def test_creation_custom(self):
        from kuavi.probes import ProbeConfig

        cfg = ProbeConfig(
            name="custom",
            num_classes=5,
            embed_dim=512,
            num_heads=8,
            num_layers=2,
            mlp_ratio=2.0,
            class_names=["a", "b", "c", "d", "e"],
            weights_path="/tmp/weights.pt",
            description="Custom probe",
        )
        assert cfg.embed_dim == 512
        assert cfg.num_heads == 8
        assert len(cfg.class_names) == 5
        assert cfg.weights_path == "/tmp/weights.pt"


class TestProbeConfigs:
    def test_all_expected_tasks_present(self):
        from kuavi.probes import PROBE_CONFIGS

        for task in ["ssv2", "k400", "diving48", "jester", "coin", "imagenet"]:
            assert task in PROBE_CONFIGS, f"Missing task: {task}"

    def test_num_classes(self):
        from kuavi.probes import PROBE_CONFIGS

        assert PROBE_CONFIGS["ssv2"].num_classes == 174
        assert PROBE_CONFIGS["k400"].num_classes == 400
        assert PROBE_CONFIGS["diving48"].num_classes == 48
        assert PROBE_CONFIGS["jester"].num_classes == 27
        assert PROBE_CONFIGS["coin"].num_classes == 180
        assert PROBE_CONFIGS["imagenet"].num_classes == 1000


# ---------------------------------------------------------------------------
# AttentiveProbe — model building
# ---------------------------------------------------------------------------


class TestAttentiveProbeModel:
    """Tests that mock torch to avoid loading real models."""

    def _make_probe(self, num_classes: int = 10, embed_dim: int = 4, num_heads: int = 2):
        from kuavi.probes import AttentiveProbe, ProbeConfig

        cfg = ProbeConfig(
            name="test",
            num_classes=num_classes,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=2,
            mlp_ratio=2.0,
        )
        return AttentiveProbe(cfg)

    def test_build_model_returns_module(self):
        import torch.nn as nn

        probe = self._make_probe()
        model = probe._build_model()
        assert isinstance(model, nn.Module)

    def test_build_model_correct_num_layers(self):
        probe = self._make_probe()
        model = probe._build_model()
        assert len(model.layers) == 2

    def test_build_model_correct_num_classes(self):
        probe = self._make_probe(num_classes=42)
        model = probe._build_model()
        assert model.classifier.out_features == 42

    def test_build_model_correct_embed_dim(self):
        probe = self._make_probe(embed_dim=4)
        model = probe._build_model()
        assert model.classifier.in_features == 4

    def test_model_query_tokens_shape(self):
        probe = self._make_probe(embed_dim=4)
        model = probe._build_model()
        # query_tokens: (1, num_query_tokens, embed_dim)
        assert model.query_tokens.shape == (1, model.num_query_tokens, 4)


# ---------------------------------------------------------------------------
# AttentiveProbe — classify
# ---------------------------------------------------------------------------


class TestAttentiveProbeClassify:
    def _make_probe(self, num_classes: int = 10):
        from kuavi.probes import AttentiveProbe, ProbeConfig

        cfg = ProbeConfig(
            name="test",
            num_classes=num_classes,
            embed_dim=4,
            num_heads=2,
            num_layers=2,
            mlp_ratio=2.0,
        )
        return AttentiveProbe(cfg)

    def test_classify_2d_input_returns_list(self):
        probe = self._make_probe(num_classes=10)
        features = _make_fake_features(num_patches=8, embed_dim=4)
        results = probe.classify(features, top_k=3)
        assert isinstance(results, list)
        assert len(results) == 3

    def test_classify_result_format(self):
        probe = self._make_probe(num_classes=10)
        features = _make_fake_features(num_patches=8, embed_dim=4)
        results = probe.classify(features, top_k=5)
        for r in results:
            assert "class_id" in r
            assert "confidence" in r
            assert isinstance(r["class_id"], int)
            assert 0.0 <= r["confidence"] <= 1.0

    def test_classify_confidences_sum_close_to_one(self):
        probe = self._make_probe(num_classes=10)
        features = _make_fake_features(num_patches=8, embed_dim=4)
        results = probe.classify(features, top_k=10)
        total = sum(r["confidence"] for r in results)
        assert abs(total - 1.0) < 0.02  # softmax sum ~1

    def test_classify_3d_input(self):
        probe = self._make_probe(num_classes=10)
        features = _make_fake_features(num_patches=8, embed_dim=4)[np.newaxis]  # (1, 8, 4)
        results = probe.classify(features, top_k=3)
        assert len(results) == 3

    def test_classify_with_class_names(self):
        from kuavi.probes import AttentiveProbe, ProbeConfig

        class_names = [f"class_{i}" for i in range(10)]
        cfg = ProbeConfig(
            name="named",
            num_classes=10,
            embed_dim=4,
            num_heads=2,
            num_layers=1,
            class_names=class_names,
        )
        probe = AttentiveProbe(cfg)
        features = _make_fake_features(num_patches=8, embed_dim=4)
        results = probe.classify(features, top_k=3)
        for r in results:
            assert "class_name" in r
            assert r["class_name"].startswith("class_")

    def test_classify_without_class_names_no_class_name_key(self):
        probe = self._make_probe(num_classes=10)
        features = _make_fake_features(num_patches=8, embed_dim=4)
        results = probe.classify(features, top_k=3)
        for r in results:
            assert "class_name" not in r


# ---------------------------------------------------------------------------
# AttentiveProbe — load_weights
# ---------------------------------------------------------------------------


class TestAttentiveProbeLoadWeights:
    def _make_probe(self):
        from kuavi.probes import AttentiveProbe, ProbeConfig

        cfg = ProbeConfig(name="test", num_classes=10, embed_dim=4, num_heads=2, num_layers=1)
        return AttentiveProbe(cfg)

    def test_load_weights_no_path_returns_false(self):
        probe = self._make_probe()
        result = probe.load_weights(path=None)
        assert result is False

    def test_load_weights_missing_file_returns_false(self, tmp_path):
        probe = self._make_probe()
        result = probe.load_weights(path=tmp_path / "nonexistent.pt")
        assert result is False

    def test_load_weights_valid_state_dict(self, tmp_path):
        import torch

        probe = self._make_probe()
        probe._build_model()
        # Save real state dict
        weights_path = tmp_path / "probe.pt"
        torch.save(probe._model.state_dict(), weights_path)

        # Load into a fresh probe
        probe2 = self._make_probe()
        result = probe2.load_weights(path=weights_path)
        assert result is True

    def test_load_weights_invalid_file_returns_false(self, tmp_path):
        probe = self._make_probe()
        bad_file = tmp_path / "bad.pt"
        bad_file.write_text("not a valid torch file")
        result = probe.load_weights(path=bad_file)
        assert result is False

    def test_load_weights_uses_config_weights_path(self, tmp_path):
        import torch

        from kuavi.probes import AttentiveProbe, ProbeConfig

        # Build probe, save weights, configure path
        cfg = ProbeConfig(name="test", num_classes=10, embed_dim=4, num_heads=2, num_layers=1)
        probe = AttentiveProbe(cfg)
        probe._build_model()
        weights_path = tmp_path / "probe.pt"
        torch.save(probe._model.state_dict(), weights_path)

        cfg2 = ProbeConfig(
            name="test",
            num_classes=10,
            embed_dim=4,
            num_heads=2,
            num_layers=1,
            weights_path=str(weights_path),
        )
        probe2 = AttentiveProbe(cfg2)
        result = probe2.load_weights()  # no path arg, uses config
        assert result is True


# ---------------------------------------------------------------------------
# AttentiveProbe — to()
# ---------------------------------------------------------------------------


class TestAttentiveProbeDevice:
    def test_to_cpu(self):
        from kuavi.probes import AttentiveProbe, ProbeConfig

        cfg = ProbeConfig(name="test", num_classes=5, embed_dim=4, num_heads=2, num_layers=1)
        probe = AttentiveProbe(cfg)
        probe._build_model()
        returned = probe.to("cpu")
        assert returned is probe
        assert probe._device == "cpu"
        assert probe._model is not None


# ---------------------------------------------------------------------------
# ProbeRegistry
# ---------------------------------------------------------------------------


class TestProbeRegistry:
    def _make_probe(self, name: str, num_classes: int = 5):
        from kuavi.probes import AttentiveProbe, ProbeConfig

        cfg = ProbeConfig(name=name, num_classes=num_classes, embed_dim=4, num_heads=2, num_layers=1)
        return AttentiveProbe(cfg)

    def test_register_and_get(self):
        from kuavi.probes import ProbeRegistry

        registry = ProbeRegistry()
        probe = self._make_probe("myprobe")
        registry.register(probe)
        retrieved = registry.get("myprobe")
        assert retrieved is probe

    def test_get_missing_returns_none(self):
        from kuavi.probes import ProbeRegistry

        registry = ProbeRegistry()
        assert registry.get("nonexistent") is None

    def test_list_probes_format(self):
        from kuavi.probes import ProbeRegistry

        registry = ProbeRegistry()
        registry.register(self._make_probe("a", num_classes=3))
        registry.register(self._make_probe("b", num_classes=7))
        listing = registry.list_probes()
        assert len(listing) == 2
        names = {p["name"] for p in listing}
        assert names == {"a", "b"}
        for p in listing:
            assert "num_classes" in p
            assert "description" in p
            assert "weights_loaded" in p

    def test_available_tasks(self):
        from kuavi.probes import ProbeRegistry

        registry = ProbeRegistry()
        registry.register(self._make_probe("x"))
        registry.register(self._make_probe("y"))
        tasks = registry.available_tasks()
        assert set(tasks) == {"x", "y"}

    def test_from_configs_creates_all_probes(self):
        from kuavi.probes import PROBE_CONFIGS, ProbeRegistry

        registry = ProbeRegistry.from_configs(configs=PROBE_CONFIGS)
        for name in PROBE_CONFIGS:
            assert registry.get(name) is not None

    def test_from_configs_no_weights_no_model_built(self):
        from kuavi.probes import PROBE_CONFIGS, ProbeRegistry

        registry = ProbeRegistry.from_configs(configs=PROBE_CONFIGS)
        # With no weights_path set, no model should be built eagerly
        for name in PROBE_CONFIGS:
            probe = registry.get(name)
            assert probe._model is None

    def test_from_configs_default(self):
        from kuavi.probes import ProbeRegistry

        registry = ProbeRegistry.from_configs()
        assert len(registry.available_tasks()) == 6


# ---------------------------------------------------------------------------
# make_classify_segment factory
# ---------------------------------------------------------------------------


class TestMakeClassifySegment:
    def test_returns_tool_dict(self):
        from kuavi.search import make_classify_segment

        index = _make_video_index()
        result = make_classify_segment(index)
        assert "tool" in result
        assert "description" in result
        assert callable(result["tool"])

    def test_error_when_no_feature_maps(self):
        from kuavi.search import make_classify_segment

        index = _make_video_index(with_feature_maps=False)
        tool = make_classify_segment(index)["tool"]
        result = tool(segment_index=0)
        assert "error" in result
        assert "store_feature_maps" in result["error"]

    def test_error_no_args(self):
        from kuavi.search import make_classify_segment

        index = _make_video_index()
        tool = make_classify_segment(index)["tool"]
        result = tool()
        assert "error" in result

    def test_error_invalid_segment_index(self):
        from kuavi.search import make_classify_segment

        index = _make_video_index(num_segments=3)
        tool = make_classify_segment(index)["tool"]
        result = tool(segment_index=99)
        assert "error" in result
        assert "99" in result["error"]

    def test_valid_segment_index_returns_predictions(self):
        from kuavi.search import make_classify_segment

        index = _make_video_index(num_segments=3)
        tool = make_classify_segment(index)["tool"]
        # Probe will use small embed_dim=4 (our fake feature maps)
        # The default registry uses embed_dim=1024, so we need a custom registry
        from kuavi.probes import AttentiveProbe, ProbeConfig, ProbeRegistry

        custom_cfg = ProbeConfig(name="k400", num_classes=5, embed_dim=4, num_heads=2, num_layers=1)
        registry = ProbeRegistry()
        registry.register(AttentiveProbe(custom_cfg))
        index._probe_registry = registry  # type: ignore[attr-defined]

        result = tool(segment_index=0, task="k400", top_k=3)
        assert "error" not in result
        assert "predictions" in result
        assert "segment" in result
        assert result["segment"]["index"] == 0
        assert result["task"] == "k400"
        assert len(result["predictions"]) == 3

    def test_time_range_resolves_segment(self):
        from kuavi.search import make_classify_segment

        index = _make_video_index(num_segments=3)
        # Segments: [0-1], [1-2], [2-3]
        from kuavi.probes import AttentiveProbe, ProbeConfig, ProbeRegistry

        custom_cfg = ProbeConfig(name="k400", num_classes=5, embed_dim=4, num_heads=2, num_layers=1)
        registry = ProbeRegistry()
        registry.register(AttentiveProbe(custom_cfg))
        index._probe_registry = registry  # type: ignore[attr-defined]

        tool = make_classify_segment(index)["tool"]
        result = tool(start_time=1.0, end_time=2.0, task="k400")
        assert "error" not in result
        assert result["segment"]["index"] == 1

    def test_unknown_task_returns_error_with_available(self):
        from kuavi.search import make_classify_segment

        index = _make_video_index()
        from kuavi.probes import AttentiveProbe, ProbeConfig, ProbeRegistry

        registry = ProbeRegistry()
        registry.register(AttentiveProbe(ProbeConfig(name="k400", num_classes=5, embed_dim=4, num_heads=2, num_layers=1)))
        index._probe_registry = registry  # type: ignore[attr-defined]

        tool = make_classify_segment(index)["tool"]
        result = tool(segment_index=0, task="unknown_task")
        assert "error" in result
        assert "unknown_task" in result["error"]
        assert "k400" in result["error"]

    def test_predictions_have_class_id_and_confidence(self):
        from kuavi.search import make_classify_segment

        index = _make_video_index(num_segments=2)
        from kuavi.probes import AttentiveProbe, ProbeConfig, ProbeRegistry

        custom_cfg = ProbeConfig(name="k400", num_classes=10, embed_dim=4, num_heads=2, num_layers=1)
        registry = ProbeRegistry()
        registry.register(AttentiveProbe(custom_cfg))
        index._probe_registry = registry  # type: ignore[attr-defined]

        tool = make_classify_segment(index)["tool"]
        result = tool(segment_index=0, task="k400", top_k=5)
        for pred in result["predictions"]:
            assert "class_id" in pred
            assert "confidence" in pred


# ---------------------------------------------------------------------------
# MCP tool signature
# ---------------------------------------------------------------------------


class TestMCPToolSignature:
    def test_kuavi_classify_segment_exists(self):
        import kuavi.mcp_server as srv

        assert hasattr(srv, "kuavi_classify_segment")
        assert callable(srv.kuavi_classify_segment)

    def test_kuavi_classify_segment_parameters(self):
        import kuavi.mcp_server as srv

        sig = inspect.signature(srv.kuavi_classify_segment)
        params = set(sig.parameters.keys())
        assert "task" in params
        assert "start_time" in params
        assert "end_time" in params
        assert "segment_index" in params
        assert "top_k" in params
        assert "video_id" in params

    def test_kuavi_classify_segment_defaults(self):
        import kuavi.mcp_server as srv

        sig = inspect.signature(srv.kuavi_classify_segment)
        assert sig.parameters["task"].default == "k400"
        assert sig.parameters["top_k"].default == 5
        assert sig.parameters["video_id"].default is None
