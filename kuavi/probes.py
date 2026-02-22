"""Attentive probe classification on frozen V-JEPA 2 features."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ProbeConfig:
    """Configuration for an attentive probe."""

    name: str  # e.g., "ssv2", "k400", "diving48"
    num_classes: int  # number of output classes
    embed_dim: int = 1024  # V-JEPA 2 embedding dimension
    num_heads: int = 16  # attention heads
    num_layers: int = 4  # cross-attention layers
    mlp_ratio: float = 4.0  # MLP expansion ratio
    class_names: list[str] = field(default_factory=list)  # optional human-readable class names
    weights_path: str | None = None  # path to pre-trained weights
    description: str = ""


# Pre-defined probe configurations for known benchmarks
PROBE_CONFIGS: dict[str, ProbeConfig] = {
    "ssv2": ProbeConfig(
        name="ssv2",
        num_classes=174,
        description="Something-Something v2: Fine-grained temporal reasoning (174 classes)",
    ),
    "k400": ProbeConfig(
        name="k400",
        num_classes=400,
        description="Kinetics-400: Human action recognition (400 classes)",
    ),
    "diving48": ProbeConfig(
        name="diving48",
        num_classes=48,
        description="Diving-48: Fine-grained diving classification (48 classes)",
    ),
    "jester": ProbeConfig(
        name="jester",
        num_classes=27,
        description="Jester: Hand gesture recognition (27 classes)",
    ),
    "coin": ProbeConfig(
        name="coin",
        num_classes=180,
        description="COIN: Comprehensive instructional video analysis (180 classes)",
    ),
    "imagenet": ProbeConfig(
        name="imagenet",
        num_classes=1000,
        description="ImageNet-1K: Image classification (1000 classes)",
    ),
}


class AttentiveProbe:
    """4-layer cross-attention probe on frozen V-JEPA 2 features.

    Architecture:
    - Learnable query tokens (CLS tokens) — one per class or a fixed set
    - 4 cross-attention layers: queries attend to V-JEPA 2 patch tokens
    - Each layer: LayerNorm → MultiheadAttention → LayerNorm → MLP
    - Final: mean-pool query tokens → linear classifier → logits

    The probe is lightweight (~10M parameters per task) and operates on
    frozen V-JEPA 2 features, requiring no backpropagation through the encoder.
    """

    def __init__(self, config: ProbeConfig):
        self.config = config
        self._model = None
        self._device = "cpu"

    def _build_model(self):
        """Build the PyTorch probe model."""
        import torch
        import torch.nn as nn

        class _AttentiveProbeModule(nn.Module):
            def __init__(self, embed_dim, num_heads, num_layers, mlp_ratio, num_classes):
                super().__init__()
                self.num_query_tokens = 8  # learnable query tokens
                self.query_tokens = nn.Parameter(
                    torch.randn(1, self.num_query_tokens, embed_dim) * 0.02
                )

                # Cross-attention layers
                self.layers = nn.ModuleList()
                for _ in range(num_layers):
                    self.layers.append(
                        nn.ModuleDict(
                            {
                                "norm_q": nn.LayerNorm(embed_dim),
                                "norm_kv": nn.LayerNorm(embed_dim),
                                "attn": nn.MultiheadAttention(
                                    embed_dim, num_heads, batch_first=True
                                ),
                                "norm_ffn": nn.LayerNorm(embed_dim),
                                "ffn": nn.Sequential(
                                    nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
                                    nn.GELU(),
                                    nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
                                ),
                            }
                        )
                    )

                self.norm_out = nn.LayerNorm(embed_dim)
                self.classifier = nn.Linear(embed_dim, num_classes)

            def forward(self, features: "torch.Tensor") -> "torch.Tensor":
                """
                Args:
                    features: (B, num_patches, D) frozen V-JEPA 2 patch tokens.
                Returns:
                    logits: (B, num_classes)
                """
                B = features.shape[0]
                queries = self.query_tokens.expand(B, -1, -1)

                for layer in self.layers:
                    # Cross-attention: queries attend to features
                    q = layer["norm_q"](queries)
                    kv = layer["norm_kv"](features)
                    attn_out, _ = layer["attn"](q, kv, kv)
                    queries = queries + attn_out

                    # FFN
                    queries = queries + layer["ffn"](layer["norm_ffn"](queries))

                # Pool and classify
                pooled = self.norm_out(queries).mean(dim=1)  # (B, D)
                return self.classifier(pooled)

        self._model = _AttentiveProbeModule(
            embed_dim=self.config.embed_dim,
            num_heads=self.config.num_heads,
            num_layers=self.config.num_layers,
            mlp_ratio=self.config.mlp_ratio,
            num_classes=self.config.num_classes,
        )
        return self._model

    def load_weights(self, path: str | Path | None = None) -> bool:
        """Load pre-trained weights.

        Args:
            path: Path to weights file (.pt/.pth). If None, uses config.weights_path.

        Returns:
            True if weights loaded successfully, False otherwise.
        """
        import torch

        if self._model is None:
            self._build_model()

        weights_path = path or self.config.weights_path
        if weights_path is None:
            logger.warning(
                "No weights path specified for probe '%s'. "
                "The probe will use random initialization. "
                "To use pre-trained weights, download from Meta's model releases "
                "and set weights_path in ProbeConfig or pass path to load_weights().",
                self.config.name,
            )
            return False

        weights_path = Path(weights_path)
        if not weights_path.exists():
            logger.warning(
                "Weights file not found at %s for probe '%s'. "
                "Please download pre-trained weights from Meta's V-JEPA 2 releases.",
                weights_path,
                self.config.name,
            )
            return False

        try:
            state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
            self._model.load_state_dict(state_dict)
            logger.info("Loaded probe weights for '%s' from %s", self.config.name, weights_path)
            return True
        except Exception as e:
            logger.warning("Failed to load probe weights for '%s': %s", self.config.name, e)
            return False

    def to(self, device: str) -> AttentiveProbe:
        """Move probe to device."""
        if self._model is None:
            self._build_model()
        self._device = device
        self._model = self._model.to(device)
        return self

    def classify(
        self,
        features: np.ndarray,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Classify video segment from V-JEPA 2 features.

        Args:
            features: (num_patches, D) or (B, num_patches, D) feature map.
            top_k: Number of top predictions to return.

        Returns:
            List of dicts with 'class_id', 'class_name' (if available),
            'confidence' (softmax probability).
        """
        import torch

        if self._model is None:
            self._build_model()
            self._model = self._model.to(self._device)

        self._model.eval()

        if features.ndim == 2:
            features = features[np.newaxis]  # add batch dim

        tensor = torch.from_numpy(features).float().to(self._device)

        with torch.no_grad():
            logits = self._model(tensor)  # (B, num_classes)
            probs = torch.softmax(logits, dim=-1)

        probs_np = probs.cpu().numpy()[0]  # first (and likely only) batch item
        top_indices = np.argsort(probs_np)[::-1][:top_k]

        results = []
        for idx in top_indices:
            entry: dict[str, Any] = {
                "class_id": int(idx),
                "confidence": round(float(probs_np[idx]), 4),
            }
            if self.config.class_names and idx < len(self.config.class_names):
                entry["class_name"] = self.config.class_names[idx]
            results.append(entry)

        return results


class ProbeRegistry:
    """Manages multiple attentive probes for different classification tasks."""

    def __init__(self):
        self._probes: dict[str, AttentiveProbe] = {}

    def register(self, probe: AttentiveProbe) -> None:
        """Register a probe by its config name."""
        self._probes[probe.config.name] = probe

    def get(self, name: str) -> AttentiveProbe | None:
        """Get a registered probe by name."""
        return self._probes.get(name)

    def list_probes(self) -> list[dict[str, Any]]:
        """List all registered probes with their configs."""
        return [
            {
                "name": p.config.name,
                "num_classes": p.config.num_classes,
                "description": p.config.description,
                "weights_loaded": p._model is not None,
            }
            for p in self._probes.values()
        ]

    def available_tasks(self) -> list[str]:
        """Return names of all registered probes."""
        return list(self._probes.keys())

    @classmethod
    def from_configs(
        cls,
        configs: dict[str, ProbeConfig] | None = None,
        device: str = "cpu",
    ) -> ProbeRegistry:
        """Create a registry from probe configs, optionally loading weights."""
        registry = cls()
        configs = configs or PROBE_CONFIGS
        for name, config in configs.items():
            probe = AttentiveProbe(config)
            if config.weights_path:
                probe._build_model()
                probe.load_weights()
                probe.to(device)
            registry.register(probe)
        return registry
