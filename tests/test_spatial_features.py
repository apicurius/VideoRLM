"""Tests for WI-7: spatial feature map preservation in V-JEPA 2 encoding."""

from __future__ import annotations

import tempfile
from unittest.mock import MagicMock, patch

import numpy as np

from kuavi.indexer import VideoIndex, VideoIndexer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_indexer(scene_model="facebook/vjepa2-vitl-fpc64-256") -> VideoIndexer:
    """Create a VideoIndexer with V-JEPA 2 scene model configured but not loaded."""
    indexer = VideoIndexer(scene_model=scene_model)
    return indexer


def _make_fake_clips(n_clips: int = 3, clip_size: int = 2) -> list[list[np.ndarray]]:
    """Create dummy BGR clips (H, W, C)."""
    return [
        [np.zeros((16, 16, 3), dtype=np.uint8) for _ in range(clip_size)]
        for _ in range(n_clips)
    ]


def _make_fake_scene_model(
    batch_size: int,
    num_patches: int = 8,
    embed_dim: int = 16,
):
    """Return a mock scene model that produces fake last_hidden_state."""
    import torch

    model = MagicMock()
    outputs = MagicMock()
    outputs.last_hidden_state = torch.randn(batch_size, num_patches, embed_dim)
    model.return_value = outputs
    return model


def _make_fake_processor(batch_size: int, num_patches: int = 8, embed_dim: int = 16):
    """Return a mock scene processor."""
    import torch

    processor = MagicMock()
    fake_inputs = {"pixel_values": torch.zeros(batch_size, 3, 8, 8)}
    result = MagicMock()
    result.to = MagicMock(return_value=fake_inputs)
    processor.return_value = result
    return processor


# ---------------------------------------------------------------------------
# VideoIndex dataclass field tests
# ---------------------------------------------------------------------------


class TestVideoIndexDataclass:
    def test_temporal_feature_maps_field_exists(self):
        """VideoIndex should have temporal_feature_maps field defaulting to None."""
        index = VideoIndex()
        assert hasattr(index, "temporal_feature_maps")
        assert index.temporal_feature_maps is None

    def test_temporal_feature_maps_can_be_set(self):
        """temporal_feature_maps can hold an ndarray."""
        maps = np.random.rand(5, 8, 16).astype(np.float32)
        index = VideoIndex(temporal_feature_maps=maps)
        assert index.temporal_feature_maps is not None
        assert index.temporal_feature_maps.shape == (5, 8, 16)

    def test_existing_fields_unchanged(self):
        """Adding the new field must not affect existing fields."""
        embs = np.ones((3, 16))
        index = VideoIndex(
            embeddings=embs,
            temporal_embeddings=np.ones((3, 16)),
        )
        assert index.embeddings is not None
        assert index.temporal_embeddings is not None
        assert index.temporal_feature_maps is None


# ---------------------------------------------------------------------------
# _encode_clips_vjepa backward-compatibility tests
# ---------------------------------------------------------------------------


class TestEncodeClipsVjepa:
    def _make_indexer_with_scene_model(
        self, n_clips: int, num_patches: int = 8, embed_dim: int = 16
    ) -> VideoIndexer:
        import torch

        indexer = _make_indexer()
        # Simulate a loaded scene model
        fake_lhs = torch.randn(n_clips, num_patches, embed_dim)
        fake_outputs = MagicMock()
        fake_outputs.last_hidden_state = fake_lhs

        fake_model = MagicMock(return_value=fake_outputs)
        fake_inputs = MagicMock()
        fake_inputs.to = MagicMock(return_value={"pixel_values": torch.zeros(n_clips, 3, 8, 8)})
        fake_processor = MagicMock(return_value=fake_inputs)

        indexer._scene_model = fake_model
        indexer._scene_processor = fake_processor
        indexer._scene_torch_device = "cpu"
        indexer._scene_embed_dim = embed_dim
        return indexer

    def test_return_full_false_returns_ndarray(self):
        """With return_full=False (default), returns a plain ndarray."""
        clips = _make_fake_clips(n_clips=2)
        indexer = self._make_indexer_with_scene_model(n_clips=2, num_patches=8, embed_dim=16)
        result = indexer._encode_clips_vjepa(clips, return_full=False)
        assert isinstance(result, np.ndarray)
        assert result.ndim == 2
        assert result.shape[0] == 2

    def test_default_returns_ndarray(self):
        """Default call (no return_full arg) should return ndarray — backward compat."""
        clips = _make_fake_clips(n_clips=2)
        indexer = self._make_indexer_with_scene_model(n_clips=2)
        result = indexer._encode_clips_vjepa(clips)
        assert isinstance(result, np.ndarray)

    def test_return_full_true_returns_tuple(self):
        """With return_full=True, returns (pooled_embs, feature_maps) tuple."""
        clips = _make_fake_clips(n_clips=3)
        indexer = self._make_indexer_with_scene_model(n_clips=3, num_patches=8, embed_dim=16)
        result = indexer._encode_clips_vjepa(clips, return_full=True)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_return_full_pooled_shape(self):
        """Pooled embeddings part must have shape (N_clips, D)."""
        n_clips, embed_dim = 4, 16
        clips = _make_fake_clips(n_clips=n_clips)
        indexer = self._make_indexer_with_scene_model(n_clips=n_clips, embed_dim=embed_dim)
        pooled, _ = indexer._encode_clips_vjepa(clips, return_full=True)
        assert pooled.shape == (n_clips, embed_dim)

    def test_return_full_feature_maps_shape(self):
        """Feature maps must be a list of arrays each shaped (num_patches, D)."""
        n_clips, num_patches, embed_dim = 3, 8, 16
        clips = _make_fake_clips(n_clips=n_clips)
        indexer = self._make_indexer_with_scene_model(
            n_clips=n_clips, num_patches=num_patches, embed_dim=embed_dim
        )
        _, feature_maps = indexer._encode_clips_vjepa(clips, return_full=True)
        assert isinstance(feature_maps, list)
        assert len(feature_maps) == n_clips
        for fm in feature_maps:
            assert isinstance(fm, np.ndarray)
            assert fm.shape == (num_patches, embed_dim)

    def test_return_full_pooled_is_normalized(self):
        """Pooled embeddings should be L2-normalized."""
        clips = _make_fake_clips(n_clips=2)
        indexer = self._make_indexer_with_scene_model(n_clips=2, embed_dim=16)
        pooled, _ = indexer._encode_clips_vjepa(clips, return_full=True)
        norms = np.linalg.norm(pooled, axis=-1)
        np.testing.assert_allclose(norms, np.ones(len(norms)), atol=1e-5)


# ---------------------------------------------------------------------------
# VideoIndex save/load round-trip with feature maps
# ---------------------------------------------------------------------------


class TestVideoIndexSaveLoad:
    def _make_index_with_feature_maps(self) -> VideoIndex:
        embs = np.random.rand(3, 16).astype(np.float32)
        temp_embs = np.random.rand(3, 16).astype(np.float32)
        feat_maps = np.random.rand(3, 8, 16).astype(np.float32)
        return VideoIndex(
            segments=[{"start_time": 0.0, "end_time": 1.0, "caption": "test"}] * 3,
            embeddings=embs,
            temporal_embeddings=temp_embs,
            temporal_feature_maps=feat_maps,
        )

    def test_save_load_roundtrip_with_feature_maps(self):
        """Feature maps survive a save/load round-trip."""
        index = self._make_index_with_feature_maps()
        with tempfile.TemporaryDirectory() as tmp_dir:
            index.save(tmp_dir)
            loaded = VideoIndex.load(tmp_dir)
        assert loaded.temporal_feature_maps is not None
        np.testing.assert_array_almost_equal(
            index.temporal_feature_maps, loaded.temporal_feature_maps
        )

    def test_save_load_without_feature_maps(self):
        """When no feature maps exist, load returns None for the field."""
        index = VideoIndex(
            segments=[{"start_time": 0.0, "end_time": 1.0, "caption": "test"}],
            embeddings=np.ones((1, 16), dtype=np.float32),
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            index.save(tmp_dir)
            loaded = VideoIndex.load(tmp_dir)
        assert loaded.temporal_feature_maps is None

    def test_save_load_feature_maps_shape_preserved(self):
        """Shape of feature maps is preserved exactly after round-trip."""
        feat_maps = np.random.rand(5, 12, 32).astype(np.float32)
        index = VideoIndex(
            segments=[{"start_time": float(i), "end_time": float(i + 1), "caption": "x"}
                      for i in range(5)],
            temporal_feature_maps=feat_maps,
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            index.save(tmp_dir)
            loaded = VideoIndex.load(tmp_dir)
        assert loaded.temporal_feature_maps.shape == (5, 12, 32)


# ---------------------------------------------------------------------------
# index_video store_feature_maps parameter tests
# ---------------------------------------------------------------------------


class TestIndexVideoStoreFeatureMaps:
    """Test that index_video correctly stores/omits feature maps."""

    def _make_loaded_video(self) -> MagicMock:
        """Create a minimal mock LoadedVideo."""
        from kuavi.loader import VideoMetadata

        lv = MagicMock()
        lv.frames = [np.zeros((16, 16, 3), dtype=np.uint8) for _ in range(4)]
        lv.timestamps = [0.0, 0.5, 1.0, 1.5]
        lv.segments = None
        meta = VideoMetadata(
            path="fake.mp4",
            total_frames=4,
            original_fps=2.0,
            duration=2.0,
            width=16,
            height=16,
            extraction_fps=2.0,
            extracted_frame_count=4,
        )
        lv.metadata = meta
        return lv

    def _make_patch_context(
        self,
        indexer: VideoIndexer,
        embed_dim: int = 16,
        encode_vjepa_return=None,
    ):
        """Return a list of patches needed for index_video to run without real models."""
        fake_embs = np.random.rand(1, embed_dim).astype(np.float32)
        return [
            patch.object(indexer, "_ensure_scene_model"),
            patch.object(indexer, "_ensure_model"),
            patch.object(
                indexer,
                "_encode_clips_vjepa",
                return_value=encode_vjepa_return if encode_vjepa_return is not None else fake_embs,
            ),
            patch.object(indexer, "_encode_frames", return_value=fake_embs),
            patch.object(indexer, "_embed_captions",
                         return_value=(fake_embs, fake_embs)),
            patch.object(indexer, "_smooth_embeddings", side_effect=lambda x, **kw: x),
            patch.object(indexer, "_check_embedding_quality"),
            patch.object(indexer, "_pre_caption_dedup"),
            patch.object(indexer, "_selective_decode"),
            patch.object(indexer, "_action_first_pass"),
            patch.object(indexer, "_build_coarse_level", return_value=([], None)),
            patch.object(indexer, "_get_transcript", return_value=[]),
            patch("kuavi.indexer.detect_scenes", return_value=[(0.0, 2.0)]),
        ]

    def test_store_feature_maps_false_default(self):
        """With store_feature_maps=False (default), feature maps NOT stored."""
        embed_dim = 16
        indexer = _make_indexer()
        lv = self._make_loaded_video()
        fake_embs = np.random.rand(1, embed_dim).astype(np.float32)

        patches = self._make_patch_context(indexer, embed_dim=embed_dim,
                                           encode_vjepa_return=fake_embs)
        # Grab the _encode_clips_vjepa patch to inspect calls
        vjepa_patch = patches[2]

        with (
            patches[0], patches[1], vjepa_patch as mock_encode,
            patches[3], patches[4], patches[5], patches[6],
            patches[7], patches[8], patches[9], patches[10],
            patches[11], patches[12],
        ):
            indexer._scene_model_name = "facebook/vjepa2-vitl-fpc64-256"
            indexer._scene_clip_size = 16
            indexer._encode_query = lambda t: np.ones(embed_dim)  # type: ignore[method-assign]
            indexer._encode_query_siglip = lambda t: np.ones(embed_dim)  # type: ignore[method-assign]
            index = indexer.index_video(lv, store_feature_maps=False, mode="fast")

        # return_full=True should NOT have been passed
        for call in mock_encode.call_args_list:
            assert call.kwargs.get("return_full", False) is False

        assert index.temporal_feature_maps is None

    def test_store_feature_maps_true_stores_maps(self):
        """With store_feature_maps=True, feature maps ARE stored on the index."""
        num_patches, embed_dim = 8, 16
        indexer = _make_indexer()
        lv = self._make_loaded_video()

        fake_pooled = np.random.rand(1, embed_dim).astype(np.float32)
        fake_maps = [np.random.rand(num_patches, embed_dim).astype(np.float32)]

        patches = self._make_patch_context(
            indexer, embed_dim=embed_dim,
            encode_vjepa_return=(fake_pooled, fake_maps),
        )

        with (
            patches[0], patches[1], patches[2],
            patches[3], patches[4], patches[5], patches[6],
            patches[7], patches[8], patches[9], patches[10],
            patches[11], patches[12],
        ):
            indexer._scene_model_name = "facebook/vjepa2-vitl-fpc64-256"
            indexer._scene_clip_size = 16
            indexer._encode_query = lambda t: np.ones(embed_dim)  # type: ignore[method-assign]
            indexer._encode_query_siglip = lambda t: np.ones(embed_dim)  # type: ignore[method-assign]
            index = indexer.index_video(lv, store_feature_maps=True, mode="fast")

        assert index.temporal_feature_maps is not None
        assert index.temporal_feature_maps.ndim == 3
        assert index.temporal_feature_maps.shape[1:] == (num_patches, embed_dim)
