"""Tests for V-JEPA 2 scene detection integration."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from kuavi.indexer import VideoIndex, VideoIndexer


def _fake_encode(frames, *, dim=4, **kw):
    """Return L2-normalized embeddings unique to each frame's content."""
    rows = []
    for f in frames:
        seed = int(np.mean(f)) + 1
        rows.append(np.random.default_rng(seed).standard_normal(dim))
    embs = np.stack(rows).astype(np.float32)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    return embs / np.maximum(norms, 1e-10)


class TestGroupFramesIntoClips:
    """Test _group_frames_into_clips helper."""

    def test_even_split(self):
        indexer = VideoIndexer()
        frames = [np.zeros((48, 64, 3), dtype=np.uint8) for _ in range(16)]
        timestamps = [float(i) for i in range(16)]

        clips, clip_ts = indexer._group_frames_into_clips(frames, timestamps, clip_size=4)

        assert len(clips) == 4
        assert len(clip_ts) == 4
        # Each clip has 4 frames
        for c in clips:
            assert len(c) == 4
        # Midpoints: clip 0 → index 2, clip 1 → index 6, etc.
        assert clip_ts[0] == timestamps[2]
        assert clip_ts[1] == timestamps[6]
        assert clip_ts[2] == timestamps[10]
        assert clip_ts[3] == timestamps[14]

    def test_remainder_clip(self):
        indexer = VideoIndexer()
        frames = [np.zeros((48, 64, 3), dtype=np.uint8) for _ in range(10)]
        timestamps = [float(i) for i in range(10)]

        clips, clip_ts = indexer._group_frames_into_clips(frames, timestamps, clip_size=4)

        assert len(clips) == 3  # 4 + 4 + 2
        assert len(clips[0]) == 4
        assert len(clips[1]) == 4
        assert len(clips[2]) == 2
        # Last clip midpoint: index 8 + 1 = 9
        assert clip_ts[2] == timestamps[9]

    def test_single_frame(self):
        indexer = VideoIndexer()
        frames = [np.zeros((48, 64, 3), dtype=np.uint8)]
        timestamps = [0.0]

        clips, clip_ts = indexer._group_frames_into_clips(frames, timestamps, clip_size=16)

        assert len(clips) == 1
        assert len(clips[0]) == 1
        assert clip_ts[0] == 0.0

    def test_clip_size_larger_than_frames(self):
        indexer = VideoIndexer()
        frames = [np.zeros((48, 64, 3), dtype=np.uint8) for _ in range(5)]
        timestamps = [float(i) for i in range(5)]

        clips, clip_ts = indexer._group_frames_into_clips(frames, timestamps, clip_size=16)

        assert len(clips) == 1
        assert len(clips[0]) == 5
        assert clip_ts[0] == timestamps[2]  # midpoint of 5 frames


class TestEncodeClipsVjepa:
    """Test _encode_clips_vjepa with mocked V-JEPA 2 model."""

    def test_encode_clips_returns_correct_shape(self):
        indexer = VideoIndexer(scene_model="facebook/vjepa2-vitl-fpc64-256")

        # Mock the scene model and processor
        mock_processor = MagicMock()
        mock_model = MagicMock()

        import torch

        # Processor returns a dict-like object with .to() support
        mock_inputs = MagicMock()
        mock_inputs.to.return_value = mock_inputs
        mock_processor.return_value = mock_inputs

        # Model returns object with last_hidden_state: (batch, seq_len, 1024)
        mock_output = MagicMock()
        mock_output.last_hidden_state = torch.randn(2, 196, 1024)
        mock_model.return_value = mock_output

        indexer._scene_model = mock_model
        indexer._scene_processor = mock_processor
        indexer._scene_torch_device = "cpu"

        clips = [
            [np.zeros((48, 64, 3), dtype=np.uint8) for _ in range(4)],
            [np.zeros((48, 64, 3), dtype=np.uint8) for _ in range(4)],
        ]

        result = indexer._encode_clips_vjepa(clips)

        assert result.shape == (2, 1024)
        # Check L2-normalized
        norms = np.linalg.norm(result, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_encode_clips_batching(self):
        """Clips exceeding batch_size=4 should be processed in multiple batches."""
        indexer = VideoIndexer(scene_model="facebook/vjepa2-vitl-fpc64-256")

        import torch

        mock_processor = MagicMock()
        mock_model = MagicMock()

        mock_inputs = MagicMock()
        mock_inputs.to.return_value = mock_inputs
        mock_processor.return_value = mock_inputs

        # Return different batch sizes based on input
        def model_forward(**kwargs):
            # The processor is called per batch, return matching batch size
            out = MagicMock()
            out.last_hidden_state = torch.randn(4, 196, 1024)
            return out

        # For the last batch of 2
        call_count = [0]

        def model_side_effect(**kwargs):
            call_count[0] += 1
            out = MagicMock()
            batch = 4 if call_count[0] == 1 else 2
            out.last_hidden_state = torch.randn(batch, 196, 1024)
            return out

        mock_model.side_effect = model_side_effect

        indexer._scene_model = mock_model
        indexer._scene_processor = mock_processor
        indexer._scene_torch_device = "cpu"

        clips = [
            [np.zeros((48, 64, 3), dtype=np.uint8) for _ in range(4)]
            for _ in range(6)
        ]

        result = indexer._encode_clips_vjepa(clips)

        assert result.shape == (6, 1024)
        assert mock_model.call_count == 2  # 4 + 2 = 6 clips in 2 batches


class TestIndexVideoWithSceneModel:
    """Test that index_video uses V-JEPA 2 when scene_model is set."""

    def _make_loaded_video(self, num_frames=16, fps=2.0):
        frames = [np.full((48, 64, 3), i * 15, dtype=np.uint8) for i in range(num_frames)]
        mock_video = MagicMock()
        mock_video.metadata.extraction_fps = fps
        mock_video.metadata.path = "/fake/video.mp4"
        mock_video.metadata.duration = num_frames / fps
        mock_video.frames = frames
        mock_video.segments = []
        return mock_video

    @patch("kuavi.indexer.detect_scenes")
    def test_vjepa_path_called_when_scene_model_set(self, mock_detect_scenes):
        """When scene_model is set, V-JEPA 2 path is used for scene detection."""
        mock_detect_scenes.return_value = [(0.0, 4.0), (4.0, 7.5)]

        indexer = VideoIndexer(scene_model="facebook/vjepa2-vitl-fpc64-256")
        loaded = self._make_loaded_video()



        with patch.object(indexer, "_ensure_scene_model"):
            with patch.object(
                indexer,
                "_encode_clips_vjepa",
                return_value=np.random.randn(1, 1024).astype(np.float32),
            ):
                with patch.object(indexer, "_ensure_model"):
                    with patch.object(indexer, "_get_transcript", return_value=[]):
                        with patch.object(indexer, "_embed_captions", return_value=(None, None)):
                            with patch.object(indexer, "_encode_frames", side_effect=_fake_encode):
                                result = indexer.index_video(loaded, caption_fn=None)

        assert isinstance(result, VideoIndex)
        # detect_scenes was called (with clip representatives, not raw frames)
        mock_detect_scenes.assert_called_once()
        call_args = mock_detect_scenes.call_args
        # First arg should be clip representatives (1 clip for 16 frames / 16 clip_size)
        clip_reps = call_args[0][0]
        assert len(clip_reps) == 1  # 16 frames / 16 clip_size = 1 clip

    @patch("kuavi.indexer.detect_scenes")
    def test_siglip_path_when_no_scene_model(self, mock_detect_scenes):
        """When scene_model is None, the existing SigLIP2 path is used."""
        mock_detect_scenes.return_value = [(0.0, 4.0), (4.0, 7.5)]

        indexer = VideoIndexer()  # no scene_model
        loaded = self._make_loaded_video()



        with patch.object(indexer, "_ensure_model"):
            with patch.object(indexer, "_get_transcript", return_value=[]):
                with patch.object(indexer, "_embed_captions", return_value=(None, None)):
                    with patch.object(indexer, "_encode_frames", side_effect=_fake_encode):
                        result = indexer.index_video(loaded, caption_fn=None)

        assert isinstance(result, VideoIndex)
        mock_detect_scenes.assert_called_once()
        # With SigLIP2 path, all raw frames are passed
        call_args = mock_detect_scenes.call_args
        assert len(call_args[0][0]) == 16  # all frames

    @patch("kuavi.indexer.detect_scenes")
    def test_vjepa_clip_timestamps_are_midpoints(self, mock_detect_scenes):
        """V-JEPA 2 path should pass midpoint timestamps to detect_scenes."""
        mock_detect_scenes.return_value = [(1.0, 5.0)]

        indexer = VideoIndexer(
            scene_model="facebook/vjepa2-vitl-fpc64-256",
            scene_clip_size=4,
        )
        loaded = self._make_loaded_video(num_frames=8, fps=1.0)



        with patch.object(indexer, "_ensure_scene_model"):
            with patch.object(
                indexer,
                "_encode_clips_vjepa",
                return_value=np.random.randn(2, 1024).astype(np.float32),
            ):
                with patch.object(indexer, "_ensure_model"):
                    with patch.object(indexer, "_get_transcript", return_value=[]):
                        with patch.object(indexer, "_embed_captions", return_value=(None, None)):
                            with patch.object(indexer, "_encode_frames", side_effect=_fake_encode):
                                indexer.index_video(loaded, caption_fn=None)

        call_args = mock_detect_scenes.call_args
        timestamps_passed = call_args[0][1]
        # 8 frames, clip_size=4 → 2 clips
        # Clip 0: frames 0-3, midpoint = index 2, timestamp = 2.0
        # Clip 1: frames 4-7, midpoint = index 6, timestamp = 6.0
        assert len(timestamps_passed) == 2
        assert timestamps_passed[0] == 2.0
        assert timestamps_passed[1] == 6.0


class TestEnsureSceneModel:
    """Test lazy loading of V-JEPA 2 scene model."""

    def test_not_loaded_on_init(self):
        indexer = VideoIndexer(scene_model="facebook/vjepa2-vitl-fpc64-256")
        assert indexer._scene_model is None
        assert indexer._scene_processor is None

    @patch("transformers.AutoModel.from_pretrained")
    @patch("transformers.AutoVideoProcessor.from_pretrained")
    def test_loads_once(self, mock_proc_cls, mock_model_cls):
        """_ensure_scene_model should load model lazily, only once."""
        mock_model = MagicMock()
        mock_model_cls.return_value.eval.return_value.to.return_value = mock_model

        indexer = VideoIndexer(
            scene_model="facebook/vjepa2-vitl-fpc64-256", device="cpu"
        )
        indexer._ensure_scene_model()
        indexer._ensure_scene_model()  # second call is no-op

        assert indexer._scene_model is not None
        mock_model_cls.assert_called_once()
        mock_proc_cls.assert_called_once_with("facebook/vjepa2-vitl-fpc64-256")
