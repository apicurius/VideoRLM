"""Tests for action-first (fast) indexing mode."""

from unittest.mock import MagicMock, patch

import numpy as np

from kuavi.indexer import VideoIndex, VideoIndexer


def _fake_encode(frames, **kw):
    """Return deterministic L2-normalized embeddings unique to each frame."""
    rows = []
    for f in frames:
        seed = int(np.mean(f)) + 1
        rows.append(np.random.default_rng(seed).standard_normal(4))
    embs = np.stack(rows).astype(np.float32)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    return embs / np.maximum(norms, 1e-10)


def _make_loaded_video(num_frames=6, fps=2.0):
    """Create a mock LoadedVideo with synthetic frames."""
    rng = np.random.RandomState(42)
    frames = [
        np.clip(
            rng.randint(50, 200, (32, 32, 3), dtype=np.uint8) + i * 10, 0, 255
        ).astype(np.uint8)
        for i in range(num_frames)
    ]
    mock_video = MagicMock()
    mock_video.metadata.extraction_fps = fps
    mock_video.metadata.path = "/fake/video.mp4"
    mock_video.frames = frames
    mock_video.segments = []
    return mock_video


def _patch_indexer(indexer, embed_captions_rv=None):
    if embed_captions_rv is None:
        embed_captions_rv = (np.eye(2, dtype=np.float32), None)
    """Context manager stack that mocks heavy dependencies on VideoIndexer."""
    from contextlib import ExitStack
    stack = ExitStack()
    stack.enter_context(patch.object(indexer, "_ensure_model"))
    stack.enter_context(patch.object(indexer, "_get_transcript", return_value=[]))
    stack.enter_context(patch.object(indexer, "_embed_captions", return_value=embed_captions_rv))
    stack.enter_context(patch.object(indexer, "_encode_frames", side_effect=_fake_encode))
    stack.enter_context(patch.object(indexer, "_pre_caption_dedup"))
    stack.enter_context(patch.object(indexer, "_selective_decode"))
    return stack


class TestFastMode:
    """Tests for index_video(mode='fast')."""

    @patch("kuavi.indexer.detect_scenes")
    def test_fast_mode_skips_caption_fn(self, mock_detect_scenes):
        """mode='fast' must not call caption_fn (segment-level captioning)."""
        mock_detect_scenes.return_value = [(0.0, 1.5), (1.5, 3.0)]

        indexer = VideoIndexer()
        loaded = _make_loaded_video()

        caption_fn = MagicMock(return_value="a scene description")
        frame_caption_fn = MagicMock(return_value="a frame caption")

        with _patch_indexer(indexer):
            indexer.index_video(
                loaded,
                caption_fn=caption_fn,
                frame_caption_fn=frame_caption_fn,
                mode="fast",
            )

        caption_fn.assert_not_called()

    @patch("kuavi.indexer.detect_scenes")
    def test_fast_mode_calls_frame_caption_fn(self, mock_detect_scenes):
        """mode='fast' must call frame_caption_fn for non-skipped segments."""
        mock_detect_scenes.return_value = [(0.0, 1.5), (1.5, 3.0)]

        indexer = VideoIndexer()
        loaded = _make_loaded_video()

        frame_caption_fn = MagicMock(return_value="a frame caption")

        with _patch_indexer(indexer):
            indexer.index_video(
                loaded,
                frame_caption_fn=frame_caption_fn,
                mode="fast",
            )

        assert frame_caption_fn.call_count >= 1

    @patch("kuavi.indexer.detect_scenes")
    def test_fast_mode_skips_refine_fn(self, mock_detect_scenes):
        """mode='fast' must not call refine_fn (Self-Refine is skipped)."""
        mock_detect_scenes.return_value = [(0.0, 1.5), (1.5, 3.0)]

        indexer = VideoIndexer()
        loaded = _make_loaded_video()

        refine_fn = MagicMock(return_value="{}")

        with _patch_indexer(indexer):
            indexer.index_video(
                loaded,
                refine_fn=refine_fn,
                mode="fast",
            )

        refine_fn.assert_not_called()

    @patch("kuavi.indexer.detect_scenes")
    def test_fast_mode_produces_video_index(self, mock_detect_scenes):
        """mode='fast' must return a VideoIndex with embeddings (searchable)."""
        mock_detect_scenes.return_value = [(0.0, 1.5), (1.5, 3.0)]

        indexer = VideoIndexer()
        loaded = _make_loaded_video()

        frame_caption_fn = MagicMock(return_value="walking down the street")
        fake_embeddings = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float32)

        with _patch_indexer(indexer, embed_captions_rv=(fake_embeddings, None)):
            result = indexer.index_video(
                loaded,
                frame_caption_fn=frame_caption_fn,
                mode="fast",
            )

        assert isinstance(result, VideoIndex)
        assert len(result.segments) == 2
        assert result.embeddings is not None
        assert result.embed_fn is not None

    @patch("kuavi.indexer.detect_scenes")
    def test_fast_mode_sets_caption_on_segments(self, mock_detect_scenes):
        """mode='fast' must set caption from frame_caption_fn on each segment."""
        mock_detect_scenes.return_value = [(0.0, 1.5), (1.5, 3.0)]

        indexer = VideoIndexer()
        loaded = _make_loaded_video()

        captions = ["first frame caption", "second frame caption"]
        frame_caption_fn = MagicMock(side_effect=lambda frames: captions.pop(0))

        with _patch_indexer(indexer):
            result = indexer.index_video(
                loaded,
                frame_caption_fn=frame_caption_fn,
                mode="fast",
            )

        # Each non-skipped segment should have a caption
        captioned = [s for s in result.segments if s.get("caption")]
        assert len(captioned) >= 1

    @patch("kuavi.indexer.detect_scenes")
    def test_fast_mode_no_frame_caption_fn(self, mock_detect_scenes):
        """mode='fast' without frame_caption_fn still produces a VideoIndex."""
        mock_detect_scenes.return_value = [(0.0, 1.5), (1.5, 3.0)]

        indexer = VideoIndexer()
        loaded = _make_loaded_video()

        with _patch_indexer(indexer, embed_captions_rv=(None, None)):
            result = indexer.index_video(loaded, mode="fast")

        assert isinstance(result, VideoIndex)
        assert len(result.segments) == 2


class TestFullModePreservation:
    """Tests that mode='full' preserves existing behavior."""

    @patch("kuavi.indexer.detect_scenes")
    def test_full_mode_calls_caption_fn(self, mock_detect_scenes):
        """mode='full' (default) must still call caption_fn."""
        mock_detect_scenes.return_value = [(0.0, 2.5), (2.5, 4.5)]

        indexer = VideoIndexer()
        loaded = _make_loaded_video(num_frames=10, fps=2.0)

        caption_fn = MagicMock(side_effect=["a cat sitting", "a dog running"])
        fake_embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

        with _patch_indexer(indexer, embed_captions_rv=(fake_embeddings, None)):
            result = indexer.index_video(loaded, caption_fn=caption_fn)

        assert caption_fn.call_count == 2
        assert result.segments[0]["caption"] == "a cat sitting"
        assert result.segments[1]["caption"] == "a dog running"

    @patch("kuavi.indexer.detect_scenes")
    def test_full_mode_is_default(self, mock_detect_scenes):
        """Calling index_video without mode= defaults to 'full' behavior."""
        mock_detect_scenes.return_value = [(0.0, 2.5), (2.5, 4.5)]

        indexer = VideoIndexer()
        loaded = _make_loaded_video(num_frames=10, fps=2.0)

        caption_fn = MagicMock(side_effect=["scene one", "scene two"])
        fake_embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

        with _patch_indexer(indexer, embed_captions_rv=(fake_embeddings, None)):
            result = indexer.index_video(loaded, caption_fn=caption_fn)

        # caption_fn should have been called — confirms full mode ran
        assert caption_fn.call_count == 2
        assert isinstance(result, VideoIndex)

    @patch("kuavi.indexer.detect_scenes")
    def test_full_mode_calls_refine_fn(self, mock_detect_scenes):
        """mode='full' must call refine_fn when provided."""
        # Segments must be >= 4s so the Self-Refine v2 minimum-duration skip doesn't filter them.
        mock_detect_scenes.return_value = [(0.0, 5.0), (5.0, 10.0)]

        indexer = VideoIndexer()
        loaded = _make_loaded_video(num_frames=10, fps=2.0)

        import json

        annotation = {
            "summary": {"brief": "test", "detailed": "test detailed"},
            "action": {"brief": "walk", "detailed": "", "actor": None},
        }
        caption_fn = MagicMock(return_value=annotation)
        refine_fn = MagicMock(return_value=json.dumps(annotation))
        fake_embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

        with _patch_indexer(indexer, embed_captions_rv=(fake_embeddings, None)):
            indexer.index_video(
                loaded,
                caption_fn=caption_fn,
                refine_fn=refine_fn,
                refine_rounds=1,
            )

        assert refine_fn.call_count > 0


class TestEnhanceIndex:
    """Tests for the enhance_index method."""

    def test_enhance_index_method_exists(self):
        """enhance_index must exist as a public method on VideoIndexer."""
        indexer = VideoIndexer()
        assert hasattr(indexer, "enhance_index")
        assert callable(indexer.enhance_index)

    def test_enhance_index_signature(self):
        """enhance_index must accept the documented keyword arguments."""
        import inspect

        sig = inspect.signature(VideoIndexer.enhance_index)
        params = set(sig.parameters.keys())
        assert "index" in params
        assert "loaded_video" in params
        assert "caption_fn" in params
        assert "frame_caption_fn" in params
        assert "refine_fn" in params
        assert "refine_rounds" in params

    @patch("kuavi.indexer.detect_scenes")
    def test_enhance_index_returns_video_index(self, mock_detect_scenes):
        """enhance_index must return a VideoIndex."""
        mock_detect_scenes.return_value = [(0.0, 1.5), (1.5, 3.0)]

        indexer = VideoIndexer()
        loaded = _make_loaded_video()

        # Build a fast-mode index first
        with _patch_indexer(indexer, embed_captions_rv=(None, None)):
            fast_index = indexer.index_video(loaded, mode="fast")

        # Now enhance it
        caption_fn = MagicMock(return_value="enhanced caption")
        fake_embeddings = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float32)

        with patch.object(indexer, "_ensure_model"):
            with patch.object(
                indexer, "_embed_captions", return_value=(fake_embeddings, None)
            ):
                with patch.object(indexer, "_encode_frames", side_effect=_fake_encode):
                    result = indexer.enhance_index(
                        fast_index,
                        loaded,
                        caption_fn=caption_fn,
                    )

        assert isinstance(result, VideoIndex)
        assert result.embed_fn is not None
