"""Tests for VideoIndexer and VideoIndex."""

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from rlm.video.video_indexer import VideoIndex, VideoIndexer


class TestVideoIndex:
    """Tests for the VideoIndex dataclass."""

    def test_default_construction(self):
        idx = VideoIndex()
        assert idx.segments == []
        assert idx.embeddings is None
        assert idx.action_embeddings is None
        assert idx.transcript == []
        assert idx.scene_boundaries == []
        assert idx.embed_fn is None

    def test_construction_with_values(self):
        emb = np.array([[1, 2], [3, 4]])
        segments = [{"start_time": 0, "end_time": 5, "caption": "hello"}]
        transcript = [{"start_time": 0, "end_time": 1, "text": "hi"}]

        idx = VideoIndex(
            segments=segments,
            embeddings=emb,
            transcript=transcript,
            scene_boundaries=[0.0, 5.0],
            embed_fn=lambda x: x,
        )

        assert idx.segments == segments
        assert np.array_equal(idx.embeddings, emb)
        assert idx.transcript == transcript
        assert idx.scene_boundaries == [0.0, 5.0]
        assert idx.embed_fn is not None


class TestVideoIndexerInit:
    """Test that __init__ stores config without loading models."""

    def test_default_init(self):
        indexer = VideoIndexer()
        assert indexer._embedding_model_name == "google/siglip2-base-patch16-256"
        assert indexer._device == "auto"
        assert indexer._model is None

    def test_custom_init(self):
        indexer = VideoIndexer(embedding_model="test-model", device="cpu")
        assert indexer._embedding_model_name == "test-model"
        assert indexer._device == "cpu"
        assert indexer._model is None


class TestVideoIndexerIndexVideo:
    """Test index_video with mocked dependencies."""

    def _make_loaded_video(self, num_frames=10, fps=2.0):
        """Create a mock LoadedVideo with synthetic frames."""
        frames = [
            np.full((48, 64, 3), i * 25, dtype=np.uint8) for i in range(num_frames)
        ]
        mock_video = MagicMock()
        mock_video.metadata.extraction_fps = fps
        mock_video.metadata.path = "/fake/video.mp4"
        mock_video.frames = frames
        mock_video.segments = []
        return mock_video

    @patch("rlm.video.video_indexer.detect_scenes")
    def test_index_video_no_caption_no_whisper(self, mock_detect_scenes):
        """Index a video without captions or ASR — scene detection still runs."""
        mock_detect_scenes.return_value = [(0.0, 2.5), (2.5, 4.5)]

        indexer = VideoIndexer()
        loaded = self._make_loaded_video(num_frames=10, fps=2.0)

        with patch.object(indexer, "_ensure_model"):
            with patch.object(indexer, "_get_transcript", return_value=[]):
                with patch.object(
                    indexer, "_embed_captions", return_value=(None, None),
                ):
                    result = indexer.index_video(loaded, caption_fn=None)

        assert isinstance(result, VideoIndex)
        assert len(result.segments) == 2
        assert result.segments[0]["start_time"] == 0.0
        assert result.segments[1]["start_time"] == 2.5
        assert result.embeddings is None  # no captions → no embeddings
        assert result.action_embeddings is None
        assert result.embed_fn is not None
        mock_detect_scenes.assert_called_once()

    @patch("rlm.video.video_indexer.detect_scenes")
    def test_index_video_with_caption(self, mock_detect_scenes):
        """When a caption_fn is provided, segments get captions and embeddings."""
        mock_detect_scenes.return_value = [(0.0, 2.5), (2.5, 4.5)]

        indexer = VideoIndexer()
        loaded = self._make_loaded_video(num_frames=10, fps=2.0)

        caption_fn = MagicMock(side_effect=["a cat sitting", "a dog running"])
        fake_embeddings = np.array([[1.0, 0.0], [0.0, 1.0]])
        fake_action_embeddings = np.array([[0.5, 0.5], [0.3, 0.7]])

        with patch.object(indexer, "_ensure_model"):
            with patch.object(
                indexer, "_embed_captions",
                return_value=(fake_embeddings, fake_action_embeddings),
            ):
                with patch.object(indexer, "_get_transcript", return_value=[]):
                    result = indexer.index_video(loaded, caption_fn=caption_fn)

        assert result.segments[0]["caption"] == "a cat sitting"
        assert result.segments[1]["caption"] == "a dog running"
        assert np.array_equal(result.embeddings, fake_embeddings)
        assert np.array_equal(result.action_embeddings, fake_action_embeddings)
        assert caption_fn.call_count == 2

    @patch("rlm.video.video_indexer.detect_scenes")
    def test_index_video_uses_loaded_segments(self, mock_detect_scenes):
        """If LoadedVideo has pre-existing segments, those are used."""
        mock_detect_scenes.return_value = [(0.0, 4.5)]

        indexer = VideoIndexer()
        loaded = self._make_loaded_video(num_frames=10, fps=2.0)

        seg1 = MagicMock()
        seg1.start_time = 0.0
        seg1.end_time = 2.0
        seg1.frames = [np.zeros((48, 64, 3), dtype=np.uint8)]

        seg2 = MagicMock()
        seg2.start_time = 2.0
        seg2.end_time = 4.0
        seg2.frames = [np.zeros((48, 64, 3), dtype=np.uint8)]

        loaded.segments = [seg1, seg2]

        with patch.object(indexer, "_ensure_model"):
            with patch.object(indexer, "_get_transcript", return_value=[]):
                with patch.object(
                    indexer, "_embed_captions", return_value=(None, None),
                ):
                    result = indexer.index_video(loaded, caption_fn=None)

        assert len(result.segments) == 2
        assert result.segments[0]["start_time"] == 0.0
        assert result.segments[1]["start_time"] == 2.0

    @patch("rlm.video.video_indexer.detect_scenes")
    def test_scene_boundaries_populated(self, mock_detect_scenes):
        mock_detect_scenes.return_value = [(0.0, 3.0), (3.0, 5.0)]

        indexer = VideoIndexer()
        loaded = self._make_loaded_video(num_frames=10, fps=2.0)

        with patch.object(indexer, "_ensure_model"):
            with patch.object(indexer, "_get_transcript", return_value=[]):
                with patch.object(
                    indexer, "_embed_captions", return_value=(None, None),
                ):
                    result = indexer.index_video(loaded, caption_fn=None)

        assert result.scene_boundaries == [0.0, 3.0]

    @patch("rlm.video.video_indexer.detect_scenes")
    def test_index_video_with_dict_caption(self, mock_detect_scenes):
        """caption_fn returning a dict populates annotation directly."""
        mock_detect_scenes.return_value = [(0.0, 4.5)]

        indexer = VideoIndexer()
        loaded = self._make_loaded_video(num_frames=10, fps=2.0)

        annotation = {
            "summary": {"brief": "cat sits", "detailed": "A cat sits on a mat."},
            "action": {"brief": "sitting", "detailed": "The cat is sitting still.", "actor": "cat"},
        }
        caption_fn = MagicMock(return_value=annotation)
        fake_embeddings = np.array([[1.0, 0.0]])
        fake_action_emb = np.array([[0.0, 1.0]])

        with patch.object(indexer, "_ensure_model"):
            with patch.object(
                indexer, "_embed_captions",
                return_value=(fake_embeddings, fake_action_emb),
            ):
                with patch.object(indexer, "_get_transcript", return_value=[]):
                    result = indexer.index_video(loaded, caption_fn=caption_fn)

        assert result.segments[0]["annotation"] == annotation
        assert result.segments[0]["caption"] == "cat sits"
        assert np.array_equal(result.action_embeddings, fake_action_emb)

    @patch("rlm.video.video_indexer.detect_scenes")
    def test_refine_fn_called_in_index_video(self, mock_detect_scenes):
        """When refine_fn is provided, it is called 3 rounds per segment."""
        mock_detect_scenes.return_value = [(0.0, 4.5)]

        indexer = VideoIndexer()
        loaded = self._make_loaded_video(num_frames=10, fps=2.0)

        caption_fn = MagicMock(return_value="hello")
        refine_fn = MagicMock(
            return_value='{"summary":{"brief":"refined","detailed":"refined"},'
            '"action":{"brief":"walk","detailed":"walking","actor":"person"}}',
        )

        with patch.object(indexer, "_ensure_model"):
            with patch.object(indexer, "_get_transcript", return_value=[]):
                with patch.object(
                    indexer, "_embed_captions", return_value=(None, None),
                ):
                    indexer.index_video(
                        loaded, caption_fn=caption_fn, refine_fn=refine_fn,
                    )

        # 3 rounds × 1 segment = 3 calls
        assert refine_fn.call_count == 3

    @patch("rlm.video.video_indexer.detect_scenes")
    def test_asr_context_injected_into_caption_fn(self, mock_detect_scenes):
        """When transcript overlaps a segment, ASR text is prepended to frames."""
        mock_detect_scenes.return_value = [(0.0, 4.5)]

        indexer = VideoIndexer()
        loaded = self._make_loaded_video(num_frames=10, fps=2.0)

        transcript = [
            {"start_time": 0.0, "end_time": 2.0, "text": "Hello world"},
        ]
        received_frames = []

        def capture_caption_fn(frames):
            received_frames.extend(frames)
            return "caption"

        with patch.object(indexer, "_ensure_model"):
            with patch.object(indexer, "_get_transcript", return_value=transcript):
                with patch.object(
                    indexer, "_embed_captions", return_value=(None, None),
                ):
                    indexer.index_video(loaded, caption_fn=capture_caption_fn)

        # First element should be the transcript context string
        assert any(isinstance(f, str) and "[transcript]" in f for f in received_frames)


class TestVideoIndexerTranscript:
    """Test transcript loading helpers."""

    def test_load_transcript_file(self, tmp_path):
        transcript_data = [
            {"start_time": 0.0, "end_time": 1.0, "text": "hello"},
            {"start_time": 1.0, "end_time": 2.0, "text": "world"},
        ]
        path = tmp_path / "transcript.json"
        path.write_text(json.dumps(transcript_data))

        result = VideoIndexer._load_transcript_file(str(path))
        assert len(result) == 2
        assert result[0]["text"] == "hello"

    def test_load_transcript_invalid_json(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("not json")

        result = VideoIndexer._load_transcript_file(str(path))
        assert result == []

    def test_load_transcript_non_list(self, tmp_path):
        path = tmp_path / "obj.json"
        path.write_text(json.dumps({"key": "value"}))

        result = VideoIndexer._load_transcript_file(str(path))
        assert result == []

    def test_transcript_for_range(self):
        transcript = [
            {"start_time": 0.0, "end_time": 2.0, "text": "hello"},
            {"start_time": 3.0, "end_time": 5.0, "text": "world"},
            {"start_time": 6.0, "end_time": 8.0, "text": "bye"},
        ]
        text = VideoIndexer._transcript_for_range(transcript, 1.0, 4.0)
        assert "hello" in text
        assert "world" in text
        assert "bye" not in text


class TestVideoIndexerEnsureModel:
    """Test lazy model loading."""

    def test_ensure_model_not_called_on_init(self):
        indexer = VideoIndexer()
        assert indexer._model is None

    @patch("transformers.AutoModel.from_pretrained")
    @patch("transformers.GemmaTokenizerFast.from_pretrained")
    @patch("transformers.SiglipImageProcessor.from_pretrained")
    def test_ensure_model_loads_once(self, mock_img_proc, mock_tokenizer, mock_model_cls):
        """_ensure_model should create the model lazily, only once."""
        mock_model = MagicMock()
        mock_model_cls.return_value.eval.return_value.to.return_value = mock_model

        indexer = VideoIndexer(embedding_model="test-model", device="cpu")
        indexer._ensure_model()
        indexer._ensure_model()  # second call should be a no-op

        assert indexer._model is not None
        # AutoModel.from_pretrained should only be called once
        mock_model_cls.assert_called_once_with("test-model")


class TestVideoIndexerStructuredAnnotations:
    """Tests for structured annotation support in captioning."""

    def _make_loaded_video(self, num_frames=10, fps=2.0):
        frames = [
            np.full((48, 64, 3), i * 25, dtype=np.uint8) for i in range(num_frames)
        ]
        mock_video = MagicMock()
        mock_video.metadata.extraction_fps = fps
        mock_video.metadata.path = "/fake/video.mp4"
        mock_video.frames = frames
        mock_video.segments = []
        return mock_video

    @patch("rlm.video.video_indexer.detect_scenes")
    def test_structured_caption_fn(self, mock_detect_scenes):
        """caption_fn returning a structured dict should populate annotation."""
        mock_detect_scenes.return_value = [(0.0, 4.5)]

        indexer = VideoIndexer()
        loaded = self._make_loaded_video()

        annotation = {
            "summary": {"brief": "A cat sitting", "detailed": "A cat sitting on a mat."},
            "action": {"brief": "sitting", "detailed": "The cat is sitting still.", "actor": "cat"},
        }
        caption_fn = MagicMock(return_value=annotation)

        with patch.object(indexer, "_ensure_model"):
            with patch.object(
                indexer, "_embed_captions", return_value=(np.array([[1.0]]), None),
            ):
                with patch.object(indexer, "_get_transcript", return_value=[]):
                    result = indexer.index_video(loaded, caption_fn=caption_fn)

        seg = result.segments[0]
        assert seg["annotation"] == annotation
        assert seg["caption"] == "A cat sitting"

    @patch("rlm.video.video_indexer.detect_scenes")
    def test_string_caption_fn_backward_compat(self, mock_detect_scenes):
        """caption_fn returning a plain string should be wrapped into annotation."""
        mock_detect_scenes.return_value = [(0.0, 4.5)]

        indexer = VideoIndexer()
        loaded = self._make_loaded_video()

        caption_fn = MagicMock(return_value="A dog running")

        with patch.object(indexer, "_ensure_model"):
            with patch.object(
                indexer, "_embed_captions", return_value=(np.array([[1.0]]), None),
            ):
                with patch.object(indexer, "_get_transcript", return_value=[]):
                    result = indexer.index_video(loaded, caption_fn=caption_fn)

        seg = result.segments[0]
        assert seg["caption"] == "A dog running"
        assert seg["annotation"]["summary"]["brief"] == "A dog running"
        assert seg["annotation"]["action"]["brief"] == ""


class TestVideoIndexerSelfRefine:
    """Tests for the Self-Refine annotation refinement."""

    def test_refine_annotations_updates_caption(self):
        indexer = VideoIndexer()
        segments = [
            {
                "start_time": 0.0,
                "end_time": 5.0,
                "caption": "old caption",
                "annotation": {
                    "summary": {"brief": "old caption", "detailed": "old detailed"},
                    "action": {"brief": "walking", "detailed": "walking slowly", "actor": "person"},
                },
            },
        ]
        transcript = [{"start_time": 0.0, "end_time": 5.0, "text": "hello world"}]

        refined = json.dumps({
            "summary": {"brief": "refined caption", "detailed": "refined detailed"},
            "action": {"brief": "running", "detailed": "running fast", "actor": "person"},
        })
        refine_fn = MagicMock(return_value=refined)

        indexer._refine_annotations(segments, transcript, refine_fn)

        # 3 rounds of refinement
        assert refine_fn.call_count == 3
        assert segments[0]["caption"] == "refined caption"
        assert segments[0]["annotation"]["action"]["brief"] == "running"

    def test_refine_annotations_none_fn_is_noop(self):
        indexer = VideoIndexer()
        segments = [{"start_time": 0.0, "end_time": 5.0, "caption": "original"}]
        indexer._refine_annotations(segments, [], None)
        assert segments[0]["caption"] == "original"

    def test_refine_annotations_invalid_json_keeps_existing(self):
        indexer = VideoIndexer()
        segments = [
            {
                "start_time": 0.0,
                "end_time": 5.0,
                "caption": "original",
                "annotation": {"summary": {"brief": "original"}},
            },
        ]
        refine_fn = MagicMock(return_value="not valid json {{{")
        indexer._refine_annotations(segments, [], refine_fn)
        assert segments[0]["caption"] == "original"


class TestVideoIndexerActionEmbeddings:
    """Test that action_embeddings field is properly handled."""

    def test_action_embeddings_default_none(self):
        idx = VideoIndex()
        assert idx.action_embeddings is None

    def test_action_embeddings_set(self):
        emb = np.array([[1, 2], [3, 4]])
        idx = VideoIndex(action_embeddings=emb)
        assert np.array_equal(idx.action_embeddings, emb)
