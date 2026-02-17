"""Tests for VideoIndexer and VideoIndex."""

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from rlm.video.video_indexer import VideoIndex, VideoIndexer, _cache_key


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


class TestVideoIndexSaveLoad:
    """Tests for VideoIndex serialization round-trip."""

    def test_save_load_round_trip(self, tmp_path):
        """Saving and loading should reproduce all serializable fields."""
        segments = [
            {"start_time": 0.0, "end_time": 5.0, "caption": "hello"},
            {"start_time": 5.0, "end_time": 10.0, "caption": "world"},
        ]
        embeddings = np.random.randn(2, 128).astype(np.float32)
        action_embeddings = np.random.randn(2, 128).astype(np.float32)
        transcript = [{"start_time": 0.0, "end_time": 3.0, "text": "hi there"}]
        scene_boundaries = [0.0, 5.0]

        idx = VideoIndex(
            segments=segments,
            embeddings=embeddings,
            action_embeddings=action_embeddings,
            transcript=transcript,
            scene_boundaries=scene_boundaries,
            embed_fn=lambda x: x,  # should NOT be saved
        )

        save_dir = tmp_path / "index_cache"
        idx.save(save_dir)

        loaded = VideoIndex.load(save_dir)

        assert loaded.segments == segments
        assert loaded.transcript == transcript
        assert loaded.scene_boundaries == scene_boundaries
        assert loaded.embed_fn is None  # callable not serialized
        np.testing.assert_array_almost_equal(loaded.embeddings, embeddings)
        np.testing.assert_array_almost_equal(loaded.action_embeddings, action_embeddings)

    def test_save_load_none_embeddings(self, tmp_path):
        """Round-trip with None embeddings should work."""
        idx = VideoIndex(
            segments=[{"start_time": 0.0, "end_time": 1.0, "caption": "x"}],
            embeddings=None,
            action_embeddings=None,
            transcript=[],
            scene_boundaries=[0.0],
        )
        save_dir = tmp_path / "no_emb"
        idx.save(save_dir)
        loaded = VideoIndex.load(save_dir)

        assert loaded.embeddings is None
        assert loaded.action_embeddings is None
        assert loaded.segments == idx.segments

    def test_save_creates_directory(self, tmp_path):
        """Save should create nested directories as needed."""
        idx = VideoIndex()
        nested = tmp_path / "a" / "b" / "c"
        idx.save(nested)
        assert (nested / "metadata.json").exists()
        assert (nested / "embeddings.npz").exists()


class TestCacheKey:
    """Tests for _cache_key helper."""

    def test_deterministic(self, tmp_path):
        video = tmp_path / "video.mp4"
        video.write_bytes(b"fake video content")
        k1 = _cache_key(str(video))
        k2 = _cache_key(str(video))
        assert k1 == k2

    def test_changes_with_content(self, tmp_path):
        video = tmp_path / "video.mp4"
        video.write_bytes(b"v1")
        k1 = _cache_key(str(video))
        video.write_bytes(b"v1 extra bytes")
        k2 = _cache_key(str(video))
        assert k1 != k2


class TestVideoIndexerCache:
    """Test VideoIndexer cache_dir integration."""

    def _make_loaded_video(self, video_path="/fake/video.mp4", num_frames=10, fps=2.0):
        frames = [
            np.full((48, 64, 3), i * 25, dtype=np.uint8) for i in range(num_frames)
        ]
        mock_video = MagicMock()
        mock_video.metadata.extraction_fps = fps
        mock_video.metadata.path = video_path
        mock_video.metadata.duration = num_frames / fps
        mock_video.frames = frames
        mock_video.segments = []
        return mock_video

    @patch("rlm.video.video_indexer.detect_scenes")
    def test_cache_dir_saves_and_loads(self, mock_detect_scenes, tmp_path):
        """First call saves to cache; second call loads from cache."""
        mock_detect_scenes.return_value = [(0.0, 2.5), (2.5, 4.5)]

        # Create a real video file so _cache_key can stat it
        video_file = tmp_path / "video.mp4"
        video_file.write_bytes(b"fake video data")

        cache_dir = tmp_path / "cache"
        indexer = VideoIndexer(cache_dir=str(cache_dir))
        loaded = self._make_loaded_video(video_path=str(video_file))

        fake_emb = np.random.randn(2, 4).astype(np.float32)

        with patch.object(indexer, "_ensure_model"):
            with patch.object(indexer, "_get_transcript", return_value=[]):
                with patch.object(
                    indexer, "_embed_captions", return_value=(fake_emb, None),
                ):
                    result1 = indexer.index_video(loaded, caption_fn=None)

        # Cache directory should now exist
        assert cache_dir.exists()
        assert mock_detect_scenes.call_count == 1

        # Second call — should load from cache, not recompute
        mock_detect_scenes.reset_mock()
        with patch.object(indexer, "_ensure_model"):
            result2 = indexer.index_video(loaded, caption_fn=None)

        mock_detect_scenes.assert_not_called()
        assert len(result2.segments) == len(result1.segments)
        if result1.embeddings is not None:
            np.testing.assert_array_almost_equal(result2.embeddings, result1.embeddings)

    def test_no_cache_dir_skips_caching(self):
        """When cache_dir is None, no caching occurs."""
        indexer = VideoIndexer(cache_dir=None)
        assert indexer._cache_dir is None


class TestDetectScenesHierarchical:
    """Tests for hierarchical multi-level scene detection."""

    def test_returns_multiple_levels(self):
        """detect_scenes_hierarchical returns one level per threshold."""
        from rlm.video.scene_detection import detect_scenes_hierarchical

        rng = np.random.default_rng(42)
        n = 30
        frames = [rng.integers(0, 255, (48, 64, 3), dtype=np.uint8) for _ in range(n)]
        timestamps = [float(i) for i in range(n)]

        def embed_fn(f):
            return rng.random((len(f), 16))

        result = detect_scenes_hierarchical(
            frames, timestamps, embed_fn,
            thresholds=(0.15, 0.30, 0.50),
            min_durations=(0.5, 2.0, 4.0),
        )

        assert "levels" in result
        assert len(result["levels"]) == 3

    def test_coarser_levels_have_fewer_scenes(self):
        """Higher thresholds should produce fewer or equal scenes."""
        from rlm.video.scene_detection import detect_scenes_hierarchical

        rng = np.random.default_rng(123)
        n = 50
        frames = [rng.integers(0, 255, (48, 64, 3), dtype=np.uint8) for _ in range(n)]
        timestamps = [float(i) for i in range(n)]

        def embed_fn(f):
            return rng.random((len(f), 16))

        result = detect_scenes_hierarchical(
            frames, timestamps, embed_fn,
            thresholds=(0.10, 0.30, 0.60),
            min_durations=(0.5, 2.0, 4.0),
        )

        levels = result["levels"]
        # Each subsequent level should have <= scenes than the previous
        for i in range(len(levels) - 1):
            assert len(levels[i]) >= len(levels[i + 1]), (
                f"Level {i} has {len(levels[i])} scenes but level {i+1} "
                f"has {len(levels[i+1])} — coarser should have fewer"
            )

    def test_empty_frames(self):
        """Empty input returns empty levels."""
        from rlm.video.scene_detection import detect_scenes_hierarchical

        result = detect_scenes_hierarchical([], [], lambda f: np.empty((0, 8)))
        assert all(lvl == [] for lvl in result["levels"])


class TestHierarchicalIndexingRoundTrip:
    """Test hierarchical indexing save/load round-trip."""

    def _make_loaded_video(self, num_frames=20, fps=2.0):
        frames = [
            np.full((48, 64, 3), i * 12, dtype=np.uint8) for i in range(num_frames)
        ]
        mock_video = MagicMock()
        mock_video.metadata.extraction_fps = fps
        mock_video.metadata.path = "/fake/video.mp4"
        mock_video.metadata.duration = num_frames / fps
        mock_video.frames = frames
        mock_video.segments = []
        return mock_video

    @patch("rlm.video.video_indexer.detect_scenes_hierarchical")
    def test_hierarchical_index_round_trip(self, mock_hier, tmp_path):
        """Hierarchical index should save and load correctly."""
        # Mock hierarchical detection to return 3 levels
        mock_hier.return_value = {
            "levels": [
                [(0.0, 2.5), (2.5, 5.0), (5.0, 7.5), (7.5, 9.5)],  # finest
                [(0.0, 5.0), (5.0, 9.5)],  # medium
                [(0.0, 9.5)],  # coarsest
            ]
        }

        indexer = VideoIndexer(hierarchical=True)
        loaded = self._make_loaded_video()

        caption_fn = MagicMock(side_effect=lambda _: "segment caption")
        fake_emb = np.random.randn(4, 8).astype(np.float32)
        fake_action_emb = np.random.randn(4, 8).astype(np.float32)

        with patch.object(indexer, "_ensure_model"):
            with patch.object(indexer, "_get_transcript", return_value=[]):
                with patch.object(
                    indexer, "_embed_captions",
                    return_value=(fake_emb, fake_action_emb),
                ):
                    result = indexer.index_video(loaded, caption_fn=caption_fn)

        # Primary segments from finest level
        assert len(result.segments) == 4
        # Hierarchy has 2 higher levels
        assert len(result.segment_hierarchy) == 2
        assert len(result.segment_hierarchy[0]) == 2  # medium
        assert len(result.segment_hierarchy[1]) == 1  # coarsest

        # Save and reload
        save_dir = tmp_path / "hier_index"
        result.save(save_dir)
        loaded_idx = VideoIndex.load(save_dir)

        assert loaded_idx.segment_hierarchy == result.segment_hierarchy
        assert len(loaded_idx.hierarchy_embeddings) == len(result.hierarchy_embeddings)
        for orig, reloaded in zip(result.hierarchy_embeddings, loaded_idx.hierarchy_embeddings):
            if orig is not None:
                np.testing.assert_array_almost_equal(reloaded, orig)

    def test_non_hierarchical_backward_compat(self):
        """Default (non-hierarchical) indexer should not populate hierarchy fields."""
        indexer = VideoIndexer(hierarchical=False)
        assert indexer._hierarchical is False

    def test_videoindex_default_hierarchy_fields(self):
        """VideoIndex default construction has empty hierarchy fields."""
        idx = VideoIndex()
        assert idx.segment_hierarchy == []
        assert idx.hierarchy_embeddings == []
