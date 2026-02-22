"""Tests for VideoIndexer and VideoIndex."""

import json
from unittest.mock import MagicMock, patch

import numpy as np

from kuavi.indexer import VideoIndex, VideoIndexer, _cache_key


def _fake_encode(frames, *, dim=4, **kw):
    """Return L2-normalized embeddings unique to each frame's content.

    Uses mean pixel value as a seed so different frames produce different
    embedding directions.  This prevents ``_pre_caption_dedup`` from
    merging segments that should remain distinct, while keeping within-
    segment diversity high enough for meaningful ``_selective_decode`` variance.
    """
    rows = []
    for f in frames:
        seed = int(np.mean(f)) + 1  # +1 to avoid seed=0
        rows.append(np.random.default_rng(seed).standard_normal(dim))
    embs = np.stack(rows).astype(np.float32)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    return embs / np.maximum(norms, 1e-10)


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
        # Use random frames (not solid color) to avoid dead-frame detection
        rng = np.random.RandomState(42 + num_frames)
        frames = [
            np.clip(rng.randint(50, 200, (48, 64, 3), dtype=np.uint8) + i * 10, 0, 255).astype(
                np.uint8
            )
            for i in range(num_frames)
        ]
        mock_video = MagicMock()
        mock_video.metadata.extraction_fps = fps
        mock_video.metadata.path = "/fake/video.mp4"
        mock_video.frames = frames
        mock_video.segments = []
        return mock_video

    @patch("kuavi.indexer.detect_scenes")
    def test_index_video_no_caption_no_whisper(self, mock_detect_scenes):
        """Index a video without captions or ASR — scene detection still runs."""
        mock_detect_scenes.return_value = [(0.0, 2.5), (2.5, 4.5)]

        indexer = VideoIndexer()
        loaded = self._make_loaded_video(num_frames=10, fps=2.0)


        with patch.object(indexer, "_ensure_model"):
            with patch.object(indexer, "_get_transcript", return_value=[]):
                with patch.object(indexer, "_embed_captions", return_value=(None, None)):
                    with patch.object(indexer, "_encode_frames", side_effect=_fake_encode):
                        result = indexer.index_video(loaded, caption_fn=None)

        assert isinstance(result, VideoIndex)
        assert len(result.segments) == 2
        assert result.segments[0]["start_time"] == 0.0
        assert result.segments[1]["start_time"] == 2.5
        assert result.embeddings is None  # no captions → no embeddings
        assert result.action_embeddings is None
        assert result.embed_fn is not None
        mock_detect_scenes.assert_called_once()

    @patch("kuavi.indexer.detect_scenes")
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
                indexer,
                "_embed_captions",
                return_value=(fake_embeddings, fake_action_embeddings),
            ):
                with patch.object(indexer, "_get_transcript", return_value=[]):
                    with patch.object(indexer, "_encode_frames", side_effect=_fake_encode):
                        result = indexer.index_video(loaded, caption_fn=caption_fn)

        assert result.segments[0]["caption"] == "a cat sitting"
        assert result.segments[1]["caption"] == "a dog running"
        assert np.array_equal(result.embeddings, fake_embeddings)
        assert np.array_equal(result.action_embeddings, fake_action_embeddings)
        assert caption_fn.call_count == 2

    @patch("kuavi.indexer.detect_scenes")
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
                with patch.object(indexer, "_embed_captions", return_value=(None, None)):
                    with patch.object(indexer, "_encode_frames", side_effect=_fake_encode):
                        result = indexer.index_video(loaded, caption_fn=None)

        assert len(result.segments) == 2
        assert result.segments[0]["start_time"] == 0.0
        assert result.segments[1]["start_time"] == 2.0

    @patch("kuavi.indexer.detect_scenes")
    def test_scene_boundaries_populated(self, mock_detect_scenes):
        mock_detect_scenes.return_value = [(0.0, 3.0), (3.0, 5.0)]

        indexer = VideoIndexer()
        loaded = self._make_loaded_video(num_frames=10, fps=2.0)


        with patch.object(indexer, "_ensure_model"):
            with patch.object(indexer, "_get_transcript", return_value=[]):
                with patch.object(indexer, "_embed_captions", return_value=(None, None)):
                    with patch.object(indexer, "_encode_frames", side_effect=_fake_encode):
                        result = indexer.index_video(loaded, caption_fn=None)

        assert result.scene_boundaries == [0.0, 3.0]

    @patch("kuavi.indexer.detect_scenes")
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
                indexer,
                "_embed_captions",
                return_value=(fake_embeddings, fake_action_emb),
            ):
                with patch.object(indexer, "_get_transcript", return_value=[]):
                    with patch.object(indexer, "_encode_frames", side_effect=_fake_encode):
                        result = indexer.index_video(loaded, caption_fn=caption_fn)

        assert result.segments[0]["annotation"] == annotation
        assert result.segments[0]["caption"] == "cat sits"
        assert np.array_equal(result.action_embeddings, fake_action_emb)

    @patch("kuavi.indexer.detect_scenes")
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
                with patch.object(indexer, "_embed_captions", return_value=(None, None)):
                    with patch.object(indexer, "_encode_frames", side_effect=_fake_encode):
                        indexer.index_video(
                            loaded,
                            caption_fn=caption_fn,
                            refine_fn=refine_fn,
                        )

        # 3 rounds × 1 segment = 3 calls
        assert refine_fn.call_count == 3

    @patch("kuavi.indexer.detect_scenes")
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
                with patch.object(indexer, "_embed_captions", return_value=(None, None)):
                    with patch.object(indexer, "_encode_frames", side_effect=_fake_encode):
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
        # Use random frames (not solid color) to avoid dead-frame detection
        rng = np.random.RandomState(42 + num_frames)
        frames = [
            np.clip(rng.randint(50, 200, (48, 64, 3), dtype=np.uint8) + i * 10, 0, 255).astype(
                np.uint8
            )
            for i in range(num_frames)
        ]
        mock_video = MagicMock()
        mock_video.metadata.extraction_fps = fps
        mock_video.metadata.path = "/fake/video.mp4"
        mock_video.frames = frames
        mock_video.segments = []
        return mock_video

    @patch("kuavi.indexer.detect_scenes")
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
                indexer,
                "_embed_captions",
                return_value=(np.array([[1.0]]), None),
            ):
                with patch.object(indexer, "_get_transcript", return_value=[]):
                    with patch.object(indexer, "_encode_frames", side_effect=_fake_encode):
                        result = indexer.index_video(loaded, caption_fn=caption_fn)

        seg = result.segments[0]
        assert seg["annotation"] == annotation
        assert seg["caption"] == "A cat sitting"

    @patch("kuavi.indexer.detect_scenes")
    def test_string_caption_fn_backward_compat(self, mock_detect_scenes):
        """caption_fn returning a plain string should be wrapped into annotation."""
        mock_detect_scenes.return_value = [(0.0, 4.5)]

        indexer = VideoIndexer()
        loaded = self._make_loaded_video()

        caption_fn = MagicMock(return_value="A dog running")


        with patch.object(indexer, "_ensure_model"):
            with patch.object(
                indexer,
                "_embed_captions",
                return_value=(np.array([[1.0]]), None),
            ):
                with patch.object(indexer, "_get_transcript", return_value=[]):
                    with patch.object(indexer, "_encode_frames", side_effect=_fake_encode):
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

        refined = json.dumps(
            {
                "summary": {"brief": "refined caption", "detailed": "refined detailed"},
                "action": {"brief": "running", "detailed": "running fast", "actor": "person"},
            }
        )
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
        # Use random frames (not solid color) to avoid dead-frame detection
        rng = np.random.RandomState(42 + num_frames)
        frames = [
            np.clip(rng.randint(50, 200, (48, 64, 3), dtype=np.uint8) + i * 10, 0, 255).astype(
                np.uint8
            )
            for i in range(num_frames)
        ]
        mock_video = MagicMock()
        mock_video.metadata.extraction_fps = fps
        mock_video.metadata.path = video_path
        mock_video.metadata.duration = num_frames / fps
        mock_video.frames = frames
        mock_video.segments = []
        return mock_video

    @patch("kuavi.indexer.detect_scenes")
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
                with patch.object(indexer, "_embed_captions", return_value=(fake_emb, None)):
                    with patch.object(indexer, "_encode_frames", side_effect=_fake_encode):
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
        from kuavi.scene_detection import detect_scenes_hierarchical

        rng = np.random.default_rng(42)
        n = 30
        frames = [rng.integers(0, 255, (48, 64, 3), dtype=np.uint8) for _ in range(n)]
        timestamps = [float(i) for i in range(n)]

        def embed_fn(f):
            return rng.random((len(f), 16))

        result = detect_scenes_hierarchical(
            frames,
            timestamps,
            embed_fn,
            thresholds=(0.15, 0.30, 0.50),
            min_durations=(0.5, 2.0, 4.0),
        )

        assert "levels" in result
        assert len(result["levels"]) == 3

    def test_coarser_levels_have_fewer_scenes(self):
        """Higher thresholds should produce fewer or equal scenes."""
        from kuavi.scene_detection import detect_scenes_hierarchical

        rng = np.random.default_rng(123)
        n = 50
        frames = [rng.integers(0, 255, (48, 64, 3), dtype=np.uint8) for _ in range(n)]
        timestamps = [float(i) for i in range(n)]

        def embed_fn(f):
            return rng.random((len(f), 16))

        result = detect_scenes_hierarchical(
            frames,
            timestamps,
            embed_fn,
            thresholds=(0.10, 0.30, 0.60),
            min_durations=(0.5, 2.0, 4.0),
        )

        levels = result["levels"]
        # Each subsequent level should have <= scenes than the previous
        for i in range(len(levels) - 1):
            assert len(levels[i]) >= len(levels[i + 1]), (
                f"Level {i} has {len(levels[i])} scenes but level {i + 1} "
                f"has {len(levels[i + 1])} — coarser should have fewer"
            )

    def test_empty_frames(self):
        """Empty input returns empty levels."""
        from kuavi.scene_detection import detect_scenes_hierarchical

        result = detect_scenes_hierarchical([], [], lambda f: np.empty((0, 8)))
        assert all(lvl == [] for lvl in result["levels"])


class TestHierarchicalIndexingRoundTrip:
    """Test hierarchical indexing save/load round-trip."""

    def _make_loaded_video(self, num_frames=20, fps=2.0):
        frames = [np.full((48, 64, 3), i * 12, dtype=np.uint8) for i in range(num_frames)]
        mock_video = MagicMock()
        mock_video.metadata.extraction_fps = fps
        mock_video.metadata.path = "/fake/video.mp4"
        mock_video.metadata.duration = num_frames / fps
        mock_video.frames = frames
        mock_video.segments = []
        return mock_video

    @patch("kuavi.indexer.detect_scenes_hierarchical")
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
                    indexer,
                    "_embed_captions",
                    return_value=(fake_emb, fake_action_emb),
                ):
                    with patch.object(indexer, "_encode_frames", side_effect=_fake_encode):
                        result = indexer.index_video(loaded, caption_fn=caption_fn)

        # Primary segments from finest level
        assert len(result.segments) == 4
        # Hierarchy has 2 higher levels + 1 coarse level (always added)
        assert len(result.segment_hierarchy) == 3
        assert len(result.segment_hierarchy[0]) == 2  # medium
        assert len(result.segment_hierarchy[1]) == 1  # coarsest
        # Third level is the fixed-duration coarse level

        # Save and reload
        save_dir = tmp_path / "hier_index"
        result.save(save_dir)
        loaded_idx = VideoIndex.load(save_dir)

        assert loaded_idx.segment_hierarchy == result.segment_hierarchy
        assert len(loaded_idx.hierarchy_embeddings) == len(result.hierarchy_embeddings)
        for orig, reloaded in zip(result.hierarchy_embeddings, loaded_idx.hierarchy_embeddings, strict=False):
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


class TestVideoIndexerSceneModel:
    """Test scene_model (V-JEPA 2) parameter handling."""

    def test_scene_model_stored(self):
        indexer = VideoIndexer(scene_model="facebook/vjepa2-vitl-fpc64-256")
        assert indexer._scene_model_name == "facebook/vjepa2-vitl-fpc64-256"
        assert indexer._scene_model is None
        assert indexer._scene_processor is None

    def test_scene_model_default_none(self):
        indexer = VideoIndexer()
        assert indexer._scene_model_name is None

    def test_scene_clip_size_default(self):
        indexer = VideoIndexer(scene_model="facebook/vjepa2-vitl-fpc64-256")
        assert indexer._scene_clip_size == 16

    def test_scene_clip_size_custom(self):
        indexer = VideoIndexer(
            scene_model="facebook/vjepa2-vitl-fpc64-256", scene_clip_size=32
        )
        assert indexer._scene_clip_size == 32


class TestEmbeddingGemmaTextEncoder:
    """Tests for Feature 1: EmbeddingGemma text encoder."""

    def test_text_embedding_model_stored(self):
        indexer = VideoIndexer(text_embedding_model="google/embedding-gemma-300m")
        assert indexer._text_embedding_model_name == "google/embedding-gemma-300m"

    def test_text_embedding_model_default_none(self):
        indexer = VideoIndexer()
        assert indexer._text_embedding_model_name is None

    def test_score_annotations_skipped_with_text_model(self):
        """_score_annotations should skip scoring when text_embedding_model is set."""
        indexer = VideoIndexer(text_embedding_model="google/embedding-gemma-300m")
        segments = [
            {"start_time": 0.0, "end_time": 5.0, "caption": "a cat sitting"},
        ]
        frames = [np.full((48, 64, 3), 128, dtype=np.uint8)] * 10
        timestamps = [float(i) * 0.5 for i in range(10)]

        with patch.object(indexer, "_ensure_model"):
            indexer._score_annotations(
                segments,
                loaded_video_frames=frames,
                timestamps=timestamps,
                caption_fn=None,
            )

        # Score should NOT be computed (skipped due to cross-space mismatch)
        assert "caption_quality_score" not in segments[0]

    def test_encode_texts_siglip_method_exists(self):
        """_encode_texts_siglip method should exist on VideoIndexer."""
        indexer = VideoIndexer()
        assert hasattr(indexer, "_encode_texts_siglip")
        assert callable(indexer._encode_texts_siglip)

    def test_encode_query_siglip_method_exists(self):
        """_encode_query_siglip method should exist on VideoIndexer."""
        indexer = VideoIndexer()
        assert hasattr(indexer, "_encode_query_siglip")
        assert callable(indexer._encode_query_siglip)


class TestMultiScaleSearch:
    """Tests for Feature 2: Multi-scale search."""

    def test_build_coarse_level_basic(self):
        """_build_coarse_level should merge segments into ~30s chunks."""
        indexer = VideoIndexer()
        segments = [
            {"start_time": float(i * 5), "end_time": float((i + 1) * 5), "caption": f"seg{i}"}
            for i in range(10)  # 10 segments of 5s each = 50s total
        ]
        embeddings = np.random.default_rng(42).random((10, 8)).astype(np.float32)
        # Normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms

        coarse_segs, coarse_embs = indexer._build_coarse_level(
            segments, embeddings, target_duration=30.0
        )

        # With 50s total and 30s target, expect 2 coarse segments
        assert len(coarse_segs) == 2
        assert coarse_embs is not None
        assert coarse_embs.shape[0] == 2
        assert coarse_segs[0]["start_time"] == 0.0
        assert coarse_segs[-1]["end_time"] == 50.0
        # Captions should be merged
        assert "seg0" in coarse_segs[0]["caption"]

    def test_build_coarse_level_empty(self):
        """Empty input returns empty result."""
        indexer = VideoIndexer()
        coarse_segs, coarse_embs = indexer._build_coarse_level([], np.array([]), target_duration=30.0)
        assert coarse_segs == []
        assert coarse_embs is None

    def test_build_coarse_level_none_embeddings(self):
        indexer = VideoIndexer()
        segs = [{"start_time": 0.0, "end_time": 5.0, "caption": "x"}]
        coarse_segs, coarse_embs = indexer._build_coarse_level(segs, None, target_duration=30.0)
        assert coarse_segs == []
        assert coarse_embs is None

    @patch("kuavi.indexer.detect_scenes")
    def test_coarse_level_added_in_index_video(self, mock_detect_scenes):
        """index_video should always add a coarse level when embeddings exist."""
        mock_detect_scenes.return_value = [(0.0, 2.5), (2.5, 4.5)]

        indexer = VideoIndexer()
        frames = [np.full((48, 64, 3), i * 25, dtype=np.uint8) for i in range(10)]
        mock_video = MagicMock()
        mock_video.metadata.extraction_fps = 2.0
        mock_video.metadata.path = "/fake/video.mp4"
        mock_video.frames = frames
        mock_video.segments = []

        fake_emb = np.random.randn(2, 4).astype(np.float32)

        with patch.object(indexer, "_ensure_model"):
            with patch.object(indexer, "_get_transcript", return_value=[]):
                with patch.object(
                    indexer, "_embed_captions", return_value=(fake_emb, None)
                ):
                    with patch.object(
                        indexer, "_encode_frames",
                        return_value=np.random.randn(2, 4).astype(np.float32),
                    ):
                        result = indexer.index_video(mock_video, caption_fn=None)

        # Should have at least 1 hierarchy level (the coarse level)
        assert len(result.segment_hierarchy) >= 1
        assert len(result.hierarchy_embeddings) >= 1


class TestSelectiveDecode:
    """Tests for Feature 4: Selective decoding."""

    def test_selective_decode_3tier(self):
        """3-tier selective decode: static → Tier 1 (1 keyframe), varied → Tier 2."""
        indexer = VideoIndexer()
        segments = [
            {"start_time": 0.0, "end_time": 5.0, "caption": "", "_frames": [f"f{i}" for i in range(10)]},
            {"start_time": 5.0, "end_time": 10.0, "caption": "", "_frames": [f"f{i}" for i in range(10)]},
        ]
        # Use textured frames (not solid) to avoid dead-frame detection
        rng = np.random.RandomState(99)
        frames = [rng.randint(50, 200, (48, 64, 3), dtype=np.uint8) for _ in range(20)]
        timestamps = [float(i) * 0.5 for i in range(20)]

        # First segment: identical unit vectors → pairwise sim = 1.0 → Tier 1
        uniform_embs = np.ones((10, 8), dtype=np.float32)
        uniform_embs = uniform_embs / np.linalg.norm(uniform_embs[0])
        # Second segment: orthogonal unit vectors → pairwise sim ≈ 0 → Tier 2
        varied_embs = np.eye(8, dtype=np.float32)
        varied_embs = np.vstack([varied_embs, varied_embs[:2]])  # 10 rows

        call_count = [0]

        def mock_encode(seg_frames):
            idx = call_count[0]
            call_count[0] += 1
            return uniform_embs if idx == 0 else varied_embs

        with patch.object(indexer, "_ensure_model"):
            with patch.object(indexer, "_encode_frames", side_effect=mock_encode):
                indexer._selective_decode(segments, frames, timestamps)

        # Static segment → Tier 1 (static-informative), NOT skipped
        assert segments[0]["_selective_tier"] == 1
        assert segments[0].get("_static_informative") is True
        assert segments[0].get("_skip_caption") is None
        # Varied segment → Tier 2 (dynamic)
        assert segments[1]["_selective_tier"] == 2
        assert segments[1]["_visual_variance"] > 0.5

    def test_selective_decode_keeps_varied(self):
        """Segments with low pairwise similarity → Tier 2 (dynamic)."""
        indexer = VideoIndexer()
        segments = [
            {"start_time": 0.0, "end_time": 5.0, "caption": ""},
        ]
        rng = np.random.RandomState(42)
        frames = [rng.randint(50, 200, (48, 64, 3), dtype=np.uint8) for _ in range(10)]
        timestamps = [float(i) * 0.5 for i in range(10)]

        # Diverse L2-normalized embeddings (low pairwise similarity)
        rng2 = np.random.default_rng(42)
        varied_embs = rng2.random((10, 8)).astype(np.float32)
        norms = np.linalg.norm(varied_embs, axis=1, keepdims=True)
        varied_embs = varied_embs / np.maximum(norms, 1e-10)

        with patch.object(indexer, "_ensure_model"):
            with patch.object(indexer, "_encode_frames", return_value=varied_embs):
                indexer._selective_decode(segments, frames, timestamps)

        assert segments[0].get("_skip_caption") is not True
        assert segments[0]["_selective_tier"] == 2

    def test_selective_decode_dead_frame_detection(self):
        """Solid black frames should be detected as Tier 0 (dead)."""
        indexer = VideoIndexer()
        segments = [
            {"start_time": 0.0, "end_time": 5.0, "caption": ""},
        ]
        frames = [np.zeros((48, 64, 3), dtype=np.uint8)] * 10
        timestamps = [float(i) * 0.5 for i in range(10)]

        with patch.object(indexer, "_ensure_model"):
            with patch.object(indexer, "_encode_frames") as mock_encode:
                indexer._selective_decode(segments, frames, timestamps)

        assert segments[0]["_selective_tier"] == 0
        assert segments[0]["_skip_caption"] is True
        assert "Dead frame" in segments[0]["caption"]
        # Should not call _encode_frames since dead was detected first
        mock_encode.assert_not_called()

    def test_selective_decode_skips_already_marked(self):
        """Segments already marked _skip_caption should not be re-processed."""
        indexer = VideoIndexer()
        segments = [
            {"start_time": 0.0, "end_time": 5.0, "caption": "dup", "_skip_caption": True},
        ]
        rng = np.random.RandomState(77)
        frames = [rng.randint(50, 200, (48, 64, 3), dtype=np.uint8) for _ in range(10)]
        timestamps = [float(i) * 0.5 for i in range(10)]

        with patch.object(indexer, "_ensure_model"):
            with patch.object(indexer, "_encode_frames") as mock_encode:
                indexer._selective_decode(segments, frames, timestamps)

        # Should not call _encode_frames since segment was already marked
        mock_encode.assert_not_called()

    def test_selective_decode_few_frames_skipped(self):
        """Segments with fewer than 3 frames → Tier 2 (no SigLIP encoding needed)."""
        indexer = VideoIndexer()
        segments = [
            {"start_time": 0.0, "end_time": 0.5, "caption": ""},
        ]
        rng = np.random.RandomState(88)
        frames = [rng.randint(50, 200, (48, 64, 3), dtype=np.uint8)]
        timestamps = [0.0]

        with patch.object(indexer, "_ensure_model"):
            with patch.object(indexer, "_encode_frames") as mock_encode:
                indexer._selective_decode(segments, frames, timestamps)

        mock_encode.assert_not_called()


class TestVideoIndexFrameEmbeddings:
    """Tests for Feature 3: frame_embeddings field on VideoIndex."""

    def test_default_none(self):
        idx = VideoIndex()
        assert idx.frame_embeddings is None
        assert idx.visual_embed_fn is None

    def test_save_load_with_frame_embeddings(self, tmp_path):
        """Frame embeddings should survive save/load round-trip."""
        frame_embs = np.random.randn(3, 16).astype(np.float32)
        idx = VideoIndex(
            segments=[
                {"start_time": float(i), "end_time": float(i + 1), "caption": f"s{i}"}
                for i in range(3)
            ],
            embeddings=np.random.randn(3, 16).astype(np.float32),
            frame_embeddings=frame_embs,
            transcript=[],
            scene_boundaries=[0.0],
        )
        save_dir = tmp_path / "frame_emb_test"
        idx.save(save_dir)
        loaded = VideoIndex.load(save_dir)
        assert loaded.frame_embeddings is not None
        np.testing.assert_array_almost_equal(loaded.frame_embeddings, frame_embs)

    def test_save_load_without_frame_embeddings(self, tmp_path):
        """Backward compat: load should work without frame_embeddings in npz."""
        idx = VideoIndex(
            segments=[{"start_time": 0.0, "end_time": 1.0, "caption": "x"}],
            transcript=[],
            scene_boundaries=[0.0],
        )
        save_dir = tmp_path / "no_frame_emb"
        idx.save(save_dir)
        loaded = VideoIndex.load(save_dir)
        assert loaded.frame_embeddings is None


class TestEmbeddingGemmaIntegration:
    """Integration tests for EmbeddingGemma text encoder cross-space guard."""

    def test_score_annotations_no_scoring_with_text_model(self):
        """When text_embedding_model is set, _score_annotations skips all segments."""
        indexer = VideoIndexer(text_embedding_model="test-model")
        segments = [
            {"start_time": 0.0, "end_time": 5.0, "caption": "a cat sitting on a mat"},
            {"start_time": 5.0, "end_time": 10.0, "caption": "a dog running in the park"},
        ]
        frames = [np.full((48, 64, 3), 128, dtype=np.uint8)] * 20
        timestamps = [float(i) * 0.5 for i in range(20)]

        with patch.object(indexer, "_ensure_model"):
            indexer._score_annotations(
                segments,
                loaded_video_frames=frames,
                timestamps=timestamps,
                caption_fn=None,
            )

        for seg in segments:
            assert "caption_quality_score" not in seg, (
                f"Segment {seg['start_time']}-{seg['end_time']} should NOT have "
                "caption_quality_score when text_embedding_model is set"
            )

    def test_score_annotations_scores_without_text_model(self):
        """Without text_embedding_model, _score_annotations computes scores."""
        indexer = VideoIndexer()  # no text_embedding_model
        segments = [
            {"start_time": 0.0, "end_time": 2.5, "caption": "a cat sitting"},
        ]
        frames = [np.full((48, 64, 3), 128, dtype=np.uint8)] * 5
        timestamps = [float(i) * 0.5 for i in range(5)]

        # Return known embeddings of matching dimensions
        text_emb = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        frame_embs = np.array(
            [[0.9, 0.1, 0.0, 0.0]] * 5, dtype=np.float32
        )
        # Normalize
        text_emb = text_emb / np.linalg.norm(text_emb, axis=1, keepdims=True)
        frame_embs = frame_embs / np.linalg.norm(frame_embs, axis=1, keepdims=True)

        with patch.object(indexer, "_ensure_model"):
            with patch.object(indexer, "_encode_texts", return_value=text_emb):
                with patch.object(indexer, "_encode_frames", return_value=frame_embs):
                    indexer._score_annotations(
                        segments,
                        loaded_video_frames=frames,
                        timestamps=timestamps,
                        caption_fn=None,
                    )

        assert "caption_quality_score" in segments[0], (
            "caption_quality_score should be set when no text_embedding_model is configured"
        )
        assert isinstance(segments[0]["caption_quality_score"], float)

    def test_visual_embed_fn_differs_from_embed_fn(self):
        """_encode_query_siglip and _encode_query should be different methods."""
        indexer = VideoIndexer(text_embedding_model="test-model")
        # They should be distinct bound methods
        assert indexer._encode_query_siglip is not indexer._encode_query
        # Verify they are actually different underlying functions
        assert (
            indexer._encode_query_siglip.__func__ is not indexer._encode_query.__func__
        )

    @patch("kuavi.indexer.detect_scenes")
    def test_index_stores_visual_embed_fn(self, mock_detect_scenes):
        """index_video should set visual_embed_fn bound to _encode_query_siglip."""
        mock_detect_scenes.return_value = [(0.0, 2.5)]

        indexer = VideoIndexer(text_embedding_model="test-model")
        rng = np.random.RandomState(33)
        frames = [rng.randint(50, 200, (48, 64, 3), dtype=np.uint8) for _ in range(5)]
        mock_video = MagicMock()
        mock_video.metadata.extraction_fps = 2.0
        mock_video.metadata.path = "/fake/video.mp4"
        mock_video.frames = frames
        mock_video.segments = []

        fake_emb = np.random.randn(1, 4).astype(np.float32)

        with patch.object(indexer, "_ensure_model"):
            with patch.object(indexer, "_get_transcript", return_value=[]):
                with patch.object(
                    indexer, "_embed_captions", return_value=(fake_emb, None)
                ):
                    with patch.object(
                        indexer, "_encode_frames",
                        return_value=np.random.randn(1, 4).astype(np.float32),
                    ):
                        result = indexer.index_video(mock_video, caption_fn=None)

        assert result.visual_embed_fn is not None
        assert result.visual_embed_fn == indexer._encode_query_siglip


class TestSelectiveDecodeEfficiency:
    """Tests proving selective decoding saves caption calls."""

    def test_uniform_segments_tier_classification(self):
        """Create 20 segments where ~70% are uniform; verify tier classification."""
        indexer = VideoIndexer()

        num_segments = 20
        fps = 1.0
        seg_duration = 5.0
        frames_per_seg = int(seg_duration * fps)
        total_frames = num_segments * frames_per_seg

        segments = [
            {
                "start_time": i * seg_duration,
                "end_time": (i + 1) * seg_duration,
                "caption": "",
                "_frames": [f"f{j}" for j in range(frames_per_seg)],
            }
            for i in range(num_segments)
        ]
        # Use textured frames to avoid dead-frame detection
        test_rng = np.random.RandomState(55)
        frames = [
            test_rng.randint(50, 200, (48, 64, 3), dtype=np.uint8)
            for _ in range(total_frames)
        ]
        timestamps = [i / fps for i in range(total_frames)]

        num_uniform = 14
        call_counter = {"n": 0}
        rng = np.random.default_rng(42)

        def mock_encode_frames(seg_frames, **kwargs):
            idx = call_counter["n"]
            call_counter["n"] += 1
            n = len(seg_frames)
            if idx < num_uniform:
                base = np.ones((1, 8), dtype=np.float32)
                base = base / np.linalg.norm(base)
                embs = np.tile(base, (n, 1))
                embs += rng.normal(0, 0.001, embs.shape).astype(np.float32)
                norms = np.linalg.norm(embs, axis=1, keepdims=True)
                embs = embs / np.maximum(norms, 1e-10)
            else:
                embs = np.eye(8, dtype=np.float32)
                embs = np.tile(embs, (max(1, n // 8 + 1), 1))[:n]
            return embs

        with patch.object(indexer, "_ensure_model"):
            with patch.object(indexer, "_encode_frames", side_effect=mock_encode_frames):
                indexer._selective_decode(segments, frames, timestamps)

        # Uniform segments → Tier 1 (static-informative), dynamic → Tier 2
        tier_1_count = sum(1 for s in segments if s.get("_selective_tier") == 1)
        tier_2_count = sum(1 for s in segments if s.get("_selective_tier") == 2)
        assert tier_1_count >= num_uniform
        assert tier_2_count >= num_segments - num_uniform

    def test_mixed_video_tier_classification(self):
        """10 segments: even=static, odd=dynamic. Verify tier classification."""
        indexer = VideoIndexer()

        num_segments = 10
        fps = 1.0
        seg_duration = 5.0
        frames_per_seg = int(seg_duration * fps)
        total_frames = num_segments * frames_per_seg

        segments = [
            {
                "start_time": i * seg_duration,
                "end_time": (i + 1) * seg_duration,
                "caption": "",
                "_frames": [f"f{j}" for j in range(frames_per_seg)],
            }
            for i in range(num_segments)
        ]
        # Use textured frames to avoid dead-frame detection
        test_rng = np.random.RandomState(66)
        frames = [
            test_rng.randint(50, 200, (48, 64, 3), dtype=np.uint8)
            for _ in range(total_frames)
        ]
        timestamps = [i / fps for i in range(total_frames)]

        call_counter = {"n": 0}

        def mock_encode_frames(seg_frames, **kwargs):
            idx = call_counter["n"]
            call_counter["n"] += 1
            n = len(seg_frames)
            if idx % 2 == 0:
                base = np.ones((1, 8), dtype=np.float32) / np.sqrt(8)
                return np.tile(base, (n, 1))
            else:
                embs = np.eye(8, dtype=np.float32)
                return np.tile(embs, (max(1, n // 8 + 1), 1))[:n]

        with patch.object(indexer, "_ensure_model"):
            with patch.object(indexer, "_encode_frames", side_effect=mock_encode_frames):
                indexer._selective_decode(segments, frames, timestamps)

        for i, seg in enumerate(segments):
            if i % 2 == 0:
                # Static → Tier 1 (static-informative)
                assert seg["_selective_tier"] == 1, (
                    f"Segment {i} (static) should be Tier 1"
                )
            else:
                # Dynamic → Tier 2
                assert seg["_selective_tier"] == 2, (
                    f"Segment {i} (dynamic) should be Tier 2"
                )
                assert seg["_visual_variance"] > 0.5, (
                    f"Segment {i} (dynamic) should have high variance"
                )
