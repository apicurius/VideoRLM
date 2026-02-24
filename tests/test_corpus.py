"""Tests for kuavi.corpus — CorpusIndex, CorpusIndexer, search_corpus, corpus_stats."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from kuavi.corpus import (
    CorpusIndex,
    CorpusIndexer,
    corpus_stats,
    discover_videos,
    search_corpus,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fake_video_index(
    video_id: str = "video1",
    num_segments: int = 3,
    emb_dim: int = 4,
    with_actions: bool = False,
) -> MagicMock:
    """Return a mock VideoIndex with embeddings, segments, and optional embed_fn."""
    idx = MagicMock()
    rng = np.random.default_rng(abs(hash(video_id)) % (2**31))
    emb = rng.standard_normal((num_segments, emb_dim)).astype(np.float32)
    # L2-normalize
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    emb = emb / np.maximum(norms, 1e-10)

    idx.embeddings = emb
    idx.embed_fn = lambda q: rng.standard_normal(emb_dim).astype(np.float32)

    segments = []
    for i in range(num_segments):
        seg: dict = {
            "start_time": float(i * 5),
            "end_time": float((i + 1) * 5),
            "caption": f"caption {video_id} seg {i}",
        }
        if with_actions:
            seg["annotation"] = {
                "action": {"brief": f"action_{i % 2}"}
            }
        segments.append(seg)
    idx.segments = segments

    # Save/load: make idx.save write a minimal VideoIndex directory
    def _fake_save(path):
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        np.savez(p / "embeddings.npz", embeddings=emb)
        meta = {
            "segments": segments,
            "transcript": [],
            "scene_boundaries": [],
            "embedding_quality": {},
            "segment_hierarchy": [],
        }
        (p / "metadata.json").write_text(json.dumps(meta))

    idx.save = _fake_save
    return idx


# ---------------------------------------------------------------------------
# CorpusIndex tests
# ---------------------------------------------------------------------------


class TestCorpusIndexProperties:
    def test_default_construction(self):
        ci = CorpusIndex()
        assert ci.num_videos == 0
        assert ci.total_segments == 0
        assert ci.total_duration == 0.0
        assert ci.corpus_embeddings is None
        assert ci.corpus_segment_map == []

    def test_num_videos(self):
        ci = CorpusIndex(
            video_indices={"a": MagicMock(), "b": MagicMock()},
            video_metadata={"a": {"duration": 10.0}, "b": {"duration": 5.0}},
        )
        assert ci.num_videos == 2

    def test_total_segments(self):
        ci = CorpusIndex(
            corpus_segment_map=[{"video_id": "a", "segment_idx": 0}] * 5
        )
        assert ci.total_segments == 5

    def test_total_duration(self):
        ci = CorpusIndex(
            video_metadata={
                "a": {"duration": 30.0},
                "b": {"duration": 45.5},
            }
        )
        assert ci.total_duration == pytest.approx(75.5)

    def test_total_duration_missing_key(self):
        ci = CorpusIndex(
            video_metadata={"a": {}, "b": {"duration": 10.0}}
        )
        assert ci.total_duration == pytest.approx(10.0)


class TestCorpusIndexSaveLoad:
    def test_save_and_load_roundtrip(self, tmp_path):
        idx_a = _make_fake_video_index("vidA", num_segments=2, emb_dim=4)
        idx_b = _make_fake_video_index("vidB", num_segments=3, emb_dim=4)

        ci = CorpusIndex(
            video_indices={"vidA": idx_a, "vidB": idx_b},
            video_metadata={
                "vidA": {"path": "/a/vidA.mp4", "duration": 10.0, "num_segments": 2},
                "vidB": {"path": "/b/vidB.mp4", "duration": 15.0, "num_segments": 3},
            },
            action_vocabulary={"walking": [{"video_id": "vidA", "segment_idx": 0}]},
            corpus_segment_map=[
                {"video_id": "vidA", "segment_idx": 0},
                {"video_id": "vidA", "segment_idx": 1},
            ],
        )

        save_dir = tmp_path / "corpus"
        ci.save(save_dir)

        # Check files exist
        assert (save_dir / "corpus_metadata.json").exists()
        assert (save_dir / "indices" / "vidA" / "metadata.json").exists()
        assert (save_dir / "indices" / "vidB" / "metadata.json").exists()

        loaded = CorpusIndex.load(save_dir)

        assert loaded.num_videos == 2
        assert set(loaded.video_indices.keys()) == {"vidA", "vidB"}
        assert loaded.video_metadata["vidA"]["duration"] == pytest.approx(10.0)
        assert loaded.action_vocabulary == {"walking": [{"video_id": "vidA", "segment_idx": 0}]}
        assert loaded.corpus_segment_map == ci.corpus_segment_map

    def test_save_and_load_with_corpus_embeddings(self, tmp_path):
        ci = CorpusIndex(
            video_indices={},
            video_metadata={},
            corpus_embeddings=np.eye(4, dtype=np.float32),
            corpus_segment_map=[],
        )
        save_dir = tmp_path / "corpus_emb"
        ci.save(save_dir)
        loaded = CorpusIndex.load(save_dir)
        assert loaded.corpus_embeddings is not None
        np.testing.assert_array_almost_equal(loaded.corpus_embeddings, np.eye(4))

    def test_load_no_corpus_embeddings(self, tmp_path):
        ci = CorpusIndex(video_indices={}, video_metadata={})
        ci.save(tmp_path / "ci")
        loaded = CorpusIndex.load(tmp_path / "ci")
        assert loaded.corpus_embeddings is None


# ---------------------------------------------------------------------------
# CorpusIndexer tests
# ---------------------------------------------------------------------------


class TestCorpusIndexerInit:
    def test_default_init(self):
        indexer = CorpusIndexer()
        assert indexer.max_workers == 4
        assert indexer._indexer_kwargs == {}

    def test_custom_init(self):
        indexer = CorpusIndexer(max_workers=2, cache_dir="/tmp/cache")
        assert indexer.max_workers == 2
        assert indexer._indexer_kwargs == {"cache_dir": "/tmp/cache"}


class TestCorpusIndexerIndexCorpus:
    """Tests for index_corpus() with mocked VideoLoader and VideoIndexer."""

    def _make_mock_loaded(self, duration: float = 10.0, fps: float = 1.0) -> MagicMock:
        loaded = MagicMock()
        loaded.metadata.duration = duration
        loaded.metadata.extracted_frame_count = int(duration * fps)
        return loaded

    def test_index_corpus_single_video(self, tmp_path):
        """Single video should be indexed without thread pool."""
        fake_path = tmp_path / "v1.mp4"
        fake_path.touch()

        mock_loaded = self._make_mock_loaded(duration=10.0)
        mock_idx = _make_fake_video_index("v1", num_segments=2)

        with (
            patch("kuavi.loader.VideoLoader") as MockLoader,
            patch("kuavi.indexer.VideoIndexer") as MockIndexer,
        ):
            MockLoader.return_value.load.return_value = mock_loaded
            MockIndexer.return_value.index_video.return_value = mock_idx

            indexer = CorpusIndexer(max_workers=1)
            corpus = indexer.index_corpus([fake_path], mode="fast")

        assert corpus.num_videos == 1
        assert "v1" in corpus.video_indices
        assert corpus.video_metadata["v1"]["duration"] == pytest.approx(10.0)

    def test_index_corpus_multiple_videos(self, tmp_path):
        """Multiple videos indexed, resulting in populated corpus."""
        paths = []
        for i in range(3):
            p = tmp_path / f"vid{i}.mp4"
            p.touch()
            paths.append(p)

        call_count = [0]

        def _mock_load(path):
            return self._make_mock_loaded(duration=5.0)

        def _mock_index_video(loaded, **kw):
            vid_id = f"vid{call_count[0]}"
            call_count[0] += 1
            return _make_fake_video_index(vid_id, num_segments=2)

        with (
            patch("kuavi.loader.VideoLoader") as MockLoader,
            patch("kuavi.indexer.VideoIndexer") as MockIndexer,
        ):
            MockLoader.return_value.load.side_effect = _mock_load
            MockIndexer.return_value.index_video.side_effect = _mock_index_video

            indexer = CorpusIndexer(max_workers=1)
            corpus = indexer.index_corpus(paths, mode="fast")

        assert corpus.num_videos == 3

    def test_index_corpus_failed_video_is_skipped(self, tmp_path):
        """A video that fails indexing does not crash the corpus."""
        p_ok = tmp_path / "ok.mp4"
        p_bad = tmp_path / "bad.mp4"
        p_ok.touch()
        p_bad.touch()

        ok_loaded = self._make_mock_loaded()
        ok_idx = _make_fake_video_index("ok", num_segments=2)

        call_idx = [0]

        def _mock_load(path):
            return ok_loaded

        def _side_effect(loaded, **kw):
            call_idx[0] += 1
            if call_idx[0] == 2:
                raise RuntimeError("fake indexing error")
            return ok_idx

        with (
            patch("kuavi.loader.VideoLoader") as MockLoader,
            patch("kuavi.indexer.VideoIndexer") as MockIndexer,
        ):
            MockLoader.return_value.load.side_effect = _mock_load
            MockIndexer.return_value.index_video.side_effect = _side_effect

            indexer = CorpusIndexer(max_workers=1)
            corpus = indexer.index_corpus([p_ok, p_bad], mode="fast")

        # Should have 1 successful + 1 failed (not in video_indices)
        assert corpus.num_videos == 1
        # Metadata records both
        assert len(corpus.video_metadata) == 2

    def test_progress_callback_called(self, tmp_path):
        p = tmp_path / "v.mp4"
        p.touch()
        mock_loaded = self._make_mock_loaded()
        mock_idx = _make_fake_video_index("v", num_segments=1)

        calls = []

        with (
            patch("kuavi.loader.VideoLoader") as MockLoader,
            patch("kuavi.indexer.VideoIndexer") as MockIndexer,
        ):
            MockLoader.return_value.load.return_value = mock_loaded
            MockIndexer.return_value.index_video.return_value = mock_idx

            indexer = CorpusIndexer(max_workers=1)
            indexer.index_corpus([p], mode="fast", progress_callback=lambda *a: calls.append(a))

        assert len(calls) == 1
        path_arg, status, elapsed = calls[0]
        assert "v.mp4" in path_arg
        assert status == "done"
        assert elapsed >= 0.0

    def test_parallel_indexing(self, tmp_path):
        """max_workers > 1 paths are exercised."""
        paths = [tmp_path / f"v{i}.mp4" for i in range(4)]
        for p in paths:
            p.touch()

        def _mock_load(path):
            return self._make_mock_loaded()

        def _mock_index(loaded, **kw):
            return _make_fake_video_index("v", num_segments=1)

        with (
            patch("kuavi.loader.VideoLoader") as MockLoader,
            patch("kuavi.indexer.VideoIndexer") as MockIndexer,
        ):
            MockLoader.return_value.load.side_effect = _mock_load
            MockIndexer.return_value.index_video.side_effect = _mock_index

            indexer = CorpusIndexer(max_workers=4)
            corpus = indexer.index_corpus(paths, mode="fast")

        assert corpus.num_videos == 4


class TestBuildActionVocabulary:
    def test_builds_vocab_from_annotations(self):
        idx = MagicMock()
        idx.segments = [
            {"start_time": 0.0, "end_time": 5.0, "annotation": {"action": {"brief": "Walking"}}},
            {"start_time": 5.0, "end_time": 10.0, "annotation": {"action": {"brief": "Running"}}},
            {"start_time": 10.0, "end_time": 15.0, "annotation": {"action": {"brief": "walking"}}},
        ]
        idx.embeddings = np.zeros((3, 4))

        corpus = CorpusIndex(video_indices={"v1": idx})
        indexer = CorpusIndexer()
        indexer._build_action_vocabulary(corpus)

        assert "walking" in corpus.action_vocabulary
        assert len(corpus.action_vocabulary["walking"]) == 2
        assert "running" in corpus.action_vocabulary
        assert len(corpus.action_vocabulary["running"]) == 1

    def test_skips_missing_brief(self):
        idx = MagicMock()
        idx.segments = [
            {"start_time": 0.0, "end_time": 5.0},  # no annotation
            {"start_time": 5.0, "end_time": 10.0, "annotation": {"action": {}}},  # empty brief
        ]
        corpus = CorpusIndex(video_indices={"v1": idx})
        indexer = CorpusIndexer()
        indexer._build_action_vocabulary(corpus)

        assert corpus.action_vocabulary == {}


class TestBuildCorpusEmbeddings:
    def test_stacks_embeddings(self):
        idx_a = _make_fake_video_index("a", num_segments=2, emb_dim=4)
        idx_b = _make_fake_video_index("b", num_segments=3, emb_dim=4)

        corpus = CorpusIndex(video_indices={"a": idx_a, "b": idx_b})
        indexer = CorpusIndexer()
        indexer._build_corpus_embeddings(corpus)

        assert corpus.corpus_embeddings is not None
        assert corpus.corpus_embeddings.shape == (5, 4)
        assert len(corpus.corpus_segment_map) == 5

    def test_segment_map_correct_order(self):
        idx_a = _make_fake_video_index("a", num_segments=2, emb_dim=4)
        idx_b = _make_fake_video_index("b", num_segments=1, emb_dim=4)

        corpus = CorpusIndex(video_indices={"a": idx_a, "b": idx_b})
        CorpusIndexer()._build_corpus_embeddings(corpus)

        # Sorted by video_id: "a" comes before "b"
        assert corpus.corpus_segment_map[0] == {"video_id": "a", "segment_idx": 0}
        assert corpus.corpus_segment_map[1] == {"video_id": "a", "segment_idx": 1}
        assert corpus.corpus_segment_map[2] == {"video_id": "b", "segment_idx": 0}

    def test_empty_corpus(self):
        corpus = CorpusIndex()
        CorpusIndexer()._build_corpus_embeddings(corpus)
        assert corpus.corpus_embeddings is None
        assert corpus.corpus_segment_map == []

    def test_skips_none_embeddings(self):
        idx = MagicMock()
        idx.embeddings = None
        idx.segments = [{"start_time": 0.0, "end_time": 5.0}]

        corpus = CorpusIndex(video_indices={"v": idx})
        CorpusIndexer()._build_corpus_embeddings(corpus)

        assert corpus.corpus_embeddings is None
        assert corpus.corpus_segment_map == []


# ---------------------------------------------------------------------------
# search_corpus tests
# ---------------------------------------------------------------------------


class TestSearchCorpus:
    def _make_corpus(self, num_videos: int = 2, num_segments: int = 3) -> CorpusIndex:
        indices = {}
        meta = {}
        for i in range(num_videos):
            vid = f"vid{i}"
            indices[vid] = _make_fake_video_index(vid, num_segments=num_segments, emb_dim=4)
            meta[vid] = {"path": f"/fake/{vid}.mp4", "duration": 10.0, "num_segments": num_segments}

        corpus = CorpusIndex(video_indices=indices, video_metadata=meta)
        CorpusIndexer()._build_corpus_embeddings(corpus)
        return corpus

    def test_returns_results(self):
        corpus = self._make_corpus()
        results = search_corpus(corpus, "person walking", top_k=3)
        assert len(results) <= 3
        for r in results:
            assert "video_id" in r
            assert "start_time" in r
            assert "end_time" in r
            assert "score" in r
            assert "caption" in r

    def test_top_k_respected(self):
        corpus = self._make_corpus(num_videos=3, num_segments=4)
        results = search_corpus(corpus, "test query", top_k=5)
        assert len(results) <= 5

    def test_video_filter_restricts_results(self):
        corpus = self._make_corpus(num_videos=3, num_segments=2)
        results = search_corpus(corpus, "test", top_k=10, video_filter=["vid0"])
        for r in results:
            assert r["video_id"] == "vid0"

    def test_empty_corpus_returns_empty(self):
        corpus = CorpusIndex()
        results = search_corpus(corpus, "query")
        assert results == []

    def test_corpus_no_embed_fn_returns_empty(self):
        idx = MagicMock()
        rng = np.random.default_rng(0)
        idx.embeddings = rng.standard_normal((2, 4)).astype(np.float32)
        idx.embed_fn = None
        idx.segments = [
            {"start_time": 0.0, "end_time": 5.0, "caption": "a"},
            {"start_time": 5.0, "end_time": 10.0, "caption": "b"},
        ]

        corpus = CorpusIndex(
            video_indices={"v": idx},
            corpus_embeddings=idx.embeddings,
            corpus_segment_map=[
                {"video_id": "v", "segment_idx": 0},
                {"video_id": "v", "segment_idx": 1},
            ],
        )
        results = search_corpus(corpus, "query")
        assert results == []

    def test_scores_are_rounded(self):
        corpus = self._make_corpus()
        results = search_corpus(corpus, "test", top_k=2)
        for r in results:
            # score should be rounded to 4 decimal places
            assert r["score"] == round(r["score"], 4)


# ---------------------------------------------------------------------------
# corpus_stats tests
# ---------------------------------------------------------------------------


class TestCorpusStats:
    def test_basic_stats(self):
        idx = _make_fake_video_index("v1", num_segments=3, with_actions=True)
        corpus = CorpusIndex(
            video_indices={"v1": idx},
            video_metadata={"v1": {"path": "/v1.mp4", "duration": 30.0, "num_segments": 3}},
        )
        CorpusIndexer()._build_action_vocabulary(corpus)
        CorpusIndexer()._build_corpus_embeddings(corpus)

        stats = corpus_stats(corpus)

        assert stats["num_videos"] == 1
        assert stats["total_segments"] == 3
        assert stats["total_duration_seconds"] == pytest.approx(30.0)
        assert "action_vocabulary_size" in stats
        assert "top_actions" in stats
        assert isinstance(stats["top_actions"], list)
        assert "videos" in stats
        assert "v1" in stats["videos"]

    def test_empty_corpus_stats(self):
        corpus = CorpusIndex()
        stats = corpus_stats(corpus)
        assert stats["num_videos"] == 0
        assert stats["total_segments"] == 0
        assert stats["total_duration_seconds"] == pytest.approx(0.0)
        assert stats["action_vocabulary_size"] == 0
        assert stats["top_actions"] == []

    def test_top_actions_capped_at_20(self):
        # Build action vocabulary with 25 unique actions
        idx = MagicMock()
        idx.segments = [
            {
                "start_time": float(i),
                "end_time": float(i + 1),
                "annotation": {"action": {"brief": f"action_{i}"}},
            }
            for i in range(25)
        ]
        corpus = CorpusIndex(video_indices={"v": idx}, video_metadata={"v": {"duration": 25.0}})
        CorpusIndexer()._build_action_vocabulary(corpus)

        stats = corpus_stats(corpus)
        assert len(stats["top_actions"]) <= 20


# ---------------------------------------------------------------------------
# discover_videos tests
# ---------------------------------------------------------------------------


class TestDiscoverVideos:
    def test_finds_video_files(self, tmp_path):
        (tmp_path / "a.mp4").touch()
        (tmp_path / "b.avi").touch()
        (tmp_path / "c.txt").touch()
        (tmp_path / "d.mov").touch()

        found = discover_videos(tmp_path)
        names = {p.name for p in found}
        assert "a.mp4" in names
        assert "b.avi" in names
        assert "d.mov" in names
        assert "c.txt" not in names

    def test_returns_sorted(self, tmp_path):
        (tmp_path / "z.mp4").touch()
        (tmp_path / "a.mp4").touch()
        (tmp_path / "m.mp4").touch()

        found = discover_videos(tmp_path)
        names = [p.name for p in found]
        assert names == sorted(names)

    def test_empty_directory(self, tmp_path):
        found = discover_videos(tmp_path)
        assert found == []


# ---------------------------------------------------------------------------
# CLI corpus subcommand tests
# ---------------------------------------------------------------------------


class TestCLICorpusSubcommand:
    def test_corpus_subcommand_exists(self):
        """kuavi corpus --help should not raise."""
        import subprocess

        result = subprocess.run(
            ["uv", "run", "python", "-m", "kuavi.cli", "corpus", "--help"],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).resolve().parent.parent),
        )
        assert result.returncode == 0
        assert "corpus" in result.stdout.lower() or "subcommand" in result.stdout.lower()

    def test_corpus_index_subcommand_exists(self):
        """kuavi corpus index --help should not raise."""
        import subprocess

        result = subprocess.run(
            ["uv", "run", "python", "-m", "kuavi.cli", "corpus", "index", "--help"],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).resolve().parent.parent),
        )
        assert result.returncode == 0

    def test_corpus_search_subcommand_exists(self):
        """kuavi corpus search --help should not raise."""
        import subprocess

        result = subprocess.run(
            ["uv", "run", "python", "-m", "kuavi.cli", "corpus", "search", "--help"],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).resolve().parent.parent),
        )
        assert result.returncode == 0
