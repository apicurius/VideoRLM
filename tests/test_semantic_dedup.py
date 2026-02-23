"""Tests for WI-8: Semantic deduplication via k-means clustering."""

from contextlib import ExitStack
from unittest.mock import MagicMock, patch

import numpy as np

from kuavi.indexer import VideoIndex, VideoIndexer
from kuavi.search import _round_robin_from_clusters, make_search_video

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_loaded_video(num_frames=10, fps=2.0):
    """Create a mock LoadedVideo with synthetic frames."""
    rng = np.random.RandomState(0)
    frames = [
        np.clip(rng.randint(50, 200, (32, 32, 3), dtype=np.uint8) + i * 5, 0, 255).astype(
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


def _fake_encode(frames, **kw):
    rows = []
    for f in frames:
        seed = int(np.mean(f)) + 1
        rows.append(np.random.default_rng(seed).standard_normal(4))
    embs = np.stack(rows).astype(np.float32)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    return embs / np.maximum(norms, 1e-10)


def _patch_indexer(indexer, embed_captions_rv=None):
    if embed_captions_rv is None:
        embed_captions_rv = (np.eye(5, dtype=np.float32), None)
    stack = ExitStack()
    stack.enter_context(patch.object(indexer, "_ensure_model"))
    stack.enter_context(patch.object(indexer, "_get_transcript", return_value=[]))
    stack.enter_context(patch.object(indexer, "_embed_captions", return_value=embed_captions_rv))
    stack.enter_context(patch.object(indexer, "_encode_frames", side_effect=_fake_encode))
    stack.enter_context(patch.object(indexer, "_pre_caption_dedup"))
    stack.enter_context(patch.object(indexer, "_selective_decode"))
    return stack


def _make_segments(n):
    return [
        {"start_time": float(i), "end_time": float(i + 1), "caption": f"caption {i}"}
        for i in range(n)
    ]


def _make_index(segments, embeddings):
    """Build a minimal VideoIndex for search tests."""
    return VideoIndex(
        segments=segments,
        embeddings=embeddings,
        action_embeddings=None,
        frame_embeddings=None,
        temporal_embeddings=None,
        transcript=[],
        scene_boundaries=[],
        embedding_quality="ok",
        embed_fn=lambda q: np.ones(embeddings.shape[1], dtype=np.float32),
    )


# ---------------------------------------------------------------------------
# Tests: _semantic_deduplicate
# ---------------------------------------------------------------------------


class TestSemanticDeduplicate:
    def test_assigns_cluster_id_to_all_segments(self):
        indexer = VideoIndexer()
        n = 10
        rng = np.random.default_rng(1)
        embeddings = rng.standard_normal((n, 8)).astype(np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings /= np.maximum(norms, 1e-10)

        segments = _make_segments(n)
        indexer._semantic_deduplicate(segments, embeddings)

        for seg in segments:
            assert "cluster_id" in seg
            assert isinstance(seg["cluster_id"], int)

    def test_marks_near_duplicates(self):
        indexer = VideoIndexer()
        n = 6
        # Create two groups of identical embeddings — all within each group are duplicates
        base1 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        base2 = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        embeddings = np.array(
            [base1, base1, base1, base2, base2, base2], dtype=np.float32
        )

        segments = _make_segments(n)
        for seg in segments:
            seg["quality_score"] = 0.5

        indexer._semantic_deduplicate(
            segments, embeddings, n_clusters=2, similarity_threshold=0.90
        )

        # Within each group of 3 identical vectors, 2 should be marked as duplicates
        dup_count = sum(1 for s in segments if s.get("is_semantic_duplicate"))
        assert dup_count >= 2

    def test_returns_none_when_embeddings_none(self):
        indexer = VideoIndexer()
        segments = _make_segments(5)
        result = indexer._semantic_deduplicate(segments, None)
        assert result is None

    def test_returns_none_when_fewer_than_3_segments(self):
        indexer = VideoIndexer()
        embeddings = np.eye(2, dtype=np.float32)
        segments = _make_segments(2)
        result = indexer._semantic_deduplicate(segments, embeddings)
        assert result is None

    def test_auto_computes_n_clusters(self):
        indexer = VideoIndexer()
        n = 20
        rng = np.random.default_rng(42)
        embeddings = rng.standard_normal((n, 8)).astype(np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings /= np.maximum(norms, 1e-10)

        segments = _make_segments(n)
        # n_clusters = max(2, 20 // 5) = 4
        with patch("sklearn.cluster.KMeans") as mock_kmeans_cls:
            mock_km = MagicMock()
            mock_km.fit_predict.return_value = np.zeros(n, dtype=int)
            mock_kmeans_cls.return_value = mock_km
            indexer._semantic_deduplicate(segments, embeddings)
            call_kwargs = mock_kmeans_cls.call_args
            assert call_kwargs[1]["n_clusters"] == 4

    def test_returns_labels_array(self):
        indexer = VideoIndexer()
        n = 6
        rng = np.random.default_rng(7)
        embeddings = rng.standard_normal((n, 4)).astype(np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings /= np.maximum(norms, 1e-10)

        segments = _make_segments(n)
        result = indexer._semantic_deduplicate(segments, embeddings)
        assert result is not None
        assert result.shape == (n,)

    def test_with_action_embeddings_combined(self):
        indexer = VideoIndexer()
        n = 6
        rng = np.random.default_rng(3)
        embeddings = rng.standard_normal((n, 4)).astype(np.float32)
        action_embeddings = rng.standard_normal((n, 4)).astype(np.float32)
        for e in (embeddings, action_embeddings):
            norms = np.linalg.norm(e, axis=1, keepdims=True)
            e /= np.maximum(norms, 1e-10)

        segments = _make_segments(n)
        # Should not raise even with combined embeddings
        result = indexer._semantic_deduplicate(segments, embeddings, action_embeddings)
        assert result is not None
        assert result.shape == (n,)


# ---------------------------------------------------------------------------
# Tests: index_video integration
# ---------------------------------------------------------------------------


class TestIndexVideoSemanticDedup:
    @patch("kuavi.indexer.detect_scenes")
    def test_semantic_dedup_parameter_exists(self, mock_detect_scenes):
        """index_video accepts semantic_dedup kwarg without error."""
        mock_detect_scenes.return_value = [(0.0, 1.5), (1.5, 3.0), (3.0, 5.0)]
        indexer = VideoIndexer()
        loaded_video = _make_loaded_video(num_frames=10)
        fake_embeddings = np.eye(3, dtype=np.float32)

        with _patch_indexer(indexer, embed_captions_rv=(fake_embeddings, None)):
            # Should not raise
            idx = indexer.index_video(loaded_video, semantic_dedup=False)
        assert idx is not None

    @patch("kuavi.indexer.detect_scenes")
    def test_semantic_dedup_false_no_cluster_ids(self, mock_detect_scenes):
        """semantic_dedup=False (default) must NOT assign cluster_id to segments."""
        mock_detect_scenes.return_value = [(0.0, 1.5), (1.5, 3.0), (3.0, 5.0)]
        indexer = VideoIndexer()
        loaded_video = _make_loaded_video(num_frames=10)
        fake_embeddings = np.eye(3, dtype=np.float32)

        with _patch_indexer(indexer, embed_captions_rv=(fake_embeddings, None)):
            idx = indexer.index_video(loaded_video, semantic_dedup=False)

        for seg in idx.segments:
            assert "cluster_id" not in seg

    @patch("kuavi.indexer.detect_scenes")
    def test_semantic_dedup_true_assigns_cluster_ids(self, mock_detect_scenes):
        """semantic_dedup=True must assign cluster_id to every segment."""
        mock_detect_scenes.return_value = [
            (0.0, 1.0),
            (1.0, 2.0),
            (2.0, 3.0),
            (3.0, 4.0),
            (4.0, 5.0),
        ]
        indexer = VideoIndexer()
        loaded_video = _make_loaded_video(num_frames=10)
        n = 5
        fake_embeddings = np.eye(n, dtype=np.float32)

        with _patch_indexer(indexer, embed_captions_rv=(fake_embeddings, None)):
            idx = indexer.index_video(loaded_video, semantic_dedup=True)

        for seg in idx.segments:
            assert "cluster_id" in seg


# ---------------------------------------------------------------------------
# Tests: search with semantic_dedup
# ---------------------------------------------------------------------------


class TestSearchSemanticDedup:
    def _make_index_with_clusters(self, n=6):
        """Return a VideoIndex where half the segments have is_semantic_duplicate=True."""
        rng = np.random.default_rng(99)
        embeddings = rng.standard_normal((n, 4)).astype(np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings /= np.maximum(norms, 1e-10)

        segments = _make_segments(n)
        # Mark odd-indexed segments as semantic duplicates
        for i, seg in enumerate(segments):
            seg["cluster_id"] = i % 2
            if i % 2 == 1:
                seg["is_semantic_duplicate"] = True

        return _make_index(segments, embeddings)

    def test_search_excludes_semantic_duplicates(self):
        # 6 segments, 3 are semantic duplicates → request only 3 results
        index = self._make_index_with_clusters(6)
        tool = make_search_video(index)["tool"]
        results = tool("query", top_k=3, diverse=False)

        assert len(results) == 3
        for r in results:
            # None of the returned segments should be semantic duplicates
            start = r["start_time"]
            matching_segs = [
                s for s in index.segments if s["start_time"] == start
            ]
            for seg in matching_segs:
                assert not seg.get("is_semantic_duplicate"), (
                    f"Segment at {start} is a semantic duplicate but was returned"
                )

    def test_cluster_diverse_uses_precomputed_cluster_ids(self):
        """When cluster_ids are present, cluster_diverse should not call KMeans."""
        n = 6
        rng = np.random.default_rng(7)
        embeddings = rng.standard_normal((n, 4)).astype(np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings /= np.maximum(norms, 1e-10)

        segments = _make_segments(n)
        for i, seg in enumerate(segments):
            seg["cluster_id"] = i % 3

        index = _make_index(segments, embeddings)
        tool = make_search_video(index)["tool"]

        with patch("sklearn.cluster.KMeans") as mock_kmeans:
            results = tool("query", top_k=3, cluster_diverse=True)
            mock_kmeans.assert_not_called()

        assert len(results) <= 3

    def test_cluster_diverse_falls_back_to_kmeans_without_cluster_ids(self):
        """Without cluster_ids, cluster_diverse falls back to query-time KMeans."""
        n = 6
        rng = np.random.default_rng(5)
        embeddings = rng.standard_normal((n, 4)).astype(np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings /= np.maximum(norms, 1e-10)

        segments = _make_segments(n)
        # No cluster_id on segments

        index = _make_index(segments, embeddings)
        tool = make_search_video(index)["tool"]

        with patch("sklearn.cluster.KMeans") as mock_kmeans_cls:
            mock_km = MagicMock()
            mock_km.fit_predict.return_value = np.array([0, 1, 2, 0, 1, 2])
            mock_kmeans_cls.return_value = mock_km
            results = tool("query", top_k=3, cluster_diverse=True)
            mock_kmeans_cls.assert_called_once()

        assert len(results) <= 3

    def test_round_robin_cluster_selection_is_diverse(self):
        """Round-robin should pick one from each cluster before revisiting."""
        scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4])
        clusters = {0: [0, 3], 1: [1, 4], 2: [2, 5]}
        top_indices = _round_robin_from_clusters(clusters, scores, top_k=3)

        # Should get exactly one from each cluster first
        assert len(top_indices) == 3
        # Each result should be from a different cluster
        result_clusters = {i % 3 for i in top_indices}
        assert len(result_clusters) == 3
