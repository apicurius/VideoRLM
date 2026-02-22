"""Tests for _align_query_dim and temporal/all field search correctness."""

from __future__ import annotations

import numpy as np
import pytest

from kuavi.search import _align_query_dim, make_search_video


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _l2_norm(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v


def _make_segments(n: int) -> list[dict]:
    return [
        {
            "start_time": float(i * 2),
            "end_time": float(i * 2 + 2),
            "caption": f"segment {i}",
        }
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Unit tests for _align_query_dim
# ---------------------------------------------------------------------------

class TestAlignQueryDim:
    def test_matching_dimensions_no_op(self):
        """Matching dims → return unchanged array."""
        q = np.random.rand(1, 768).astype(np.float32)
        m = np.random.rand(5, 768).astype(np.float32)
        result = _align_query_dim(q, m)
        np.testing.assert_array_equal(result, q)

    def test_768_query_vs_1024_matrix_pads(self):
        """768-d query vs 1024-d matrix → output has 1024 columns."""
        q = np.random.rand(1, 768).astype(np.float32)
        m = np.random.rand(5, 1024).astype(np.float32)
        result = _align_query_dim(q, m)
        assert result.shape == (1, 1024)

    def test_256_query_vs_1024_matrix_pads(self):
        """256-d query vs 1024-d matrix → output has 1024 columns."""
        q = np.random.rand(1, 256).astype(np.float32)
        m = np.random.rand(5, 1024).astype(np.float32)
        result = _align_query_dim(q, m)
        assert result.shape == (1, 1024)

    def test_padded_output_is_l2_normalized(self):
        """After padding, the output vector should be L2-normalized."""
        q = np.ones((1, 768), dtype=np.float32)
        m = np.random.rand(5, 1024).astype(np.float32)
        result = _align_query_dim(q, m)
        norm = np.linalg.norm(result)
        assert abs(norm - 1.0) < 1e-5, f"Expected unit norm, got {norm:.6f}"

    def test_padded_prefix_matches_original(self):
        """The first d_q values in the padded vector are derived from the original."""
        q = np.array([[1.0, 2.0, 3.0]], dtype=np.float64)
        m = np.random.rand(5, 6)
        result = _align_query_dim(q, m)
        assert result.shape == (1, 6)
        # The tail should be zeros (before normalization), so after normalization
        # result[0, 3:] should all be zero.
        np.testing.assert_array_equal(result[0, 3:], np.zeros(3))

    def test_larger_query_truncates(self):
        """Query dim > matrix dim → truncate to matrix dim."""
        q = np.random.rand(1, 1024).astype(np.float32)
        m = np.random.rand(5, 768).astype(np.float32)
        result = _align_query_dim(q, m)
        assert result.shape == (1, 768)
        np.testing.assert_array_equal(result[0], q[0, :768])


# ---------------------------------------------------------------------------
# Integration test: field="temporal"
# ---------------------------------------------------------------------------

class TestSearchVideoTemporal:
    def _make_index(self):
        n = 5
        rng = np.random.default_rng(42)

        temporal = rng.standard_normal((n, 1024)).astype(np.float32)
        temporal /= np.linalg.norm(temporal, axis=1, keepdims=True)

        summary_embs = rng.standard_normal((n, 768)).astype(np.float32)
        summary_embs /= np.linalg.norm(summary_embs, axis=1, keepdims=True)

        def embed_fn(text: str) -> np.ndarray:  # noqa: ARG001
            v = rng.standard_normal(768).astype(np.float32)
            return v / np.linalg.norm(v)

        class MockIndex:
            segments = _make_segments(n)
            embeddings = summary_embs
            action_embeddings = None
            frame_embeddings = None
            temporal_embeddings = temporal
            transcript = []

        idx = MockIndex()
        idx.embed_fn = embed_fn
        idx.visual_embed_fn = embed_fn
        return idx

    def test_temporal_search_returns_results(self):
        """field='temporal' with 768-d embed_fn vs 1024-d embeddings must not raise."""
        idx = self._make_index()
        tool = make_search_video(idx)["tool"]
        results = tool("test query", field="temporal", top_k=3)
        assert isinstance(results, list)
        assert len(results) > 0

    def test_temporal_search_result_schema(self):
        """Each result has required keys."""
        idx = self._make_index()
        tool = make_search_video(idx)["tool"]
        results = tool("movement", field="temporal", top_k=2)
        for r in results:
            assert "start_time" in r
            assert "end_time" in r
            assert "score" in r
            assert "caption" in r

    def test_temporal_search_no_value_error(self):
        """Specifically verify no ValueError from cosine_similarity dim mismatch."""
        idx = self._make_index()
        tool = make_search_video(idx)["tool"]
        try:
            tool("walking person", field="temporal")
        except ValueError as e:
            pytest.fail(f"search_video raised ValueError: {e}")


# ---------------------------------------------------------------------------
# Integration test: field="all"
# ---------------------------------------------------------------------------

class TestSearchVideoAll:
    def _make_index(self):
        n = 5
        rng = np.random.default_rng(7)

        def rand_embs(rows, cols):
            e = rng.standard_normal((rows, cols)).astype(np.float32)
            e /= np.linalg.norm(e, axis=1, keepdims=True)
            return e

        def embed_fn(text: str) -> np.ndarray:  # noqa: ARG001
            v = rng.standard_normal(768).astype(np.float32)
            return v / np.linalg.norm(v)

        class MockIndex:
            segments = _make_segments(n)
            embeddings = rand_embs(n, 768)
            action_embeddings = rand_embs(n, 768)
            frame_embeddings = rand_embs(n, 768)
            temporal_embeddings = rand_embs(n, 1024)  # 1024-d V-JEPA
            transcript = []

        idx = MockIndex()
        idx.embed_fn = embed_fn
        idx.visual_embed_fn = embed_fn
        return idx

    def test_all_field_returns_results(self):
        """field='all' with mismatched temporal dims must not raise."""
        idx = self._make_index()
        tool = make_search_video(idx)["tool"]
        results = tool("test query", field="all", top_k=3)
        assert isinstance(results, list)
        assert len(results) > 0

    def test_all_field_no_value_error(self):
        """Verify no ValueError when temporal embeddings have higher dimensionality."""
        idx = self._make_index()
        tool = make_search_video(idx)["tool"]
        try:
            tool("action scene", field="all")
        except ValueError as e:
            pytest.fail(f"search_video raised ValueError on field='all': {e}")

    def test_all_field_result_schema(self):
        """Each result has required keys."""
        idx = self._make_index()
        tool = make_search_video(idx)["tool"]
        results = tool("running", field="all", top_k=2)
        for r in results:
            assert "start_time" in r
            assert "end_time" in r
            assert "score" in r
