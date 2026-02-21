"""Tests for video search tool factory functions."""

import sys
from unittest.mock import MagicMock

import numpy as np

# Ensure sklearn is available (real or shimmed) so the lazy import inside
# make_search_video succeeds even in environments without scikit-learn.
try:
    from sklearn.metrics.pairwise import cosine_similarity as _real_cos_sim  # noqa: F401
except ImportError:
    # Provide a minimal shim so the production import does not fail.
    def _cosine_similarity(a, b):
        a = np.asarray(a)
        b = np.asarray(b)
        a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-10)
        b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-10)
        return a_norm @ b_norm.T

    _pairwise_mod = type(sys)("sklearn.metrics.pairwise")
    _pairwise_mod.cosine_similarity = _cosine_similarity
    _metrics_mod = type(sys)("sklearn.metrics")
    _metrics_mod.pairwise = _pairwise_mod
    _sklearn_mod = type(sys)("sklearn")
    _sklearn_mod.metrics = _metrics_mod

    class _KMeans:
        def __init__(self, n_clusters=2, random_state=None, n_init=10):
            self.n_clusters = n_clusters

        def fit_predict(self, X):
            X = np.asarray(X)
            return np.array([i % self.n_clusters for i in range(len(X))])

    _cluster_mod = type(sys)("sklearn.cluster")
    _cluster_mod.KMeans = _KMeans
    sys.modules.setdefault("sklearn", _sklearn_mod)
    sys.modules.setdefault("sklearn.metrics", _metrics_mod)
    sys.modules.setdefault("sklearn.metrics.pairwise", _pairwise_mod)
    sys.modules.setdefault("sklearn.cluster", _cluster_mod)

from kuavi.search import (
    make_discriminative_vqa,
    make_get_scene_list,
    make_get_transcript,
    make_search_transcript,
    make_search_video,
)


def _make_index(
    segments=None,
    embeddings=None,
    action_embeddings=None,
    transcript=None,
    embed_fn=None,
    frame_embeddings=None,
    temporal_embeddings=None,
    visual_embed_fn=None,
):
    """Create a mock VideoIndex with the given attributes."""
    index = MagicMock()
    index.segments = segments or []
    index.embeddings = embeddings
    index.action_embeddings = action_embeddings
    index.frame_embeddings = frame_embeddings
    index.temporal_embeddings = temporal_embeddings
    index.visual_embed_fn = visual_embed_fn
    index.transcript = transcript or []
    index.embed_fn = embed_fn
    return index


# ---------------------------------------------------------------
# search_video
# ---------------------------------------------------------------


class TestSearchVideo:
    def test_basic_semantic_search(self):
        # Segment 0 embedding close to query, segment 1 far away
        emb0 = np.array([1.0, 0.0, 0.0])
        emb1 = np.array([0.0, 1.0, 0.0])
        embeddings = np.stack([emb0, emb1])

        segments = [
            {"start_time": 0.0, "end_time": 5.0, "caption": "a cat"},
            {"start_time": 5.0, "end_time": 10.0, "caption": "a dog"},
        ]

        # embed_fn returns vector close to emb0
        embed_fn = MagicMock(return_value=np.array([0.9, 0.1, 0.0]))

        index = _make_index(
            segments=segments,
            embeddings=embeddings,
            embed_fn=embed_fn,
        )

        tool_dict = make_search_video(index)
        search_video = tool_dict["tool"]
        results = search_video("cat", top_k=2)

        assert len(results) == 2
        # First result should be segment 0 (higher similarity)
        assert results[0]["caption"] == "a cat"
        assert results[0]["score"] > results[1]["score"]
        embed_fn.assert_called_once_with("cat")

    def test_empty_embeddings_returns_empty(self):
        index = _make_index(embeddings=None)
        tool_dict = make_search_video(index)
        assert tool_dict["tool"]("query") == []

    def test_empty_embeddings_array_returns_empty(self):
        index = _make_index(embeddings=np.array([]))
        tool_dict = make_search_video(index)
        assert tool_dict["tool"]("query") == []

    def test_top_k_limits_results(self):
        n = 5
        embeddings = np.random.default_rng(42).random((n, 3))
        segments = [
            {"start_time": float(i), "end_time": float(i + 1), "caption": f"seg{i}"}
            for i in range(n)
        ]
        embed_fn = MagicMock(return_value=np.array([1.0, 0.0, 0.0]))
        index = _make_index(segments=segments, embeddings=embeddings, embed_fn=embed_fn)

        results = make_search_video(index)["tool"]("q", top_k=2)
        assert len(results) == 2

    def test_field_action_uses_action_embeddings(self):
        """field='action' should search against action_embeddings."""
        summary_emb = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        action_emb = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

        segments = [
            {"start_time": 0.0, "end_time": 5.0, "caption": "seg0"},
            {"start_time": 5.0, "end_time": 10.0, "caption": "seg1"},
        ]
        # Query close to action_emb[0]
        embed_fn = MagicMock(return_value=np.array([0.9, 0.1, 0.0]))
        index = _make_index(
            segments=segments,
            embeddings=summary_emb,
            action_embeddings=action_emb,
            embed_fn=embed_fn,
        )

        results = make_search_video(index)["tool"]("running", top_k=2, field="action")
        assert results[0]["caption"] == "seg0"
        assert results[0]["score"] > results[1]["score"]

    def test_field_all_weighted_composite(self):
        """field='all' should use weighted composite (summary 0.4, action 0.2, visual 0.2, temporal 0.2)."""
        summary_emb = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        action_emb = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])

        segments = [
            {"start_time": 0.0, "end_time": 5.0, "caption": "seg0"},
            {"start_time": 5.0, "end_time": 10.0, "caption": "seg1"},
        ]
        # Query [1, 0, 0]: seg0 has summary=1.0, action=0.0; seg1 has summary=0.0, action=1.0
        embed_fn = MagicMock(return_value=np.array([1.0, 0.0, 0.0]))
        index = _make_index(
            segments=segments,
            embeddings=summary_emb,
            action_embeddings=action_emb,
            embed_fn=embed_fn,
        )

        results = make_search_video(index)["tool"]("q", top_k=2, field="all")
        assert len(results) == 2
        # With only summary (0.4) and action (0.2) available:
        # seg0: (0.4*1.0 + 0.2*0.0) / 0.6 = 0.667
        # seg1: (0.4*0.0 + 0.2*1.0) / 0.6 = 0.333
        assert results[0]["caption"] == "seg0"
        assert results[0]["score"] > results[1]["score"]

    def test_result_includes_annotation(self):
        """Results should include the annotation dict when present."""
        annotation = {
            "summary": {"brief": "A cat", "detailed": "A cat sitting."},
            "action": {"brief": "sitting", "detailed": "sitting still", "actor": "cat"},
        }
        segments = [
            {"start_time": 0.0, "end_time": 5.0, "caption": "A cat", "annotation": annotation},
        ]
        embeddings = np.array([[1.0, 0.0]])
        embed_fn = MagicMock(return_value=np.array([1.0, 0.0]))
        index = _make_index(segments=segments, embeddings=embeddings, embed_fn=embed_fn)

        results = make_search_video(index)["tool"]("cat", top_k=1)
        assert results[0]["annotation"] == annotation

    def test_field_action_fallback_when_action_embeddings_none(self):
        """field='action' falls back to summary embeddings when action_embeddings is None."""
        summary_emb = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        segments = [
            {"start_time": 0.0, "end_time": 5.0, "caption": "seg0"},
            {"start_time": 5.0, "end_time": 10.0, "caption": "seg1"},
        ]
        embed_fn = MagicMock(return_value=np.array([0.9, 0.1, 0.0]))
        index = _make_index(
            segments=segments,
            embeddings=summary_emb,
            action_embeddings=None,
            embed_fn=embed_fn,
        )

        # Should not crash — falls back to summary embeddings
        results = make_search_video(index)["tool"]("q", top_k=2, field="action")
        assert len(results) == 2

    def test_field_summary_default_uses_embeddings(self):
        """field='summary' (default) uses index.embeddings."""
        summary_emb = np.array([[1.0, 0.0], [0.0, 1.0]])
        action_emb = np.array([[0.0, 1.0], [1.0, 0.0]])
        segments = [
            {"start_time": 0.0, "end_time": 5.0, "caption": "seg0"},
            {"start_time": 5.0, "end_time": 10.0, "caption": "seg1"},
        ]
        # Query aligned with summary_emb[0]
        embed_fn = MagicMock(return_value=np.array([1.0, 0.0]))
        index = _make_index(
            segments=segments,
            embeddings=summary_emb,
            action_embeddings=action_emb,
            embed_fn=embed_fn,
        )

        # Default field="summary": seg0 should rank highest
        results = make_search_video(index)["tool"]("q", top_k=2)
        assert results[0]["caption"] == "seg0"

    def test_caption_field_always_present(self):
        """Backward compat: results always include a caption field."""
        segments = [
            {"start_time": 0.0, "end_time": 5.0, "caption": "hello"},
        ]
        embeddings = np.array([[1.0, 0.0]])
        embed_fn = MagicMock(return_value=np.array([1.0, 0.0]))
        index = _make_index(segments=segments, embeddings=embeddings, embed_fn=embed_fn)

        results = make_search_video(index)["tool"]("q", top_k=1)
        assert "caption" in results[0]
        assert results[0]["caption"] == "hello"


# ---------------------------------------------------------------
# search_transcript
# ---------------------------------------------------------------


class TestSearchTranscript:
    def test_keyword_match(self):
        transcript = [
            {"start_time": 0.0, "end_time": 2.0, "text": "Hello world"},
            {"start_time": 2.0, "end_time": 4.0, "text": "Goodbye moon"},
            {"start_time": 4.0, "end_time": 6.0, "text": "Hello again"},
        ]
        index = _make_index(transcript=transcript)
        search = make_search_transcript(index)["tool"]

        results = search("hello")
        assert len(results) == 2
        assert results[0]["text"] == "Hello world"
        assert results[1]["text"] == "Hello again"

    def test_case_insensitive(self):
        transcript = [{"start_time": 0.0, "end_time": 1.0, "text": "FOO bar"}]
        index = _make_index(transcript=transcript)
        results = make_search_transcript(index)["tool"]("foo")
        assert len(results) == 1

    def test_context_includes_neighbours(self):
        transcript = [
            {"start_time": 0.0, "end_time": 1.0, "text": "before"},
            {"start_time": 1.0, "end_time": 2.0, "text": "match keyword"},
            {"start_time": 2.0, "end_time": 3.0, "text": "after"},
        ]
        index = _make_index(transcript=transcript)
        results = make_search_transcript(index)["tool"]("keyword")
        assert len(results) == 1
        context = results[0]["context"]
        assert "before" in context
        assert "match keyword" in context
        assert "after" in context

    def test_empty_transcript(self):
        index = _make_index(transcript=[])
        results = make_search_transcript(index)["tool"]("anything")
        assert results == []

    def test_no_match(self):
        transcript = [{"start_time": 0.0, "end_time": 1.0, "text": "Hello"}]
        index = _make_index(transcript=transcript)
        results = make_search_transcript(index)["tool"]("xyz")
        assert results == []


# ---------------------------------------------------------------
# get_transcript
# ---------------------------------------------------------------


class TestGetTranscript:
    def test_time_range_filter(self):
        transcript = [
            {"start_time": 0.0, "end_time": 2.0, "text": "first"},
            {"start_time": 2.0, "end_time": 4.0, "text": "second"},
            {"start_time": 4.0, "end_time": 6.0, "text": "third"},
            {"start_time": 6.0, "end_time": 8.0, "text": "fourth"},
        ]
        index = _make_index(transcript=transcript)
        get_fn = make_get_transcript(index)["tool"]

        result = get_fn(1.5, 5.0)
        assert "first" in result
        assert "second" in result
        assert "third" in result
        assert "fourth" not in result

    def test_empty_transcript_returns_empty_string(self):
        index = _make_index(transcript=[])
        result = make_get_transcript(index)["tool"](0.0, 10.0)
        assert result == ""

    def test_no_overlap_returns_empty(self):
        transcript = [{"start_time": 10.0, "end_time": 12.0, "text": "late"}]
        index = _make_index(transcript=transcript)
        result = make_get_transcript(index)["tool"](0.0, 5.0)
        assert result == ""


# ---------------------------------------------------------------
# get_scene_list
# ---------------------------------------------------------------


class TestGetSceneList:
    def test_returns_all_segments(self):
        segments = [
            {"start_time": 0.0, "end_time": 5.0, "caption": "intro"},
            {"start_time": 5.0, "end_time": 10.0, "caption": "main"},
        ]
        index = _make_index(segments=segments)
        scenes = make_get_scene_list(index)["tool"]()

        assert len(scenes) == 2
        assert scenes[0]["scene_index"] == 0
        assert scenes[0]["start_time"] == 0.0
        assert scenes[0]["caption"] == "intro"
        assert scenes[1]["scene_index"] == 1

    def test_empty_segments(self):
        index = _make_index(segments=[])
        scenes = make_get_scene_list(index)["tool"]()
        assert scenes == []

    def test_missing_caption_defaults_to_empty(self):
        segments = [{"start_time": 0.0, "end_time": 5.0}]
        index = _make_index(segments=segments)
        scenes = make_get_scene_list(index)["tool"]()
        assert scenes[0]["caption"] == ""

    def test_returns_annotations(self):
        """get_scene_list should include annotation when present."""
        annotation = {
            "summary": {"brief": "A cat", "detailed": "A cat sitting."},
            "action": {"brief": "sitting", "detailed": "sitting still", "actor": "cat"},
        }
        segments = [
            {"start_time": 0.0, "end_time": 5.0, "caption": "A cat", "annotation": annotation},
        ]
        index = _make_index(segments=segments)
        scenes = make_get_scene_list(index)["tool"]()
        assert scenes[0]["annotation"] == annotation

    def test_description_present(self):
        index = _make_index()
        tool_dict = make_get_scene_list(index)
        assert "description" in tool_dict
        assert isinstance(tool_dict["description"], str)


# ---------------------------------------------------------------
# discriminative_vqa
# ---------------------------------------------------------------


class TestDiscriminativeVqa:
    def _build_index(self):
        """Build a mock index with 3 segments and known embeddings."""
        segments = [
            {"start_time": 0.0, "end_time": 5.0, "caption": "person cooking"},
            {"start_time": 5.0, "end_time": 10.0, "caption": "person reading"},
            {"start_time": 10.0, "end_time": 15.0, "caption": "person exercising"},
        ]
        # Each segment embedding is a unit vector along a different axis
        embeddings = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )

        def embed_fn(text: str) -> np.ndarray:
            # Return embedding close to segment 0 for "cooking", segment 1 for "reading", etc.
            if "cooking" in text:
                return np.array([0.9, 0.1, 0.0])
            elif "reading" in text:
                return np.array([0.1, 0.9, 0.0])
            elif "exercising" in text:
                return np.array([0.0, 0.1, 0.9])
            return np.array([0.33, 0.33, 0.33])

        return _make_index(
            segments=segments,
            embeddings=embeddings,
            embed_fn=embed_fn,
        )

    def test_basic_multiple_choice(self):
        index = self._build_index()
        tool_dict = make_discriminative_vqa(index)
        vqa = tool_dict["tool"]

        results = vqa("What is the person doing?", ["cooking", "reading", "exercising"])
        assert len(results) == 3
        # Results should be sorted by confidence descending
        assert results[0]["confidence"] >= results[1]["confidence"]
        assert results[1]["confidence"] >= results[2]["confidence"]
        # Each result has the expected keys
        for r in results:
            assert "answer" in r
            assert "confidence" in r
            assert "best_segment" in r
            assert "start_time" in r["best_segment"]
            assert "end_time" in r["best_segment"]
            assert "caption" in r["best_segment"]

    def test_returns_empty_for_no_candidates(self):
        index = self._build_index()
        vqa = make_discriminative_vqa(index)["tool"]
        assert vqa("question", []) == []

    def test_returns_empty_when_no_embeddings(self):
        index = _make_index(embeddings=None)
        vqa = make_discriminative_vqa(index)["tool"]
        assert vqa("question", ["a", "b"]) == []

    def test_returns_empty_when_no_embed_fn(self):
        index = _make_index(
            embeddings=np.array([[1.0, 0.0]]),
            embed_fn=None,
            segments=[{"start_time": 0.0, "end_time": 5.0}],
        )
        vqa = make_discriminative_vqa(index)["tool"]
        assert vqa("question", ["a", "b"]) == []

    def test_time_range_filter(self):
        index = self._build_index()
        vqa = make_discriminative_vqa(index)["tool"]

        # Only look at segments 0-4.9s (segment 0: cooking)
        results = vqa("What is happening?", ["cooking", "reading"], time_range=(0.0, 4.9))
        assert len(results) == 2
        # Both should reference segment 0 since it's the only active one
        for r in results:
            assert r["best_segment"]["start_time"] == 0.0
            assert r["best_segment"]["end_time"] == 5.0

    def test_time_range_no_overlap_returns_empty(self):
        index = self._build_index()
        vqa = make_discriminative_vqa(index)["tool"]
        results = vqa("question", ["a", "b"], time_range=(100.0, 200.0))
        assert results == []

    def test_description_present(self):
        index = self._build_index()
        tool_dict = make_discriminative_vqa(index)
        assert "description" in tool_dict
        assert "multiple-choice" in tool_dict["description"]


# ---------------------------------------------------------------
# Visual field search (Feature 3)
# ---------------------------------------------------------------


class TestSearchVideoVisualField:
    """Tests for field='visual' search using frame embeddings."""

    def test_visual_field_basic(self):
        """field='visual' should search against frame_embeddings."""
        frame_emb0 = np.array([1.0, 0.0, 0.0])
        frame_emb1 = np.array([0.0, 1.0, 0.0])
        frame_embeddings = np.stack([frame_emb0, frame_emb1])

        segments = [
            {"start_time": 0.0, "end_time": 5.0, "caption": "a cat"},
            {"start_time": 5.0, "end_time": 10.0, "caption": "a dog"},
        ]

        # visual_embed_fn returns vector close to frame_emb0
        visual_fn = MagicMock(return_value=np.array([0.9, 0.1, 0.0]))
        embed_fn = MagicMock(return_value=np.array([0.1, 0.9, 0.0]))

        index = _make_index(
            segments=segments,
            embeddings=np.stack([frame_emb0, frame_emb1]),
            embed_fn=embed_fn,
        )
        index.frame_embeddings = frame_embeddings
        index.visual_embed_fn = visual_fn

        results = make_search_video(index)["tool"]("cat", top_k=2, field="visual")

        assert len(results) == 2
        assert results[0]["caption"] == "a cat"
        visual_fn.assert_called_once_with("cat")

    def test_visual_field_fallback_to_summary(self):
        """field='visual' without frame_embeddings should fall back to summary."""
        segments = [
            {"start_time": 0.0, "end_time": 5.0, "caption": "a cat"},
        ]
        embeddings = np.array([[1.0, 0.0]])
        embed_fn = MagicMock(return_value=np.array([1.0, 0.0]))

        index = _make_index(segments=segments, embeddings=embeddings, embed_fn=embed_fn)
        index.frame_embeddings = None

        results = make_search_video(index)["tool"]("cat", top_k=1, field="visual")
        assert len(results) == 1
        assert results[0]["caption"] == "a cat"

    def test_visual_field_excludes_duplicates(self):
        """field='visual' should suppress duplicate segments."""
        frame_embeddings = np.array([[1.0, 0.0], [0.9, 0.1]])
        segments = [
            {"start_time": 0.0, "end_time": 5.0, "caption": "a cat"},
            {"start_time": 5.0, "end_time": 10.0, "caption": "a cat dup", "is_duplicate": True},
        ]

        visual_fn = MagicMock(return_value=np.array([1.0, 0.0]))
        index = _make_index(
            segments=segments, embeddings=frame_embeddings, embed_fn=visual_fn
        )
        index.frame_embeddings = frame_embeddings
        index.visual_embed_fn = visual_fn

        results = make_search_video(index)["tool"]("cat", top_k=2, field="visual")
        # First result should be non-duplicate, second should have -inf score
        assert results[0]["caption"] == "a cat"

    def test_description_mentions_visual(self):
        """Tool description should mention 'visual' field."""
        index = _make_index(embeddings=np.array([[1.0]]))
        tool_dict = make_search_video(index)
        assert "visual" in tool_dict["description"]


# ---------------------------------------------------------------
# Frame embeddings bypass caption quality bottleneck
# ---------------------------------------------------------------


class TestFrameEmbeddingsBypassCaptionQuality:
    """Prove that field='visual' bypasses bad/misleading captions."""

    def test_bad_captions_good_frames(self):
        """Visual search finds the right segment even when captions are misleading.

        Setup: segment 0 caption says "a dog running" but its *frame*
        embedding actually matches "cat".  Summary embeddings match the
        (wrong) caption text.  Visual search should find segment 0 for
        "cat" while summary search should *not*.
        """
        segments = [
            {"start_time": 0.0, "end_time": 5.0, "caption": "a dog running"},
            {"start_time": 5.0, "end_time": 10.0, "caption": "a cat sleeping"},
            {"start_time": 10.0, "end_time": 15.0, "caption": "a bird flying"},
        ]

        # Summary (caption) embeddings — aligned with the caption text:
        #   seg0 → "dog" direction, seg1 → "cat" direction, seg2 → "bird"
        summary_emb = np.array([
            [1.0, 0.0, 0.0],  # dog direction
            [0.0, 1.0, 0.0],  # cat direction
            [0.0, 0.0, 1.0],  # bird direction
        ])

        # Frame embeddings — reflect the *actual visual content*:
        #   seg0 actually shows a cat, seg1 shows a dog, seg2 shows a bird
        frame_emb = np.array([
            [0.0, 1.0, 0.0],  # cat direction (mislabelled as dog)
            [1.0, 0.0, 0.0],  # dog direction (mislabelled as cat)
            [0.0, 0.0, 1.0],  # bird direction (correct)
        ])

        # Query "cat" → cat direction [0, 1, 0]
        cat_query = np.array([0.0, 0.95, 0.05])

        visual_fn = MagicMock(return_value=cat_query)
        embed_fn = MagicMock(return_value=cat_query)

        index = _make_index(
            segments=segments,
            embeddings=summary_emb,
            embed_fn=embed_fn,
        )
        index.frame_embeddings = frame_emb
        index.visual_embed_fn = visual_fn

        search = make_search_video(index)["tool"]

        # Visual search: should find seg0 (frame embedding matches "cat")
        visual_results = search("cat", top_k=3, field="visual")
        assert visual_results[0]["caption"] == "a dog running"  # seg0
        assert visual_results[0]["start_time"] == 0.0

        # Summary search: should find seg1 (caption embedding matches "cat")
        summary_results = search("cat", top_k=3, field="summary")
        assert summary_results[0]["caption"] == "a cat sleeping"  # seg1
        assert summary_results[0]["start_time"] == 5.0

        # The key insight: visual and summary searches disagree on the top
        # result, proving visual search bypasses caption quality.
        assert visual_results[0]["caption"] != summary_results[0]["caption"]

    def test_visual_search_independent_of_caption_text(self):
        """Visual search differentiates segments even when all captions are identical.

        When every segment has the same generic caption, summary embeddings
        are identical and cannot distinguish between segments.  Frame
        embeddings, derived from actual pixels, remain distinct.
        """
        segments = [
            {"start_time": 0.0, "end_time": 5.0, "caption": "generic scene"},
            {"start_time": 5.0, "end_time": 10.0, "caption": "generic scene"},
            {"start_time": 10.0, "end_time": 15.0, "caption": "generic scene"},
        ]

        # Summary embeddings are all the same (caption text is identical)
        same_vec = np.array([0.577, 0.577, 0.577])
        summary_emb = np.stack([same_vec, same_vec, same_vec])

        # Frame embeddings are distinct (actual visual content differs)
        frame_emb = np.array([
            [1.0, 0.0, 0.0],  # outdoor scene
            [0.0, 1.0, 0.0],  # indoor scene
            [0.0, 0.0, 1.0],  # underwater scene
        ])

        # Query targets seg1 (indoor) direction
        indoor_query = np.array([0.05, 0.95, 0.0])

        visual_fn = MagicMock(return_value=indoor_query)
        embed_fn = MagicMock(return_value=indoor_query)

        index = _make_index(
            segments=segments,
            embeddings=summary_emb,
            embed_fn=embed_fn,
        )
        index.frame_embeddings = frame_emb
        index.visual_embed_fn = visual_fn

        search = make_search_video(index)["tool"]

        # Visual search: can identify seg1 as the best match
        visual_results = search("indoor scene", top_k=3, field="visual")
        assert visual_results[0]["start_time"] == 5.0  # seg1
        # Scores should be meaningfully different
        assert visual_results[0]["score"] > visual_results[1]["score"]
        assert visual_results[0]["score"] > visual_results[2]["score"]

        # Summary search: all scores are essentially identical (can't differentiate)
        summary_results = search("indoor scene", top_k=3, field="summary")
        score_spread = abs(summary_results[0]["score"] - summary_results[-1]["score"])
        assert score_spread < 0.01, (
            f"Summary scores should be nearly identical but spread was {score_spread}"
        )


# ---------------------------------------------------------------
# Multi-scale search efficiency (Feature: coarse hierarchy)
# ---------------------------------------------------------------


class TestMultiScaleSearchEfficiency:
    """Verify that multi-scale search reduces search space via coarse hierarchy."""

    def test_coarse_level_fewer_segments(self):
        """_build_coarse_level merges 30 fine 5s segments into 5 coarse 30s segments."""
        from kuavi.indexer import VideoIndexer

        n_fine = 30
        seg_dur = 5.0
        segments = [
            {
                "start_time": float(i * seg_dur),
                "end_time": float((i + 1) * seg_dur),
                "caption": f"seg{i}",
            }
            for i in range(n_fine)
        ]
        embeddings = np.random.default_rng(42).random((n_fine, 8))
        # Normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.maximum(norms, 1e-10)

        indexer = VideoIndexer()
        coarse_segs, coarse_embs = indexer._build_coarse_level(
            segments, embeddings, target_duration=30.0
        )

        # 150s total / 30s target = 5 coarse segments
        assert len(coarse_segs) == 5
        assert coarse_embs is not None
        assert coarse_embs.shape[0] == 5

        # Each coarse segment should span 30s
        for seg in coarse_segs:
            duration = seg["end_time"] - seg["start_time"]
            assert duration == 30.0, f"Expected 30s, got {duration}s"

    def test_coarse_search_returns_broader_ranges(self):
        """Searching at level=1 returns wider time ranges than level=0."""
        dim = 4
        rng = np.random.default_rng(99)

        # Fine segments: 10 segments of 5s each
        n_fine = 10
        fine_segments = [
            {
                "start_time": float(i * 5),
                "end_time": float((i + 1) * 5),
                "caption": f"fine{i}",
            }
            for i in range(n_fine)
        ]
        fine_embs = rng.random((n_fine, dim)).astype(np.float32)
        norms = np.linalg.norm(fine_embs, axis=1, keepdims=True)
        fine_embs = fine_embs / np.maximum(norms, 1e-10)

        # Coarse segments: 2 segments of 25s each
        coarse_segments = [
            {"start_time": 0.0, "end_time": 25.0, "caption": "coarse0"},
            {"start_time": 25.0, "end_time": 50.0, "caption": "coarse1"},
        ]
        coarse_embs = rng.random((2, dim)).astype(np.float32)
        cnorms = np.linalg.norm(coarse_embs, axis=1, keepdims=True)
        coarse_embs = coarse_embs / np.maximum(cnorms, 1e-10)

        embed_fn = MagicMock(return_value=rng.random(dim).astype(np.float32))

        index = _make_index(
            segments=fine_segments,
            embeddings=fine_embs,
            embed_fn=embed_fn,
        )
        index.segment_hierarchy = [coarse_segments]
        index.hierarchy_embeddings = [coarse_embs]

        tool = make_search_video(index)["tool"]

        results_fine = tool("query", top_k=3, level=0)
        results_coarse = tool("query", top_k=2, level=1)

        # Fine results have 5s ranges, coarse have 25s ranges
        fine_widths = [r["end_time"] - r["start_time"] for r in results_fine]
        coarse_widths = [r["end_time"] - r["start_time"] for r in results_coarse]

        assert all(w == 5.0 for w in fine_widths)
        assert all(w == 25.0 for w in coarse_widths)
        assert min(coarse_widths) > max(fine_widths)

    def test_search_space_reduction_ratio(self):
        """Coarse level has at least 3x fewer segments than fine level for 150s/5s video."""
        from kuavi.indexer import VideoIndexer

        n_fine = 30  # 30 segments * 5s = 150s
        seg_dur = 5.0
        segments = [
            {
                "start_time": float(i * seg_dur),
                "end_time": float((i + 1) * seg_dur),
                "caption": f"seg{i}",
            }
            for i in range(n_fine)
        ]
        embeddings = np.random.default_rng(7).random((n_fine, 8))
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.maximum(norms, 1e-10)

        indexer = VideoIndexer()
        coarse_segs, coarse_embs = indexer._build_coarse_level(
            segments, embeddings, target_duration=30.0
        )

        # 30 fine / 5 coarse = 6x reduction, well above 3x minimum
        reduction = n_fine / len(coarse_segs)
        assert reduction >= 3.0, f"Expected >=3x reduction, got {reduction}x"
        assert coarse_embs is not None
        assert coarse_embs.shape[0] == len(coarse_segs)
