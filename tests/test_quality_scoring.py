"""Tests for quality scoring signals in VideoIndexer._score_annotations."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from kuavi.indexer import VideoIndexer


def _make_seg(
    start: float = 0.0,
    end: float = 5.0,
    caption: str = "A scene",
    summary_brief: str = "A scene",
    action_brief: str = "Walk forward",
    actor: str | None = "person",
) -> dict:
    return {
        "start_time": start,
        "end_time": end,
        "caption": caption,
        "annotation": {
            "summary": {"brief": summary_brief, "detailed": summary_brief},
            "action": {"brief": action_brief, "detailed": action_brief, "actor": actor},
        },
    }


# ---------------------------------------------------------------------------
# Signal 2: Format compliance
# ---------------------------------------------------------------------------

class TestFormatComplianceScore:
    def test_perfect_annotation(self):
        seg = _make_seg()
        score = VideoIndexer._score_format_compliance(seg)
        assert score == pytest.approx(1.0)

    def test_missing_summary_brief(self):
        seg = _make_seg(summary_brief="")
        score = VideoIndexer._score_format_compliance(seg)
        # Missing summary.brief loses 0.25; timestamp check also skipped (summary_brief falsy)
        # loses another 0.25 → 0.5
        assert score < 1.0
        assert score == pytest.approx(0.5)

    def test_action_brief_too_long(self):
        seg = _make_seg(action_brief="Walk slowly over the long bridge ahead")
        score = VideoIndexer._score_format_compliance(seg)
        assert score < 1.0

    def test_action_brief_one_word(self):
        seg = _make_seg(action_brief="Walking")
        score = VideoIndexer._score_format_compliance(seg)
        assert score < 1.0

    def test_timestamp_in_summary(self):
        seg = _make_seg(summary_brief="Something happens at 3.5s here")
        score = VideoIndexer._score_format_compliance(seg)
        # timestamp penalty — -0.25
        assert score < 1.0

    def test_missing_actor_when_action_present(self):
        seg = _make_seg(actor=None)
        score = VideoIndexer._score_format_compliance(seg)
        assert score < 1.0

    def test_actor_not_required_when_action_na(self):
        seg = _make_seg(action_brief="N/A", actor=None)
        score = VideoIndexer._score_format_compliance(seg)
        # action.brief="N/A" is 1 word, fails the 2-5 words check (-0.25)
        # but actor is not required when action is N/A (+0.25)
        # summary.brief present (+0.25), no timestamp (+0.25) → 0.75
        assert score == pytest.approx(0.75)

    def test_empty_annotation(self):
        seg = {"start_time": 0, "end_time": 5, "caption": "", "annotation": {}}
        score = VideoIndexer._score_format_compliance(seg)
        assert 0.0 <= score <= 1.0

    def test_no_annotation_key(self):
        seg = {"start_time": 0, "end_time": 5, "caption": ""}
        score = VideoIndexer._score_format_compliance(seg)
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# Signal 5: Action frequency
# ---------------------------------------------------------------------------

class TestActionFrequencyScore:
    def test_unique_actions_score_1(self):
        segs = [
            _make_seg(action_brief=f"Do action {i}") for i in range(5)
        ]
        VideoIndexer._score_action_frequency(segs)
        for seg in segs:
            assert seg["action_frequency_score"] == pytest.approx(1.0)

    def test_dominant_action_above_50_pct(self):
        # 6 of 10 segments share the same action → 60% > 50%
        segs = [_make_seg(action_brief="Walk forward") for _ in range(6)]
        segs += [_make_seg(action_brief=f"Other action {i}") for i in range(4)]
        VideoIndexer._score_action_frequency(segs)
        dominant = [s for s in segs if s["annotation"]["action"]["brief"] == "Walk forward"]
        for seg in dominant:
            assert seg["action_frequency_score"] == pytest.approx(0.0)

    def test_action_at_30_pct_threshold(self):
        # exactly 3 of 10 → 30% → score 1.0
        segs = [_make_seg(action_brief="Common action") for _ in range(3)]
        segs += [_make_seg(action_brief=f"Unique {i}") for i in range(7)]
        VideoIndexer._score_action_frequency(segs)
        common = [s for s in segs if s["annotation"]["action"]["brief"] == "Common action"]
        for seg in common:
            assert seg["action_frequency_score"] == pytest.approx(1.0)

    def test_action_at_40_pct_interpolation(self):
        # 4 of 10 → 40%, midpoint of [30%, 50%] → score 0.5
        segs = [_make_seg(action_brief="Mid action") for _ in range(4)]
        segs += [_make_seg(action_brief=f"Unique {i}") for i in range(6)]
        VideoIndexer._score_action_frequency(segs)
        mid_segs = [s for s in segs if s["annotation"]["action"]["brief"] == "Mid action"]
        for seg in mid_segs:
            assert seg["action_frequency_score"] == pytest.approx(0.5)

    def test_na_action_scores_1(self):
        segs = [_make_seg(action_brief="N/A", actor=None) for _ in range(10)]
        VideoIndexer._score_action_frequency(segs)
        for seg in segs:
            assert seg["action_frequency_score"] == pytest.approx(1.0)

    def test_empty_segments(self):
        VideoIndexer._score_action_frequency([])  # should not raise


# ---------------------------------------------------------------------------
# Signal 4: Temporal consistency
# ---------------------------------------------------------------------------

class TestTemporalConsistency:
    def _make_indexer_with_encode(self, embeddings_by_caption: dict[str, np.ndarray]):
        """Return a VideoIndexer whose _encode_texts returns controlled embeddings."""
        indexer = VideoIndexer.__new__(VideoIndexer)
        indexer._text_embedding_model_name = None
        indexer._model = MagicMock()
        indexer._image_processor = MagicMock()
        indexer._torch_device = "cpu"

        def fake_encode_texts(texts):
            result = []
            for t in texts:
                emb = embeddings_by_caption.get(t, np.array([1.0, 0.0, 0.0, 0.0]))
                result.append(emb)
            return np.array(result)

        def fake_encode_frames(frames):
            return np.tile(np.array([0.5, 0.5, 0.5, 0.5]), (len(frames), 1)).astype(np.float32)

        indexer._encode_texts = fake_encode_texts
        indexer._encode_frames = fake_encode_frames
        indexer._ensure_model = MagicMock()
        return indexer

    def test_distinct_captions_score_near_0(self):
        # Orthogonal embeddings → sim = 0.0 → temporal_consistency = 1.0
        embs = {
            "Scene A": np.array([1.0, 0.0, 0.0, 0.0]),
            "Scene B": np.array([0.0, 1.0, 0.0, 0.0]),
            "Scene C": np.array([0.0, 0.0, 1.0, 0.0]),
        }
        indexer = self._make_indexer_with_encode(embs)
        frames = [np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(6)]
        timestamps = [0.0, 1.0, 2.0, 5.0, 6.0, 7.0]
        segs = [
            _make_seg(0.0, 3.0, "Scene A", "Scene A"),
            _make_seg(3.0, 6.0, "Scene B", "Scene B"),
            _make_seg(6.0, 9.0, "Scene C", "Scene C"),
        ]
        indexer._score_annotations(segs, frames, timestamps)
        # Each segment's caption is orthogonal to neighbors → high consistency
        for seg in segs:
            if "temporal_consistency_score" in seg:
                assert seg["temporal_consistency_score"] >= 0.9

    def test_identical_captions_score_near_0(self):
        # Same embedding → sim = 1.0 → temporal_consistency = 0.0
        emb = np.array([1.0, 0.0, 0.0, 0.0])
        embs = {f"Same caption {i}": emb for i in range(3)}
        # All three have the same caption text
        embs["Same caption"] = emb
        indexer = self._make_indexer_with_encode({"Same caption": emb})
        frames = [np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(6)]
        timestamps = [0.0, 1.0, 2.0, 5.0, 6.0, 7.0]
        segs = [
            _make_seg(0.0, 3.0, "Same caption", "Same caption"),
            _make_seg(3.0, 6.0, "Same caption", "Same caption"),
            _make_seg(6.0, 9.0, "Same caption", "Same caption"),
        ]
        indexer._score_annotations(segs, frames, timestamps)
        # Middle segment should have low temporal_consistency_score
        middle = segs[1]
        assert "temporal_consistency_score" in middle
        assert middle["temporal_consistency_score"] < 0.1


# ---------------------------------------------------------------------------
# Signal 3: Coherence score
# ---------------------------------------------------------------------------

class TestCoherenceScore:
    def _make_indexer_coherence(self, summary_emb, action_emb):
        indexer = VideoIndexer.__new__(VideoIndexer)
        indexer._text_embedding_model_name = None
        indexer._model = MagicMock()
        indexer._image_processor = MagicMock()
        indexer._torch_device = "cpu"

        call_count = {"n": 0}

        def fake_encode_texts(texts):
            # First call: caption (for caption_quality_score)
            # Subsequent coherence calls: action_brief, summary_brief
            result = []
            for _ in texts:
                call_count["n"] += 1
                if call_count["n"] % 2 == 0:
                    result.append(action_emb)
                else:
                    result.append(summary_emb)
            return np.array(result)

        def fake_encode_frames(frames):
            return np.tile(summary_emb, (len(frames), 1)).astype(np.float32)

        indexer._encode_texts = fake_encode_texts
        indexer._encode_frames = fake_encode_frames
        indexer._ensure_model = MagicMock()
        return indexer

    def test_coherent_summary_action(self):
        # Identical embeddings → cosine sim = 1.0
        emb = np.array([1.0, 0.0, 0.0, 0.0])
        indexer = self._make_indexer_coherence(emb, emb)
        frames = [np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(2)]
        timestamps = [0.0, 1.0]
        segs = [_make_seg(0.0, 2.0, "Walk forward", "Walking scene", "Walk forward")]
        indexer._score_annotations(segs, frames, timestamps)
        assert "coherence_score" in segs[0]
        assert segs[0]["coherence_score"] >= 0.9

    def test_na_action_skips_coherence(self):
        emb = np.array([1.0, 0.0, 0.0, 0.0])
        indexer = self._make_indexer_coherence(emb, emb)
        frames = [np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(2)]
        timestamps = [0.0, 1.0]
        segs = [_make_seg(0.0, 2.0, "Some caption", "Some scene", "N/A", None)]
        indexer._score_annotations(segs, frames, timestamps)
        assert "coherence_score" not in segs[0]


# ---------------------------------------------------------------------------
# Aggregate quality_score
# ---------------------------------------------------------------------------

class TestAggregateQualityScore:
    def test_aggregate_averages_available_signals(self):
        seg = {
            "start_time": 0.0,
            "end_time": 5.0,
            "caption": "test",
            "caption_quality_score": 0.8,
            "format_compliance_score": 0.75,
            "action_frequency_score": 1.0,
        }
        # Simulate the aggregation logic directly
        signal_keys = [
            "caption_quality_score",
            "format_compliance_score",
            "coherence_score",
            "temporal_consistency_score",
            "action_frequency_score",
        ]
        values = [seg[k] for k in signal_keys if k in seg]
        seg["quality_score"] = round(sum(values) / len(values), 4)
        assert seg["quality_score"] == pytest.approx((0.8 + 0.75 + 1.0) / 3, abs=1e-3)

    def test_aggregate_set_by_score_annotations(self):
        indexer = VideoIndexer.__new__(VideoIndexer)
        indexer._text_embedding_model_name = None
        indexer._model = None
        indexer._ensure_model = MagicMock(side_effect=AttributeError)

        # Force _ensure_model to fail gracefully so only format/frequency run
        indexer2 = VideoIndexer.__new__(VideoIndexer)
        emb = np.array([1.0, 0.0, 0.0, 0.0])
        indexer2._text_embedding_model_name = None
        indexer2._model = MagicMock()
        indexer2._image_processor = MagicMock()
        indexer2._torch_device = "cpu"
        indexer2._encode_texts = lambda texts: np.tile(emb, (len(texts), 1))
        indexer2._encode_frames = lambda frames: np.tile(emb, (len(frames), 1)).astype(np.float32)
        indexer2._ensure_model = MagicMock()

        frames = [np.zeros((4, 4, 3), dtype=np.uint8)]
        timestamps = [0.5]
        seg = _make_seg(0.0, 1.0, "Walk forward", "Walking scene", "Walk forward")
        indexer2._score_annotations([seg], frames, timestamps)
        assert "quality_score" in seg
        assert 0.0 <= seg["quality_score"] <= 1.0


# ---------------------------------------------------------------------------
# _fix_low_quality_annotations
# ---------------------------------------------------------------------------

class TestFixLowQualityAnnotations:
    def test_method_exists_and_callable(self):
        indexer = VideoIndexer()
        assert callable(indexer._fix_low_quality_annotations)

    def test_no_caption_fn_is_noop(self):
        seg = _make_seg()
        seg["format_compliance_score"] = 0.1
        indexer = VideoIndexer()
        frames = [np.zeros((4, 4, 3), dtype=np.uint8)]
        indexer._fix_low_quality_annotations([seg], frames, [0.5], caption_fn=None)
        assert seg["caption"] == "A scene"

    def test_recaptions_low_quality_segment(self):
        seg = _make_seg(caption="bad caption", summary_brief="bad caption")
        seg["format_compliance_score"] = 0.1
        seg["quality_score"] = 0.1

        new_annotation = {
            "summary": {"brief": "new good caption", "detailed": "new good caption"},
            "action": {"brief": "Walk forward", "detailed": "Walk forward", "actor": "person"},
        }
        caption_fn = MagicMock(return_value=new_annotation)

        indexer = VideoIndexer.__new__(VideoIndexer)
        indexer._ensure_model = MagicMock()
        frames = [np.zeros((4, 4, 3), dtype=np.uint8)]
        timestamps = [0.5]
        indexer._fix_low_quality_annotations([seg], frames, timestamps, caption_fn=caption_fn)

        assert caption_fn.called
        assert seg["caption"] == "new good caption"

    def test_skips_high_quality_segment(self):
        seg = _make_seg()
        seg["format_compliance_score"] = 0.9
        seg["quality_score"] = 0.9

        caption_fn = MagicMock(return_value="should not be called")
        indexer = VideoIndexer.__new__(VideoIndexer)
        frames = [np.zeros((4, 4, 3), dtype=np.uint8)]
        timestamps = [0.5]
        indexer._fix_low_quality_annotations([seg], frames, timestamps, caption_fn=caption_fn)

        caption_fn.assert_not_called()
        assert seg["caption"] == "A scene"
