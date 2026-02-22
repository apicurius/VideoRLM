"""Tests for Self-Refine Protocol v2 (WI-3)."""

import json
from unittest.mock import MagicMock


def _make_indexer():
    """Return a VideoIndexer with transcript helper stubbed (no real models needed)."""
    from kuavi.indexer import VideoIndexer

    indexer = VideoIndexer()
    indexer._transcript_for_range = MagicMock(return_value="")
    return indexer


def _make_seg(start, end, caption="cap", frame_caption="frame"):
    return {
        "start_time": start,
        "end_time": end,
        "caption": caption,
        "frame_caption": frame_caption,
        "annotation": {"summary": {"brief": caption, "detailed": caption}, "action": {"brief": "walk"}},
    }


class TestShortSegmentSkip:
    """1. Segments shorter than 4s must not be passed to refine_fn."""

    def test_short_segments_skipped(self):
        indexer = _make_indexer()
        refine_fn = MagicMock(return_value=json.dumps({"summary": {"brief": "x", "detailed": "x"}, "action": {"brief": "y", "detailed": "y", "actor": None}}))

        segments = [
            _make_seg(0.0, 2.0, "short seg"),   # 2s — should be skipped
            _make_seg(2.0, 3.5, "short seg 2"),  # 1.5s — should be skipped
            _make_seg(3.5, 9.0, "long seg"),     # 5.5s — should be processed
        ]

        indexer._refine_annotations(segments, [], refine_fn, rounds=1)

        assert refine_fn.call_count == 1, (
            f"refine_fn should be called once (for the 5.5s segment), got {refine_fn.call_count}"
        )

    def test_exactly_4s_is_not_skipped(self):
        indexer = _make_indexer()
        refine_fn = MagicMock(return_value=json.dumps({"summary": {"brief": "x", "detailed": "x"}, "action": {"brief": "y", "detailed": "y", "actor": None}}))

        segments = [_make_seg(0.0, 4.0, "exactly 4s")]  # 4.0s — must NOT be skipped

        indexer._refine_annotations(segments, [], refine_fn, rounds=1)

        assert refine_fn.call_count == 1

    def test_all_short_no_calls(self):
        indexer = _make_indexer()
        refine_fn = MagicMock()

        segments = [_make_seg(0.0, 1.0), _make_seg(1.0, 2.5), _make_seg(2.5, 3.9)]

        indexer._refine_annotations(segments, [], refine_fn, rounds=2)

        refine_fn.assert_not_called()

    def test_multiple_rounds_skips_consistently(self):
        indexer = _make_indexer()
        refine_fn = MagicMock(return_value=json.dumps({"summary": {"brief": "x", "detailed": "x"}, "action": {"brief": "y", "detailed": "y", "actor": None}}))

        segments = [
            _make_seg(0.0, 2.0),  # short
            _make_seg(2.0, 8.0),  # long
        ]

        indexer._refine_annotations(segments, [], refine_fn, rounds=3)

        # 3 rounds × 1 long segment = 3 calls
        assert refine_fn.call_count == 3


class TestMarkdownTreeFormat:
    """2. The tree text passed to refine_fn must use Markdown format."""

    def test_tree_uses_markdown_headers(self):
        indexer = _make_indexer()
        captured_contexts = []

        def capturing_refine(draft, context, effort=None):
            captured_contexts.append(context)
            return json.dumps({"summary": {"brief": "x", "detailed": "x"}, "action": {"brief": "y", "detailed": "y", "actor": None}})

        segments = [_make_seg(0.0, 5.0, "a scene", "a frame")]
        indexer._refine_annotations(segments, [], capturing_refine, rounds=1)

        assert captured_contexts, "refine_fn was not called"
        context = captured_contexts[0]

        assert "## Tree of Captions" in context
        assert "### Seg 0 [0.0s-5.0s]" in context

    def test_tree_uses_bold_labels(self):
        indexer = _make_indexer()
        captured = []

        def capturing_refine(draft, context, effort=None):
            captured.append(context)
            return json.dumps({"summary": {"brief": "x", "detailed": "x"}, "action": {"brief": "y", "detailed": "y", "actor": None}})

        segments = [_make_seg(0.0, 6.0, "scene cap", "frame cap")]
        indexer._refine_annotations(segments, [], capturing_refine, rounds=1)

        context = captured[0]
        assert "**Frame**:" in context
        assert "**Segment**:" in context

    def test_old_indented_format_not_present(self):
        """Make sure the old '  Seg 0 [...]:\n    Frame:' format is gone."""
        indexer = _make_indexer()
        captured = []

        def capturing_refine(draft, context, effort=None):
            captured.append(context)
            return json.dumps({"summary": {"brief": "x", "detailed": "x"}, "action": {"brief": "y", "detailed": "y", "actor": None}})

        segments = [_make_seg(0.0, 5.0, "cap", "frame")]
        indexer._refine_annotations(segments, [], capturing_refine, rounds=1)

        context = captured[0]
        # Old format had 4-space indented lines like "    Frame: ..."
        assert "    Frame:" not in context
        assert "    Segment:" not in context


class TestJsonSchemaInPrompt:
    """3. The JSON schema must appear in the draft prompt."""

    def test_schema_in_round_0_draft(self):
        indexer = _make_indexer()
        captured_drafts = []

        def capturing_refine(draft, context, effort=None):
            captured_drafts.append(draft)
            return json.dumps({"summary": {"brief": "x", "detailed": "x"}, "action": {"brief": "y", "detailed": "y", "actor": None}})

        segments = [_make_seg(0.0, 5.0)]
        indexer._refine_annotations(segments, [], capturing_refine, rounds=1)

        assert captured_drafts
        draft = captured_drafts[0]
        assert "Output Format (strict JSON)" in draft
        assert '"summary"' in draft
        assert '"action"' in draft

    def test_schema_in_round_1_draft(self):
        indexer = _make_indexer()
        captured_drafts = []

        def capturing_refine(draft, context, effort=None):
            captured_drafts.append(draft)
            return json.dumps({"summary": {"brief": "x", "detailed": "x"}, "action": {"brief": "y", "detailed": "y", "actor": None}})

        segments = [_make_seg(0.0, 5.0)]
        indexer._refine_annotations(segments, [], capturing_refine, rounds=2)

        # captured_drafts[1] is round 1
        assert len(captured_drafts) >= 2
        round1_draft = captured_drafts[1]
        assert "Output Format (strict JSON)" in round1_draft
        assert '"summary"' in round1_draft


class TestRoundSpecificPrompting:
    """4. Round 0 uses 'analyze', round 1+ uses 'verify'."""

    def test_round_0_uses_analyze_keyword(self):
        indexer = _make_indexer()
        captured_drafts = []

        def capturing_refine(draft, context, effort=None):
            captured_drafts.append(draft)
            return json.dumps({"summary": {"brief": "x", "detailed": "x"}, "action": {"brief": "y", "detailed": "y", "actor": None}})

        segments = [_make_seg(0.0, 6.0)]
        indexer._refine_annotations(segments, [], capturing_refine, rounds=1)

        round0 = captured_drafts[0]
        assert "Analyze" in round0 or "analyze" in round0

    def test_round_1_uses_verify_keyword(self):
        indexer = _make_indexer()
        captured_drafts = []

        def capturing_refine(draft, context, effort=None):
            captured_drafts.append(draft)
            return json.dumps({"summary": {"brief": "x", "detailed": "x"}, "action": {"brief": "y", "detailed": "y", "actor": None}})

        segments = [_make_seg(0.0, 6.0)]
        indexer._refine_annotations(segments, [], capturing_refine, rounds=2)

        round1 = captured_drafts[1]
        assert "verify" in round1.lower() or "Verify" in round1

    def test_round_0_does_not_contain_previous_draft_label(self):
        indexer = _make_indexer()
        captured_drafts = []

        def capturing_refine(draft, context, effort=None):
            captured_drafts.append(draft)
            return json.dumps({"summary": {"brief": "x", "detailed": "x"}, "action": {"brief": "y", "detailed": "y", "actor": None}})

        segments = [_make_seg(0.0, 5.0)]
        indexer._refine_annotations(segments, [], capturing_refine, rounds=1)

        # Round 0 should say "Current annotation:", not "Previous draft:"
        assert "Current annotation:" in captured_drafts[0]

    def test_round_1_contains_previous_draft_label(self):
        indexer = _make_indexer()
        captured_drafts = []

        def capturing_refine(draft, context, effort=None):
            captured_drafts.append(draft)
            return json.dumps({"summary": {"brief": "x", "detailed": "x"}, "action": {"brief": "y", "detailed": "y", "actor": None}})

        segments = [_make_seg(0.0, 5.0)]
        indexer._refine_annotations(segments, [], capturing_refine, rounds=2)

        assert "Previous draft:" in captured_drafts[1]


class TestBackwardCompatibility:
    """5. Method signature and refine_fn interface must remain backward-compatible."""

    def test_none_refine_fn_returns_early(self):
        indexer = _make_indexer()
        segments = [_make_seg(0.0, 10.0)]
        # Should not raise
        indexer._refine_annotations(segments, [], None, rounds=3)
        # Annotations unchanged
        assert segments[0]["annotation"]["summary"]["brief"] == "cap"

    def test_refine_fn_without_effort_arg(self):
        """refine_fn(draft, context) without effort keyword must still work."""
        indexer = _make_indexer()
        call_count = [0]

        def two_arg_refine(draft, context):
            call_count[0] += 1
            return json.dumps({"summary": {"brief": "new", "detailed": "new"}, "action": {"brief": "go", "detailed": "go", "actor": None}})

        segments = [_make_seg(0.0, 5.0)]
        indexer._refine_annotations(segments, [], two_arg_refine, rounds=1)
        assert call_count[0] == 1

    def test_annotation_updated_from_refine_fn(self):
        indexer = _make_indexer()
        new_ann = {"summary": {"brief": "refined", "detailed": "refined detail"}, "action": {"brief": "run fast", "detailed": "run fast sentence", "actor": "athlete"}}

        def refine_fn(draft, context, effort=None):
            return json.dumps(new_ann)

        segments = [_make_seg(0.0, 5.0, "original")]
        indexer._refine_annotations(segments, [], refine_fn, rounds=1)

        assert segments[0]["annotation"] == new_ann
        assert segments[0]["caption"] == "refined"

    def test_invalid_json_from_refine_fn_is_silently_ignored(self):
        indexer = _make_indexer()

        def bad_refine(draft, context, effort=None):
            return "not valid json {{{"

        segments = [_make_seg(0.0, 5.0, "original")]
        indexer._refine_annotations(segments, [], bad_refine, rounds=1)

        # Original annotation should be preserved
        assert segments[0]["caption"] == "original"

    def test_video_metadata_optional(self):
        indexer = _make_indexer()
        refine_fn = MagicMock(return_value=json.dumps({"summary": {"brief": "x", "detailed": "x"}, "action": {"brief": "y", "detailed": "y", "actor": None}}))

        segments = [_make_seg(0.0, 5.0)]
        # video_metadata defaults to None — should not raise
        indexer._refine_annotations(segments, [], refine_fn)
        assert refine_fn.called
