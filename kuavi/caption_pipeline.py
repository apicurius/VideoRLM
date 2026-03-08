"""Captioning pipeline: selective decode, Tree-of-Captions, Self-Refine, quality scoring."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def transcript_for_range(
    transcript: list[dict],
    start: float,
    end: float,
) -> str:
    """Return concatenated transcript text overlapping a time range."""
    return " ".join(
        e["text"] for e in transcript if e["end_time"] >= start and e["start_time"] <= end
    )


def selective_decode(
    segments: list[dict],
    frames: list[np.ndarray],
    timestamps: list[float],
    encode_frames_fn: Callable,
    similarity_threshold: float = 0.98,
    temporal_clip_embeddings: np.ndarray | None = None,
    temporal_clip_timestamps: list[float] | None = None,
) -> None:
    """3-tier selective decoding to optimize captioning cost.

    Tier 0 — DEAD: Skip captioning entirely (black/blank frames).
    Tier 1 — STATIC-INFORMATIVE: Caption with 1 keyframe only (slides, charts).
    Tier 2 — DYNAMIC: Full captioning pipeline (no change).

    V-JEPA temporal variance can promote Tier 1 → Tier 2 when subtle motion
    is detected that SigLIP2 misses.
    """
    import cv2

    tier_0_count = 0
    tier_1_count = 0
    tier_2_count = 0

    for seg in segments:
        if seg.get("_skip_caption"):
            continue
        seg_frames = [
            f
            for f, t in zip(frames, timestamps, strict=False)
            if seg["start_time"] <= t <= seg["end_time"]
        ]
        if not seg_frames:
            continue

        # --- Tier 0: DEAD frame detection ---
        # Sample 1-2 frames and check pixel variance + edge density
        sample_indices = [len(seg_frames) // 2]
        if len(seg_frames) >= 4:
            sample_indices.append(len(seg_frames) // 4)
        is_dead = True
        for si in sample_indices:
            sample = seg_frames[si]
            gray = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)
            pixel_std = float(gray.std())
            laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            edge_density = laplacian_var / max(gray.mean(), 1.0)
            if pixel_std >= 5.0 and edge_density >= 0.01:
                is_dead = False
                break

        if is_dead:
            seg["_skip_caption"] = True
            seg["_selective_tier"] = 0
            seg["caption"] = "Dead frame (black/blank)"
            seg["annotation"] = {
                "summary": {
                    "brief": "Dead frame (black/blank)",
                    "detailed": "This segment contains dead frames with no visual content.",
                },
                "action": {"brief": "N/A", "detailed": "", "actor": None},
            }
            seg["is_non_action"] = True
            tier_0_count += 1
            continue

        # --- Compute SigLIP2 visual similarity for Tier 1/2 ---
        if len(seg_frames) < 3:
            seg["_selective_tier"] = 2
            tier_2_count += 1
            continue

        try:
            embs = encode_frames_fn(seg_frames)
            sim_matrix = embs @ embs.T
            n = len(sim_matrix)
            if n < 2:
                seg["_selective_tier"] = 2
                tier_2_count += 1
                continue
            mean_sim = float((sim_matrix.sum() - np.trace(sim_matrix)) / (n * (n - 1)))
            seg["_visual_variance"] = round(1.0 - mean_sim, 6)
        except Exception:
            seg["_selective_tier"] = 2
            tier_2_count += 1
            continue

        if mean_sim <= similarity_threshold:
            # Dynamic — full captioning
            seg["_selective_tier"] = 2
            tier_2_count += 1
            continue

        # --- Tier 1 candidate: check V-JEPA temporal variance for promotion ---
        if temporal_clip_embeddings is not None and temporal_clip_timestamps is not None:
            clip_indices = [
                i
                for i, ct in enumerate(temporal_clip_timestamps)
                if seg["start_time"] <= ct <= seg["end_time"]
            ]
            if len(clip_indices) >= 2:
                clip_embs = temporal_clip_embeddings[clip_indices]
                temporal_var = float(np.var(clip_embs, axis=0).mean())
                seg["_temporal_variance"] = round(temporal_var, 6)
                if temporal_var > 0.05:
                    # Subtle motion detected — promote to dynamic
                    seg["_selective_tier"] = 2
                    tier_2_count += 1
                    continue

        # --- Tier 1: STATIC-INFORMATIVE — keep only middle keyframe ---
        seg["_selective_tier"] = 1
        seg["_static_informative"] = True
        real_frames = [f for f in seg.get("_frames", []) if not isinstance(f, str)]
        if real_frames:
            mid_frame = real_frames[len(real_frames) // 2]
            str_tokens = [f for f in seg.get("_frames", []) if isinstance(f, str)]
            seg["_frames"] = str_tokens + [mid_frame]
        tier_1_count += 1

    logger.info(
        "Selective decode: Tier 0 (dead): %d, Tier 1 (static-informative): %d, "
        "Tier 2 (dynamic): %d out of %d segments",
        tier_0_count,
        tier_1_count,
        tier_2_count,
        len(segments),
    )


def action_first_pass(
    segment_infos: list[dict],
    frame_caption_fn: Callable | None,
) -> None:
    """Set brief frame captions for fast-mode indexing (action-first pass).

    For each non-skipped segment, extracts the midpoint keyframe and calls
    ``frame_caption_fn`` to produce a brief caption.  Sets ``caption``,
    ``frame_caption``, and a minimal ``annotation`` structure so that
    ``_embed_captions`` can produce searchable embeddings immediately,
    without running the full Tree-of-Captions or Self-Refine pipeline.

    Skipped segments (Tier-0 dead frames or pre-caption dedup) have their
    ``_frames`` key removed; caption propagation from representatives is
    handled by the caller after this method returns.
    """
    caption_tasks = []
    for seg in segment_infos:
        seg_frames = seg.pop("_frames", [])
        if seg.get("_skip_caption"):
            # Already captioned (Tier 0) or dedup'd — propagation handled by caller
            continue
        real_frames = [f for f in seg_frames if not isinstance(f, str)]
        if real_frames:
            mid_frame = real_frames[len(real_frames) // 2]
            caption_tasks.append((seg, mid_frame))

    if frame_caption_fn is None or not caption_tasks:
        return

    def _caption_one(args):
        seg, mid_frame = args
        try:
            result = frame_caption_fn([mid_frame])
            caption = result if isinstance(result, str) else str(result)
        except Exception:
            logger.warning("Fast-mode frame caption failed", exc_info=True)
            caption = ""
        return seg, caption

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(_caption_one, task) for task in caption_tasks]
        for future in as_completed(futures):
            try:
                seg, caption = future.result()
                seg["frame_caption"] = caption
                seg["caption"] = caption
                seg["annotation"] = {
                    "summary": {"brief": caption, "detailed": caption},
                    "action": {"brief": "", "detailed": "", "actor": None},
                }
            except Exception:
                logger.warning("Fast-mode caption future raised an exception", exc_info=True)


def filter_edge_frames(
    seg_frames: list,
    encode_frames_fn: Callable,
    threshold: float = 0.5,
) -> list:
    """Filter visually dissimilar edge frames from a segment."""
    real_frames = [f for f in seg_frames if not isinstance(f, str)]
    if len(real_frames) < 5:
        return seg_frames

    str_tokens = [f for f in seg_frames if isinstance(f, str)]

    try:
        embs = encode_frames_fn(real_frames)
    except AttributeError:
        return seg_frames
    n = len(real_frames)

    start_20 = max(1, int(n * 0.2))
    end_80 = min(n - 1, int(n * 0.8))

    central_embs = embs[start_20:end_80]
    central_mean = central_embs.mean(axis=0)
    norm = np.linalg.norm(central_mean)
    if norm > 1e-10:
        central_mean = central_mean / norm

    keep_indices = set(range(start_20, end_80))
    for i in list(range(0, start_20)) + list(range(end_80, n)):
        sim = float(np.dot(embs[i], central_mean))
        if sim >= threshold:
            keep_indices.add(i)

    filtered_real = [real_frames[i] for i in sorted(keep_indices)]
    return str_tokens + filtered_real


def refine_annotations(
    segments: list[dict],
    transcript: list[dict],
    refine_fn: Callable | None,
    video_metadata=None,
    rounds: int = 3,
) -> None:
    """Iteratively refine segment annotations using the Self-Refine pattern."""
    if refine_fn is None:
        return

    global_context = ""
    if len(segments) > 1:
        first_cap = segments[0].get("caption", "")
        last_cap = segments[-1].get("caption", "")
        global_context = f"Video starts with: {first_cap}\nVideo ends with: {last_cap}"

    metadata_text = ""
    if video_metadata:
        path = getattr(video_metadata, "path", "") or ""
        duration = float(getattr(video_metadata, "duration", 0) or 0)
        metadata_text = f"Video: {Path(path).name}, Duration: {duration:.1f}s"

    _JSON_SCHEMA = (
        "### Output Format (strict JSON)\n"
        "{\n"
        '  "summary": {"brief": "<single sentence, ~20 words>", "detailed": "<~95 words>"},\n'
        '  "action": {"brief": "<imperative verb phrase, 2-5 words>", '
        '"detailed": "<imperative sentence>", "actor": "<noun phrase or null>"}\n'
        "}"
    )

    def _build_tree_text(segs: list[dict]) -> str:
        lines = ["## Tree of Captions"]
        for j, s in enumerate(segs):
            fc = s.get("frame_caption", "")
            sc = s.get("caption", "")
            lines.append(f"### Seg {j} [{s['start_time']:.1f}s-{s['end_time']:.1f}s]")
            if fc:
                lines.append(f"- **Frame**: {fc}")
            if sc:
                lines.append(f"- **Segment**: {sc}")
        return "\n".join(lines)

    for _round in range(rounds):
        tree_text = _build_tree_text(segments)
        refine_tasks = []
        skipped = 0
        for i, seg in enumerate(segments):
            seg_duration = seg["end_time"] - seg["start_time"]
            if seg_duration < 4.0:
                skipped += 1
                continue
            neighbors = segments[max(0, i - 1) : i + 2]
            neighbor_text = " | ".join(n.get("caption", "") for n in neighbors if n is not seg)
            transcript_text = transcript_for_range(
                transcript,
                seg["start_time"],
                seg["end_time"],
            )
            context = f"""# Video Metadata
{metadata_text}

# Global Video Context
{global_context}

{tree_text}

# Neighbor Segments
{neighbor_text}

# Transcript
{transcript_text}"""
            annotation_json = json.dumps(seg.get("annotation", {}))
            if _round > 0:
                draft = (
                    "Carefully analyze, verify, and revise the previous draft. "
                    "Correct factual errors, resolve inconsistencies, and remove "
                    "unsupported statements.\n\n"
                    "### Verification Checklist\n"
                    "- Remove any claims not supported by at least 2 frame observations\n"
                    "- Remove names, speech content, or internal states unless directly visible\n"
                    "- Ensure chronological ordering without timestamps\n"
                    "- Verify action.brief is an imperative verb phrase (2-5 words)\n\n"
                    f"{_JSON_SCHEMA}\n\n"
                    f"Previous draft:\n{annotation_json}"
                )
            else:
                draft = (
                    "Analyze this video segment annotation and produce a refined version.\n\n"
                    "### Task 1: Summarization\n"
                    "Generate summary.brief (single sentence, ~20 words) and summary.detailed (~95 words).\n"
                    "Describe events in chronological order. Do not mention exact timestamps.\n\n"
                    "### Task 2: Action Identification\n"
                    "Identify the primary action (action.brief: imperative verb phrase, 2-5 words).\n"
                    "Describe the actor performing the action (action.actor).\n"
                    "Use 'N/A' for action.brief if no identifiable action exists.\n\n"
                    "### Anti-Hallucination Rules\n"
                    "- Be cautious and conservative. Rely on majority consensus across frame captions.\n"
                    "- Do not add visually unobservable information (speech content, names, internal states).\n"
                    "- Use global context and metadata only for disambiguation, not for adding new claims.\n"
                    "- If frame captions conflict, describe only what is consistently observed.\n\n"
                    f"{_JSON_SCHEMA}\n\n"
                    f"Current annotation:\n{annotation_json}"
                )
            refine_tasks.append((i, seg, draft, context))
        if skipped:
            logger.debug("Self-Refine round %d: skipped %d short segments (< 4s)", _round, skipped)

        effort = "high" if _round == 0 else "low"

        def _refine_one(args, _effort=effort):
            i, seg, draft, context = args
            try:
                refined = refine_fn(draft, context, _effort)
            except TypeError:
                refined = refine_fn(draft, context)
            return i, refined

        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(_refine_one, task) for task in refine_tasks]
            results = {}
            for future in as_completed(futures):
                try:
                    i, refined = future.result()
                    results[i] = refined
                except Exception:
                    logger.warning("Refine future raised an exception", exc_info=True)

        for i, seg in enumerate(segments):
            refined = results.get(i)
            if refined is None:
                continue
            try:
                seg["annotation"] = json.loads(refined)
                seg["caption"] = (
                    seg["annotation"].get("summary", {}).get("brief", seg.get("caption", ""))
                )
            except (json.JSONDecodeError, TypeError):
                pass


def score_format_compliance(seg: dict) -> float:
    """Score annotation format compliance (0.0-1.0, pure string checks)."""
    import re

    annotation = seg.get("annotation", {})
    summary = annotation.get("summary", {}) if isinstance(annotation, dict) else {}
    action = annotation.get("action", {}) if isinstance(annotation, dict) else {}

    score = 0.0

    # summary.brief exists and is non-empty (0.25)
    summary_brief = summary.get("brief", "") if isinstance(summary, dict) else ""
    if summary_brief and isinstance(summary_brief, str) and summary_brief.strip():
        score += 0.25

    # action.brief is 2-5 words starting with imperative verb (0.25)
    action_brief = action.get("brief", "") if isinstance(action, dict) else ""
    if action_brief and isinstance(action_brief, str) and action_brief.strip():
        words = action_brief.strip().split()
        if 2 <= len(words) <= 5 and words[0][0].isupper():
            score += 0.25

    # No timestamps in summary text (0.25)
    if summary_brief and not re.search(r"\bat\s+\d+(?:\.\d+)?s\b", summary_brief):
        score += 0.25

    # action.actor is present when action.brief is not "N/A" (0.25)
    if isinstance(action, dict):
        ab = action.get("brief", "")
        if ab and ab != "N/A":
            actor = action.get("actor")
            if actor is not None and str(actor).strip():
                score += 0.25
        else:
            # action is N/A — actor field not required
            score += 0.25

    return round(score, 4)


def score_action_frequency(segments: list[dict]) -> None:
    """Score each segment's action.brief frequency across all segments (in-place)."""
    action_counts: dict[str, int] = {}
    total = len(segments)
    if total == 0:
        return

    for seg in segments:
        annotation = seg.get("annotation", {})
        action = annotation.get("action", {}) if isinstance(annotation, dict) else {}
        ab = action.get("brief", "") if isinstance(action, dict) else ""
        if ab and ab != "N/A":
            action_counts[ab] = action_counts.get(ab, 0) + 1

    for seg in segments:
        annotation = seg.get("annotation", {})
        action = annotation.get("action", {}) if isinstance(annotation, dict) else {}
        ab = action.get("brief", "") if isinstance(action, dict) else ""
        if not ab or ab == "N/A":
            seg["action_frequency_score"] = 1.0
            continue

        freq = action_counts.get(ab, 0) / total
        if freq <= 0.30:
            freq_score = 1.0
        elif freq >= 0.50:
            freq_score = 0.0
        else:
            freq_score = 1.0 - (freq - 0.30) / 0.20

        seg["action_frequency_score"] = round(freq_score, 4)


def score_annotations(
    segments: list[dict],
    loaded_video_frames: list[np.ndarray],
    timestamps: list[float],
    encode_frames_fn: Callable,
    encode_texts_fn: Callable,
    text_embedding_model_name: str | None,
    min_similarity: float = 0.3,
) -> None:
    """Score annotation quality using model-free signals."""
    # Signal 2: Format compliance — no model needed
    for seg in segments:
        seg["format_compliance_score"] = score_format_compliance(seg)

    # Signal 5: Action frequency — no model needed, needs all segments
    score_action_frequency(segments)

    # Collect caption embeddings for signals 3 and 4 (text model required)
    caption_embeddings: dict[int, np.ndarray] = {}

    for idx, seg in enumerate(segments):
        caption = seg.get("caption", "")
        if not caption:
            continue

        if text_embedding_model_name is not None:
            # Skip signal 1 and signal 3 when using separate text embedding model
            continue

        seg_frames = [
            f
            for f, t in zip(loaded_video_frames, timestamps, strict=False)
            if seg["start_time"] <= t <= seg["end_time"]
        ]
        if not seg_frames:
            continue

        try:
            caption_emb = encode_texts_fn([caption])
            frame_embs = encode_frames_fn(seg_frames)
        except AttributeError:
            continue
        mean_frame_emb = frame_embs.mean(axis=0, keepdims=True)
        norm = np.linalg.norm(mean_frame_emb, axis=1, keepdims=True)
        mean_frame_emb = mean_frame_emb / np.maximum(norm, 1e-10)

        similarity = float(np.dot(caption_emb[0], mean_frame_emb[0]))
        seg["caption_quality_score"] = round(similarity, 4)

        # Store caption embedding for signals 3 and 4
        caption_embeddings[idx] = caption_emb[0]

        # Signal 3: Summary-Action Coherence
        annotation = seg.get("annotation", {})
        action = annotation.get("action", {}) if isinstance(annotation, dict) else {}
        action_brief = action.get("brief", "") if isinstance(action, dict) else ""
        if action_brief and action_brief != "N/A":
            summary = annotation.get("summary", {}) if isinstance(annotation, dict) else {}
            summary_brief = summary.get("brief", "") if isinstance(summary, dict) else ""
            if summary_brief:
                try:
                    action_emb = encode_texts_fn([action_brief])
                    summary_emb = encode_texts_fn([summary_brief])
                    coherence = float(np.dot(action_emb[0], summary_emb[0]))
                    seg["coherence_score"] = round(coherence, 4)
                except AttributeError:
                    pass

    # Signal 4: Temporal consistency (needs all caption embeddings)
    for idx, seg in enumerate(segments):
        if idx not in caption_embeddings:
            continue
        emb = caption_embeddings[idx]
        sims = []
        if idx - 1 in caption_embeddings:
            sims.append(float(np.dot(emb, caption_embeddings[idx - 1])))
        if idx + 1 in caption_embeddings:
            sims.append(float(np.dot(emb, caption_embeddings[idx + 1])))
        if sims:
            max_sim = max(sims)
            seg["temporal_consistency_score"] = round(max(0.0, min(1.0, 1.0 - max_sim)), 4)

    # Aggregate quality_score: average of all available signals
    signal_keys = [
        "caption_quality_score",
        "format_compliance_score",
        "coherence_score",
        "temporal_consistency_score",
        "action_frequency_score",
    ]
    for seg in segments:
        values = [seg[k] for k in signal_keys if k in seg]
        if values:
            seg["quality_score"] = round(sum(values) / len(values), 4)


def fix_low_quality_annotations(
    segments: list[dict],
    loaded_video_frames: list[np.ndarray],
    timestamps: list[float],
    caption_fn: Callable | None = None,
    threshold: float = 0.3,
    num_retries: int = 3,
) -> None:
    """Re-caption segments where any quality signal is below *threshold*."""
    if caption_fn is None:
        return

    signal_keys = [
        "caption_quality_score",
        "format_compliance_score",
        "coherence_score",
        "temporal_consistency_score",
        "action_frequency_score",
    ]

    for seg in segments:
        low_quality = any(seg.get(k, 1.0) < threshold for k in signal_keys if k in seg)
        if not low_quality:
            continue

        seg_frames = [
            f
            for f, t in zip(loaded_video_frames, timestamps, strict=False)
            if seg["start_time"] <= t <= seg["end_time"]
        ]
        if not seg_frames:
            continue

        best_annotation = seg.get("annotation", {})
        best_caption = seg.get("caption", "")

        for _ in range(num_retries):
            try:
                result = caption_fn(seg_frames)
                if isinstance(result, str):
                    new_annotation = {
                        "summary": {"brief": result, "detailed": result},
                        "action": {"brief": "", "detailed": "", "actor": None},
                    }
                    new_caption = result
                else:
                    new_annotation = result
                    new_caption = result.get("summary", {}).get("brief", "")

                if new_caption and new_caption != best_caption:
                    best_annotation = new_annotation
                    best_caption = new_caption
                    break
            except Exception:
                logger.warning("_fix_low_quality_annotations re-caption failed", exc_info=True)

        if best_caption and best_caption != seg.get("caption", ""):
            seg["annotation"] = best_annotation
            seg["caption"] = best_caption
            logger.info(
                "Fixed low-quality segment %.1f-%.1fs",
                seg["start_time"],
                seg["end_time"],
            )
