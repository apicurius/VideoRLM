# Unified Integration Plan: V-JEPA 2 + Action100M for RLM & KUAVi

## Executive Summary

This document merges the V-JEPA 2 integration plan (8 integrations) and Action100M integration plan (7 integrations) into a single prioritized roadmap with full impact analysis. It resolves overlaps, identifies the critical path, and maps every change to specific files in both the `kuavi/` and `rlm/` packages.

**Scope**: 15 original integrations collapse into **13 unified work items** (12 + 1 critical bug fix) after merging overlaps. The VL-JEPA plan is referenced for future alignment but not included in this roadmap — it depends on the V-JEPA 2 foundation being in place first.

**Key insight from merging**: The two plans are highly complementary. V-JEPA 2 upgrades the **encoder foundation** (what features we extract). Action100M upgrades the **annotation pipeline** (how we label and search those features). Together they transform KUAVi from a caption-dependent single-video tool into an action-aware, offline-capable, corpus-scale analysis system.

**Critical findings from codebase analysis**:
1. **Temporal search is broken** — `field="temporal"` queries V-JEPA 1024-d embeddings with SigLIP2 768-d text encoder. `cosine_similarity((1, 768), (N, 1024))` raises a dimension mismatch error. This must be fixed before any V-JEPA upgrades (Work Item 0).
2. **Captioner callables are already loose** — `frame_caption_fn`, `caption_fn`, `refine_fn` are passed as Callables to `index_video()`, making the CaptionerBackend abstraction (Work Item 5) a thin wrapper, not a deep refactor.
3. **cluster_diverse already exists** — The parameter is on `kuavi_search_video` and `search.py` already has KMeans at query time. Work Item 8 moves clustering to index time (better).
4. **Self-Refine v2 is mostly format alignment** — `_refine_annotations()` already has round-specific prompting and verification checklists. The 4-second minimum and JSON schema are small additions.
5. **RLM mirror overhead** — `rlm/video/video_indexer.py` is a near-identical copy of `kuavi/indexer.py`. Every indexer change must be manually mirrored, adding ~30-50% to effort estimates.
6. **V-JEPA 2-AC weights may not generalize** — The world model (Work Item 11) was trained on 62 hours of robot manipulation data. Applying it to general video is untested and may not work.
7. **Predictor weights unverified** — HuggingFace `AutoModel.from_pretrained()` for V-JEPA 2 may only expose the encoder, not the predictor. Needs verification before Work Items 9/11.

---

## Overlap Resolution

Three areas of overlap were identified between the two plans:

| Overlap | V-JEPA 2 Plan | Action100M Plan | Resolution |
|---------|--------------|----------------|------------|
| **Gemini replacement** | Integration 5: V-JEPA 2 + LLM for VidQA captioning | Integration 1: Llama-3.2-Vision + PLM-3B + GPT-OSS | **Start with Action100M's approach** (pragmatic, validated at 147M scale). Evolve toward V-JEPA 2's approach when projector weights are available. Merged into Work Item 5. |
| **Overlapping V-JEPA windows** | Integration 3: 64-frame clips with stride | Integration 5: 64-frame windows with 8-frame stride, per-frame averaging | **Use Action100M's per-frame averaging** on top of V-JEPA 2's clip configuration. Merged into Work Item 4. |
| **Scene detection improvements** | Integration 3: Ward clustering threshold adjustment | Integration 5: Per-frame Ward clustering | **Combined** — per-frame embeddings from overlapping windows feed into threshold-adapted Ward clustering. Part of Work Item 4. |

---

## The 13 Unified Work Items

### Priority Tier 0: Critical Bug Fix (Immediate)

---

#### Work Item 0: Fix Temporal Search Dimension Mismatch
**Source**: Pre-existing bug discovered during analysis

`field="temporal"` search is broken. `search.py:121-125` uses `visual_embed_fn` (SigLIP2, 768-d) to encode text queries, then computes `cosine_similarity((1, 768), (N, 1024))` against V-JEPA temporal embeddings — this raises a sklearn `ValueError` at runtime.

**Changes**:

| File | Change | Breaking? |
|------|--------|-----------|
| `kuavi/search.py:121-125` | Add linear projection layer (768→1024) for temporal queries, OR use a separate temporal query encoder | Fixes broken behavior |
| `rlm/video/video_search_tools.py` | Mirror fix | Same |

**Options**:
- **(a) Linear projection**: Train or initialize a `nn.Linear(768, 1024)` to project SigLIP2 queries into V-JEPA space. Quick but lossy.
- **(b) Disable temporal text search**: Only allow temporal search via embedding (e.g., `discriminative_vqa` with visual candidates). Honest about the space mismatch.
- **(c) Wait for VL-JEPA**: VL-JEPA's Predictor produces query embeddings in V-JEPA's native space — the proper fix. But blocked on VL-JEPA implementation.

**Recommended**: Option (a) as stopgap with option (c) as the proper long-term fix.

**Risk**: Low — fixing a broken feature
**Effort**: 1 day
**This blocks**: Work Items 1 and 6 (model upgrades make the dimension problem worse — ViT-g produces 1536-d embeddings)

---

### Priority Tier 1: Quick Wins (Week 1-2)

These are independent, low-risk, high-value changes that can be done in any order.

---

#### Work Item 1: V-JEPA 2 Model Presets
**Source**: V-JEPA 2 Integration 1

Make V-JEPA 2 model variant configurable instead of hardcoded.

**Changes**:

| File | Change | Breaking? |
|------|--------|-----------|
| `kuavi/types.py` | Add `VJEPA2_PRESETS` dict (`fast`/`balanced`/`quality`), add `scene_model_preset` to `KUAViConfig` | No |
| `kuavi/indexer.py:144-145` | Load model from preset instead of hardcoded string | No |
| `kuavi/indexer.py:1509-1523` | `_group_frames_into_clips()` — support 32/64 frame clips from preset | No |
| `kuavi/indexer.py:1474-1507` | `_encode_clips_vjepa()` — adjust batch_size for larger models (2 for ViT-g) | No |
| `kuavi/mcp_server.py` | `kuavi_index_video` gets `scene_model_preset` parameter | No |
| `rlm/video/video_indexer.py` | Mirror all changes | No |

**KUAVi impact**: Current ViT-L (300M, 256px, 16 frames) becomes `fast` preset. Users can opt into `balanced` (ViT-H, 600M) or `quality` (ViT-g, 1B, 384px, 64 frames) for +3.1 avg accuracy improvement.

**RLM impact**: `video_indexer.py` mirrors preset system. `run_video.py` gains `--scene-model-preset` CLI arg.

**Risk**: Low — drop-in replacement, current behavior preserved as default. **Caveat**: ViT-g produces different embedding dimensions (likely 1536-d vs current 1024-d) — cached `.npz` indexes are incompatible across presets. Cache invalidation via content hash mitigates this (different model = different cache key).
**Effort**: 2 days (+ RLM mirror)
**VRAM**: fast=1.2GB (current), balanced=2.4GB, quality=4.0GB
**Prerequisite**: Work Item 0 (temporal search fix) — dimension mismatch gets worse with ViT-g

---

#### Work Item 2: Action-First Indexing
**Source**: Action100M Integration 2

Restructure indexing pipeline to produce searchable action embeddings first (fast), with detailed captions as a lazy second pass.

**Changes**:

| File | Change | Breaking? |
|------|--------|-----------|
| `kuavi/indexer.py` | New `_action_first_pass()` method — frame caption + action brief extraction | No |
| `kuavi/indexer.py` | New `enhance_index()` method for lazy Pass 2 (full captioning) | No |
| `kuavi/search.py` | Default `field` changes to `action` when action embeddings exist | Behavioral (search results may differ) |
| `kuavi/mcp_server.py` | New `kuavi_enhance_index` tool, `mode` param on `kuavi_index_video` (`fast`/`full`) | No |
| `rlm/video/video_indexer.py` | Mirror `_action_first_pass()` and two-pass flow | No |
| `rlm/video/video_search_tools.py` | Default field may change | Behavioral |

**KUAVi impact**: 5-10x faster first searchable index. Action brief (~3 words) becomes primary search signal — validated by Action100M as best-performing annotation type on 5/8 benchmarks. Detailed annotations generated only when needed (get_scene_list, analyze_shards, or explicit request).

**RLM impact**: VideoRLM benefits from faster indexing in REPL flow. `search_video()` defaults to action field when available. Existing `full` mode preserved for backward compatibility.

**Risk**: Low — additive, two-pass design preserves full pipeline as Pass 2
**Effort**: 3-4 days

---

#### Work Item 3: Self-Refine Protocol v2
**Source**: Action100M Integration 4

Adopt Action100M's validated Self-Refine protocol (proven at 147M segments).

**Changes**:

| File | Change | Breaking? |
|------|--------|-----------|
| `kuavi/indexer.py:967-1090` | Refactor `_refine_annotations` → v2: 4s minimum skip, depth-first Markdown tree, explicit JSON schema, round-specific prompting | Annotation output may differ |
| `kuavi/indexer.py:991-1001` | Replace tree text formatting with depth-first Markdown | No |
| `rlm/video/video_indexer.py:1047-1183` | Mirror all changes | Same |

**KUAVi impact**: Skip segments < 4 seconds (Action100M finding: too short for meaningful aggregation) → saves LLM calls. Structured JSON schema in prompt reduces parse failures. Depth-first Markdown format matches Action100M's validated template.

**RLM impact**: Mirror changes in `video_indexer.py`. No search tool changes.

**Risk**: Low — can A/B test v1 vs v2, prompt-only change. The 4-second minimum alone is a 30-minute change with immediate cost savings (Action100M finding: ~64% of segments are 0-3 seconds). JSON schema is a 1-hour change. Highest ROI-per-hour of any integration.
**Effort**: 1-2 days (+ RLM mirror)

---

#### Work Item 4: Quality Scoring & Audit
**Source**: Action100M Integration 7

Multi-signal annotation quality scoring with automatic re-captioning for degenerate outputs.

**Changes**:

| File | Change | Breaking? |
|------|--------|-----------|
| `kuavi/indexer.py` | New `_quality_score_annotations()` after Stage 7 — 5 signals: action-visual alignment, format compliance, summary-action coherence, temporal consistency, degenerate detection | No |
| `kuavi/indexer.py` | New `_fix_low_quality_annotations()` — re-caption degenerate/misaligned segments | No |
| `kuavi/mcp_server.py` | New `kuavi_quality_audit` tool | No |
| `rlm/video/video_indexer.py` | Mirror quality methods | No |

**KUAVi impact**: Extends existing `_score_annotations()` (lines 1092+) which already checks caption-visual alignment and re-captions below 0.3 threshold. Adds 4 new signals on top. Signal 2 (format compliance) and Signal 5 (per-action frequency degenerate detection) are 30-minute additions to the existing method.

**RLM impact**: Mirror methods. Can expose quality info in REPL.

**Risk**: Low — extends existing quality scoring, advisory only
**Effort**: 1-2 days (+ RLM mirror)

---

### Priority Tier 2: Foundation Upgrades (Week 2-4)

These require more engineering but unlock significant capabilities.

---

#### Work Item 5: Captioner Abstraction + Open VLM Support
**Source**: Action100M Integration 1 + V-JEPA 2 Integration 5 (merged)

Replace hardcoded Gemini dependency with pluggable captioning backend. Start with Action100M's validated model cascade, provide path to V-JEPA 2 + LLM when projector weights are available.

**Changes**:

| File | Change | Breaking? |
|------|--------|-----------|
| New: `kuavi/captioners.py` | `CaptionerBackend` protocol, `GeminiCaptioner`, `LocalVLMCaptioner`, `AggregatorBackend`, `GeminiAggregator`, `LocalLLMAggregator` | No |
| `kuavi/types.py` | Add `CAPTION_PRESETS` (api/local-full/local-efficient/local-minimal), `caption_preset` config field | No |
| `kuavi/indexer.py:401-510` | Stage 5: Replace hardcoded `frame_caption_fn`/`caption_fn` with `CaptionerBackend` interface | No (api preset = current behavior) |
| `kuavi/indexer.py:967-1090` | Stage 6: Replace hardcoded `refine_fn` with `AggregatorBackend` interface | No |
| `kuavi/mcp_server.py` | `kuavi_index_video` gets `caption_preset` parameter | No |
| `rlm/video/video_indexer.py` | Mirror captioner abstraction | No |
| Future: `kuavi/vidqa.py` | V-JEPA 2 + MLP projector + LLM (when weights available) | N/A |

**KUAVi impact**: Zero API cost indexing with `local-*` presets. Tiered GPU support: 8GB (local-minimal) to 48GB (local-full). Existing Gemini path preserved as `api` preset — zero breaking changes.

**RLM impact**: `video_indexer.py` gains backend selection. `run_video.py` gains `--caption-preset` arg.

**Model cascade (Action100M validated)**:
- Frame captioner: Llama-3.2-Vision-11B (or PLM-3B for efficient preset)
- Segment captioner: PerceptionLM-3B
- Aggregator: GPT-OSS-120B (or Llama-3.3-8B for efficient preset)

**Risk**: Medium — model loading/inference engineering, VRAM management
**Effort**: 5-7 days
**VRAM**: api=0GB, local-minimal=8GB, local-efficient=16GB, local-full=48GB

---

#### Work Item 6: Overlapping V-JEPA 2 Windows with Per-Frame Averaging
**Source**: V-JEPA 2 Integration 3 + Action100M Integration 5 (merged)

Replace non-overlapping 16-frame clip encoding with overlapping 64-frame windows (8-frame stride) producing per-frame averaged embeddings. This is the encoding strategy Action100M validated on 1.2M videos.

**Changes**:

| File | Change | Breaking? |
|------|--------|-----------|
| `kuavi/indexer.py` | New `_encode_frames_overlapping_vjepa()` — replaces `_group_frames_into_clips` + `_encode_clips_vjepa` | Scene boundaries change |
| `kuavi/scene_detection.py` | New `detect_scenes_perframe()` — Ward clustering on per-frame embeddings instead of per-clip | Scene boundaries change |
| `kuavi/scene_detection.py:46-91` | Threshold adjustment: 0.30 → 0.20 for 64-frame overlapping | Threshold change |
| `kuavi/types.py` | New `scene_stride` (default 8), updated `scene_clip_size` default to 64 | No |
| `rlm/video/video_indexer.py` | Mirror encoding changes | Same |

**KUAVi impact**: Each frame appears in up to 8 windows → 8x more context per frame. Smoother embedding trajectory → more accurate scene boundaries. Per-frame temporal resolution vs current per-clip. Cached indexes need re-generation.

**RLM impact**: Scene detection quality improvement benefits all downstream REPL tools. Mirror encoding changes.

**Dependencies**: Benefits greatly from Work Item 1 (ViT-g handles 64 frames natively). Can work with current ViT-L using smaller windows (e.g., 16-frame windows, stride 4).

**Risk**: Medium — memory scaling (each frame in up to 8 windows), compute increase, threshold tuning needed
**Effort**: 4-5 days

---

#### Work Item 7: Preserve Spatial Feature Maps
**Source**: V-JEPA 2 Integration 2

Stop discarding the spatial feature map. Store both pooled embeddings (for search) and full patch tokens (for VidQA, anticipation, classification).

**Changes**:

| File | Change | Breaking? |
|------|--------|-----------|
| `kuavi/indexer.py:1474-1507` | `_encode_clips_vjepa()` — optional `return_full` parameter | No |
| `kuavi/indexer.py:563-583` | `VideoIndex` gains `temporal_feature_maps: np.ndarray | None` | No |
| Index save/load | Include feature maps in `.npz` cache (optional, compressed) | No |
| `rlm/video/video_indexer.py` | Mirror `return_full` parameter | No |

**KUAVi impact**: Enables all downstream capabilities that need spatial info (VidQA, classification, anticipation). Only stores when explicitly requested — zero overhead by default.

**RLM impact**: Enables richer `llm_query_batched()` prompts with spatial features.

**Risk**: Low — optional, additive
**Effort**: 1-2 days
**Note**: This is a prerequisite for Work Items 9, 10, and 11.

---

#### Work Item 8: Semantic Deduplication via Clustering
**Source**: Action100M Integration 3

Replace pairwise cosine dedup with EmbeddingGemma k-means clustering for semantic dedup + cluster-aware search diversity.

**Changes**:

| File | Change | Breaking? |
|------|--------|-----------|
| `kuavi/indexer.py` | New `_semantic_deduplicate()` after `_embed_captions()` — text hash dedup + k-means clustering | No (opt-in) |
| `kuavi/search.py` | Cluster-aware diversity in `_rerank_mmr()` or new `_cluster_diverse_rerank()` | No (opt-in) |
| `kuavi/mcp_server.py` | Wire up existing `cluster_diverse` parameter on `kuavi_search_video` | No |
| `rlm/video/video_indexer.py` | Mirror dedup method | No |
| `rlm/video/video_search_tools.py` | Cluster diversity parameter passthrough | No |

**KUAVi impact**: Catches semantic duplicates (e.g., "stir pot" appearing 15 times in cooking video with different frames). Prevents repetitive actions from dominating search results. `cluster_diverse` parameter already exists on `kuavi_search_video` — just needs wiring.

**RLM impact**: Cluster diversity available in REPL search tools.

**Dependencies**: Benefits from Work Item 2 (action embeddings to cluster)
**Risk**: Low — opt-in, disabled by default
**Effort**: 2-3 days

---

### Priority Tier 3: New Capabilities (Week 4-6)

These unlock entirely new functionality.

---

#### Work Item 9: Action Anticipation via V-JEPA 2 Predictor
**Source**: V-JEPA 2 Integration 4

Use V-JEPA 2's predictor (already in checkpoint, currently unused) to predict what happens next in a video.

**Changes**:

| File | Change | Breaking? |
|------|--------|-----------|
| `kuavi/indexer.py` | Load predictor alongside encoder in `_ensure_scene_model()` | No |
| `kuavi/search.py` | New `make_anticipate_action()` factory | No |
| `kuavi/mcp_server.py` | New `kuavi_anticipate_action` tool | No |
| `rlm/video/video_search_tools.py` | New `anticipate_action` tool for REPL | No |

**KUAVi impact**: New "what happens next?" capability. Given context frames, predicts future frame representation and matches against candidates or finds nearest segment.

**RLM impact**: High value — LM can reason about temporal progression in REPL code. `anticipate_action()` enables predictive reasoning loops.

**Dependencies**: Work Item 7 (needs full feature maps), verification that HuggingFace checkpoint includes predictor weights

**BLOCKER**: `_ensure_scene_model()` loads via `AutoModel.from_pretrained()`. The V-JEPA 2 predictor is a separate `VisionTransformerPredictor` that may NOT be included in the HF checkpoint. Must verify against the HF model card and/or the original `github.com/facebookresearch/vjepa2` repo before any implementation. If predictor weights aren't in the HF checkpoint, this requires custom weight loading from the original repo.

**Risk**: Medium-High — predictor loading untested, mask token creation for arbitrary "future" timestamps is undocumented in HF Transformers
**Effort**: 3-5 days if weights accessible, **blocked entirely if not**

---

#### Work Item 10: Attentive Probe Classification
**Source**: V-JEPA 2 Integration 6

Lightweight 4-layer attentive probe on frozen V-JEPA 2 features for high-accuracy video classification.

**Changes**:

| File | Change | Breaking? |
|------|--------|-----------|
| New: `kuavi/probes.py` | Attentive probe architecture + weight loading | No |
| `kuavi/mcp_server.py` | New `kuavi_classify_segment` tool | No |
| `kuavi/search.py` | Probe-based classification as optional enhancer | No |
| `rlm/video/video_search_tools.py` | New `classify_segment` tool for REPL | No |

**KUAVi impact**: Sub-second classification across 6 pre-trained tasks (SSv2 174 classes, K400 400 classes, Diving-48 48 classes, Jester 27 classes, COIN ~180 classes, ImageNet 1K classes). More powerful than current `discriminative_vqa`.

**RLM impact**: Automated labeling in REPL analysis loops.

**Dependencies**: Work Item 7 (full feature maps), pre-trained probe weights from Meta
**Risk**: Medium — conditional on weight availability
**Effort**: 3-4 days (with weights), 7-10 days (if retraining)

---

#### Work Item 11: Predictive Video Understanding (World Model)
**Source**: V-JEPA 2 Integration 7

Temporal coherence verification and "what-if" reasoning using V-JEPA 2's predictive capability.

**Changes**:

| File | Change | Breaking? |
|------|--------|-----------|
| `kuavi/mcp_server.py` | New `kuavi_verify_coherence`, `kuavi_predict_future` tools | No |
| `kuavi/search.py` | New `make_predict_future()` factory | No |
| `rlm/video/video_search_tools.py` | New `predict_future()`, `verify_coherence()` tools | No |

**KUAVi impact**: Anomaly detection (segments where predicted ≠ actual), video summarization (keep surprising segments), temporal ordering, cut detection. Novel capabilities not available in any current video analysis tool.

**RLM impact**: Highest value for VideoRLM — enables predictive reasoning loops: "what happens if the person continues this action?"

**Dependencies**: Work Item 9 (predictor loading), potentially V-JEPA 2-AC specific weights

**BLOCKER**: V-JEPA 2-AC was trained on 62 hours of **robot manipulation** data only. The energy landscape analysis in the paper (Figure 9) is measured on robot tasks. Applying coherence scores to general video (human actions, nature footage, etc.) is a completely untested extrapolation — the model may produce meaningless results. V-JEPA 2-AC weights may not be publicly released at all.

**Risk**: High — novel untested application, generalization from robot data to general video is highly uncertain, no validation path without ground truth
**Effort**: 1-2 weeks of experimentation (mostly to discover whether it generalizes)

---

#### Work Item 12: Corpus-Level Indexing
**Source**: Action100M Integration 6

Multi-video indexing pipeline with cross-video action vocabulary and corpus-wide search.

**Changes**:

| File | Change | Breaking? |
|------|--------|-----------|
| New: `kuavi/corpus.py` | `CorpusIndexer`, `CorpusIndex` — parallel video indexing, cross-video action vocabulary, corpus search | No |
| `kuavi/mcp_server.py` | New `kuavi_index_corpus`, `kuavi_search_corpus`, `kuavi_corpus_stats` tools | No |
| `kuavi/cli.py` | New `kuavi corpus index <dir>` and `kuavi corpus search <query>` commands | No |

**KUAVi impact**: Multi-video analysis workflows (surveillance, lectures, dataset curation). Cross-video action vocabulary provides corpus-level semantic understanding. Parallel indexing with ThreadPoolExecutor.

**RLM impact**: VideoRLM is single-video oriented. Could add `--corpus` mode but this is more natural for KUAVi's MCP paradigm. Lower RLM priority.

**Dependencies**: Benefits from Work Items 2 (action-first for speed) and 8 (semantic dedup across corpus)
**Risk**: Medium — new architecture, cross-video state management
**Effort**: 5-7 days

---

### Deferred: Progressive Indexing
**Source**: V-JEPA 2 Integration 8

Three-tier progressive indexing (Scan → Index → Deep Index) is architecturally elegant but lower priority given that Work Item 2 (action-first indexing) already provides the "fast first results" capability. Deferred until the foundation (Work Items 1-7) is in place. Can be implemented as a natural extension of Work Item 1 (presets define the tiers).

---

## Full Impact Analysis

### Files Modified (by frequency)

| File | Work Items | Total Changes |
|------|-----------|---------------|
| `kuavi/indexer.py` | 1, 2, 3, 4, 5, 6, 7, 8, 9 | 9 work items (heaviest) |
| `kuavi/mcp_server.py` | 1, 2, 4, 5, 9, 10, 11, 12 | 8 work items |
| `rlm/video/video_indexer.py` | 1, 2, 3, 4, 5, 6, 7, 8 | 8 work items (mirror) |
| `kuavi/types.py` | 1, 5, 6 | 3 work items |
| `kuavi/search.py` | 2, 8, 9, 10, 11 | 5 work items |
| `kuavi/scene_detection.py` | 6 | 1 work item |
| `rlm/video/video_search_tools.py` | 2, 8, 9, 10, 11 | 5 work items |

### New Files

| File | Work Item | Purpose |
|------|----------|---------|
| `kuavi/captioners.py` | 5 | CaptionerBackend protocol + implementations |
| `kuavi/probes.py` | 10 | Attentive probe architecture |
| `kuavi/corpus.py` | 12 | Corpus-level indexing |
| `kuavi/vidqa.py` | 5 (future) | V-JEPA 2 + LLM alignment wrapper |

### Breaking Changes

| Work Item | What Changes | Mitigation |
|-----------|-------------|------------|
| 2 | Default search field may change to `action` | Only when action embeddings present; explicit `field` param still works |
| 3 | Annotation output format slightly different (4s minimum, new tree format) | Only affects new indexes; cached indexes unchanged |
| 6 | Scene boundaries change with per-frame embeddings | Existing indexes remain valid; re-index for new quality level |

### VRAM Requirements

| Configuration | Models Loaded | VRAM |
|--------------|--------------|------|
| Current default | V-JEPA 2 ViT-L + SigLIP2 | ~2.0 GB |
| + Work Item 1 (quality preset) | V-JEPA 2 ViT-g + SigLIP2 | ~4.8 GB |
| + Work Item 5 (local-minimal) | + PLM-3B + Llama-3.2-3B | ~12.8 GB |
| + Work Item 5 (local-full) | + Llama-3.2-Vision-11B + PLM-3B + Llama-3.3-8B | ~32.8 GB |
| + Work Item 10 (probes) | + 6 attentive probes (~10M each) | +0.25 GB |
| Full stack (quality + local-full) | All models | ~33 GB |

---

## Dependency Graph

```
  ┌──────────────┐
  │ WI-0: Fix    │ ◄── CRITICAL PREREQUISITE
  │ temporal dim │
  └──────┬───────┘
         │
         ▼ (blocks WI-1 and WI-6)
  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
  │ WI-1: Model  │  │ WI-2: Action │  │ WI-3: Self-  │  │ WI-4: Quality│
  │   Presets    │  │   First      │  │  Refine v2   │  │   Scoring    │
  └──────┬───────┘  └──────┬───────┘  └──────────────┘  └──────────────┘
         │                 │
         │                 ├───────────────┐
         ▼                 ▼               ▼
  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
  │ WI-6: Overlap│  │ WI-8: Sem.   │  │ WI-12:Corpus │
  │  Windows     │  │  Dedup       │  │  Indexing    │
  └──────────────┘  └──────────────┘  └──────────────┘

  ┌──────────────┐
  │ WI-5: Open   │ (independent, but benefits from WI-3)
  │  Captioners  │
  └──────────────┘

  ┌──────────────┐
  │ WI-7: Spatial│ (independent, prereq for 9, 10, 11)
  │  Features    │
  └──────┬───────┘
         │
         ├──────────────────────────────┐
         ▼                              ▼
  ┌──────────────┐               ┌──────────────┐
  │ WI-9: Action │               │ WI-10:Probes │
  │ Anticipation │ ⚠ BLOCKER:    │              │ ⚠ BLOCKER:
  │              │ HF predictor  └──────────────┘ probe weights
  └──────┬───────┘ weights?
         │
         ▼
  ┌──────────────┐
  │ WI-11: World │ ⚠ BLOCKER: V-JEPA 2-AC generalization
  │   Model      │ untested (robot → general video)
  └──────────────┘
```

---

## Critical Path

The critical path determines the minimum time to unlock the most valuable capabilities:

```
Day 1:   WI-0 (fix temporal search dimension mismatch)
         → Unblocks WI-1 and WI-6; fixes currently broken feature
         [PREREQUISITE — must go first]

Week 1:  WI-2 (action-first) + WI-3 (self-refine v2) + WI-4 (quality) + WI-1 (presets)
         → Immediately better indexing: faster, higher quality, configurable
         [4 quick wins in parallel after WI-0]

Week 2:  WI-7 (spatial features) + WI-5 (captioner abstraction) + WI-8 (semantic dedup)
         → Offline-capable indexing, better search diversity
         [3 items in parallel]
         ALSO: Verify HF checkpoint for predictor weights (unblocks WI-9)
         ALSO: Check Meta releases for attentive probe weights (unblocks WI-10)

Week 3:  WI-6 (overlapping windows) + WI-9 (action anticipation, IF unblocked)
         → Best-quality scene detection, "what happens next?" capability
         [WI-6 benefits from WI-1, WI-9 needs WI-7 + verified predictor weights]

Week 4+: WI-10 (probes, IF weights available) + WI-12 (corpus)
         → Classification, multi-video
         [WI-10 needs WI-7 + weights, WI-12 benefits from WI-2+WI-8]

Future:  WI-11 (world model) — only after proof-of-concept validates generalization
         → Predictive understanding (HIGH RISK — may not work on general video)
```

**Total estimated effort**: 30-45 days of engineering across 13 work items (including RLM mirror overhead).

**External blockers to resolve in parallel**:
- [ ] Verify V-JEPA 2 predictor weights in HF checkpoint (blocks WI-9, WI-11)
- [ ] Check Meta releases for attentive probe weights (blocks WI-10 quick path)
- [ ] Verify PLM-3B (`facebook/Perception-LM-3B`) HuggingFace availability (blocks WI-5 local-full)
- [ ] Assess V-JEPA 2-AC generalization to non-robot video (blocks WI-11)

---

## RLM vs KUAVi Impact Matrix

| Work Item | KUAVi Impact | RLM Impact | Shared? |
|-----------|:---:|:---:|:---:|
| 1. Model Presets | HIGH | HIGH | Mirror in video_indexer.py |
| 2. Action-First | HIGH | HIGH | Mirror + search default |
| 3. Self-Refine v2 | HIGH | HIGH | Mirror in video_indexer.py |
| 4. Quality Scoring | MEDIUM | LOW | Mirror, advisory only |
| 5. Open Captioners | HIGH | HIGH | Mirror + backend selection |
| 6. Overlapping Windows | HIGH | HIGH | Mirror encoding changes |
| 7. Spatial Features | MEDIUM | MEDIUM | Mirror, enables future |
| 8. Semantic Dedup | MEDIUM | LOW | Mirror, opt-in parameter |
| 9. Action Anticipation | HIGH | **VERY HIGH** | New REPL tool |
| 10. Attentive Probes | HIGH | MEDIUM | New REPL tool |
| 11. World Model | HIGH | **VERY HIGH** | Predictive REPL reasoning |
| 12. Corpus Indexing | HIGH | LOW | KUAVi-native, not natural for REPL |

**Key RLM differentiator**: Work Items 9 and 11 (action anticipation + world model) have the highest RLM impact because VideoRLM's REPL loop naturally supports iterative predictive reasoning — "predict what happens next, search for similar, compare, predict further." This is exactly the kind of multi-step analysis that RLMs excel at but standard single-shot tools cannot do.

---

## Risk Summary

| Risk Level | Work Items | Mitigation |
|-----------|-----------|------------|
| **Low** | 0, 2, 3, 4, 7, 8 | Bug fix, additive/configurable, existing behavior preserved |
| **Medium** | 1, 5, 6, 12 | Embedding dim changes, model engineering, threshold tuning |
| **Medium-High** | 9, 10 | **External blockers**: predictor weights unverified in HF checkpoint; probe weights may need retraining (2-3 weeks) |
| **High** | 11 | Novel research — V-JEPA 2-AC trained on 62hrs robot data, generalization to general video is untested and uncertain |

---

## Relationship to VL-JEPA Plan

The VL-JEPA plan (7 integrations) is **not included** in this roadmap because it depends on the V-JEPA 2 foundation being upgraded first. Specifically:

- VL-JEPA's Predictor uses V-JEPA 2 as its frozen X-Encoder → benefits from Work Item 1 (ViT-g upgrade)
- VL-JEPA's unified embedding space → requires Work Item 7 (spatial features) to be in place
- VL-JEPA's caption-free indexing → synergizes with Work Item 2 (action-first already reduces caption dependency)

**Recommended sequencing**: Complete this unified plan (V-JEPA 2 + Action100M) first, then integrate VL-JEPA as a third phase that builds on the upgraded foundation.

```
Phase 1 (this plan):  V-JEPA 2 encoder upgrades + Action100M pipeline improvements
Phase 2 (future):     VL-JEPA unified embedding space + caption-free search
Phase 3 (future):     Full stack — V-JEPA 2 encoder → VL-JEPA predictor → Action100M annotations
```

---

## Quick Reference: What to Build First

If time is limited, these 5 work items deliver the most value with the least risk:

1. **WI-0: Fix temporal search** — Fix broken feature, unblock model upgrades (1 day)
2. **WI-3: Self-Refine v2** — 4-second minimum is a 30-minute change saving ~64% of LLM calls. Highest ROI per hour. (1-2 days)
3. **WI-2: Action-First Indexing** — 5-10x faster indexing, better search signal (2-3 days)
4. **WI-4: Quality Scoring** — Format compliance + degenerate detection are 30-minute extensions to existing method (1-2 days)
5. **WI-7: Spatial Features** — 1 day of work that unblocks 3 future capabilities

Total: ~8 days for the foundation that enables everything else.

**Bonus quick win** (from V-JEPA 2 analyst): Changing `scene_clip_size=16` to `scene_clip_size=64` is a **1-line change** that uses the V-JEPA 2 model (`vjepa2-vitl-fpc64-256`, note the `fpc64`) at its intended capacity. Zero risk, immediate quality improvement.
