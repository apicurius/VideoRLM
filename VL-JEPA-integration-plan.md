# VL-JEPA Integration Plan for VideoRLM & KUAVi

## Executive Summary

VL-JEPA is not just an incremental improvement — it fundamentally changes what's possible in our pipeline. Currently, KUAVi uses a 3-model stack (V-JEPA 2 for scenes, SigLIP2 for embeddings, Gemini for captioning) with a multi-stage caption-then-embed workflow. VL-JEPA collapses much of this into a single forward pass: visual input + text query → predicted embedding, with text generation only when needed. This plan identifies seven integration opportunities, ordered from lowest-risk/highest-impact to most ambitious.

---

## The Core Problem VL-JEPA Solves

Today's KUAVi pipeline has a fundamental architectural tension:

```
Current pipeline (caption-then-embed):
  frames → SigLIP2 → frame_embeddings (768-d)
  frames → Gemini captioning → text → SigLIP2/EmbeddingGemma → caption_embeddings
  frames → V-JEPA 2 clips → temporal_embeddings (1024-d)
```

Three problems:
1. **Captioning is the bottleneck** — Stage 5-6 (Tree-of-Captions + Self-Refine) consumes 80%+ of indexing time and all Gemini API cost
2. **Embedding space mismatch** — temporal search queries V-JEPA embeddings (1024-d) with SigLIP2 text encoder (768-d) — these live in different spaces
3. **Information loss through text** — visual content → text caption → text embedding loses spatial/temporal detail that never survives verbalization

VL-JEPA solves all three: it predicts target embeddings directly from visual input, in a unified space, without text as an intermediary.

---

## Integration 1: Fix the Temporal Search Mismatch (Quick Win)

### Problem

`kuavi/search.py` field="temporal" currently does:
```python
query_embedding = siglip2_text_encode(query)          # 768-d SigLIP2 space
scores = cosine_sim(query_embedding, vjepa_embeddings)  # 1024-d V-JEPA space
```

This is comparing vectors from **completely different embedding spaces**. It works at all only because both spaces have some shared structure from being trained on similar data, but it's fundamentally broken — like comparing CLIP embeddings to BERT embeddings.

### Solution

Use VL-JEPA's Predictor to produce query embeddings **in V-JEPA's native space**:

```python
# VL-JEPA approach: query → Predictor → V-JEPA-aligned embedding
query_embedding = vl_jepa_predictor(
    visual_context=None,  # query-only mode, or use representative frame
    query=query_text
)  # → 1536-d shared space (projected from V-JEPA 1024-d)
scores = cosine_sim(query_embedding, temporal_embeddings)
```

Alternatively, since VL-JEPA's Y-Encoder produces text embeddings in the shared space, we can:
```python
query_embedding = vl_jepa_y_encoder(query)  # EmbeddingGemma-300M → 1536-d
# Now query and temporal embeddings live in the SAME trained space
```

### Impact
- Fixes a correctness bug in temporal search
- No pipeline changes needed — just swap the query encoder for field="temporal"
- Expected: significant improvement on motion-centric queries

### Files to modify
- `kuavi/indexer.py`: Re-embed temporal embeddings through VL-JEPA's projection layer during Stage 7
- `kuavi/search.py`: Use VL-JEPA Y-Encoder for temporal field query encoding
- `rlm/video/video_search_tools.py`: Same change for VideoRLM path

---

## Integration 2: VL-JEPA as a Unified Embedding Space (Medium Effort)

### Problem

KUAVi currently maintains **four separate embedding spaces**:

| Field | Embedding Source | Query Encoder | Dim |
|-------|-----------------|---------------|-----|
| summary | SigLIP2 or EmbeddingGemma text encode(caption) | Same | 768 or 256 |
| action | SigLIP2 or EmbeddingGemma text encode(action_brief) | Same | 768 or 256 |
| visual | SigLIP2 vision encode(frame) | SigLIP2 text | 768 |
| temporal | V-JEPA 2 clip encode | SigLIP2 text (!) | 1024 |

The `field="all"` weighted composite (0.4/0.2/0.2/0.2) is a hack — it averages scores across incompatible spaces. And summary/action embeddings require captioning first (expensive).

### Solution

Replace all four with VL-JEPA's unified embedding space:

```
VL-JEPA Unified Pipeline:
  frames → X-Encoder (frozen V-JEPA 2) → S_V
  (S_V, query) → Predictor → Ŝ_Y  (1536-d, shared space)
  text → Y-Encoder (EmbeddingGemma) → S_Y  (1536-d, same space)
```

**New search architecture:**

```python
def search_unified(query, index, field="all"):
    # All embeddings now live in VL-JEPA's 1536-d shared space
    query_emb = vl_jepa_y_encoder(query)  # text → shared space

    if field == "visual":
        # VL-JEPA Predictor: (visual_embs, "describe this") → shared space
        scores = cosine_sim(query_emb, index.vl_jepa_visual_embeddings)
    elif field == "temporal":
        # VL-JEPA Predictor: (temporal_clips, "what is happening") → shared space
        scores = cosine_sim(query_emb, index.vl_jepa_temporal_embeddings)
    elif field == "semantic":
        # VL-JEPA Predictor: (visual_embs, "summarize this scene") → shared space
        scores = cosine_sim(query_emb, index.vl_jepa_semantic_embeddings)
    elif field == "all":
        # NOW this is a proper weighted average — all in same space
        scores = weighted_mean([visual_scores, temporal_scores, semantic_scores])
```

### Key Insight

VL-JEPA's Predictor is **query-conditioned**. By varying the query prompt while keeping the same visual input, we can produce **different semantic projections** of the same segment — all in the same embedding space. This is far more principled than the current approach of captioning → text embedding.

```python
# Same visual input, different semantic views
visual_emb = vl_jepa_predict(frames, "describe what you see")
action_emb = vl_jepa_predict(frames, "what action is being performed")
object_emb = vl_jepa_predict(frames, "what objects are present")
# All in same 1536-d space — composable, comparable
```

### Impact
- Eliminates the multi-space problem entirely
- `field="all"` becomes mathematically sound
- Opens up **arbitrary query-conditioned search fields** without re-indexing
- Reduces index storage (one space instead of four)

### Files to modify
- `kuavi/indexer.py`: New `_embed_vl_jepa()` method in Stage 7, producing unified embeddings
- `kuavi/search.py`: Rewrite search routing to use unified space
- `kuavi/types.py`: New config fields for VL-JEPA model paths
- `rlm/video/video_indexer.py`: Mirror changes

---

## Integration 3: Caption-Free Indexing via Direct Embedding Prediction (High Impact)

### Problem

Stages 5-6 (Tree-of-Captions + 3 rounds of Self-Refine) are:
- **Slow**: 8 parallel Gemini API calls per segment × 3 refinement rounds = 24 LLM calls per segment
- **Expensive**: Gemini API cost dominates total indexing cost
- **Lossy**: Visual information → text → embedding loses spatial detail

The entire reason captions exist is to produce searchable text embeddings. But VL-JEPA can produce those embeddings **directly from frames**, skipping text entirely.

### Solution: Two-Track Indexing

```
Track A — VL-JEPA Direct (fast, caption-free):
  Stage 1: Frame extraction
  Stage 2: Scene detection (V-JEPA 2, unchanged)
  Stage 3: ASR transcript (unchanged)
  Stage 4: Selective decoding (enhanced, see Integration 4)
  Stage 5-NEW: VL-JEPA Embedding
    For each segment:
      visual_emb = vl_jepa_predict(frames, "describe this scene in detail")
      action_emb = vl_jepa_predict(frames, "what actions are being performed")
      object_emb = vl_jepa_predict(frames, "what objects and people are visible")
  Stage 6: SKIP (no self-refine needed — no text to refine)
  Stage 7: Store embeddings (already produced in Stage 5-NEW)
  Stage 8: Hierarchy (unchanged)

Track B — Full Pipeline (when text captions needed):
  Same as current pipeline, used only when:
  - User explicitly requests scene descriptions
  - get_scene_list() is called (needs text annotations)
  - Debugging / explainability required
```

### Lazy Caption Generation

Instead of captioning at index time, generate captions **on demand** when `get_scene_list()` is called:

```python
def get_scene_list(index):
    for segment in index.segments:
        if segment.caption is None:
            # Only now invoke Gemini to produce text
            segment.caption = gemini_caption(segment.frames)
        yield segment
```

This inverts the current flow: embeddings are the primary representation, text is a derived view.

### Impact
- **10-50x faster indexing** (eliminate all Gemini API calls during indexing)
- **Zero API cost** for indexing (all models run locally)
- **No information loss** — embeddings capture visual detail that text can't express
- Captions still available on demand for explainability

### Complexity
- Requires VL-JEPA model weights (~1.6B params, but Predictor is only 490M)
- Need to validate search quality without captions
- `get_scene_list()` needs lazy caption generation path

### Files to modify
- `kuavi/indexer.py`: New `_index_vl_jepa()` fast path, lazy `_generate_caption()`
- `kuavi/search.py`: Updated to use VL-JEPA embeddings
- `kuavi/mcp_server.py`: `kuavi_index_video` gets `--mode fast|full` flag
- `run_video.py`: `--caption-free` flag for VideoRLM

---

## Integration 4: VL-JEPA-Powered Selective Decoding (Elegant Upgrade)

### Problem

KUAVi's current selective decoding (Tier 0/1/2) is a heuristic system:
- Tier 0 (DEAD): pixel_std < 5 or edge_density < 0.01 — pure pixel statistics
- Tier 1 (STATIC): SigLIP2 sim > 0.98 — visual similarity
- Tier 2 (DYNAMIC): everything else

This works but is coarse. The paper shows VL-JEPA achieves **2.85x decoding reduction** with its embedding-guided selective decoding on EgoExo4D.

### Solution

Replace the heuristic tiers with VL-JEPA's **semantic change detection**:

```python
def selective_decode_vl_jepa(segments, vl_jepa_model):
    """VL-JEPA-native selective decoding.

    Instead of pixel heuristics, monitor the VL-JEPA embedding stream.
    Decode only when semantic content changes significantly.
    """
    embeddings = []
    for segment in segments:
        emb = vl_jepa_model.predict(segment.frames, query="describe this")
        embeddings.append(emb)

    # Agglomerative clustering on embedding stream
    # (exactly as described in VL-JEPA paper Section 4.6)
    clusters = ward_clustering(
        embeddings,
        connectivity="temporal",  # only adjacent segments can merge
        variance_threshold=0.05   # semantic coherence threshold
    )

    for cluster in clusters:
        if cluster.internal_variance < STATIC_THRESHOLD:
            # Semantically uniform — caption only the midpoint
            cluster.tier = "STATIC"
            cluster.decode_point = cluster.midpoint
        else:
            # Semantically varying — full captioning needed
            cluster.tier = "DYNAMIC"

    return clusters
```

### Key Advantage Over Current System

The current tier system makes **per-segment** decisions independently. VL-JEPA's approach considers the **embedding stream holistically** — it identifies semantically coherent *regions* (potentially spanning multiple segments) and decodes at segment boundaries only.

```
Current (per-segment):     [T2][T1][T1][T2][T0][T2][T1][T2]
                           (each segment decided independently)

VL-JEPA (stream-based):   [──STATIC──][──DYNAMIC──][STATIC]
                              ↑ decode    ↑↑ decode    ↑ decode
                           (regions identified, decode at boundaries)
```

### Impact
- More accurate than pixel heuristics
- Reduces captioning further (merge adjacent similar segments)
- Aligns with how the paper achieves 2.85x reduction
- Can be applied during **both** indexing and **real-time streaming**

### Files to modify
- `kuavi/indexer.py`: Replace `_selective_decode()` with `_selective_decode_vl_jepa()`
- `kuavi/scene_detection.py`: VL-JEPA embeddings as an alternative to SigLIP2 for scene boundaries

---

## Integration 5: Native Discriminative VQA via VL-JEPA Predictor (Drop-in Upgrade)

### Problem

Current `kuavi_discriminative_vqa` implementation:
```python
# Current: encode "question + candidate" as flat text, compare to caption embeddings
for candidate in candidates:
    text = f"{question} {candidate}"
    emb = text_encoder(text)  # SigLIP2 or EmbeddingGemma
    score = cosine_sim(emb, segment_embeddings)  # caption embeddings
```

This is suboptimal because:
- Caption embeddings are a lossy proxy for visual content
- The question isn't conditioned on the visual input
- Text-only encoding can't capture visual nuance

### Solution

Use VL-JEPA's Predictor for **visually-grounded discriminative VQA**:

```python
def discriminative_vqa_vl_jepa(question, candidates, frames, vl_jepa):
    # VL-JEPA: predict answer embedding conditioned on BOTH visual input AND question
    predicted_emb = vl_jepa.predict(frames, question)  # (S_V, X_Q) → Ŝ_Y

    # Encode each candidate through Y-Encoder
    candidate_embs = [vl_jepa.y_encode(c) for c in candidates]

    # Score by proximity in shared space
    scores = [cosine_sim(predicted_emb, c_emb) for c_emb in candidate_embs]
    return sorted(zip(candidates, scores), key=lambda x: -x[1])
```

This is **exactly** how VL-JEPA is evaluated on GQA, TallyQA, POPE — the paper reports strong results (61.5% GQA, 69.9% TallyQA, 85.7% POPE).

### Impact
- Much more accurate VQA — visually grounded, question-conditioned
- Still zero LLM generation cost (pure embedding comparison)
- Drop-in replacement for existing discriminative VQA tool

### Files to modify
- `kuavi/search.py`: `make_discriminative_vqa()` enhanced with VL-JEPA path
- `rlm/video/video_search_tools.py`: Same
- `kuavi/mcp_server.py`: No changes needed (same tool interface)

---

## Integration 6: Real-Time Streaming for VideoRLM (New Capability)

### Problem

VideoRLM currently processes video in batch:
1. Index entire video upfront
2. Run RLM REPL loop against static index
3. No support for live/streaming video

### Solution: VL-JEPA Streaming Mode

VL-JEPA's non-autoregressive nature enables a **continuous semantic monitoring** mode:

```python
class StreamingVideoRLM:
    def __init__(self, vl_jepa_model):
        self.vl_jepa = vl_jepa_model
        self.embedding_buffer = SlidingWindowBuffer(window_size=30)  # 30s
        self.last_decoded = None

    def process_frame(self, frame, timestamp):
        # 1. Encode frame (fast, non-autoregressive)
        emb = self.vl_jepa.predict([frame], query="what is happening now")
        self.embedding_buffer.append(emb, timestamp)

        # 2. Check for semantic shift
        if self.embedding_buffer.variance() > SHIFT_THRESHOLD:
            # Semantic change detected — decode and notify
            description = self.vl_jepa.decode(self.embedding_buffer.mean())
            self.last_decoded = description
            self.notify_rlm(description, timestamp)

        # 3. Always available for discriminative queries
        # (no decoding needed, just embedding comparison)

    def answer_question(self, question, candidates):
        """Real-time VQA against current buffer — no generation needed."""
        predicted = self.vl_jepa.predict(
            self.embedding_buffer.frames, question
        )
        return rank_candidates(predicted, candidates)
```

### Architecture

```
Live Video Stream
     │
     ▼ (frame-by-frame)
┌─────────────────────────────┐
│  VL-JEPA X-Encoder          │ (frozen V-JEPA 2, runs continuously)
│  → S_V embedding stream     │
└──────────┬──────────────────┘
           │
     ┌─────▼─────┐
     │ Predictor  │ (490M params, fast forward pass)
     │ + query    │
     └─────┬─────┘
           │
     Ŝ_Y embedding stream
           │
     ┌─────▼──────────────┐
     │ Semantic Monitor    │
     │ - sliding window    │
     │ - change detection  │
     │ - selective decode  │
     └─────┬──────────────┘
           │
     ┌─────▼──────────────┐
     │ RLM REPL Loop       │
     │ - tools available:  │
     │   search_buffer()   │
     │   get_current()     │
     │   answer_question() │
     │   extract_frame()   │
     └────────────────────┘
```

### Impact
- Enables real-time video analysis (security cameras, sports, live events)
- VL-JEPA paper shows this works well on EgoExo4D (procedural activities)
- RLM agent can react to semantic changes as they happen
- Discriminative VQA works in real-time without any text generation

### Files to modify
- `rlm/video/video_rlm.py`: New `StreamingVideoRLM` class
- `rlm/video/video_indexer.py`: Streaming-compatible incremental indexing
- `kuavi/mcp_server.py`: New `kuavi_start_stream` / `kuavi_query_stream` tools

---

## Integration 7: World Model Prediction (Most Ambitious)

### Background

VL-JEPA_SFT achieves **65.7% SOTA on WorldPrediction-WM** — predicting what happens next given initial and final state images. This suggests VL-JEPA learns a latent world model.

### Application in KUAVi

Currently, KUAVi's agent only reasons about **what has happened** (post-hoc analysis). With VL-JEPA's predictive capabilities, we can add **forward reasoning**:

```python
# New MCP tool: kuavi_predict_next
def predict_next(current_frames, candidates):
    """Given current visual state, predict what happens next.

    Uses VL-JEPA's world model capabilities to rank candidate
    future states by likelihood.
    """
    state_emb = vl_jepa.predict(
        current_frames,
        "what will happen next in this video"
    )

    candidate_embs = [vl_jepa.y_encode(c) for c in candidates]
    return rank_by_similarity(state_emb, candidate_embs)

# New MCP tool: kuavi_predict_action
def predict_action(before_frames, after_frames, candidate_actions):
    """Given before/after states, identify the action that occurred.

    This is exactly the WorldPrediction-WM task where VL-JEPA
    achieves 65.7% SOTA.
    """
    state_emb = vl_jepa.predict(
        concat(before_frames, after_frames),
        "what action explains the transition between these states"
    )

    action_embs = [vl_jepa.y_encode(a) for a in candidate_actions]
    return rank_by_similarity(state_emb, action_embs)
```

### Use Cases
- **Procedural video understanding**: "What step comes next?"
- **Anomaly detection**: Predicted next state vs actual next state → anomaly score
- **Action anticipation**: EPIC-Kitchens-style "what will the person do in 1/4/10 seconds?"
- **Causal reasoning**: "Why did X happen?" → identify the transition that explains the observed state change

### Impact
- Transforms KUAVi from reactive analysis to predictive understanding
- Enables new agent capabilities (planning, anticipation)
- Paper proves SOTA on the relevant benchmarks

### Files to modify
- `kuavi/mcp_server.py`: New `kuavi_predict_next`, `kuavi_predict_action` tools
- `kuavi/search.py`: New prediction factories
- `.claude/agents/video-analyst.md`: Updated prompt for predictive reasoning

---

## Implementation Roadmap

### Phase 1: Foundation (1-2 weeks)

```
1.1 Add VL-JEPA model loading to kuavi/indexer.py
    - Lazy loading pattern (like existing _ensure_model())
    - Support frozen V-JEPA 2 X-Encoder (already loaded) + Predictor + Y-Encoder
    - Config via KUAViConfig: vl_jepa_predictor_path, vl_jepa_y_encoder_path

1.2 Fix temporal search mismatch (Integration 1)
    - Use VL-JEPA Y-Encoder for temporal field query encoding
    - Validate on existing test suite

1.3 Upgrade discriminative VQA (Integration 5)
    - Add VL-JEPA-backed VQA path alongside existing
    - A/B compare on GQA-style queries
```

### Phase 2: Unified Embeddings (2-3 weeks)

```
2.1 Implement VL-JEPA embedding stage (Integration 2)
    - New _embed_vl_jepa() in Stage 7
    - Query-conditioned multi-view embeddings
    - Validate search quality vs current pipeline

2.2 Implement caption-free indexing (Integration 3)
    - --mode fast flag for kuavi_index_video
    - Lazy caption generation for get_scene_list()
    - Benchmark indexing speed vs current pipeline

2.3 VL-JEPA selective decoding (Integration 4)
    - Replace heuristic tiers with embedding-stream clustering
    - Validate on diverse video types
```

### Phase 3: Advanced Capabilities (3-4 weeks)

```
3.1 Streaming mode for VideoRLM (Integration 6)
    - StreamingVideoRLM class
    - Incremental indexing
    - Real-time semantic monitoring

3.2 World model prediction tools (Integration 7)
    - kuavi_predict_next, kuavi_predict_action
    - Action anticipation benchmarks
    - Agent prompt updates
```

---

## Model Requirements

| Component | Source | Params | VRAM | Already in KUAVi? |
|-----------|--------|--------|------|--------------------|
| X-Encoder (V-JEPA 2 ViT-L) | `facebook/vjepa2-vitl-fpc64-256` | 304M | ~1.2 GB | Yes (scene detection) |
| Predictor (Llama 3.2-1B layers 8-16) | VL-JEPA release | 490M | ~2.0 GB | No — new |
| Y-Encoder (EmbeddingGemma-300M) | `google/embeddinggemma-300m` | 300M | ~0.6 GB | Yes (optional text encoder) |
| Y-Decoder | VL-JEPA release | ~300M | ~0.6 GB | No — new (only needed for text generation) |
| **Total new** | | **~790M** | **~2.6 GB** | |

The Predictor is the only truly new component. Y-Decoder is only needed for Integrations 3 (lazy caption generation) and 6 (streaming decode).

---

## Risk Assessment

| Integration | Risk | Mitigation |
|-------------|------|------------|
| 1. Temporal fix | Low — pure query encoder swap | Keep SigLIP2 as fallback |
| 2. Unified space | Medium — changes search semantics | A/B test against current pipeline |
| 3. Caption-free | Medium — loses text explainability | Lazy caption generation preserves it |
| 4. Selective decode | Low — better version of existing | Keep heuristic tiers as fast fallback |
| 5. Discriminative VQA | Low — drop-in upgrade | Existing path as fallback |
| 6. Streaming | High — new capability, untested | Prototype on short clips first |
| 7. World model | High — novel application | Start with WorldPrediction-WM benchmark |

---

## Summary

VL-JEPA offers KUAVi/VideoRLM three categories of improvement:

**Correctness fixes:**
- Integration 1 fixes the temporal search space mismatch (currently broken)
- Integration 5 gives visually-grounded VQA (currently text-only proxy)

**Efficiency gains:**
- Integration 3 eliminates captioning during indexing (10-50x faster, zero API cost)
- Integration 4 provides principled selective decoding (2.85x reduction proven)

**New capabilities:**
- Integration 2 enables arbitrary query-conditioned search (no re-indexing)
- Integration 6 enables real-time streaming analysis
- Integration 7 enables predictive reasoning (what happens next)

The remarkable thing is that VL-JEPA achieves all this with **fewer total parameters** than the current 3-model + Gemini setup, and the two most expensive current components (Gemini captioning, SigLIP2 text encoding) become optional rather than required.
