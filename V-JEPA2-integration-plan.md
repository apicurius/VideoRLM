# V-JEPA 2 Integration Plan for VideoRLM & KUAVi

## Executive Summary

V-JEPA 2 is the foundational vision encoder already at the heart of KUAVi's scene detection — but the paper reveals we are using roughly **10% of its capabilities**. We load the smallest model (ViT-L, 300M) at the lowest resolution (256px) with the shortest clips (16 frames), use it only for scene boundary clustering, and discard everything else. Meanwhile, the paper proves this same architecture — when scaled and properly leveraged — achieves SOTA on action classification, action anticipation, video QA, and even zero-shot robot manipulation.

This plan identifies eight integrations organized around three themes: **scaling what we have**, **unlocking latent capabilities**, and **enabling new paradigms**.

---

## Current State: How KUAVi Uses V-JEPA 2

```
Current pipeline:
  frames → group into 16-frame clips
         → V-JEPA 2 ViT-L (300M, 256px)
         → mean-pool patch tokens → L2-norm → (N_clips, 1024)
         → Ward clustering (threshold=0.3) → scene boundaries
         → average per-segment → temporal_embeddings (N_segs, 1024)
         → search temporal field: SigLIP2 text query (768-d) vs V-JEPA embs (1024-d)
```

**What's wrong:**
1. **Smallest model**: ViT-L (300M) when ViT-g (1B) exists with +1.5 avg accuracy
2. **Lowest resolution**: 256px when 384px yields +0.9 avg accuracy
3. **Shortest clips**: 16 frames when 64 frames adds +0.7 avg accuracy
4. **Mean-pool discards spatial info**: `patch_tokens.mean(dim=1)` throws away the 16×16 spatial feature map
5. **Cross-space search**: SigLIP2 (768-d, vision-language space) queries V-JEPA (1024-d, self-supervised motion space) — these are semantically incompatible embedding spaces even if dimensions were handled
6. **Scene detection only**: The predictor (which predicts masked/future representations) is never used
7. **No action recognition**: Despite V-JEPA 2 being SOTA on SSv2 (77.3%), EK100 action anticipation (39.7 recall@5)
8. **No VidQA**: Despite V-JEPA 2 + LLM being SOTA on PerceptionTest, TempCompass, TOMATO

---

## Theme 1: Scale What We Have

### Integration 1: Model Upgrade — ViT-L → ViT-g at 384px, 64 Frames

#### Problem

KUAVi hardcodes `facebook/vjepa2-vitl-fpc64-256` — the smallest V-JEPA 2 variant. The paper proves every scaling axis independently improves performance:

| Current | Upgrade | Paper Evidence |
|---------|---------|----------------|
| ViT-L (300M) | ViT-g (1B) | +1.5 avg accuracy (Table 4) |
| 256px | 384px | +0.9 avg accuracy (Figure 5) |
| 16 frames | 64 frames | +0.7 avg accuracy (Figure 5, Right) |
| **Total** | | **+3.1 cumulative** |

On motion-specific benchmarks the gains are even larger:
- SSv2: 73.7 → 77.3 (+3.6 points)
- EK100 action anticipation: 32.7 → 39.7 (+7.0 points)

#### Solution

Make V-JEPA 2 model variant configurable with tiered presets:

```python
# kuavi/types.py
VJEPA2_PRESETS = {
    "fast": {
        "model": "facebook/vjepa2-vitl-fpc64-256",  # Current (300M, 256px)
        "clip_size": 16,
        "resolution": 256,
        "vram_gb": 1.2,
    },
    "balanced": {
        "model": "facebook/vjepa2-vith-fpc64-256",  # 600M, 256px
        "clip_size": 32,
        "resolution": 256,
        "vram_gb": 2.4,
    },
    "quality": {
        "model": "facebook/vjepa2-vitg-fpc64-384",  # 1B, 384px
        "clip_size": 64,
        "resolution": 384,
        "vram_gb": 4.0,
    },
}
```

#### VRAM and Speed Trade-offs

| Preset | Params | VRAM | Clips/sec (est.) | Avg Accuracy |
|--------|--------|------|:-:|:-:|
| fast (current) | 300M | ~1.2 GB | ~12 | 86.0 |
| balanced | 600M | ~2.4 GB | ~6 | 86.4 |
| quality | 1B | ~4.0 GB | ~3 | 88.2 |

For indexing (one-time cost), `quality` is worth it. For real-time, `fast` remains appropriate.

#### Files to Modify
- `kuavi/types.py`: Add `VJEPA2_PRESETS`, new config field `scene_model_preset`
- `kuavi/indexer.py:144-145`: Use preset to set model name and clip size
- `kuavi/indexer.py:1509-1523`: `_group_frames_into_clips()` — support 32/64 frame clips
- `kuavi/indexer.py:1474-1507`: `_encode_clips_vjepa()` — adjust batch size for larger models (batch_size=2 for ViT-g)
- `kuavi/mcp_server.py`: `kuavi_index_video` gets `scene_model_preset` parameter
- `rlm/video/video_indexer.py`: Mirror all changes

---

### Integration 2: Preserve Spatial Feature Maps (Stop Discarding Information)

#### Problem

Current code does `patch_tokens.mean(dim=1)` — collapsing the spatial feature map `(batch, H×W×T, 1024)` into a single `(batch, 1024)` vector. For ViT-L at 256px with 16 frames, this throws away a `16×16×8 = 2048` element feature map (each with 1024 dims).

The paper uses the **full spatiotemporal feature map** for:
- Action anticipation (probe attends to full patch features)
- V-JEPA 2-AC world model (predictor operates on `16×16×1408` per frame)
- VidQA (visual tokens fed to LLM without pooling)

#### Solution

Store both pooled AND full feature maps, with a lazy loading pattern:

```python
class VideoIndex:
    temporal_embeddings: np.ndarray          # (N_segments, 1024) — pooled, for search
    temporal_feature_maps: np.ndarray | None # (N_clips, H*W*T, 1024) — full, for advanced use
```

**When to use each:**
- **Pooled** (current): Search, scene detection, selective decoding — fast, low-memory
- **Full feature maps**: VidQA (feed as visual tokens to LLM), action anticipation (attentive probe), pixel-level analysis, V-JEPA 2-AC planning

```python
def _encode_clips_vjepa(self, clips, return_full=False):
    # ... existing code ...
    outputs = self._scene_model(**inputs)
    patch_tokens = outputs.last_hidden_state  # (batch, seq_len, 1024)

    pooled = patch_tokens.mean(dim=1)
    pooled = pooled / pooled.norm(p=2, dim=-1, keepdim=True)

    if return_full:
        return pooled.cpu().numpy(), patch_tokens.cpu().numpy()
    return pooled.cpu().numpy(), None
```

#### Impact
- Enables all downstream integrations (VidQA, anticipation, planning) that need spatial information
- Minimal cost: only store when explicitly requested (`return_full=True`)
- Feature maps can be saved to disk alongside embeddings (`.npz` format already used)

#### Files to Modify
- `kuavi/indexer.py:1474-1507`: `_encode_clips_vjepa()` — optional `return_full` parameter
- `kuavi/indexer.py:563-583`: Store both pooled and full maps in VideoIndex
- Index save/load: Include feature maps in `.npz` cache (optional, compressed)

---

### Integration 3: 64-Frame Temporal Context for Scene Detection

#### Problem

With 16-frame clips at typical video FPS, each clip covers only ~0.5-2 seconds. This is too short to capture many meaningful actions (e.g., "person walks across room" may take 5-10 seconds). The paper shows 64-frame clips improve downstream tasks by +0.7 avg.

More importantly, 64-frame clips fundamentally change what scene detection captures:

```
16-frame clip: "hand reaches toward cup"
64-frame clip: "person picks up cup, drinks, puts it down"
```

#### Solution

With the `quality` preset (Integration 1), clips are already 64 frames. But we also need to adapt the scene detection algorithm:

```python
# Overlapping clips for smoother scene boundaries
def _group_frames_into_clips_overlapping(self, frames, timestamps, clip_size, stride):
    """Overlapping clips with configurable stride for finer temporal resolution."""
    clips, clip_timestamps = [], []
    for i in range(0, len(frames) - clip_size + 1, stride):
        clip = frames[i : i + clip_size]
        mid = i + clip_size // 2
        clips.append(clip)
        clip_timestamps.append(timestamps[mid])
    return clips, clip_timestamps
```

**Current**: Non-overlapping 16-frame clips → coarse temporal boundaries
**Upgraded**: 64-frame clips with stride=16 → 4x overlap → much smoother scene boundaries, better temporal resolution

#### Ward Clustering Adaptation

Longer clips = smoother embedding trajectories = different optimal distance thresholds:

| Clip Size | Recommended Threshold | Min Duration |
|-----------|:---------------------:|:------------:|
| 16 frames | 0.30 (current) | 0.5s |
| 32 frames | 0.25 | 1.0s |
| 64 frames | 0.20 | 2.0s |

#### Files to Modify
- `kuavi/indexer.py:1509-1523`: New `_group_frames_into_clips_overlapping()`
- `kuavi/scene_detection.py:46-91`: Adjust default threshold based on clip length
- `kuavi/types.py`: Add `scene_clip_stride` config

---

## Theme 2: Unlock Latent Capabilities

### Integration 4: V-JEPA 2 Predictor for Action Anticipation

#### Background

The V-JEPA 2 paper's most striking result is action anticipation on EK100: **39.7 recall@5** — a **44% relative improvement** over PlausiVL (8B). This uses a frozen V-JEPA 2 encoder + predictor with mask tokens for the future frame, feeding into a lightweight attentive probe.

KUAVi already has the encoder loaded. The predictor is **part of the V-JEPA 2 checkpoint** but we never use it.

#### How It Works

```
Context frames (t-N ... t-1)
    │
    ▼
V-JEPA 2 Encoder (frozen)
    │
    ├── encoder output: patch tokens for context frames
    │
    └── predictor input: [encoder output] + [mask tokens for t+1]
              │
              ▼
        V-JEPA 2 Predictor
              │
              ▼
        Predicted representation of future frame (t+1)
```

The predictor literally **imagines** what the next frame's representation will look like, without generating pixels.

#### Application in KUAVi

New MCP tool: `kuavi_anticipate_action`

```python
def kuavi_anticipate_action(
    start_time: float,
    end_time: float,
    anticipation_time: float = 1.0,
    candidates: list[str] | None = None,
    video_id: str | None = None,
) -> dict:
    """Predict what action will happen next after the given time range.

    Uses V-JEPA 2's predictor to predict the representation of the
    future frame at (end_time + anticipation_time), then matches against
    candidate action embeddings or returns nearest segment.

    Args:
        start_time: Start of context window
        end_time: End of context window
        anticipation_time: How far into the future to predict (seconds)
        candidates: Optional action labels to rank

    Returns:
        Predicted action ranking or nearest matching segment
    """
    # 1. Extract context frames and encode
    context_frames = extract_frames(start_time, end_time)
    context_repr = vjepa2_encoder(context_frames)

    # 2. Create mask tokens for future frame
    future_mask = create_mask_tokens(anticipation_time)

    # 3. Run predictor to get future representation
    future_repr = vjepa2_predictor(context_repr, future_mask)

    # 4. If candidates provided, rank by similarity (discriminative)
    if candidates:
        candidate_embs = [encode_action(c) for c in candidates]
        return rank_by_similarity(future_repr, candidate_embs)

    # 5. Otherwise, find nearest segment in the indexed video
    scores = cosine_sim(future_repr, index.temporal_embeddings)
    return {"predicted_segment": segments[scores.argmax()]}
```

#### Integration with VideoRLM

In the REPL loop, the LLM can call `anticipate_action()` to reason about **what happens next**:

```python
# In REPL:
context = extract_frames(10.0, 15.0)
prediction = anticipate_action(10.0, 15.0, anticipation_time=2.0,
                                candidates=["opens door", "picks up phone", "sits down"])
# → {"ranking": [("opens door", 0.87), ("picks up phone", 0.45), ("sits down", 0.31)]}
```

#### Impact
- New capability: temporal prediction without any generation model
- Leverages model weights we already download but never use
- Paper proves SOTA performance on this exact task

#### Files to Modify
- `kuavi/indexer.py`: Load predictor alongside encoder in `_ensure_scene_model()`
- `kuavi/search.py`: New `make_anticipate_action()` factory
- `kuavi/mcp_server.py`: New `kuavi_anticipate_action` tool
- `rlm/video/video_search_tools.py`: New `anticipate_action` tool for REPL

---

### Integration 5: V-JEPA 2 as Vision Encoder for VidQA (Replace Gemini Captioning)

#### Background

Section 7 of the paper proves V-JEPA 2 can be aligned with an LLM (Qwen2-7B) for SOTA video question answering, **without any language supervision during pretraining**. This outperforms SigLIP2 and PE-Core as vision encoders.

Currently, KUAVi's most expensive operation is Gemini-based captioning (Stages 5-6). What if we replace it with a local V-JEPA 2 + small LLM pipeline?

#### Architecture

```
Current (expensive):
  frames → Gemini API → text caption → SigLIP2/Gemma text encode → embeddings
  Cost: $0.01-0.05 per segment, 2-5 seconds latency per API call

Proposed (local):
  frames → V-JEPA 2 encoder → visual tokens → MLP projector → LLM → caption
  Cost: $0 (local), ~0.5 seconds per segment on GPU
```

#### Two Modes

**Mode A — Direct Visual Tokens (no captioning needed):**
V-JEPA 2 patch tokens can be fed directly to the LLM as visual tokens (LLaVA-style). For search, we don't even need to generate text — the visual token representations already capture the semantics.

```python
# V-JEPA 2 visual tokens for segment
visual_tokens = vjepa2_encoder(segment_frames)  # (N_patches, 1024)
projected = mlp_projector(visual_tokens)          # (N_patches, LLM_dim)
# These ARE the segment representation — no captioning needed
```

**Mode B — Local Captioning (when text needed):**
For `get_scene_list()` or explainability, run the full V-JEPA 2 + LLM pipeline locally:

```python
# Local VidQA-style captioning
visual_tokens = vjepa2_encoder(segment_frames)
projected = mlp_projector(visual_tokens)
caption = local_llm.generate(
    visual_tokens=projected,
    prompt="Describe this video segment in detail."
)
```

#### LLM Options

| LLM | Params | VRAM | Quality (Table 8 Avg) |
|-----|--------|------|:---------------------:|
| Qwen2-7B | 7B | ~14 GB | 54.4 |
| Llama 3.1-8B | 8B | ~16 GB | 59.5 |
| Qwen2-1.5B (quantized) | 1.5B | ~3 GB | ~45 (estimated) |

For captioning quality comparable to Gemini, Qwen2-7B or Llama-3.1-8B is needed. For resource-constrained environments, a quantized smaller model provides a reasonable trade-off.

#### Impact
- **Eliminates Gemini API dependency** for indexing (zero cost, offline-capable)
- Paper proves V-JEPA 2 + LLM achieves SOTA VidQA — captions will be at least as good
- Visual tokens capture spatial/temporal information that text captions lose
- Enables fully offline video analysis pipeline

#### Files to Modify
- `kuavi/indexer.py`: New `_caption_vjepa2_llm()` method as alternative to Gemini
- `kuavi/types.py`: Config for local LLM path, projector weights
- `kuavi/mcp_server.py`: New `kuavi_index_video` option `--caption-backend local`
- New file: `kuavi/vidqa.py` — V-JEPA 2 + LLM alignment wrapper

---

### Integration 6: Attentive Probe for Zero-Shot Video Classification

#### Background

The paper uses a 4-layer attentive probe on frozen V-JEPA 2 features for classification. The probe replaces the last self-attention layer with a **cross-attention layer using a learnable query token**, which attends to all patch features and outputs a classification logit.

This architecture is lightweight (~10M params) and can be finetuned rapidly for new classification tasks.

#### Application in KUAVi

Currently, `kuavi_discriminative_vqa` ranks candidates by embedding similarity — a zero-shot approach. An attentive probe would provide a more powerful **few-shot** classification capability:

```python
# New tool: kuavi_classify_segment
def kuavi_classify_segment(
    start_time: float,
    end_time: float,
    task: str = "action",  # "action", "scene", "object", "custom"
    labels: list[str] | None = None,
    video_id: str | None = None,
) -> dict:
    """Classify a video segment using V-JEPA 2 + attentive probe.

    For standard tasks (action, scene, object), uses pre-trained probes.
    For custom labels, falls back to discriminative VQA.
    """
    # Extract V-JEPA 2 features (full feature map, not pooled)
    features = vjepa2_encode_full(start_time, end_time)  # (clips, patches, 1024)

    if task in PRETRAINED_PROBES:
        probe = load_probe(task)  # 4-layer attentive probe
        logits = probe(features)
        return {"predictions": top_k(logits, labels)}
    else:
        # Custom labels → discriminative approach
        return discriminative_vqa(features, labels)
```

#### Pre-trained Probes

The paper provides probes for 6 tasks. We can distribute these as optional downloads:

| Task | Dataset | Labels | V-JEPA 2 Accuracy |
|------|---------|--------|:-:|
| Motion understanding | SSv2 | 174 classes | 77.3% |
| Action recognition | Kinetics-400 | 400 classes | 87.3% |
| Fine-grained action | Diving-48 | 48 classes | 90.2% |
| Gesture recognition | Jester | 27 classes | 97.8% |
| Instructional steps | COIN | ~180 classes | 91.1% |
| Object recognition | ImageNet | 1000 classes | 85.1% |

#### Impact
- High-accuracy classification without LLM generation
- Sub-second inference (probe is tiny)
- Useful for automated video categorization, content filtering, action labeling

#### Files to Modify
- New file: `kuavi/probes.py` — Attentive probe architecture + loading
- `kuavi/mcp_server.py`: New `kuavi_classify_segment` tool
- `kuavi/search.py`: Probe-based classification as optional search enhancer

---

## Theme 3: Enable New Paradigms

### Integration 7: V-JEPA 2-AC World Model for Predictive Video Understanding

#### Background

V-JEPA 2-AC is a 300M-parameter action-conditioned predictor that **predicts future frame representations** given past frames and actions. Trained on only 62 hours of robot data, it enables zero-shot planning on real robots. While robot manipulation isn't KUAVi's domain, the underlying capability — **predicting what the future looks like given the present** — has profound implications for video understanding.

#### Key Insight: Prediction as Understanding

The V-JEPA 2-AC energy landscape (Figure 9 in the paper) reveals that the model learns a smooth, locally convex energy function. The minimum of this function corresponds to the actual outcome. This means we can ask: **"Given what we've seen so far, which of these possible futures is most likely?"**

#### Application: Temporal Coherence Verification

```python
# New tool: kuavi_verify_temporal_coherence
def kuavi_verify_temporal_coherence(
    segment_a_time: tuple[float, float],
    segment_b_time: tuple[float, float],
    video_id: str | None = None,
) -> dict:
    """Check if segment B is a plausible temporal continuation of segment A.

    Uses V-JEPA 2's predictive capability to measure whether the
    transition from A to B is natural (low energy) or unlikely (high energy).

    Returns:
        coherence_score: 0.0 (impossible transition) to 1.0 (natural continuation)
        energy: Raw L1 distance between predicted and actual representations
    """
    repr_a = vjepa2_encode(segment_a_time)
    repr_b = vjepa2_encode(segment_b_time)

    # Predict what should follow segment A
    predicted_b = vjepa2_predictor(repr_a, mask_for_future)

    # Compare prediction to actual segment B
    energy = l1_distance(predicted_b, repr_b)
    coherence = 1.0 / (1.0 + energy)

    return {"coherence_score": coherence, "energy": energy}
```

#### Use Cases

1. **Anomaly detection**: Segments where predicted ≠ actual indicate unexpected events
2. **Video summarization**: Keep segments that are surprising (high energy), skip predictable ones
3. **Temporal ordering**: Given shuffled segments, order by minimizing total transition energy
4. **Cut detection**: Scene cuts produce very high energy transitions
5. **Narrative structure**: Build a graph of segment transitions weighted by coherence

#### Application: What-If Reasoning

```python
# In VideoRLM REPL:
# "What would happen if the person continued their current action?"
context = extract_frames(10.0, 15.0)
future_repr = vjepa2_predict_future(context, steps=4)  # 4 seconds ahead

# Find the most similar segment in the video to the predicted future
similar = search_by_embedding(future_repr, index.temporal_embeddings)
print(f"Predicted future most similar to segment at {similar.time_range}")
print(f"That segment shows: {similar.caption}")
```

#### Files to Modify
- `kuavi/indexer.py`: Load V-JEPA 2 predictor (already in checkpoint, just unused)
- `kuavi/mcp_server.py`: New `kuavi_verify_coherence`, `kuavi_predict_future` tools
- `kuavi/search.py`: New `make_predict_future()` factory
- `rlm/video/video_search_tools.py`: `predict_future()` tool for REPL

---

### Integration 8: Progressive Indexing with Resolution Tiers

#### Background

The paper's progressive resolution training (Section 2.4) achieves 8.4x compute reduction by training on low-res short clips first, then upgrading. We can apply the same principle to video indexing.

#### Problem

Currently, indexing is all-or-nothing: run the full pipeline at a single resolution. For a 2-hour video, this can take 30+ minutes. Users often want quick results first, then refinement.

#### Solution: Three-Tier Progressive Indexing

```
Tier 1 — Scan (seconds):
  1 fps, 256px, 16-frame clips, SigLIP2 only (no V-JEPA)
  → Rough scene boundaries, basic frame embeddings
  → Enough for coarse search ("find the car scene")

Tier 2 — Index (minutes):
  fps auto, 256px, 16-frame clips, V-JEPA 2 ViT-L
  → Accurate scene detection, temporal embeddings
  → Full search including temporal field
  → Current KUAVi quality level

Tier 3 — Deep Index (10+ minutes):
  fps auto, 384px, 64-frame clips, V-JEPA 2 ViT-g
  → SOTA-quality temporal embeddings
  → Full spatial feature maps stored
  → Action anticipation capability
  → Ready for VidQA alignment
```

Each tier builds on the previous — Tier 2 reuses Tier 1's frame extractions, Tier 3 reuses Tier 2's scene boundaries (just refines embeddings).

```python
# MCP usage
kuavi_index_video(path, tier=1)   # Fast scan, ready in seconds
# User searches, finds relevant segments
kuavi_upgrade_index(tier=2)        # Full index, only re-processes what changed
# User needs deep analysis
kuavi_upgrade_index(tier=3)        # Deep index with ViT-g
```

#### Impact
- Interactive workflow: get quick results immediately, refine as needed
- Aligns with how users actually work (scan → investigate → deep dive)
- Reduces upfront latency from minutes to seconds for initial queries

#### Files to Modify
- `kuavi/indexer.py`: New `upgrade_index()` method, tier-aware pipeline stages
- `kuavi/mcp_server.py`: New `kuavi_upgrade_index` tool, `tier` param on `kuavi_index_video`
- Index format: Store tier metadata, support incremental enhancement

---

## Implementation Roadmap

### Phase 1: Scaling Foundation (1 week)

```
1.1 Model upgrade presets (Integration 1)
    - Add VJEPA2_PRESETS to kuavi/types.py
    - Make scene_model configurable in indexer
    - Test with ViT-g at 384px on sample videos
    - Benchmark VRAM usage and indexing speed

1.2 Preserve spatial feature maps (Integration 2)
    - Optional return_full in _encode_clips_vjepa()
    - Store in VideoIndex when requested
    - Update .npz save/load format

1.3 64-frame overlapping clips (Integration 3)
    - Implement _group_frames_into_clips_overlapping()
    - Adjust Ward clustering thresholds
    - Validate scene detection quality
```

### Phase 2: Latent Capabilities (2-3 weeks)

```
2.1 Action anticipation (Integration 4)
    - Load V-JEPA 2 predictor from checkpoint
    - Implement mask token creation for future frames
    - New kuavi_anticipate_action MCP tool
    - Validate on EK100-style queries

2.2 Local VidQA captioning (Integration 5)
    - Implement V-JEPA 2 + LLM alignment wrapper
    - MLP projector training or use pre-trained weights
    - New --caption-backend local option
    - A/B test caption quality vs Gemini

2.3 Attentive probe classification (Integration 6)
    - Port attentive probe architecture
    - Load pre-trained probes for 6 tasks
    - New kuavi_classify_segment MCP tool
```

### Phase 3: New Paradigms (3-4 weeks)

```
3.1 Predictive understanding (Integration 7)
    - Temporal coherence verification tool
    - What-if reasoning in VideoRLM REPL
    - Anomaly detection via prediction error

3.2 Progressive indexing (Integration 8)
    - Three-tier pipeline with incremental upgrade
    - kuavi_upgrade_index MCP tool
    - Validate tier transitions preserve results
```

---

## Relationship to VL-JEPA Integration Plan

This plan complements (not replaces) the VL-JEPA integration plan. The two papers address different layers:

| Aspect | V-JEPA 2 (this plan) | VL-JEPA (previous plan) |
|--------|---------------------|------------------------|
| **Focus** | Vision encoder + predictor | Vision-language alignment |
| **Training** | Self-supervised (no language) | InfoNCE with text targets |
| **Key strength** | Motion understanding, prediction | Unified search space, selective decoding |
| **Primary value** | Better temporal embeddings, action anticipation, world model | Caption-free embedding, discriminative VQA, streaming |

**Combined integration strategy:**

```
V-JEPA 2 (this plan)               VL-JEPA (previous plan)
     │                                    │
     ├── Upgrade encoder to ViT-g          ├── Unified embedding space
     ├── 64-frame clips                    ├── Caption-free indexing
     ├── Spatial feature maps              ├── Selective decoding
     ├── Action anticipation               ├── VL-JEPA discriminative VQA
     ├── Predictive understanding          ├── Real-time streaming
     └── Progressive indexing              └── World model prediction
          │                                    │
          └──────────── SHARED ────────────────┘
                         │
              V-JEPA 2 encoder (frozen)
              serves as X-Encoder for BOTH
```

The V-JEPA 2 encoder is the shared foundation. V-JEPA 2 improvements (ViT-g, 384px, 64 frames) automatically benefit VL-JEPA's Predictor, since VL-JEPA uses V-JEPA 2 as its frozen X-Encoder.

---

## Resource Requirements

| Component | Source | Params | VRAM | New? |
|-----------|--------|--------|------|------|
| V-JEPA 2 ViT-g encoder | `facebook/vjepa2-vitg-fpc64-384` | 1B | ~4 GB | Upgrade |
| V-JEPA 2 predictor | Same checkpoint (unused weights) | ~50M | ~0.2 GB | Already downloaded |
| Attentive probes (6 tasks) | Meta release / retrained | ~10M each | ~0.04 GB | New |
| MLP projector (VidQA) | Train or download | ~5M | ~0.02 GB | New |
| Local LLM (optional) | Qwen2-7B-Instruct | 7B | ~14 GB | New (optional) |

**Minimum new VRAM for Phase 1-2**: ~3 GB (ViT-g upgrade + predictor)
**Full stack with local LLM**: ~18 GB

---

## Risk Assessment

| Integration | Risk | Mitigation |
|-------------|------|------------|
| 1. Model upgrade | Low — drop-in replacement | Keep ViT-L as `fast` preset fallback |
| 2. Spatial features | Low — additive, optional | Only store when explicitly requested |
| 3. 64-frame clips | Low — parameterized | Configurable clip size, existing 16-frame as default |
| 4. Action anticipation | Medium — predictor loading untested | Verify HuggingFace checkpoint includes predictor weights |
| 5. Local VidQA | Medium — needs projector weights | Start with Gemini as fallback, train projector on captioning data |
| 6. Attentive probes | Medium — need pre-trained weights | Train on standard datasets if not publicly released |
| 7. Predictive understanding | High — novel application | Start with coherence scoring (simpler), then expand |
| 8. Progressive indexing | Medium — architecture change | Careful tier transition testing, backward compatibility |

---

## Summary

V-JEPA 2 in KUAVi today is like buying a Tesla and only using it as a paperweight. The model we already load contains:

**What we use:**
- Patch token mean-pooling for scene boundary clustering

**What we're ignoring:**
- SOTA motion understanding (77.3% SSv2 — better than any other encoder)
- SOTA action anticipation (39.7 recall@5 EK100 — 44% better than 8B models)
- A predictor that can imagine future frame representations
- Spatial feature maps that capture fine-grained visual detail
- The ability to serve as a vision encoder for SOTA video question answering
- Smooth energy landscapes for temporal reasoning

This plan unlocks all of it, progressively, starting with simple scaling wins and building toward predictive video understanding.
