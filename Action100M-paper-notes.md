# Action100M: A Large-scale Video Action Dataset

**Paper:** [arXiv:2601.10592v1](https://arxiv.org/abs/2601.10592v1) (January 2026)

**Authors:** Delong Chen, Tejaswi Kasarla, Yejin Bang, Mustafa Shukor, Willy Chung, Jade Yu, Allen Bolourchi, Theo Moutakanni, Pascale Fung (Meta FAIR, HKUST, University of Amsterdam, Sorbonne Universite)

**Code/Dataset:** https://github.com/facebookresearch/Action100M

---

## Core Thesis

Existing video action datasets are either manually annotated (high quality but small scale, <1M instances) or ASR-derived (large scale but noisy and weakly aligned to visual content). Action100M bridges this gap with a **fully automated annotation pipeline** that produces 147M temporally localized, structured action annotations from 1.2M HowTo100M instructional videos (14.6 years of video). The pipeline uses V-JEPA 2 for hierarchical segmentation, multi-level vision-language captioning organized as a Tree-of-Captions, and LLM aggregation with Self-Refine.

---

## Pipeline Architecture (Section 3)

Three stages, each designed to be independently scalable:

```
Raw Video (1.2M HowTo100M videos)
        │
        ▼
┌─────────────────────────────────────────────┐
│  Stage 1: Hierarchical Temporal Segmentation │
│                                              │
│  V-JEPA 2 ViT-g-384 (1B params)             │
│  64-frame windows, 8-frame stride            │
│  Spatial average pooling → per-frame embed   │
│  Overlapping windows → averaged per frame    │
│                                              │
│  Hierarchical agglomerative clustering       │
│  Ward linkage, local temporal connectivity   │
│  Multiple distance thresholds → tree of      │
│  segments (fine atomic → coarse procedural)  │
│  Discard nodes < 0.5 seconds                 │
└──────────────────────┬──────────────────────┘
                       ▼
┌─────────────────────────────────────────────┐
│  Stage 2: Tree-of-Captions Generation        │
│                                              │
│  Leaf nodes (atomic segments):               │
│    Llama-3.2-Vision-11B                      │
│    Input: midpoint keyframe                  │
│    Prompt: "Describe this image in detail."  │
│    → Frame caption                           │
│                                              │
│  Higher-level nodes (longer segments):       │
│    PerceptionLM-3B                           │
│    Input: 32 evenly-spaced frames @ 320²     │
│    Prompt: "Describe this video in detail."  │
│    → Segment caption                         │
│                                              │
│  Both models: max 1024 tokens generation     │
│  Both run on single NVIDIA V100 32GB GPU     │
└──────────────────────┬──────────────────────┘
                       ▼
┌─────────────────────────────────────────────┐
│  Stage 3: LLM Aggregation + Self-Refine      │
│                                              │
│  GPT-OSS-120B (open-source reasoning model)  │
│                                              │
│  Input per node:                             │
│    - Current node's Tree-of-Captions         │
│      (depth-first, Markdown format)          │
│    - Global root node captions               │
│    - Video metadata (title, description)     │
│    - ASR transcript                          │
│                                              │
│  3 rounds of Self-Refine:                    │
│    Round 1: High reasoning effort → initial  │
│             structured annotation draft      │
│    Rounds 2-3: "Verify and revise previous   │
│             draft" — correct factual errors,  │
│             resolve inconsistencies, remove   │
│             unsupported statements            │
│                                              │
│  Nodes < 4 seconds: discarded                │
│  Non-action content: labeled "N/A" (3.23%)   │
│                                              │
│  Output: 5 structured fields per segment     │
└──────────────────────┬──────────────────────┘
                       ▼
              Structured Annotations
```

### Output Annotation Schema

```json
{
  "summary": {
    "brief": "Single sentence video caption.",
    "detailed": "Detailed, comprehensive description."
  },
  "action": {
    "brief": "A single verb phrase (no -ing forms) briefly summarizing the overall action content.",
    "detailed": "A single imperative sentence describing how the action is performed with more details.",
    "actor": "Single sentence or informative noun phrase describing who is performing the action."
  }
}
```

### Anti-Hallucination Guidelines (from Appendix A)

The LLM aggregation prompt enforces:
- Focus on what is **visually observable** — physical motion, procedural actions, appearance information
- Video metadata timestamps are reliable; **timestamps inside captions are not**
- Use global captions only for disambiguation — do not add **visually unobservable information**
- Be cautious and conservative — rely on **majority consensus** among captions
- Ignore captions from very short edges at segment start/end (scene transition artifacts)

---

## Stage 1: Temporal Segmentation Details

### V-JEPA 2 Encoding

| Parameter | Value |
|-----------|-------|
| Model | V-JEPA 2 ViT-g-384 (`facebook/vjepa2-vitg-fpc64-384`) |
| Parameters | 1B |
| Window size | 64 frames |
| Window stride | 8 frames |
| Spatial resolution | 384 × 384 |
| Frame sampling | 1 out of every 4 raw frames |
| Output | Spatial average pooled → per-frame embedding |
| Overlap handling | Multiple representations per shared frame → averaged |

### Hierarchical Agglomerative Clustering

- **Ward linkage**: Minimizes within-cluster variance at every merge step
- **Local temporal connectivity constraint**: Each frame links only to immediate neighbors — merges only occur between contiguous time spans
- **Result**: A hierarchical tree decomposition where:
  - Lower levels = fine-grained atomic motions
  - Higher levels = coarser activities/procedures
- **Minimum duration filter**: Nodes < 0.5 seconds are discarded

---

## Stage 2: Caption Generation Details

### Two Complementary Captioning Modes

| Mode | Model | Input | When Used |
|------|-------|-------|-----------|
| **Mid-frame captioning** | Llama-3.2-Vision-11B | Single keyframe (midpoint) | Leaf nodes (smallest contiguous segments) |
| **Video-segment captioning** | PerceptionLM-3B | 32 evenly-spaced frames @ 320² | Higher-level nodes (longer temporal spans) |

Both models:
- Generation limit: 1024 tokens
- Run on a single NVIDIA V100 32GB GPU
- The two modes are complementary: frame captions capture fine-grained spatial details, segment captions capture temporal dynamics

---

## Stage 3: LLM Aggregation Details

### Self-Refine Procedure

```
Round 1: GPT-OSS-120B with HIGH reasoning effort
         Input: Tree-of-Captions + metadata + ASR
         Output: Initial structured annotation draft

Round 2: GPT-OSS-120B revisits Round 1 output
         "Carefully analyze, verify, and revise the previous draft
          so that it is fully accurate, faithful to the provided
          content, and strictly adheres to all stated guidelines."

Round 3: Same revision process on Round 2 output
         → Final structured annotation
```

### Prompt Structure (Appendix A)

```
# Video metadata
{video_metadata}

# Global video context
{formatted_global_tree_of_captions}

# Current segment to be processed
{formatted_current_tree_of_captions}

# Your Task
[Extract structured information from video segment...]

### Task 1. Summarization
[Generate brief and detailed captions...]

### Task 2. Action Identification
[Identify main actor and physical action...]

# Response Formats
## output
{JSON schema for summary + action fields}
```

Tree-of-Captions are formatted as **depth-first traversal** into a **Markdown-style text stream**.

---

## Dataset Statistics (Section 4)

### Source Data

| Metric | Value |
|--------|-------|
| Source | HowTo100M (face-blurred version) |
| Videos | 1,199,096 |
| Total duration | ~14.6 years |
| ASR transcripts recovered | 72% of videos (10.6 years) |
| Content domains | 12 WikiHow categories (Food & Entertaining, Home & Garden, Hobbies & Crafts, etc.) |
| Dominant topics | "make", "recipe", "DIY", "easy", "cake", "chocolate" |

### Generated Annotations

| Metric | Value |
|--------|-------|
| Total annotated segments | 147,092,653 |
| Total words | 21.3 billion |
| Brief action avg length | 3.2 words |
| Brief caption avg length | 19.2 words |
| Detailed action avg length | 27.8 words |
| Detailed caption avg length | 95.3 words |
| Word breakdown | 0.46B (brief actions) + 2.83B (brief captions) + 3.96B (detailed actions) + 14.02B (detailed captions) |
| Non-action segments ("N/A") | 3.23% |
| Storage (all annotations + Tree-of-Captions) | ~205 GB |

### Segment Duration Distribution

| Duration | Percentage |
|----------|-----------|
| 0–3 seconds | 64% |
| 3–10 seconds | 23.8% |
| 10s–1 minute | 10.2% |
| > 1 minute | ~2% |

---

## Compute Requirements

| Stage | GPU Type | GPU Hours |
|-------|----------|-----------|
| Segmentation + Captioning | V100 | ~1.3 million |
| LLM Aggregation | H100/H200 | ~0.3 million |

---

## Comparison with Existing Datasets (Table 1)

### vs Action Recognition Datasets

| Dataset | Duration | #Videos | #Clips | Annotation |
|---------|----------|---------|--------|------------|
| COIN | 476 hours | 11.8K | 46.3K | Manual |
| YouCook2 | 176 hours | 2K | 14K | Manual |
| THUMOS14 | 30 hours | 2,584 | 20,108 | Manual |
| ActivityNet | 849 hours | 20K | 100K | Manual |
| FineAction | 705 hours | 17K | 103K | — |
| Assembly101 | 513 hours | 4,321 | 1M | Manual |
| **Action100M Brief** | **14.6 years** | **1.2M** | **147M** | **Automated (PLM-3B, Llama-3.2-11B, GPT-OSS-120B)** |
| **Action100M Detailed** | **14.6 years** | **1.2M** | **147M** | **Automated (PLM-3B, Llama-3.2-11B, GPT-OSS-120B)** |

### vs Caption Datasets

| Dataset | Duration | #Videos | #Clips | Avg Text Len | Annotation |
|---------|----------|---------|--------|:---:|------------|
| HD-VILA-100M | 42.4 years | 3.3M | 103M | 32.5 | ASR |
| InternVid | 86.8 years | 7.1M | 234M | 17.6 | Tag2Text, BLIP2 |
| Koala-36M | 19.6 years | — | 36M | 202.1 | GPT-4V |
| PLM-Video-Auto | 6.06 years | 6.4M | — | — | Llama-3.3-70B, LLama-3-405B |
| **Action100M Brief** | **14.6 years** | **1.2M** | **147M** | **18.4** | **PLM-3B, Llama-3.2-11B, GPT-OSS-120B** |
| **Action100M Detailed** | **14.6 years** | **1.2M** | **147M** | **150.2** | **PLM-3B, Llama-3.2-11B, GPT-OSS-120B** |

Action100M is unique in having both **action annotations** (structured verb phrases) and **caption annotations** (detailed descriptions), unlike prior datasets which have one or the other.

---

## Semantic Resampling (Section 5.4)

Addresses the long-tailed distribution of action frequencies in the dataset.

### Procedure

1. Embed all brief action descriptions using **EmbeddingGemma-300M**
2. Deduplicate by text hashing (7.58M duplicate groups → 141.8M duplicate instances)
3. Apply **k-means clustering** with k ∈ {10³, 10⁴, 10⁵}
4. From each cluster, sample uniformly with replacement until target dataset size is reached
5. This ensures both frequent and rare actions are adequately represented

### Most Duplicated Actions

| Action | Count |
|--------|-------|
| Speak to camera | 2.13M |
| Speak | 1.39M |
| Speak and gesture | 894.9K |
| Stir mixture | 719.4K |
| Talk to camera | 644.2K |
| Talk | 428.6K |
| Stir pot | 422.5K |
| Gesture | 419.1K |

### Effectiveness (Figure 10)

| Setting | Avg Accuracy (8 datasets) |
|---------|:---:|
| No resampling | 40.79 |
| k=100,000 | 40.77 |
| k=10,000 | 41.07 |
| k=1,000 | **41.41** |

Smaller k (coarser clusters) = more aggressive rebalancing = better performance.

---

## Experiments: Training VL-JEPA on Action100M (Section 5)

### Three-Stage Training Procedure (Table 2)

| Stage | Vision Encoder | #Frames | Training Data | Batch Size | #Iterations |
|-------|:---:|:---:|---|:---:|:---:|
| Stage 1 | Frozen V-JEPA 2 | 1 | Image-text (DataComp-1B, YFCC-100M) | 24,576 | 100k |
| Stage 2 | Frozen V-JEPA 2 | 8 | Action100M (all 4 fields, detailed downsampled ½) + PLM-3B segment captions | 12,288 | 60k |
| Stage 3 | **Unfrozen** V-JEPA 2 | 32 | Action100M | 3,072 × 4 (grad accum) | 10k |

### Main Results (Table 3)

#### Zero-shot Action Recognition (Top-1 Accuracy, 8 datasets)

| Model | Params | #Samples Seen | #Frames | SSv2 | EK100 | EgoExo4D | K400 | COIN(SR) | COIN(TR) | CrossTask(SR) | CrossTask(TR) | **Avg** |
|-------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| CLIP RN50 | 75M | 12.8B | 8 | 2.1 | 1.5 | 2.1 | 41.4 | 8.6 | 39.0 | 10.9 | 68.7 | 21.8 |
| CLIP ViT-B | 124M | | | 3.1 | 1.3 | 2.8 | 49.5 | 11.2 | 47.3 | 16.2 | 71.5 | 25.4 |
| CLIP ViT-L (336px) | 389M | | | 3.5 | 3.7 | 2.6 | 58.3 | 14.7 | 63.5 | 20.8 | 78.5 | 30.7 |
| SigLIP2 ViT-B | 224M | 40B | 8 | 3.9 | 5.2 | 2.3 | 4.3 | 57.8 | 20.6 | 60.9 | 27.7 | 92.9 | 34.9 |
| SigLIP2 ViT-L (384px) | 882M | | | 5.5 | 4.5 | 6.4 | 63.6 | 21.4 | 78.5 | 25.1 | 90.8 | 38.6 |
| SigLIP2 ViT-g (384px) | 1.9B | | | 5.8 | 6.1 | 5.6 | 68.0 | 26.0 | 80.4 | 35.1 | 90.8 | 39.8 |
| PE-Core ViT-B | 448M | 58B | 8 | 3.7 | 5.2 | 5.3 | 3.3 | 6.0 | 65.4 | 21.5 | 77.1 | 26.9 | 91.8 | 37.2 |
| PE-Core ViT-L (336px) | 671M | | | 9.3 | 6.0 | 11.6 | 73.4 | 27.1 | **83.1** | 47.3 | 99.4 | 42.9 |
| **PE-Core ViT-G (448px)** | **2.3B** | **86B** | **8** | 9.9 | 6.4 | 13.6 | **76.4** | 29.0 | **86.0** | **40.3** | **97.2** | **44.7** |
| **VL-JEPA Stage 1** | 1.6B | 2.4B | 1 | 3.2 | 3.9 | 1.0 | 3.2 | 39.5 | 12.1 | 40.6 | 20.9 | 50.9 | 21.5 |
| **VL-JEPA Stage 2** | 1.6B | 3.1B | 8 | 48.0 | 18.4 | 16.8 | 25.7 | 61.2 | 43.1 | 72.7 | 62.0 | 83.9 | 59.5 | **48.0** |
| **VL-JEPA Stage 3** | 1.6B | 3.5B | 32 | **19.3** | **21.8** | **33.2** | 64.8 | 47.4 | 79.4 | **64.5** | 89.6 | **63.7** | **52.5** |

**Key findings:**
- VL-JEPA with Action100M achieves **52.5% average** — significantly outperforming PE-Core ViT-G (44.7%) which has 2.3B params and saw 86B samples
- VL-JEPA uses only **1.6B params**, **3.5B samples**, and **ViT-L at 256px** — much smaller and more efficient
- Particularly strong on **motion-focused tasks**: SSv2 (19.3 vs 9.9), EK100 (21.8 vs 6.4), EgoExo4D (33.2 vs 13.6)
- Stage 1→2 jump is dramatic — image-only training is insufficient for action recognition

#### Zero-shot Text-to-Video Retrieval (Recall@1, 8 datasets)

| Model | MSR-VTT | ActivityNet | DiDeMo | MSVD | YouCook2 | PVDBench | Dream-1K | VDC-1K | **Avg** |
|-------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| CLIP ViT-L | 35.3 | 35.9 | 23.4 | 30.7 | 41.9 | 7.9 | 30.7 | 56.8 | 28.3 |
| SigLIP2 ViT-g | 43.4 | 43.3 | 33.9 | 38.9 | 56.0 | 22.2 | 60.4 | 73.0 | 47.5 |
| PE-Core ViT-G | **58.1** | **51.6** | **51.9** | 41.7 | 40.8 | **58.7** | 26.0 | 77.0 | 89.2 | 51.6 |
| **VL-JEPA Stage 3** | 40.0 | **64.9** | **50.0** | **49.0** | **40.4** | **83.1** | **93.3** | **88.8** | **63.7** |

- VL-JEPA achieves **63.7% average recall@1** vs 51.6% for PE-Core (which uses larger backbone + 86B samples)
- Despite seeing far fewer samples (3.5B vs 86B) and using a smaller backbone

---

## Pipeline Ablations (Section 5.3, Figure 9)

Controlled comparison: all models initialized from Stage 1 checkpoint, trained for 20k steps (20.48M samples).

### Data Source Comparison (performance improvement over Stage 1)

Best performing annotations on each benchmark (from Figure 9):

| Data Source | SSv2 | EK100 | EgoExo4D | K400 | COIN(SR) | COIN(TR) | CrossTask(SR) | CrossTask(TR) |
|-------------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Mix (all Action100M)** | 10.07 | 10.21 | 14.86 | 16.99 | 12.42 | 34.72 | 33.47 | 46.84 |
| Brief Action | 9.33 | **15.26** | **17.91** | **20.41** | **16.54** | **40.53** | **38.75** | **49.42** |
| Brief Caption | 8.04 | 8.67 | 11.79 | 10.31 | 10.68 | 26.40 | 22.97 | 42.72 |
| Detailed Action | 3.85 | 3.23 | 10.04 | 8.16 | 4.72 | 26.40 | 21.68 | 28.16 |
| Detailed Caption | 4.44 | 7.02 | 5.11 | -0.89 | 7.42 | 15.79 | 10.10 | 45.57 |
| PLM-Video-Auto | 1.59 | 3.11 | 3.01 | 10.78 | — | 18.97 | 4.84 | 27.85 |
| Ego4D Narrations | 4.94 | **19.76** | **18.53** | -18.69 | -1.02 | -6.15 | 0.87 | -5.96 |

**Key findings:**
- **Brief action descriptions** are the most effective single annotation type for zero-shot action recognition
- They outperform direct PLM-3B pseudo-labeling, demonstrating the value of hierarchical captioning + LLM aggregation
- **Detailed captions** also show advantages over PLM-Video-Auto on most benchmarks
- **Mix** (combining all annotation types) is not always best — brief actions alone often outperform
- Ego4D Narrations boost egocentric benchmarks (EK100, EgoExo4D) but hurt others

---

## Training Stage Analysis (Figure 8)

Performance evolution across stages:

| Dataset | Stage 1 (images) | Stage 2 (8 frames) | Stage 3 (32 frames, unfrozen) |
|---------|:---:|:---:|:---:|
| SSv2 | 3.9 | 13.0 | **19.3** |
| EK100 | 1.0 | 16.8 | **21.8** |
| EgoExo4D | 3.2 | 25.7 | **33.2** |
| K400 | 39.5 | 61.2 | **64.8** |
| COIN (step) | 12.1 | 43.3 | **47.4** |
| COIN (task) | 40.6 | 72.7 | **79.4** |
| CrossTask (step) | 20.9 | 62.0 | **64.5** |
| CrossTask (task) | 50.9 | 83.9 | **89.6** |

- Stage 1→2: Massive jump — **video training is essential** for action recognition
- Stage 2→3: Consistent improvement from unfreezing V-JEPA 2 encoder and increasing to 32 frames
- Scaling is consistent across all benchmarks — data scaling works

---

## Scaling Behavior (Figure 1)

VL-JEPA performance improves consistently as Action100M training data increases (log-scale samples seen). At the same data budget, VL-JEPA (ViT-L, 256px, 8 frames) outperforms:
- **CLIP** (ViT-L, 336px) — which has seen 12.8B samples
- **SigLIP2** (ViT-g-OPT, 384px) — which has 1.9B params and seen 40B samples
- **Perception Encoder** (ViT-G, 448px) — which has 2.3B params and seen 86B samples

This demonstrates that **data quality (structured action annotations) matters more than data quantity or model size**.

---

## Redundancy Analysis (Appendix B)

### Deduplication Statistics

- **7.58 million duplicate groups** identified
- These account for **141.8 million duplicate instances**
- Remaining action texts are unique (each occurring only once)

### UMAP Visualization (Figure 13)

Using k=10⁴ clusters, UMAP plots show that Action100M clusters provide broad and diverse coverage that overlaps well with multiple downstream datasets:
- COIN, CrossTask: Well-covered procedural actions
- Epic-Kitchens-100: Dense coverage of egocentric cooking actions
- Kinetics-400: Broad coverage of general actions
- YouCook2: Focused culinary action coverage
- EgoExo4D: Multi-perspective activity coverage

---

## Direct Relevance to KUAVi

Action100M's pipeline is essentially the **dataset-scale version** of KUAVi's single-video indexing pipeline. The correspondence is remarkably precise:

| Action100M Component | KUAVi Equivalent | KUAVi File |
|---------------------|------------------|------------|
| V-JEPA 2 ViT-g-384, 64-frame windows, 8-frame stride | V-JEPA 2 ViT-L-256, 16-frame clips | `scene_detection.py` |
| Ward linkage + local temporal connectivity | Ward linkage clustering | `scene_detection.py:46-91` |
| Hierarchical multi-level segmentation | Hierarchical scene detection (multiple thresholds) | `scene_detection.py:94-148` |
| Llama-3.2-Vision-11B frame captions | Gemini frame captions (midpoint keyframe) | `indexer.py` Stage 5a |
| PerceptionLM-3B segment captions | Gemini segment captions (structured annotation) | `indexer.py` Stage 5b |
| GPT-OSS-120B LLM aggregation | Gemini LLM aggregation | `indexer.py` Stage 6 |
| 3-round Self-Refine | 3-round Self-Refine | `indexer.py` Stage 6 |
| Tree-of-Captions (depth-first Markdown) | Tree-of-Captions | `indexer.py` Stage 5 |
| Structured {summary, action} annotations | Same structured format | `indexer.py` |
| Nodes < 0.5s discarded | Minimum segment duration filter | `scene_detection.py` |
| Nodes < 4s: no LLM aggregation | Tier 0/1/2 selective decoding | `indexer.py` Stage 4 |
| EmbeddingGemma-300M for semantic resampling | EmbeddingGemma-300M optional text encoder | `indexer.py` |
| Anti-hallucination guidelines in LLM prompt | Anti-hallucination rules in Self-Refine prompts | `indexer.py` Stage 6 |

### Key Differences

| Aspect | Action100M | KUAVi |
|--------|-----------|-------|
| **Scale** | 1.2M videos, 147M segments | Single video at a time |
| **V-JEPA 2 model** | ViT-g-384 (1B params, 64 frames) | ViT-L-256 (300M params, 16 frames) |
| **Frame captioner** | Llama-3.2-Vision-11B (local) | Gemini (API) |
| **Segment captioner** | PerceptionLM-3B (local) | Gemini (API) |
| **LLM aggregator** | GPT-OSS-120B (local) | Gemini (API) |
| **Deduplication** | Pre-caption: text hashing | Pre-caption: SigLIP2 cosine sim > 0.90 |
| **Selective decoding** | All segments captioned | 3-tier: DEAD/STATIC/DYNAMIC |
| **Post-refine dedup** | Not mentioned | Adjacent >0.95, global >0.90 |
| **Quality scoring** | Not mentioned | Re-caption if sim < 0.3 |

### Implications

1. **KUAVi's pipeline is validated at scale** — Action100M proves the same architecture works on 147M segments
2. **Brief actions are the most useful annotation type** — KUAVi should prioritize action brief embeddings for search
3. **V-JEPA 2 ViT-g-384 upgrade** — Action100M uses the larger model; KUAVi currently uses ViT-L-256
4. **Semantic resampling concept** — Could be applied to KUAVi's search reranking (EmbeddingGemma k-means diversity)
5. **Self-Refine is confirmed effective** — +1.3 avg accuracy over single-pass annotation at dataset scale
6. **Tree-of-Captions hierarchy is key** — outperforms single-level captioning by +1.5 avg accuracy
