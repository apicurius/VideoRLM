# V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning

**Paper:** [arXiv:2506.09985v1](https://arxiv.org/abs/2506.09985v1) (June 2025)

**Authors:** Mahmoud Assran, Adrien Bardes, David Fan, Quentin Garrido, Russell Howes, Mojtaba Komeili, Matthew Muckley, Ammar Rizvi, Claire Roberts, Koustuv Sinha, Artem Zholus, + many more (FAIR at Meta)

**Code:** https://github.com/facebookresearch/vjepa2

---

## Core Thesis

Self-supervised learning on internet-scale video, combined with a small amount of robot interaction data, can produce a world model capable of **understanding** (classification, VQA), **predicting** (action anticipation), and **planning** (zero-shot robot manipulation). This is built on the Joint-Embedding Predictive Architecture (JEPA).

---

## V-JEPA 2 Pretraining (Section 2)

### Objective: Mask-Denoising in Representation Space

The model predicts the learned representation of masked video patches from visible ones — no pixel reconstruction.

```
minimize_{θ,φ,Δ_y}  ||P_φ(Δ_y, E_θ(x)) - sg(E_θ̄(y))||₁
```

- `E_θ`: Encoder (ViT) — extracts representations from masked video
- `P_φ`: Predictor (ViT-small) — predicts representations of masked patches
- `E_θ̄`: EMA encoder — provides targets via exponential moving average
- `Δ_y`: Learnable mask tokens indicating positions of dropped patches
- `sg()`: Stop-gradient on target encoder

**Architecture:** Vision Transformer with **3D-RoPE** (rotary position embeddings for temporal, height, width axes). Input: tubelets of size `2 × 16 × 16 (T × H × W)`.

### Four Scaling Ingredients

| Ingredient | Change | Impact |
|-----------|--------|--------|
| **Data scaling** | 2M → 22M videos (VideoMix22M) | +1.0 avg accuracy |
| **Model scaling** | ViT-L (300M) → ViT-g (1B) | +1.5 avg accuracy |
| **Longer training** | 90K → 252K iterations (warmup-constant-decay schedule) | +0.8 avg accuracy |
| **Higher resolution** | 256→384 spatial, 16→64 frames, progressive | +0.9 avg accuracy |

**Cumulative gain: +4.0 points** (84.2% → 88.2% avg across 6 classification tasks).

### Pretraining Dataset: VideoMix22M

| Source | Samples | Type | Hours |
|--------|---------|------|-------|
| SSv2 | 168K | Ego-video | 168 |
| Kinetics | 733K | Exo-video | 614 |
| HowTo100M | 1.1M | Tutorials | 134K |
| YT-Temporal-1B | 19M | General YouTube (curated) | 1.6M |
| ImageNet | 1M | Images (as 16-frame static video) | n/a |

**Data curation for YT1B**: Retrieval-based filtering — extract scenes, compute embeddings, cluster-select scenes matching target distribution (Kinetics, SSv2, COIN, EpicKitchen). Yields +1.4 point improvement.

### Progressive Resolution Training

Train on low-res short clips during warmup/constant phases, then increase resolution and duration during cooldown:
- Warmup: 16 frames, 256×256, 12K iterations
- Constant: 256×256, 228K iterations
- Cooldown: 64 frames, 384×384, 12K iterations

**8.4× reduction in GPU time** vs full-resolution training throughout.

### Model Family

| Model | Params | Avg. Accuracy (6 tasks) |
|-------|--------|:-----------------------:|
| V-JEPA 2 ViT-L | 300M | 86.0 |
| V-JEPA 2 ViT-H | 600M | 86.4 |
| V-JEPA 2 ViT-g | 1B | 87.5 |
| V-JEPA 2 ViT-g₃₈₄ | 1B (384px) | 88.2 |

---

## V-JEPA 2-AC: Action-Conditioned World Model (Section 3)

After pretraining, the V-JEPA 2 encoder is frozen and a new **action-conditioned predictor** is trained on top, using only **62 hours of unlabeled robot manipulation data** from the Droid dataset.

### Architecture

- **Predictor**: 300M params, 24 layers, 16 heads, 1024 hidden dim, GELU activations
- **Block-causal attention**: Each patch at time `t` attends to patches, actions, and end-effector states from current and previous timesteps
- **Inputs**: Interleaved sequence of `(action_k, state_k, z_k)` where:
  - `z_k = E(x_k)` — frozen V-JEPA 2 frame encoding (16×16×1408 feature map)
  - `s_k` — 7D end-effector state (3D position + 3D orientation + gripper)
  - `a_k` — 7D action (delta end-effector state between frames)

### Training Objective

**Teacher-forcing loss** + **Rollout loss**:

```
L(φ) = L_teacher-forcing(φ) + L_rollout(φ)

L_teacher-forcing = (1/T) Σ ||P_φ(a_t, s_t, E(x_l))_{l≤k} - E(x_{k+1})||₁

L_rollout = ||P_φ(a_{1:T}; s_1, z_1) - z_{T+1}||₁
```

- Teacher-forcing: predict next frame representation from ground-truth context
- Rollout: predict T steps ahead from autoregressive rollout (T=2 in practice, differentiate through 1 recurrent step)

### Planning via Energy Minimization (Section 3.2)

Given a **goal image**, plan actions by minimizing:

```
E(â_{1:T}; z_k, s_k, z_g) := ||P(â_{1:T}; s_k, z_k) - z_g||₁
```

Optimized via **Cross-Entropy Method (CEM)**:
1. Sample action sequences from Gaussian distributions (zero mean, unit variance)
2. Evaluate energy for each sequence
3. Update distribution statistics using top-k trajectories
4. Repeat for several iterations
5. Execute first action only, then replan (receding horizon control)

---

## Robot Planning Results (Section 4)

Zero-shot deployment on **two Franka Emika Panda arms** in different labs (neither in training data).

### Single-Goal Reaching
- Moves end-effector within **<4 cm of target** in all cases
- Monotonic decrease in position error across planning steps
- Energy landscape is smooth and locally convex near ground truth

### Prehensile Manipulation (Table 2)

| Task | V-JEPA 2-AC (avg) | Octo baseline (avg) |
|------|:-:|:-:|
| Reach | 100% | 100% |
| Grasp (cup) | 65% | 15% |
| Grasp (box) | 25% | 0% |
| Reach w/ Object | 75% | 15% |
| Pick-&-Place (cup) | 80% | 15% |
| Pick-&-Place (box) | 65% | 10% |

### V-JEPA 2-AC vs Cosmos World Model (Table 3)

| | V-JEPA 2-AC | Cosmos |
|--|:-:|:-:|
| Planning time per action | **16 sec** | 4 min |
| Samples per refinement | 800 | 80 |
| Reach | 100% | 80% |
| Grasp (cup) | 60% | 0% |
| Pick-&-Place (cup) | 80% | 0% |

V-JEPA 2-AC is **15x faster** per planning step and achieves higher success across all manipulation tasks.

### Limitations
- **Camera sensitivity**: Must manually find a good camera position (no explicit calibration)
- **Long horizon**: Autoregressive error accumulation limits reliable prediction to ~16 seconds
- **Image goals only**: No language-based goal specification yet

---

## Understanding: Probe-based Classification (Section 5)

**Evaluation protocol**: Freeze V-JEPA 2 encoder, train 4-layer attentive probe on top.

### Classification Results (Table 4)

| Method | Params | SSv2 | Diving-48 | Jester | K400 | COIN | IN1K | Avg |
|--------|--------|:----:|:---------:|:------:|:----:|:----:|:----:|:---:|
| DINOv2 | 1.1B | 50.7 | 82.5 | 93.4 | 83.6 | 90.7 | 86.1 | 81.1 |
| PE_Core G | 1.9B | 55.4 | 76.9 | 90.0 | 88.5 | **95.3** | 87.6 | 82.3 |
| SigLIP2 | 1.2B | 49.9 | 75.3 | 91.0 | 87.3 | 95.1 | **88.0** | 81.1 |
| V-JEPA 2 ViT-g | 1B | 75.3 | 90.1 | 97.7 | 86.6 | 90.7 | 84.6 | 87.5 |
| **V-JEPA 2 ViT-g₃₈₄** | **1B** | **77.3** | **90.2** | **97.8** | **87.3** | 91.1 | 85.1 | **88.2** |

V-JEPA 2 **dominates motion understanding** (SSv2: 77.3 vs 55.4 for PE-Core, Diving-48: 90.2 vs 82.5) while remaining competitive on appearance tasks.

---

## Prediction: Action Anticipation (Section 6)

**Task**: Predict what action occurs next, given context video ending 1 second before the action starts. Evaluated on Epic-Kitchens-100 (EK100).

**Method**: Freeze V-JEPA 2 encoder + predictor. The predictor takes context frames and mask tokens for the future frame. Outputs are fed to a 4-layer attentive probe with 3 query tokens (verb, noun, action classifiers).

### Results (Table 5)

| Method | Params | Verb | Noun | Action |
|--------|--------|:----:|:----:|:------:|
| InAViT | 160M | 51.9 | 52.0 | 25.8 |
| Video-LLaMA | 7B | 52.9 | 52.0 | 26.0 |
| PlausiVL | 8B | 55.6 | 54.2 | 27.6 |
| V-JEPA 2 ViT-L | 300M | 57.8 | 53.8 | 32.7 |
| V-JEPA 2 ViT-g | 1B | 61.2 | 55.7 | 38.0 |
| **V-JEPA 2 ViT-g₃₈₄** | **1B** | **63.6** | **57.1** | **39.7** |

**44% relative improvement** over PlausiVL (8B params) using only 1B params. Performance scales linearly with model size.

---

## Video Question Answering (Section 7)

V-JEPA 2 is aligned with an LLM (Qwen2-7B-Instruct) via LLaVA-style visual instruction tuning. **First video encoder pretrained without language supervision to be used for VidQA.**

### Controlled Comparison — Frozen Encoder (Table 6)

| Encoder | Avg | PerceptionTest | MVP | TempCompass | TemporalBench | TVBench | TOMATO | MVBench |
|---------|:---:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| DINOv2 ViT-g₅₁₈ | 45.7 | 67.1 | 22.4 | 62.3 | 26.8 | 47.6 | 32.0 | 61.8 |
| SigLIP2 ViT-g₃₈₄ | 48.1 | **72.4** | 26.2 | 66.8 | 25.7 | 48.7 | 33.2 | 64.0 |
| PE ViT-G/14₄₄₈ | 49.1 | 72.3 | 26.7 | 67.0 | 27.5 | 51.6 | 34.0 | 64.7 |
| **V-JEPA 2 ViT-g₅₁₂** | **52.3** | 72.0 | **31.1** | **69.2** | **33.3** | **55.9** | **37.0** | **67.0** |

V-JEPA 2 outperforms all image encoders, especially on **temporal understanding** (TempCompass +2.2, TemporalBench +5.8, TVBench +4.3).

### Scaling Results (Table 7 — End-to-End)

| Model | Avg | PerceptionTest | TempCompass | TemporalBench | TVBench | TOMATO |
|-------|:---:|:-:|:-:|:-:|:-:|:-:|
| V-JEPA 2 ViT-L₂₅₆ | 51.7 | 74.6 | 70.1 | 30.2 | 50.9 | 36.5 |
| V-JEPA 2 ViT-g₃₈₄ | 54.0 | 76.5 | 71.7 | **33.1** | 56.5 | **39.0** |
| V-JEPA 2 ViT-g₅₁₂ | **54.4** | **77.7** | **33.7** | 32.3 | **57.5** | 38.5 |

### State-of-the-Art (Table 8 — Full 88.5M training data, Llama 3.1 8B backbone)

| Model | Avg | PerceptionTest | MVP | TempCompass | TemporalBench | TOMATO | TVBench | MVBench |
|-------|:---:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| PLM 8B | 56.7 | 82.7 | 39.7 | 72.7 | 28.3 | 33.2 | 63.5 | 77.1 |
| **V-JEPA 2 ViT-g₃₈₄ 8B** | **59.5** | **84.0** | **44.5** | **76.9** | **36.7** | **40.3** | 60.6 | 73.5 |

SOTA on PerceptionTest (+1.3), MVP (+4.8), TempCompass (+4.2), TemporalBench (+8.4), TOMATO (+7.1).

---

## Conclusion & Future Work

- **Hierarchical world models** needed for longer-horizon planning (beyond 16 seconds)
- **Language-based goals** instead of image goals — align V-JEPA 2-AC with LLMs
- **Scaling beyond 1B** — previous work shows vision encoders up to 20B, but sustainable scaling recipes needed

---

## Direct Relevance to KUAVi

This is the **exact model** KUAVi already uses for scene detection (`facebook/vjepa2-vitl-fpc64-256`). Key implications:

| Current KUAVi Usage | What V-JEPA 2 Paper Reveals |
|---------------------|----------------------------|
| Scene detection only (Ward clustering on clip embeddings) | The same embeddings achieve SOTA on 6 classification tasks — hugely underutilized |
| ViT-L (300M) at 256px, 16 frames | ViT-g (1B) at 384px, 64 frames is significantly better (+2.2 avg) |
| Temporal embeddings queried with SigLIP2 text encoder | V-JEPA 2 embeddings are in a **completely different space** than SigLIP2 — this is broken |
| Not used for action recognition | SOTA on action anticipation (39.7 recall@5 on EK100) |
| Not used for VQA | When aligned with LLM, achieves SOTA VidQA |
| No predictive capabilities | V-JEPA 2-AC enables zero-shot planning with goal images |
| 16-frame clips only | 64-frame clips capture longer temporal context (+0.7 avg) |

**The V-JEPA 2-AC world model** (Section 3-4) is particularly relevant to the **Integration 7 (World Model Prediction)** identified in the VL-JEPA integration plan — it provides the concrete architecture and training recipe for action-conditioned prediction that VL-JEPA's WorldPrediction-WM results also demonstrated.
