# VL-JEPA: Joint Embedding Predictive Architecture for Vision-Language

**Paper:** [arXiv:2512.10942v2](https://arxiv.org/abs/2512.10942v2) (Feb 2026)

**Authors:** Delong Chen, Mustafa Shukor, Theo Moutakanni, Willy Chung, Jade Yu, Tejaswi Kasarla, Yejin Bang, Allen Bolourchi, Yann LeCun, Pascale Fung (Meta FAIR, HKUST, Sorbonne, NYU)

---

## Core Idea

VL-JEPA replaces the standard autoregressive token-generation paradigm of Vision Language Models (VLMs) with **continuous embedding prediction**. Instead of predicting text tokens one-by-one, VL-JEPA predicts the *embedding* of the target text in a shared latent space. This has three key advantages:

1. **Simplified learning target** — multiple valid textual answers map to nearby points in embedding space (even if they share no overlapping tokens), making training more efficient
2. **No heavy decoder during training** — the text decoder is only needed at inference time
3. **Non-autoregressive** — enables selective decoding and real-time streaming

---

## Architecture (4 components)

| Component | Function | Initialization |
|-----------|----------|---------------|
| **X-Encoder** (X_V → S_V) | Encodes visual input into embeddings | Frozen **V-JEPA 2** ViT-L (304M params) |
| **Predictor** (S_V, X_Q → Ŝ_Y) | Maps visual embeddings + text query → predicted target embedding | Last 8 layers of **Llama-3.2-1B** (490M trainable) |
| **Y-Encoder** (Y → S_Y) | Encodes textual target into embedding space | **EmbeddingGemma-300M** |
| **Y-Decoder** (Ŝ_Y → Ŷ) | Translates predicted embedding back to text (inference only) | Lightweight decoder |

**Training objective:** Bi-directional **InfoNCE loss** between predicted embedding Ŝ_Y and target embedding S_Y — alignment + uniformity, no representation collapse.

**Key insight:** The Predictor uses bi-directional attention (not causal), so vision and query tokens jointly attend to each other.

---

## Multi-task Support (single unified architecture)

- **Vision-text-to-text generation** (captioning, open VQA): Predict embedding → decode with Y-Decoder. Supports *selective decoding* — only decode when the embedding stream shifts significantly.
- **Discriminative VQA / Classification**: Encode candidates with Y-Encoder, pick nearest to predicted embedding Ŝ_Y. No generation needed.
- **Text-to-video retrieval**: Rank candidate videos by similarity of their predicted embeddings to a retrieval query.

---

## Two-Stage Training

### Stage 1 — Large-scale pretraining (VL-JEPA_BASE)

- Query-free caption alignment on Datacomp + YFCC-100M (images) and Action100M (video)
- Image-only first (100k iters, 1 frame, batch 24k), then video (60k iters with 8 frames, 10k with 32 frames)
- 4 weeks on 24 nodes x 8 H200 GPUs
- Evaluated on zero-shot classification and retrieval

### Stage 2 — Supervised finetuning (VL-JEPA_SFT)

- PLM data mixture: 25M VQA + 2.8M captioning + 1.8M classification + pretraining data
- 83k steps, ~2.5 days on 24 nodes
- Evaluated on VQA benchmarks

---

## Key Experimental Results

### Classification & Retrieval (Table 1)

- VL-JEPA_BASE (1.6B params) achieves **52.5% average classification accuracy** vs 44.7% for best PE-Core baseline
- **63.7% average retrieval recall@1** vs 58.1 for PE-Core
- Particularly strong on motion-centric benchmarks (SSv2, EK-100, EgoExo4D)

### VQA (Table 2)

- VL-JEPA_SFT matches or approaches specialist VLMs on GQA (61.5%), TallyQA (69.9%), POPE (85.7%), POPEv2 (86.3%)
- Comparable to InstructBLIP, Qwen-VL despite being non-generative and only 1.6B params

### WorldPrediction-WM (Table 3)

- VL-JEPA_SFT achieves **65.7% SOTA** — surpasses GPT-4o (52.0%), Claude-3.5 (53.3%), Gemini-2 (55.6%)

### Action Anticipation (Tables 4-5)

- SOTA on COIN next-step forecasting: **56.2%** (vs 53.6 for ProVideLLM)
- Competitive on EPIC-Kitchens-100 across all anticipation times

---

## Embedding Prediction vs Token Prediction (Section 4.5)

Strictly controlled comparison (same encoder, data, batch size, schedule — only the loss differs):

- VL-JEPA learns **much faster** — sharper performance increase early in training
- At 5M samples: VL-JEPA reaches 14.7 CIDEr and 35.3% top-5 classification; VLM baseline: 7.1 CIDEr, 27.2%
- Gap persists at 15M samples
- VL-JEPA uses **50% fewer trainable parameters** (0.5B predictor vs 1B LLM)
- Inference: decouples "Encoder + Predictor" (126ms) from "Text Decoding" (203ms) — can skip decoding when unnecessary

---

## Selective Decoding (Section 4.6)

For streaming video, VL-JEPA can monitor embedding changes and only decode when semantics shift:

- Uses agglomerative clustering on the embedding stream to find segment boundaries
- Decodes at segment midpoints only
- At 0.35 Hz selective decoding: matches 1 Hz uniform decoding quality → **~2.85x fewer decode operations**
- Average pooling within segments further improves quality

---

## Y-Encoder Analysis (Section 4.7)

- VL-JEPA_BASE achieves **63.9% on SugarCrepe++** and **42.9% on VISLA** — more resilient to text hard-negatives than CLIP (44.5%), SigLIP2 (56.5%), and PE-Core (58.6%)
- After SFT, slight drop on SugarCrepe++ (58.4%) but maintains VISLA (39.5%)

---

## Ablation Studies (Table 7)

Key findings:

- **Pretraining is critical**: -21.7 classification, -17.3 retrieval without it
- **Y-Encoder learning rate multiplier**: Sweet spot at 0.05-0.10 (slower than main model)
- **Loss function**: InfoNCE best overall; cosine loss wins VQA but can't work with unfrozen Y-Encoder
- **Predictor size**: More layers = better, especially for VQA; layers 8-16 is best config
- **Bi-directional attention** helps (+1.9 VQA vs causal)
- **Llama-3 initialization** helps VQA but slightly hurts classification/retrieval vs random init

---

## Architecture Comparison

| | CLIP | VLM | VL-JEPA |
|---|---|---|---|
| Generation | No | Yes | Yes |
| Classification | Yes | No | Yes |
| Retrieval | Yes | No | Yes |

VL-JEPA is the first model to support all three task families in a single architecture.

---

## Relevance to KUAVi

This paper is directly relevant to the KUAVi codebase — it describes the **VL-JEPA architecture** that extends the V-JEPA 2 vision encoder already used in KUAVi for scene detection. Key connections:

- KUAVi uses **V-JEPA 2** for scene boundary detection and temporal embeddings — VL-JEPA adds a Predictor + Y-Encoder on top of the same frozen encoder
- KUAVi uses **SigLIP2** for frame/text embeddings — VL-JEPA shows that embedding prediction (with InfoNCE) outperforms SigLIP2's contrastive approach
- VL-JEPA's **discriminative VQA** (candidate ranking by cosine similarity) is exactly the approach used in KUAVi's `kuavi_discriminative_vqa` tool
- VL-JEPA's **selective decoding** paradigm aligns with KUAVi's selective decoding in the indexer (Tier 0/1/2 system based on SigLIP2 similarity)
- The **EmbeddingGemma-300M** Y-Encoder is the same optional text encoder available in KUAVi via `--text-embedding-model`
