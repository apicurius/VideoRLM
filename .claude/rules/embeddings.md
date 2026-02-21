---
paths:
  - "kuavi/indexer.py"
  - "kuavi/scene_detection.py"
---

# Embedding & Model Conventions

## Three-Model Architecture
- **V-JEPA 2** (`facebook/vjepa2-vitl-fpc64-256`): Scene boundary detection ONLY — clusters 16-frame clip embeddings via Ward linkage
- **SigLIP2** (`google/siglip2-base-patch16-256`): Vision-language embeddings for `field="visual"` search; also the default text encoder
- **EmbeddingGemma** (`google/embeddinggemma-300m`): Optional dedicated text encoder for caption/action embeddings

## Embedding Functions
- `embed_fn(texts)` — Encodes text queries (captions, actions) into dense vectors; uses EmbeddingGemma when available, falls back to SigLIP2
- `visual_embed_fn(images)` — Encodes images via SigLIP2 for visual search; never use for text
- Do not mix these two functions — they produce embeddings in different spaces when EmbeddingGemma is active

## Batch Processing
- Always batch inputs to avoid per-item overhead
- Group V-JEPA 2 clips by frame count before batching (clips with different lengths cannot be in the same batch)
- Use `torch.no_grad()` for all inference

## GPU Memory
- Delete tensors and call `torch.cuda.empty_cache()` after large batch operations
- Move results to CPU before storing in index structures
- Load models with `torch.float16` when possible to reduce VRAM usage
