# Design: Separate indexer.py Into Logical Modules

## Problem

`kuavi/indexer.py` is 2,980 lines with a 2,792-line `VideoIndexer` class containing 43 methods. The class handles six distinct concerns (model management, neural encoding, captioning, deduplication, embedding, ASR transcription) plus pipeline orchestration, all in one file.

## Approach

Extract each concern into its own module. `VideoIndexer` stays in `indexer.py` as a slim orchestrator that delegates to the extracted modules. All existing imports from `kuavi.indexer` continue to work.

## New Module Structure

```
kuavi/
├── indexer.py          # VideoIndex, VideoIndexer (orchestrator), _StageCache, _cache_key
│                       # ~700 lines (down from 2,980)
├── encoding.py         # Frame/text encoding (SigLIP2, EmbeddingGemma, V-JEPA 2 clips)
├── captioning.py       # Selective decode, captioning, Self-Refine, quality scoring
├── dedup.py            # Pre-caption dedup, adjacent dedup, global dedup, semantic dedup
├── embedding.py        # Caption embedding, smoothing, quality checks, coarse levels, prediction
├── transcript.py       # ASR pipeline (Qwen3-ASR, faster-whisper, audio extraction)
```

## Method Allocation

### `encoding.py` (~266 lines)
- `_encode_frames` → `encode_frames(model, processor, frames, ...)`
- `_encode_texts` → `encode_texts(model_name, text_model, text_tokenizer, model, processor, texts)`
- `_encode_texts_siglip` → `encode_texts_siglip(model, processor, texts)`
- `_encode_query_siglip` → `encode_query_siglip(model, processor, query)`
- `_encode_query` → `encode_query(model_name, text_model, text_tokenizer, model, processor, query)`
- `_encode_clips_vjepa` → `encode_clips_vjepa(scene_model, clips, ...)`
- `_group_frames_into_clips` → `group_frames_into_clips(frames, timestamps, clip_size)`
- `_encode_frames_overlapping_vjepa` → `encode_frames_overlapping_vjepa(scene_model, frames, ...)`

### `captioning.py` (~659 lines)
- `_selective_decode`
- `_action_first_pass`
- `_filter_edge_frames`
- `_refine_annotations`
- `_score_format_compliance`
- `_score_action_frequency`
- `_score_annotations`
- `_fix_low_quality_annotations`
- `_transcript_for_range`

### `dedup.py` (~166 lines)
- `_pre_caption_dedup`
- `_deduplicate_segments`
- `_global_deduplicate`
- `_semantic_deduplicate`

### `embedding.py` (~185 lines)
- `_embed_captions`
- `_smooth_embeddings`
- `_check_embedding_quality`
- `_build_coarse_level`
- `_predict_future_embedding`

### `transcript.py` (~382 lines)
- `_get_transcript`
- `_load_transcript_file`
- `_extract_audio`
- `_ensure_asr_model`
- `_split_audio_chunks`
- `_run_asr`
- `_run_faster_whisper`
- `_collect_transcript_segments`
- `_is_faster_whisper_model` (module-level)

## Pattern: Methods become module-level functions

Methods that only access `self` for model references become standalone functions that receive those models as parameters. Methods that need broader `self` state stay as `VideoIndexer` methods that delegate to the module functions.

Example:
```python
# Before (in indexer.py):
def _encode_frames(self, frames, temporal_window=1, stride=None):
    ...use self._model, self._processor...

# After (in encoding.py):
def encode_frames(model, processor, frames, temporal_window=1, stride=None):
    ...

# VideoIndexer delegates:
def _encode_frames(self, frames, ...):
    self._ensure_model()
    return encode_frames(self._model, self._processor, frames, ...)
```

## Backward Compatibility

- `VideoIndex`, `VideoIndexer`, `_cache_key` remain importable from `kuavi.indexer`
- `kuavi/__init__.py` lazy imports unchanged
- All test imports unchanged
- Internal methods on `VideoIndexer` keep their names (thin wrappers to new functions)

## Constraints

- No changes to MCP tool signatures
- No changes to public API
- All 780 tests must pass
