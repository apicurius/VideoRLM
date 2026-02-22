# KUAVi Architecture

## Package Structure

```
kuavi/                  # Main package
├── __init__.py         # Public API exports (lazy imports)
├── __main__.py         # python -m kuavi → MCP server
├── types.py            # KUAViConfig dataclass
├── loader.py           # VideoLoader, LoadedVideo, VideoSegment, VideoMetadata
├── indexer.py          # VideoIndexer, VideoIndex (scene detection, embedding, ASR)
├── search.py           # Search tool factories (make_search_video, etc.)
├── scene_detection.py  # Scene boundary detection algorithms
├── context.py          # VideoContext, frame encoding, make_extract_frames
├── captioners.py       # Pluggable captioner backends (Gemini, OpenAI, local)
├── probes.py           # AttentiveProbe, ProbeRegistry (cross-attention classifiers)
├── corpus.py           # CorpusIndex, CorpusIndexer (multi-video indexing)
├── prompts.py          # VIDEO_ANALYSIS_PROMPT for Claude Code integration
├── mcp_server.py       # FastMCP stdio server with 27 tools
└── cli.py              # CLI: kuavi index/search/analyze/corpus
```

## 3 Models

- **V-JEPA 2** (`facebook/vjepa2-vitl-fpc64-256`): Scene boundary detection, temporal embeddings, action anticipation predictor, coherence verification. Supports overlapping windows (configurable stride) and 3 presets (fast/balanced/quality)
- **SigLIP2** (`google/siglip2-base-patch16-256`): Vision-language embeddings for `field="visual"` search + default text encoder + semantic dedup clustering
- **EmbeddingGemma** (`google/embeddinggemma-300m`): Optional separate text encoder

## Claude Code Integration

- **MCP Server**: `.mcp.json` registers `kuavi` MCP server (stdio)
- **Skills**: `.claude/skills/kuavi-{index,search,analyze,compare,deep-analyze,pixel-analysis,deep-search,predictive,corpus}/SKILL.md`
- **Agents**:
  - `video-triage.md` — Fast entry point (Haiku, 6 turns, search-only, escalates to video-analyst)
  - `video-analyst.md` — Full analysis with frame inspection (Sonnet, 20 turns, can spawn sub-agents)
  - `video-decomposer.md` — Question decomposition (Haiku, 8 turns, search-only tools)
  - `video-segment-analyst.md` — Parallel temporal analysis (Sonnet, 12 turns, background)
  - `video-synthesizer.md` — Result aggregation (Sonnet, 8 turns)
- **Hooks**:
  - `validate_transcript_claims.sh` — Anti-hallucination enforcement on transcript searches
  - `validate_visual_confirmation.sh` — Validates final output cites frame evidence
  - `validate_analysis.sh` — Checks for timestamps, evidence, confidence markers
  - `kuavi_trace_logger.sh` — JSONL trajectory logging

## MCP Tools

| Tool | Purpose |
|---|---|
| `kuavi_index_video` | Index a video file |
| `kuavi_search_video` | Semantic search (fields: summary, action, visual, all) |
| `kuavi_search_transcript` | Keyword search over ASR transcript |
| `kuavi_get_transcript` | Get transcript for a time range |
| `kuavi_get_scene_list` | List detected scenes with annotations |
| `kuavi_discriminative_vqa` | Embedding-based multiple-choice VQA |
| `kuavi_extract_frames` | Extract frames as base64 images |
| `kuavi_get_index_info` | Metadata about current index |
| `kuavi_get_session_stats` | Usage statistics for current session |
| `kuavi_zoom_frames` | Extract frames at preset zoom levels (1-3) |
| `kuavi_load_index` | Load a saved .kuavi index directory |
| `kuavi_crop_frame` | Crop image region using percentage coordinates |
| `kuavi_diff_frames` | Absolute pixel difference between two frames |
| `kuavi_blend_frames` | Average multiple frames into composite |
| `kuavi_threshold_frame` | Binary mask with contour detection |
| `kuavi_frame_info` | Image metadata and brightness/color stats |
| `kuavi_eval` | Execute Python in persistent namespace with tools |
| `kuavi_analyze_shards` | Parallel temporal shard analysis via LLM |
| `kuavi_anticipate_action` | Predict next action using V-JEPA 2 predictor or embedding similarity |
| `kuavi_predict_future` | Predict future video content with temporal continuation fallback |
| `kuavi_verify_coherence` | Score temporal coherence; detect anomalies and surprising transitions |
| `kuavi_classify_segment` | Classify segment via attentive probes (SSv2, K400, Diving48, etc.) |
| `kuavi_index_corpus` | Index multiple videos in parallel for cross-video search |
| `kuavi_search_corpus` | Semantic search across all videos in a corpus index |
| `kuavi_corpus_stats` | Statistics for the current corpus (video count, segment count, action vocabulary) |
| `kuavi_set_budget` | Configure tool-call, time, and token limits for the session |
| `kuavi_set_llm_config` | Route primary and secondary LLM calls to specific backends/models |
| `kuavi_orient` | Compound: get_index_info + get_scene_list in one call |
| `kuavi_search_all` | Compound: multi-field search + transcript search in parallel |
| `kuavi_inspect_segment` | Compound: extract_frames + get_transcript for a time range |
| `kuavi_quick_answer` | Compound: search_all + inspect_segment for top hits in one call |
