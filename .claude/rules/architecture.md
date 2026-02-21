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
├── prompts.py          # VIDEO_ANALYSIS_PROMPT for Claude Code integration
├── mcp_server.py       # FastMCP stdio server with 18 tools
└── cli.py              # CLI: kuavi index/search/analyze
```

## 3 Models

- **V-JEPA 2** (`facebook/vjepa2-vitl-fpc64-256`): Scene boundary detection ONLY
- **SigLIP2** (`google/siglip2-base-patch16-256`): Vision-language embeddings for `field="visual"` search + default text encoder
- **EmbeddingGemma** (`google/embeddinggemma-300m`): Optional separate text encoder

## Claude Code Integration

- **MCP Server**: `.mcp.json` registers `kuavi` MCP server (stdio)
- **Skills**: `.claude/skills/kuavi-{index,search,analyze,compare,deep-analyze,pixel-analysis,deep-search}/SKILL.md`
- **Agents**:
  - `video-analyst.md` — Orchestrator (Sonnet, 30 turns, can spawn sub-agents)
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
