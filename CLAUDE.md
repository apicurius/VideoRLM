# KUAVi: Agentic Vision Intelligence

See @README.md for project overview and @pyproject.toml for package configuration.

## Architecture & Tools
@.claude/rules/architecture.md

## Development
@.claude/rules/development.md

## Quick Reference

IMPORTANT: Use `uv run` for all Python commands, never bare `python`.

```bash
uv run python -m pytest tests/           # Run tests
uv run python -m kuavi.mcp_server        # Start MCP server
uv run python -m kuavi.cli index <video> # Index a video
```

IMPORTANT: Never modify MCP tool signatures without updating both `mcp_server.py` and `search.py`.

## Multi-Agent Architecture

KUAVi uses a decompose-analyze-synthesize pattern for complex video questions:

| Agent | Role | Model |
|-------|------|-------|
| `video-triage` | Fast entry point — answers from captions/transcript or escalates to video-analyst | Haiku |
| `video-analyst` | Full analysis with frame inspection, orchestrates sub-agents for complex questions | Sonnet |
| `video-decomposer` | Breaks complex questions into sub-questions with time ranges | Haiku |
| `video-segment-analyst` | Analyzes one temporal region (runs in background for parallelism) | Sonnet |
| `video-synthesizer` | Aggregates per-segment results into a final answer | Sonnet |

Use `video-triage` as the default entry point for video questions. It answers simple questions directly from search results on Haiku (fast path), and escalates to `video-analyst` (Sonnet) only when visual frame inspection or complex orchestration is needed.

## Skills

| Skill | When to use |
|-------|-------------|
| `kuavi-pixel-analysis` | Counting, motion detection, brightness tracking via `kuavi_eval` |
| `kuavi-deep-search` | Iterative query refinement when initial search fails |
| `kuavi-search` | Standard multi-field search |
| `kuavi-analyze` | End-to-end analysis |
| `kuavi-deep-analyze` | Multi-pass with shard analysis |
| `kuavi-predictive` | Action anticipation, future prediction, coherence verification |
| `kuavi-corpus` | Multi-video corpus indexing and cross-video search |

## Hooks

- **Anti-hallucination**: `validate_transcript_claims.sh` warns when transcript results contain names/numbers that need visual confirmation
- **Visual confirmation**: `validate_visual_confirmation.sh` checks that final output cites frame evidence for numeric claims
- **Trace logging**: `kuavi_trace_logger.sh` logs all tool calls to JSONL for trajectory visualization
- **Compile check**: `py_compile_check.sh` validates Python files after edits

## Compaction

When compacting, always preserve: the list of modified files, current task context, active video path and index info, sub-agent dispatch state, and KUAVi MCP tool names (`kuavi_index_video`, `kuavi_search_video`, `kuavi_search_transcript`, `kuavi_get_transcript`, `kuavi_get_scene_list`, `kuavi_discriminative_vqa`, `kuavi_extract_frames`, `kuavi_get_index_info`, `kuavi_anticipate_action`, `kuavi_predict_future`, `kuavi_verify_coherence`, `kuavi_classify_segment`).
