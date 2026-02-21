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

## Compaction

When compacting, always preserve: the list of modified files, current task context, and KUAVi MCP tool names (`kuavi_index_video`, `kuavi_search_video`, `kuavi_search_transcript`, `kuavi_get_transcript`, `kuavi_get_scene_list`, `kuavi_discriminative_vqa`, `kuavi_extract_frames`, `kuavi_get_index_info`).
