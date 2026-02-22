---
name: kuavi-deep-analyze
description: Deep multi-pass video analysis with parallel sharding and zoom
agent: video-analyst
context: fork
argument-hint: <video-path> <question>
disable-model-invocation: true
---

# KUAVi Deep Analyze

Perform comprehensive, multi-pass video analysis with parallel temporal sharding and progressive zoom.

## Instructions

1. Parse `$ARGUMENTS` for: video path and question/task.
2. **Index**: If the video is not yet indexed, call `kuavi_index_video`.
3. **Orient**: Call `kuavi_get_scene_list` to understand the video structure.
4. **Parallel Shard Analysis**: Call `kuavi_analyze_shards` with the question to get per-shard answers from parallel LLM workers. Use this to identify which temporal regions are most relevant.
5. **Pass 1 — Visual Search**: Use `kuavi_search_video` across multiple fields (summary, action, visual, temporal) to find relevant segments. Cross-reference with shard analysis results.
6. **Pass 2 — Transcript Search**: Use `kuavi_search_transcript` and `kuavi_get_transcript` to find spoken evidence. Note any discrepancies with visual findings.
7. **Pass 3 — Cross-Reference**: For key findings, use the 3-pass zoom protocol:
   - Level 1 zoom to locate
   - Level 2 zoom to read
   - Level 3 zoom to confirm specific values
8. **Pixel Analysis**: When counting, measuring, or comparing, use pixel tools (`crop_frame`, `diff_frames`, `threshold_frame`, `frame_info`) for deterministic evidence.
9. **Code Reasoning**: Use `kuavi_eval` for complex computations, iterating over frames, or chaining multiple operations.
10. **Budget Check**: Monitor `kuavi_get_session_stats` — synthesize after 15 tool calls or 120 seconds.
11. **Synthesize**: Combine all evidence into a comprehensive, timestamped answer.

## Arguments

The first argument is the video path. Remaining text is the question.

Example: `/kuavi-deep-analyze /path/to/video.mp4 What are the key events and when do they occur?`
