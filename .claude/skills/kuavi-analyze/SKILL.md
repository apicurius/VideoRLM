---
name: kuavi-analyze
description: Full end-to-end video analysis with KUAVi
agent: video-analyst
context: fork
argument-hint: <video-path> <question>
disable-model-invocation: true
---

# KUAVi Analyze Video

Perform comprehensive video analysis: index, search, reason, and answer.

## Instructions

1. Parse `$ARGUMENTS` for: video path and question/task.
2. **Index**: If the video is not yet indexed, call `kuavi_index_video`.
3. **Orient**: Call `kuavi_get_scene_list` to understand the video structure.
4. **Search**: Use the SEARCH-FIRST strategy to find relevant content:
   - Decompose the question into search sub-queries
   - Search across multiple fields (summary, action, visual, temporal)
   - Search transcript for spoken content
5. **Inspect**: Extract frames for the most relevant segments.
6. **Reason**: Synthesize findings from visual evidence and transcript.
7. **Verify**: Cross-reference visual and audio evidence.
8. **Answer**: Provide a clear, evidence-based answer.

## Arguments

The first argument is the video path. Remaining text is the question.

Example: `/kuavi-analyze /path/to/video.mp4 What is the main topic of this presentation?`
