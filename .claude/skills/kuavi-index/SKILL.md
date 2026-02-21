---
name: kuavi-index
description: Index a video file for KUAVi analysis
argument-hint: <video-path>
disable-model-invocation: true
---

# KUAVi Index Video

Index a video file to enable semantic search, scene detection, and analysis.

## Instructions

1. The user provides a video file path via `$ARGUMENTS` or in the conversation.
2. Call the `kuavi_index_video` MCP tool with the video path.
3. Report the indexing results: number of segments, scenes, transcript entries, duration.
4. Suggest next steps: `/kuavi-search` for finding specific content, or `/kuavi-analyze` for full analysis.

## Example

User: `/kuavi-index /path/to/video.mp4`

Call `kuavi_index_video` with `video_path="/path/to/video.mp4"`, then report:
- Video duration
- Number of segments detected
- Number of scene boundaries
- Whether transcript was generated
- Whether embeddings are available
