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

## Advanced Options

- **Overlapping V-JEPA 2 windows**: Set `scene_stride` for per-frame averaged embeddings (smoother scene detection)
- **Feature map storage**: Enable `store_feature_maps=True` for spatial analysis
- **Auto FPS**: Set `auto_fps=True` to automatically compute FPS from video duration (targets 120 frames)

## Corpus Indexing

For multi-video workflows, use `kuavi_index_corpus` instead:
```
kuavi_index_corpus(video_paths=["/path/to/video1.mp4", "/path/to/video2.mp4"])
```
This builds a cross-video index enabling `kuavi_search_corpus` for cross-video semantic search.
