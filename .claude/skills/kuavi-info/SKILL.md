---
name: kuavi-info
description: Show metadata about the currently indexed video
disable-model-invocation: true
---

Call the `kuavi_get_index_info` MCP tool with no arguments.

Present the returned metadata as a formatted summary covering:
- Number of segments and scenes
- Video duration
- Number of transcript entries
- Models used for indexing (scene detection, embeddings, text encoding)
- Any other relevant metadata returned by the tool
