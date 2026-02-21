---
name: kuavi-search
description: Search indexed video for specific content
argument-hint: <search-query>
---

# KUAVi Search Video

Search an indexed video using semantic search, transcript search, and frame extraction.

## Instructions

1. The user provides a search query via `$ARGUMENTS` or in the conversation.
2. If no video is indexed yet, ask the user to run `/kuavi-index` first.
3. Execute a multi-field search strategy:
   a. `kuavi_search_video` with `field="all"` for broad results
   b. `kuavi_search_transcript` for spoken content matches
   c. For specific values/text, also search with `field="visual"`
4. For the top results, use `kuavi_extract_frames` to get visual evidence.
5. Present results with timestamps, captions, and confidence scores.
6. Cross-reference visual results with transcript using `kuavi_get_transcript`.

## Example

User: `/kuavi-search person cooking pasta`

1. `kuavi_search_video(query="person cooking pasta", field="all", top_k=5)`
2. `kuavi_search_video(query="cooking pasta", field="action", top_k=3)`
3. `kuavi_search_transcript(query="pasta")`
4. For top hits: `kuavi_extract_frames(start_time, end_time)`
5. Report findings with timestamps and visual descriptions
