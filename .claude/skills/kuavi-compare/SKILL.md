---
name: kuavi-compare
description: Compare video segments across time ranges
argument-hint: <query>
disable-model-invocation: true
---

# KUAVi Compare Segments

Compare content across different time ranges in an indexed video.

## Instructions

1. Parse `$ARGUMENTS` for time ranges or content descriptions to compare.
2. If not indexed, ask the user to run `/kuavi-index` first.
3. For each comparison target:
   a. Search for the relevant segments using `kuavi_search_video`
   b. Extract frames using `kuavi_extract_frames`
   c. Get transcript using `kuavi_get_transcript`
4. Present a structured comparison:
   - Visual differences/similarities
   - Transcript differences/similarities
   - Temporal progression or changes over time
5. Synthesize findings into a clear comparison summary.

## Example

User: `/kuavi-compare How does the scene change between the beginning and end?`

1. Search for opening content: `kuavi_search_video("beginning opening", top_k=3)`
2. Search for ending content: `kuavi_search_video("ending conclusion", top_k=3)`
3. Extract frames from both time ranges
4. Compare visual content and transcript
5. Report changes, progression, and key differences
