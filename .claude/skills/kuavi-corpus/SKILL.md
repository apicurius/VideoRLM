---
name: kuavi-corpus
description: Multi-video corpus indexing and cross-video search
---

# Corpus Analysis

Index and search across multiple videos simultaneously.

## Tools

### kuavi_index_corpus
Index multiple videos in parallel for cross-video search.

```
kuavi_index_corpus(video_paths=["/path/to/video1.mp4", "/path/to/video2.mp4"])
```

### kuavi_search_corpus
Semantic search across all videos in the corpus.

```
kuavi_search_corpus(query="person presenting slides", field="summary", top_k=10)
```

### kuavi_corpus_stats
Get statistics about the current corpus.

```
kuavi_corpus_stats()
```

Returns: video count, total segments, total duration, action vocabulary.

## Example Workflows

### Cross-Video Comparison
1. `kuavi_index_corpus(video_paths=[...])` to build the corpus
2. `kuavi_search_corpus(query="topic X")` to find relevant segments across videos
3. Compare timestamps and content across different videos

### Finding Common Themes
1. `kuavi_corpus_stats()` to get action vocabulary
2. `kuavi_search_corpus(query="common action", field="action")` across all videos
3. Identify patterns that appear in multiple videos
