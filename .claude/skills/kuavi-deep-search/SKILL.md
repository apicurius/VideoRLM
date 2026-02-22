---
name: kuavi-deep-search
description: Multi-pass search with query refinement for hard-to-find video content. Use when initial search returns low-confidence results or no matches.
---

# Deep Search with Query Refinement

When a simple `kuavi_search_video` call returns poor results (scores < 0.3 or no relevant matches), escalate through these refinement passes.

## Pass 1: Field Rotation

Try different search fields — the same query may match differently depending on how content was indexed:

```
kuavi_search_video("your query", field="summary", top_k=5)
kuavi_search_video("your query", field="action", top_k=5)
kuavi_search_video("your query", field="visual", top_k=5)
kuavi_search_video("your query", field="temporal", top_k=5)
```

If `field="summary"` misses, `field="visual"` may catch it through raw frame embeddings.

## Pass 2: Query Reformulation

Rephrase the query with synonyms, more general terms, or more specific terms:

| Original | Reformulations |
|----------|---------------|
| "person cooking pasta" | "making food", "kitchen activity", "preparing meal" |
| "final score" | "scoreboard", "results table", "end of game" |
| "presenter introduction" | "title slide", "speaker name", "opening remarks" |

Try 2-3 reformulations across different fields.

## Pass 3: Transcript Search

Switch from visual to audio modality:
```
kuavi_search_transcript("keyword from question")
```

Transcript search is keyword-based — use specific nouns, names, or distinctive terms rather than descriptions.

## Pass 4: Hierarchy Level Search

Search at coarser temporal granularity for broad localization:
```
kuavi_search_video("query", field="all", level=1, top_k=5)
```

Level 1 segments are ~30s chunks. Once you find the right region, search again at level 0 within that range.

## Pass 5: Exhaustive Scan via kuavi_eval

As a last resort, compute similarity against ALL segments programmatically:

```python
kuavi_eval("""
scenes = get_scene_list()
best_score = -1
best_scene = None
for s in scenes:
    # Check caption content directly
    caption = s.get('caption', '') + ' ' + str(s.get('annotation', ''))
    if 'keyword' in caption.lower():
        print(f"Direct match: scene {s['scene_index']} at {s['start_time']:.1f}s")
        print(f"  Caption: {caption[:200]}")
""")
```

## Pass 6: Temporal Elimination

If you know when the content is NOT (e.g., "not in the first half"), search only the remaining region:

```python
kuavi_eval("""
# Search only the second half of the video
scenes = get_scene_list()
mid = scenes[-1]['end_time'] / 2
late_scenes = [s for s in scenes if s['start_time'] > mid]
for s in late_scenes:
    print(f"Scene {s['scene_index']}: {s['start_time']:.1f}-{s['end_time']:.1f}s: {s.get('caption', '')[:100]}")
""")
```

## Decision Flow

```
Initial search (field="all") → scores > 0.4? → DONE
                             ↓ no
Field rotation (summary/action/visual/temporal) → found? → DONE
                             ↓ no
Query reformulation (3 variants) → found? → DONE
                             ↓ no
Transcript search → found? → DONE
                             ↓ no
Hierarchy level 1 search → found region? → search level 0 within region → DONE
                             ↓ no
Exhaustive scan via kuavi_eval → found? → DONE
                             ↓ no
Report: "Content not found after exhaustive search across all fields and modalities."
```

## Key Principle

Each pass costs budget. Stop as soon as you find relevant results. Do not run all passes if pass 1 succeeds.
