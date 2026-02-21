# Video Analysis Memory Template

Use this template when saving analysis results to agent memory. Copy the structure and fill in the values for each video you analyze.

## Template

```markdown
## Video: [filename]

**Path**: [full path]
**Duration**: [seconds]s ([human readable])
**Scenes**: [count] scenes, [segment count] segments
**Has transcript**: [yes/no]
**Index cached at**: [cache dir or "not cached"]

### Content Structure
- [timestamp range]: [topic/content summary]
- [timestamp range]: [topic/content summary]
- ...

### Effective Queries
| Question Type | Best Field | Example Query |
|---|---|---|
| [type] | [field] | [query] |

### Confirmed Values
| Value | Timestamp | Source |
|---|---|---|
| [name/number] | [time] | [frame/transcript/both] |

### Patterns Learned
- [observation about what search strategies work for this video type]
```

## Example

```markdown
## Video: lecture_ml_101.mp4

**Path**: /data/videos/lecture_ml_101.mp4
**Duration**: 3620s (1h 0m 20s)
**Scenes**: 12 scenes, 45 segments
**Has transcript**: yes
**Index cached at**: ./cache/lecture_ml_101

### Content Structure
- 0-120s: Title slide and introduction, speaker name visible
- 120-900s: Linear regression theory with equations on slides
- 900-1800s: Live coding demonstration in Jupyter notebook
- 1800-2700s: Neural network fundamentals with diagrams
- 2700-3400s: Q&A session with audience
- 3400-3620s: Summary and closing remarks

### Effective Queries
| Question Type | Best Field | Example Query |
|---|---|---|
| Speaker identity | visual | "title slide speaker name" |
| Equations | visual | "equation formula" |
| Code content | visual | "code notebook python" |
| Concepts discussed | summary | "neural network" |
| Audience questions | action | "question asking" |

### Confirmed Values
| Value | Timestamp | Source |
|---|---|---|
| "Dr. Sarah Chen" | 5.0s | frame (title slide) |
| "97.3% accuracy" | 1650s | frame (notebook output) |
| "batch size 32" | 1420s | frame + transcript |

### Patterns Learned
- Slide-heavy lectures: field="visual" outperforms field="summary"
- Code segments: high-res extraction (1280x960) needed to read code
- Speaker name was in transcript as "Sarah Chen" (correct) — ASR got it right here
```
