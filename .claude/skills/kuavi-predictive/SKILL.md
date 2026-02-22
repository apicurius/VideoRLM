---
name: kuavi-predictive
description: Predictive video understanding — anticipate actions, predict future content, verify coherence, classify activities
---

# Predictive Video Analysis

Use V-JEPA 2-powered predictive tools for forward-looking video understanding.

## Tools

### kuavi_anticipate_action
Predict what happens next after a given timestamp.
- Uses V-JEPA 2 predictor when available, falls back to embedding similarity
- Returns predicted action, confidence, and supporting evidence

```
kuavi_anticipate_action(time_point=45.0)
```

### kuavi_predict_future
Predict future video content from a time range.
- Temporal continuation using V-JEPA 2 predictor embeddings
- Returns predicted content description with confidence

```
kuavi_predict_future(start_time=30.0, end_time=45.0)
```

### kuavi_verify_coherence
Score temporal coherence across video segments and detect anomalies.
- Compares predicted vs actual embeddings at segment boundaries
- Flags surprising transitions where prediction diverges from reality

```
kuavi_verify_coherence()
```

### kuavi_classify_segment
Classify a video segment using attentive probes trained on benchmark tasks.
- Available tasks: SSv2, K400, Diving48, and more
- Returns top-K class labels with confidence scores

```
kuavi_classify_segment(start_time=10.0, end_time=20.0)
```

## Example Workflows

### "What happens next?"
1. `kuavi_search_video("current activity", field="action")` to locate the moment
2. `kuavi_anticipate_action(time_point=<end_of_activity>)` to predict next action
3. `kuavi_extract_frames` around the predicted time to verify

### "Is this video coherent?"
1. `kuavi_verify_coherence()` to get per-segment coherence scores
2. Look for segments with low coherence (anomalies)
3. `kuavi_extract_frames` around anomalous transitions to inspect

### "Classify this activity"
1. `kuavi_classify_segment(start_time, end_time)` for benchmark labels
2. Cross-reference with `kuavi_search_video(field="action")` for caption-based description
