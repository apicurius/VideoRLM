---
name: kuavi-pixel-analysis
description: Compositional pixel analysis patterns using kuavi_eval for counting, motion detection, change tracking, and visual comparison. Use when questions require precise measurements, object counting, or frame-by-frame comparison.
---

# Pixel Analysis with kuavi_eval

Use `kuavi_eval(code)` to compose pixel tools programmatically when discrete MCP tool calls are insufficient. The persistent Python namespace has `np`, `cv2`, and all kuavi tools pre-loaded.

## When to Use This

- **Counting objects** in a frame (people, items, markers)
- **Detecting motion** between frames (what changed?)
- **Measuring visual properties** (brightness trends, color distribution)
- **Comparing regions** across time (same area, different moments)
- **Reading values** from charts, tables, or graphs
- **Iterating** over multiple frames with the same analysis
- **Motion-focused search**: Use `field="temporal"` with `kuavi_search_video` to find motion/dynamics-heavy segments before pixel analysis

## Pattern 1: Object Counting

```python
# Extract frames, threshold, count contours
kuavi_eval("""
frames = extract_frames(10.0, 15.0, fps=2.0, max_frames=5)
counts = []
for f in frames:
    mask = threshold_frame(f, value=120)
    counts.append(mask['contour_count'])
print(f"Object counts per frame: {counts}")
print(f"Average: {sum(counts)/len(counts):.1f}, Max: {max(counts)}")
""")
```

## Pattern 2: Motion Detection Between Frames

```python
# Compare consecutive frames to find when motion occurs
kuavi_eval("""
frames = extract_frames(20.0, 30.0, fps=1.0, max_frames=10)
for i in range(len(frames)-1):
    diff = diff_frames(frames[i], frames[i+1])
    print(f"Frame {i}→{i+1}: changed={diff['changed_pct']:.1f}%, mean_diff={diff['mean_diff']:.1f}")
""")
```

## Pattern 3: Region-of-Interest Tracking

```python
# Crop the same region across frames and compare
kuavi_eval("""
frames = extract_frames(5.0, 25.0, fps=0.5, max_frames=10)
for i, f in enumerate(frames):
    roi = crop_frame(f, x1_pct=0.6, y1_pct=0.1, x2_pct=0.95, y2_pct=0.3)
    info = frame_info(roi)
    print(f"Frame {i}: brightness={info['brightness']['mean']:.0f}, size={info['width']}x{info['height']}")
""")
```

## Pattern 4: Brightness/Color Trend Analysis

```python
# Track brightness over time to detect scene changes
kuavi_eval("""
frames = extract_frames(0.0, 60.0, fps=0.5, max_frames=30)
brightness = []
for f in frames:
    info = frame_info(f)
    brightness.append(info['brightness']['mean'])
print(f"Brightness over time: {[f'{b:.0f}' for b in brightness]}")
# Find sudden changes (potential scene boundaries)
for i in range(1, len(brightness)):
    delta = abs(brightness[i] - brightness[i-1])
    if delta > 30:
        print(f"  Sharp change at frame {i}: delta={delta:.0f}")
""")
```

## Pattern 5: Composite/Blend for Background Extraction

```python
# Blend multiple frames to extract static background
kuavi_eval("""
frames = extract_frames(10.0, 20.0, fps=1.0, max_frames=10)
composite = blend_frames(frames)
# Now diff a single frame against the composite to isolate foreground
single = frames[5]
fg = diff_frames(composite, single)
print(f"Foreground coverage: {fg['changed_pct']:.1f}%")
""")
```

## Pattern 6: Parallel LLM Analysis of Frames

```python
# Ask an LLM to describe specific content in multiple frames
kuavi_eval("""
frames = extract_frames(30.0, 40.0, fps=2.0, max_frames=6)
prompts = []
for i, f in enumerate(frames):
    prompts.append(f"What text or numbers are visible in this frame? Be precise.")
# llm_query_batched sends all prompts in parallel
descriptions = llm_query_batched(prompts)
for i, desc in enumerate(descriptions):
    print(f"Frame {i}: {desc}")
""")
```

## Pattern 7: Iterative Search + Extract Pipeline

```python
# Programmatically search, then extract and analyze hits
kuavi_eval("""
hits = search_video("scoreboard results table", field="visual", top_k=5)
for h in hits:
    frames = extract_frames(h['start_time'], h['end_time'], fps=4.0, width=1280, height=960, max_frames=3)
    for f in frames:
        # Crop the likely scoreboard region (right side of frame)
        roi = crop_frame(f, x1_pct=0.5, y1_pct=0.0, x2_pct=1.0, y2_pct=0.5)
        info = frame_info(roi)
        print(f"  [{h['start_time']:.1f}s] ROI brightness={info['brightness']['mean']:.0f}")
""")
```

## Key Reminders

- Variables persist across `kuavi_eval` calls — set values in one call, use them in the next.
- Use `SHOW_VARS()` to inspect what's in the namespace.
- `llm_query(prompt)` for single LLM calls; `llm_query_batched(prompts)` for parallel.
- All pixel tools return dicts with metadata — check the keys before accessing.
- Keep each eval block focused on one task for clarity.
