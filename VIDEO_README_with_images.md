# VideoRLM: Long-Form Video Understanding

**Agentic video analysis that indexes, searches, and reasons over arbitrarily long videos.**

## Overview

VideoRLM turns a video into a searchable, structured index and then lets an LLM iteratively explore it using tool calls. Instead of sampling random frames or cramming an entire video into a single context window, the system builds a multi-scale representation of the video's temporal structure — scenes, segments, captions, embeddings, and transcripts — and exposes it through search tools.

The LLM operates as an autonomous agent: it decides which parts of the video to examine, at what granularity, and through which modality (visual, semantic, or spoken). It can start with a coarse search to localize relevant sections, narrow down to individual scenes, extract high-resolution frames, and query a sub-LLM for detailed visual analysis. This recursive tool-use loop continues until the agent has gathered enough evidence to answer the question.

VideoRLM handles videos of any length. Long videos are automatically segmented using temporal-aware scene detection, and optionally sharded across parallel sub-agent calls for efficient analysis.

## Architecture

Three specialized models handle different aspects of video understanding, each operating in its own embedding space:

![Diagram 1](diagram_1.png)

- **V-JEPA 2** processes 16-frame video clips to detect scene boundaries via temporal embedding similarity and Ward linkage clustering. It understands motion and temporal dynamics, not just visual appearance.
- **SigLIP2** encodes representative frames into a shared vision-language embedding space, enabling visual search where text queries match directly against image features.
- **EmbeddingGemma** encodes LLM-generated captions and action descriptions into a separate semantic embedding space, enabling rich text-to-text search over video content.

## Indexing Pipeline

![Diagram 2](diagram_2.png)

1. **Frame Extraction** — Frames are decoded at a configurable FPS (or auto-computed to hit a target frame count).
2. **V-JEPA 2 Scene Detection** — 16-frame clips are encoded and clustered using Ward linkage to find temporally coherent scene boundaries, replacing naive fixed-interval segmentation.
3. **Selective Decode** — Segments with low visual variance (< 0.02) are flagged as uniform and skipped during captioning, avoiding ~70% of redundant LLM calls on static content.
4. **Pre-Caption Dedup** — Visually near-identical segments (cosine similarity > 0.90) are deduplicated before captioning.
5. **LLM Captioning** — A VLM (e.g. Gemini) generates structured captions (summary + actions) for each non-trivial segment.
6. **Self-Refine** — Captions are refined over 3 rounds with neighbor context, improving coherence and reducing hallucination by grounding each caption against adjacent segments.
7. **ASR Transcript Injection** — Whisper-generated (or pre-existing) speech transcripts are prepended to captions, making spoken content searchable.
8. **Dual Embedding** — Frames are embedded with SigLIP2 (visual space) and captions with EmbeddingGemma (semantic space). The two spaces are kept separate to avoid cross-contamination.
9. **Embedding Smoothing** — A moving average (window=3) enforces temporal coherence across adjacent segment embeddings.
10. **Multi-Scale Hierarchy** — Segments are grouped into coarse chunks (~30s) with merged captions and averaged embeddings, enabling fast broad search before fine-grained drill-down.

## Agentic Search

![Diagram 3](diagram_3.png)

The LLM receives a set of tools and autonomously decides how to explore the video:

- **`get_scene_list()`** — Returns all segments with timestamps and captions for a structural overview.
- **`search_video(query, field, level, top_k)`** — Embedding search across visual, summary, or action fields at coarse or fine granularity.
- **`extract_frames(start_time, end_time)`** — Pulls high-resolution frames from a specific time range for detailed inspection.
- **`llm_query(prompt, images)`** — Sends frames to a sub-LLM for visual question answering.
- **`search_transcript(query)`** — Full-text search over ASR transcripts.

The typical pattern is: coarse search to localize, fine search to pinpoint, frame extraction to inspect, LLM query to reason. The agent loops until it has sufficient evidence.

## Multi-Scale Search

![Diagram 4](diagram_4.png)

The index stores two levels of granularity:

- **Coarse (level=1)** — ~30-second merged chunks with aggregated captions and averaged embeddings. Fast for broad localization.
- **Fine (level=0)** — Individual scene-based segments from V-JEPA 2 detection. Precise temporal boundaries.

Three search fields are available at each level: `visual` (SigLIP2 frame embeddings), `summary` (EmbeddingGemma caption embeddings), and `action` (EmbeddingGemma action embeddings). The agent can also use `field="all"` to search across all fields simultaneously.

## Key Optimizations

- **V-JEPA 2 scene detection** — Temporal-aware boundaries from video clip embeddings, not just frame-level visual similarity.
- **Selective decoding** — Segments with low visual variance (< 0.02) are skipped during captioning, avoiding ~70% of redundant LLM calls on static content.
- **Pre-caption dedup** — Visually near-identical segments (cosine similarity > 0.90) share captions instead of invoking the VLM redundantly.
- **Self-Refine** — 3 rounds of neighbor-aware caption refinement reduce hallucination and improve cross-segment coherence.
- **ASR transcript injection** — Whisper transcripts are prepended to captions, making spoken words searchable without a separate index.
- **Dual embedding spaces** — Visual (SigLIP2) and semantic (EmbeddingGemma) embeddings are kept separate, avoiding the quality degradation of cross-modal contamination.
- **Embedding smoothing** — Moving-average smoothing (window=3) enforces temporal coherence between adjacent segments.

## Quick Start

### CLI

```bash
# Comprehensive analysis (default prompt)
uv run python run_video.py --video path/to/video.mp4

# Ask a specific question
uv run python run_video.py --video path/to/video.mp4 \
    --question "What is the OOLONG score of RLM shown in this video?"

# Lightweight run — no scene detection or text embedding
uv run python run_video.py --video path/to/video.mp4 --no-scene-model --no-text-embedding

# Auto-compute FPS for long videos
uv run python run_video.py --video path/to/long_video.mp4 --auto-fps

# Use a different backend
uv run python run_video.py --video path/to/video.mp4 --backend portkey --model "@openai/gpt-5-nano"
```

**All CLI flags:**

| Flag | Default | Description |
|---|---|---|
| `--video` | *(required)* | Path to the video file |
| `--question` | comprehensive analysis prompt | Question to ask about the video |
| `--backend` | `gemini` | LLM backend (`gemini`, `portkey`, `openai`, `anthropic`) |
| `--model` | `gemini-3-flash-preview` | Model name |
| `--fps` | `0.5` | Frames per second to extract |
| `--num-segments` | `5` | Number of temporal segments |
| `--max-frames-per-segment` | `3` | Max frames per segment in LLM context |
| `--max-iterations` | `15` | Max REPL iterations per completion |
| `--embedding-model` | `google/siglip2-base-patch16-256` | SigLIP2 vision-text embedding model |
| `--no-search` | *(off)* | Disable semantic search tools |
| `--no-scene-model` | *(off)* | Disable V-JEPA 2 scene detection |
| `--no-text-embedding` | *(off)* | Disable EmbeddingGemma text encoder |
| `--cache-dir` | `None` | Directory to cache video indexes |
| `--auto-fps` | *(off)* | Auto-compute FPS based on video duration |

**Test scripts** (same flags as `run_video.py`):

```bash
uv run python run_test_oolong.py --video test_video.mp4   # Fine-grained detail (OOLONG score)
uv run python run_test_blend.py  --video test_video.mp4   # blend_frames + llm_query_batched
uv run python run_test_pixel.py  --video test_video.mp4   # Pixel manipulation tools
uv run python run_test_vqa.py    --video test_video.mp4   # discriminative_vqa + search_video
```

### Python API

```python
from rlm.logger import RLMLogger
from rlm.video import VideoRLM

logger = RLMLogger(log_dir="./logs")

vrlm = VideoRLM(
    backend="gemini",
    backend_kwargs={"model_name": "gemini-3-flash-preview", "timeout": 300.0},
    fps=0.5,
    num_segments=5,
    max_frames_per_segment=3,
    resize=(640, 480),
    max_iterations=15,
    logger=logger,
    verbose=True,
    enable_search=True,
    embedding_model="google/siglip2-base-patch16-256",
    scene_model="facebook/vjepa2-vitl-fpc64-256",
    text_embedding_model="google/embeddinggemma-300m",
)

result = vrlm.completion(
    "path/to/video.mp4",
    prompt="Provide a comprehensive analysis of this video.",
)

print(result.response)
```

## Configuration

| Parameter | Type | Default | Description |
|---|---|---|---|
| `backend` | `str` | `"openai"` | LLM backend (`"openai"`, `"anthropic"`, `"gemini"`) |
| `fps` | `float` | `1.0` | Frame extraction rate (frames per second) |
| `num_segments` | `int \| None` | `None` | Number of temporal segments (None = no segmentation) |
| `max_frames_per_segment` | `int \| None` | `None` | Cap on frames per segment in LLM context |
| `resize` | `tuple[int,int] \| None` | `None` | Resize frames to (width, height) |
| `max_iterations` | `int` | `30` | Maximum REPL iterations per completion |
| `scene_model` | `str \| None` | `None` | V-JEPA 2 model for scene detection (e.g. `"facebook/vjepa2-vitl-fpc64-256"`) |
| `text_embedding_model` | `str \| None` | `None` | Text encoder model (e.g. `"google/embeddinggemma-300m"`) |
| `enable_search` | `bool` | `True` | Build search index and expose search tools |
| `embedding_model` | `str` | `"google/siglip2-base-patch16-256"` | SigLIP2 model for frame embeddings |
| `auto_fps` | `bool` | `False` | Auto-compute FPS based on video duration |
| `target_frames` | `int` | `120` | Target frame count when `auto_fps=True` |
| `enable_sharding` | `bool` | `False` | Shard long videos across parallel sub-agents |
| `shard_max_segments` | `int` | `5` | Max segments per shard |
| `whisper_model` | `str` | `"base"` | Whisper model size for ASR |
| `token_budget` | `int \| None` | `None` | Max tokens before wrap-up signal |
