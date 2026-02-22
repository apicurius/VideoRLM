
---

<h1 align="center" style="font-size:2.8em">
<span>Recursive Language Models (<span style="color:orange">RLM</span>s)</span><br/>
<span style="font-size:0.5em; color:#888;">+ KUAVi: Agentic Vision Intelligence</span>
</h1>

<p align="center" style="font-size:1.3em">
  <a href="https://arxiv.org/abs/2512.24601">Full Paper</a> •
  <a href="https://alexzhang13.github.io/blog/2025/rlm/">Blogpost</a> •
  <a href="https://alexzhang13.github.io/rlm/">Documentation</a> •
  <a href="https://github.com/alexzhang13/rlm-minimal">RLM Minimal</a>
</p>

<p align="center">
  <a href="https://github.com/alexzhang13/rlm/actions/workflows/style.yml">
    <img src="https://github.com/alexzhang13/rlm/actions/workflows/style.yml/badge.svg" alt="Style" />
  </a>
  <a href="https://github.com/alexzhang13/rlm/actions/workflows/test.yml">
    <img src="https://github.com/alexzhang13/rlm/actions/workflows/test.yml/badge.svg" alt="Test" />
  </a>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2512.24601">
    <img src="media/paper_preview.png" alt="Paper Preview" width="300"/>
  </a>
</p>

---

## Overview

This repository contains two tightly integrated systems:

**Recursive Language Models (RLMs)** are a task-agnostic inference paradigm that replaces `llm.completion(prompt)` with a REPL-loop where the language model *programmatically* examines, decomposes, and recursively calls itself over its input. RLMs offload context as a variable in a REPL environment, enabling near-infinite context handling through code execution and sub-LM calls.

**KUAVi (Agentic Vision Intelligence)** extends RLMs to video understanding through a multi-model indexing pipeline, MCP tool server, and multi-agent orchestration system. KUAVi builds a searchable neural index from raw video and exposes it to LM agents as callable tools — whether through RLM's REPL injection (VideoRLM) or Claude Code's MCP protocol.

> [!NOTE]
> This repository is maintained by the authors of the paper from the MIT OASYS lab. Open-source contributions are welcome.

---

## Table of Contents

- [Quick Setup](#quick-setup)
- [RLM Architecture](#rlm-architecture)
  - [Completion Flow](#completion-flow)
  - [Recursion Model](#recursion-model)
  - [REPL Environments](#repl-environments)
  - [Model Providers](#model-providers)
- [VideoRLM](#videorlm)
- [KUAVi: Agentic Vision Intelligence](#kuavi-agentic-vision-intelligence)
  - [Three-Model Architecture](#three-model-architecture)
  - [Indexing Pipeline](#indexing-pipeline)
  - [MCP Tool Server](#mcp-tool-server)
  - [Multi-Agent Orchestration](#multi-agent-orchestration)
  - [Pixel Analysis Tools](#pixel-analysis-tools)
  - [Anti-Hallucination System](#anti-hallucination-system)
- [Usage](#usage)
  - [RLM Quick Start](#rlm-quick-start)
  - [VideoRLM CLI](#videorlm-cli)
  - [KUAVi CLI](#kuavi-cli)
  - [KUAVi MCP Server](#kuavi-mcp-server)
- [Trajectory Logging & Visualization](#trajectory-logging--visualization)
- [Citation](#citation)

---

## Quick Setup

```bash
pip install rlms          # RLM from PyPI
```

```bash
# Full development setup with KUAVi
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync                   # Install all dependencies
```

---

## RLM Architecture

RLMs replace the single-shot `llm.completion(prompt)` pattern with a **REPL-augmented loop** where the language model writes and executes code, inspects results, and launches recursive sub-LM calls.

### Completion Flow

```
┌─────────────────────────────────────────────────────┐
│                 RLM Completion Loop                 │
│                                                     │
│  prompt + context                                   │
│       │                                             │
│       ▼                                             │
│  ┌──────────┐  repl code  ┌──────────────┐          │
│  │   LLM    │────────────▶│     REPL     │          │
│  │ Response │   blocks    │ Environment  │          │
│  └──────────┘◀── results ─└──────────────┘          │
│       │                    │                        │
│       │ (up to 30 iters)   │ llm_query()            │
│       │                    ▼                        │
│       │               ┌──────────────┐              │
│       │               │   Sub-LLM    │              │
│       │               │  (via TCP)   │              │
│       │               └──────────────┘              │
│       │                                             │
│       ▼                                             │
│  FINAL(answer) or FINAL_VAR(var_name)               │
└─────────────────────────────────────────────────────┘
```

```python
from rlm import RLM

rlm = RLM(
    backend="openai",
    backend_kwargs={"model_name": "gpt-5-nano"},
    verbose=True,
)

result = rlm.completion("Print me the first 100 powers of two, each on a newline.")
print(result.response)
```

**How it works:**

1. The LM receives the prompt and context as REPL variables
2. It writes Python code blocks (` ```repl ... ``` `) to examine, transform, or query the context
3. Code executes in a persistent REPL environment; results are appended to the conversation
4. The LM can call `llm_query(prompt)` from code to spawn sub-LM calls (depth=1)
5. The loop terminates when the LM outputs `FINAL(answer)`, exceeds the token budget, or reaches max iterations

### Recursion Model

Sub-LM calls use a **socket-based communication protocol** between the REPL environment and an `LMHandler` TCP server:

```
Root LLM (depth=0)
  │
  │  writes code with llm_query(prompt)
  ▼
REPL Environment
  │
  │  TCP socket request (4-byte length prefix + JSON)
  ▼
LMHandler (ThreadingTCPServer)
  │
  │  routes by depth: depth=0 → default client, depth=1 → other_backend
  ▼
Sub-LLM (depth=1) → response string returned to REPL
```

- **Max depth**: Currently 1 (one level of recursion)
- **Batched sub-calls**: `llm_query_batched(prompts)` runs multiple sub-LM calls concurrently via `asyncio.gather()`
- **Token budget**: When exceeded, a wrap-up signal forces the LM to synthesize a final answer

### REPL Environments

RLMs support both local and fully isolated execution environments:

```python
rlm = RLM(
    environment="local",    # "local", "docker", "modal", "prime", "daytona", "e2b"
    environment_kwargs={...},
)
```

| Environment | Isolation | How it works |
|-------------|-----------|-------------|
| **`local`** (default) | Same process | Python `exec` with sandboxed builtins whitelist. Shares host virtual environment. |
| **`docker`** | Container | Launches `python:3.11-slim` (or custom image). Flask broker routes LLM requests to host via `host.docker.internal`. |
| **`modal`** | Cloud | [Modal Sandboxes](https://modal.com/docs/guide/sandboxes). Flask broker inside sandbox; state persisted via `dill`. |
| **`prime`** | Cloud | [Prime Intellect Sandboxes](https://docs.primeintellect.ai/sandboxes/overview). Same broker pattern. |
| **`daytona`** | Cloud | Daytona sandbox environment. |
| **`e2b`** | Cloud | E2B code interpreter sandbox. |

All isolated environments use a **broker pattern**: a Flask server inside the sandbox receives LLM requests from executing code and forwards them via HTTP to the `LMHandler` on the host process.

### Model Providers

| Backend | Notes |
|---------|-------|
| `openai` | Also supports vLLM, OpenRouter, Vercel |
| `gemini` | Thinking levels, retry on 500/504 |
| `anthropic` | Claude models |
| `portkey` | Router platform |
| `litellm` | Aggregator platform |
| `azure_openai` | Azure-hosted OpenAI |

All providers support multimodal content (base64 images), usage tracking, and async execution. See [`rlm/clients/`](rlm/clients/) for implementations.

---

## VideoRLM

VideoRLM extends the RLM paradigm to video understanding by **injecting video analysis tools directly into the REPL environment**. The LM can search, extract frames, and reason about video content using the same code-execution loop.

```
┌──────────────────────────────────────────────────────────┐
│                      VideoRLM Flow                       │
│                                                          │
│  Video File                                              │
│     │                                                    │
│     ├──▶ VideoLoader ──▶ frames + metadata               │
│     │                                                    │
│     ├──▶ VideoIndexer ──▶ VideoIndex                     │
│     │       │  V-JEPA 2 scene detection                  │
│     │       │  SigLIP2 frame/text embeddings             │
│     │       │  Qwen3-ASR transcription                   │
│     │       │  Tree-of-Captions + Self-Refine            │
│     │                                                    │
│     └──▶ Build extra_tools:                              │
│            search_video()         ─┐                     │
│            search_transcript()     │                     │
│            get_transcript()        │  injected into      │
│            get_scene_list()        ├─ REPL namespace     │
│            discriminative_vqa()    │  as functions       │
│            extract_frames()        │                     │
│            crop_frame()            │                     │
│            diff_frames()          ─┘                     │
│                    │                                     │
│                    ▼                                     │
│     rlm.completion(context, extra_tools=tools)           │
│                    │                                     │
│                    ▼                                     │
│     Standard RLM loop with video tools available         │
│     LM: results = search_video("person walking")         │
│     LM: frames = extract_frames(10.0, 20.0)              │
│     LM: analysis = llm_query_batched(shard_prompts)      │
└──────────────────────────────────────────────────────────┘
```

**Sharded analysis for long videos**: VideoRLM divides long videos into temporal shards, encodes each shard's frames as multimodal content, and runs `llm_query_batched()` for parallel analysis across all shards simultaneously.

```bash
# Run VideoRLM
uv run python run_video.py --video path/to/video.mp4 \
    --model gemini-3.1-pro-preview --question "What happens in this video?"
```

---

## KUAVi: Agentic Vision Intelligence

KUAVi re-exposes the VideoRLM tool set as an **MCP (Model Context Protocol) server**, enabling Claude Code agents to call the same video analysis tools via structured tool calls rather than REPL code injection. On top of this, KUAVi adds a **multi-agent orchestration layer** for complex video questions.

```
┌─────────────────────────────────────────────────────┐
│             KUAVi Architecture Overview             │
│                                                     │
│  ┌───────────────────────────────────────────────┐  │
│  │               Claude Code Agent               │  │
│  │                                               │  │
│  │  ┌────────┐                                   │  │
│  │  │ video- │ Haiku — fast path from captions   │  │
│  │  │triage  │ or escalate ──┐                   │  │
│  │  └────────┘               ▼                   │  │
│  │  ┌────────┐ ┌──────────┐ ┌──────────────┐     │  │
│  │  │ video- │ │  video-  │ │video-segment │     │  │
│  │  │analyst │─│decomposer│ │ analyst (xN) │     │  │
│  │  │(orch.) │ │  (plan)  │ │(parallel BG) │     │  │
│  │  └────────┘ └──────────┘ └──────────────┘     │  │
│  │      │    ┌──────────┐                        │  │
│  │      └────│  video-  │                        │  │
│  │           │synthesizr│                        │  │
│  │           └──────────┘                        │  │
│  └───────────────────────┬───────────────────────┘  │
│                        │ MCP Protocol (stdio)       │
│  ┌───────────────────────▼───────────────────────┐  │
│  │             KUAVi MCP Tool Server             │  │
│  │  30 tools: search, extract, pixel, compound   │  │
│  │  Result caching, budget gates, tracing        │  │
│  └───────────────────────┬───────────────────────┘  │
│                        │                            │
│  ┌───────────────────────▼───────────────────────┐  │
│  │                   VideoIndex                  │  │
│  │  Segments + 4 embedding spaces + ASR          │  │
│  │  Built by VideoIndexer (3-model pipeline)     │  │
│  └───────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘
```

### Three-Model Architecture

KUAVi uses three specialized neural models, each handling a distinct aspect of video understanding:

```
Video Frames
     │
     ├──▶ V-JEPA 2 (facebook/vjepa2-vitl-fpc64-256)
     │      16-frame clips → 1024-d embeddings
     │      Ward linkage clustering → scene boundaries
     │      Also produces temporal_embeddings for motion search
     │
     ├──▶ SigLIP2 (google/siglip2-base-patch16-256)
     │      Per-frame → 768-d vision-language embeddings
     │      Used for: visual search, text encoding (default),
     │                pre-caption dedup, selective decoding, quality scoring
     │
     └──▶ EmbeddingGemma (google/embeddinggemma-300m) [optional]
            Text-only → richer caption/action embeddings
            Replaces SigLIP2's text tower for summary/action search
```

| Model | Role | Embedding Dim | When Loaded |
|-------|------|:---:|-------------|
| **V-JEPA 2** | Scene boundary detection + temporal motion embeddings | 1024 | Lazy; skipped with `--no-scene-model` |
| **SigLIP2** | Frame embeddings, default text encoder, dedup, quality scoring | 768 | Always (lazy on first use) |
| **EmbeddingGemma** | Dedicated text encoder for summary/action search | 256 | Only when `--text-embedding-model` is set |

### Indexing Pipeline

The indexing pipeline converts raw video into a searchable `VideoIndex` through 8 stages:

```
Raw Video
  │
  ▼
┌──────────────────────────────────────────────────────┐
│ Stage 1: Frame Extraction                            │
│   VideoLoader → decode at fixed/auto FPS → arrays    │
│   Auto-FPS: target_frames / duration, clamped 0.1-5  │
└─────────────────────────┬────────────────────────────┘
                          ▼
┌──────────────────────────────────────────────────────┐
│ Stage 2: Scene Detection                             │
│   V-JEPA 2: 16-frame clips → encode → Ward cluster   │
│   Fallback: SigLIP2 frame embeddings → clustering    │
│   Hierarchical: multiple thresholds → multi-level    │
└─────────────────────────┬────────────────────────────┘
                          ▼
┌──────────────────────────────────────────────────────┐
│ Stage 3: Transcript Extraction                       │
│   ffmpeg → PCM audio → Qwen3-ASR + ForcedAligner     │
│   Or load pre-computed transcript from JSON file     │
└─────────────────────────┬────────────────────────────┘
                          ▼
┌──────────────────────────────────────────────────────┐
│ Stage 4: Deduplication & Selective Decoding          │
│   4a. Pre-caption dedup: SigLIP2 cos sim > 0.90      │
│       → skip captioning, propagate from rep.         │
│   4b. 3-tier selective decode:                       │
│       Tier 0 (DEAD): pixel_std < 5 or edges < 0.01   │
│       Tier 1 (STATIC): SigLIP2 > 0.98 + low VJEPA    │
│       Tier 2 (DYNAMIC): full captioning              │
│       V-JEPA can promote Tier 1 → 2 on motion        │
└─────────────────────────┬────────────────────────────┘
                          ▼
┌──────────────────────────────────────────────────────┐
│ Stage 5: Tree-of-Captions                            │
│   5a. Frame-level: midpoint keyframe caption (leaf)  │
│   5b. Segment-level: structured annotation (node)    │
│       → {summary: {brief, detailed},                 │
│          action: {brief, detailed, actor}}           │
│   Edge frame filter (cos sim < 0.5 to cluster)       │
│   Parallel: ThreadPoolExecutor(max_workers=8)        │
└─────────────────────────┬────────────────────────────┘
                          ▼
┌──────────────────────────────────────────────────────┐
│ Stage 6: Self-Refine                                 │
│   3 rounds of iterative annotation refinement:       │
│   Round 0: "Analyze and produce refined version"     │
│   Round 1+: "Verify and revise previous draft"       │
│   Context: video metadata + neighbor captions + ASR  │
│   Anti-hallucination rules enforced per round        │
│   Post-refine dedup: adjacent >0.95, global >0.90    │
│   Quality scoring: re-caption if sim < 0.3           │
└─────────────────────────┬────────────────────────────┘
                          ▼
┌──────────────────────────────────────────────────────┐
│ Stage 7: Embedding                                   │
│   Summary captions → embeddings (SigLIP2 or Gemma)   │
│   Action briefs → action_embeddings                  │
│   Representative frames → frame_embeddings           │
│   V-JEPA clips → temporal_embeddings (segment avg)   │
│   Smoothed: moving average window=3, L2-normalized   │
│   Quality: mean pairwise sim > 0.99 = DEGENERATE     │
└─────────────────────────┬────────────────────────────┘
                          ▼
┌──────────────────────────────────────────────────────┐
│ Stage 8: Hierarchy                                   │
│   Optional multi-level from hierarchical detection   │
│   Always builds coarse level ~30s for level=1 search │
│   Saved to disk: embeddings .npz + metadata JSON     │
└─────────────────────────┬────────────────────────────┘
                          ▼
                     VideoIndex
  {segments, embeddings, action_embeddings, frame_embeddings,
   temporal_embeddings, transcript, hierarchy, embed_fn}
```

Indexes are cached by content hash (`MD5(path|size|mtime)`) — repeated queries on the same video skip re-indexing entirely.

### MCP Tool Server

KUAVi exposes 30 MCP tools via a FastMCP stdio server, organized into six categories:

#### Search Tools

| Tool | Description |
|------|-------------|
| `kuavi_search_video` | Semantic search over segments. Fields: `summary`, `action`, `visual`, `temporal`, `all`. MMR diversity reranking. Multi-level hierarchy (`level=0` fine, `level=1+` coarse). |
| `kuavi_search_transcript` | Case-insensitive keyword search over ASR transcript with context windows. |
| `kuavi_get_transcript` | Retrieve transcript text for a specific time range. |
| `kuavi_get_scene_list` | List all detected scenes with structured annotations. |
| `kuavi_discriminative_vqa` | Embedding-based multiple-choice VQA. Ranks candidates by cosine similarity — no LLM generation needed. |
| `kuavi_anticipate_action` | Predict what happens next after a given time point using V-JEPA 2 predictor or embedding similarity fallback. |
| `kuavi_predict_future` | Predict future video content from a time range using V-JEPA 2 predictor with temporal continuation fallback. |
| `kuavi_verify_coherence` | Score temporal coherence across segments; detect anomalies and surprising transitions. |
| `kuavi_classify_segment` | Classify a segment using attentive probes trained on benchmark tasks (SSv2, K400, etc.). |

**Search field routing:**

| Field | Embeddings | Query Encoder | Best For |
|-------|-----------|---------------|----------|
| `summary` | Caption embeddings | Text encoder | General content queries |
| `action` | Action brief embeddings | Text encoder | Activity/behavior queries |
| `visual` | SigLIP2 frame embeddings | SigLIP2 text | Appearance/object queries |
| `temporal` | V-JEPA 2 clip embeddings | SigLIP2 text | Motion/dynamics queries |
| `all` | Weighted composite (0.4/0.2/0.2/0.2) | Both | Broad queries |

#### Frame Extraction Tools

| Tool | Description |
|------|-------------|
| `kuavi_extract_frames` | Extract base64 JPEG frames. Configurable FPS, resolution, max frames. Results cached for index-referencing. |
| `kuavi_zoom_frames` | Preset zoom levels: L1 (480x360, overview), L2 (720x540, detail), L3 (1280x960, high-res). |

#### Pixel Analysis Tools

| Tool | Description |
|------|-------------|
| `kuavi_crop_frame` | Crop region using percentage coordinates (0.0-1.0). |
| `kuavi_diff_frames` | Absolute pixel difference. Returns `mean_diff`, `max_diff`, `changed_pct`. |
| `kuavi_blend_frames` | Average multiple frames into composite (motion summary / background extraction). |
| `kuavi_threshold_frame` | Binary threshold + contour detection. Returns `white_pct`, `contour_count`, `contour_areas`. |
| `kuavi_frame_info` | Image metadata: dimensions, brightness stats, color channel means. |

All pixel tools accept frame references by **integer index** into the last `extract_frames` result, avoiding redundant base64 re-transmission.

#### Compute Tools

| Tool | Description |
|------|-------------|
| `kuavi_eval` | Persistent Python REPL with `numpy`, `cv2`, all KUAVi tools as short names, `llm_query()`, `llm_query_batched()`. Namespace protected against accidental overwrites. |
| `kuavi_analyze_shards` | Split video into 30s shards, analyze each in parallel (4 workers) via LLM with optional multimodal frames. |

#### Management Tools

| Tool | Description |
|------|-------------|
| `kuavi_index_video` | Index a video file (runs full pipeline). |
| `kuavi_load_index` | Load a saved `.kuavi` index directory. |
| `kuavi_get_index_info` | Video metadata: segments, duration, resolution, embedding types. |
| `kuavi_get_session_stats` | Tool calls, frames extracted, searches performed, budget status. |
| `kuavi_set_budget` | Configure tool-call, time, and token limits. Hard limits block all gated tools. |
| `kuavi_set_llm_config` | Route primary (shard analysis) and secondary (eval's `llm_query`) to specific backends. |
| `kuavi_index_corpus` | Index multiple videos in parallel for cross-video search. |
| `kuavi_search_corpus` | Semantic search across all videos in a corpus index. |
| `kuavi_corpus_stats` | Statistics for the current corpus (video count, segment count, action vocabulary). |

#### Compound Tools

Compound tools batch common multi-call patterns into single MCP calls, reducing agent API round-trips by ~60%. Results from search and metadata tools are cached per video — repeated calls return instantly.

| Tool | Description |
|------|-------------|
| `kuavi_orient` | Get video overview: `get_index_info` + `get_scene_list` in one call. Cached per video. |
| `kuavi_search_all` | Multi-field search + transcript search in parallel via `ThreadPoolExecutor`. Cached by query parameters. |
| `kuavi_inspect_segment` | Extract frames + get transcript for a time range in one call. Supports zoom level presets (1-3). |

### Multi-Agent Orchestration

KUAVi uses a **triage-first** architecture with a **decompose-analyze-synthesize** pattern for complex video questions, coordinated by five specialized agents:

```
                       User Question
                            │
                            ▼
                  ┌──────────────────┐
                  │  video-triage   │ Haiku, 6 turns
                  │  (fast entry)   │
                  └────────┬─────────┘
                           │
              ┌────────────┴────────────┐
              │                         │
          ANSWERABLE                NEEDS FRAMES
        from captions              or is complex
              │                         │
              ▼                         ▼
      Direct answer             ┌──────────────────┐
      (Haiku, 1-2 turns)       │  video-analyst   │ Sonnet, 20 turns
      orient + search_all       │  (orchestrator)  │
      → answer from captions    └────────┬─────────┘
                                         │
                                Is this complex?
                                         │
                            ┌────────────┴────────────┐
                            │                         │
                        SIMPLE                    COMPLEX
                            │                         │
                            ▼                         ▼
                    Turn 1: orient +         ┌──────────────┐
                    search_all (parallel)    │  Phase 1:    │
                    Turn 2: inspect_segment  │  DECOMPOSE   │
                    (parallel per hit)       └──────┬───────┘
                    Turn 3: verify + answer
                                             ┌──────▼───────────┐
                                             │video-decomposer  │
                                             │Haiku, 8 turns    │
                                             └──────┬───────────┘

                                             JSON plan with
                                             sub-questions,
                                             time ranges,
                                             dependencies
                                                    │
                                             ┌──────▼──────┐
                                             │  Phase 2:   │
                                             │  ANALYZE    │
                                             └──────┬──────┘
                                                    │
                                     ┌──────────────┼──────────────┐
                                     ▼              ▼              ▼
                              ┌────────────┐ ┌────────────┐ ┌────────────┐
                              │  segment-  │ │  segment-  │ │  segment-  │
                              │ analyst #1 │ │ analyst #2 │ │ analyst #3 │
                              │(background)│ │(background)│ │(background)│
                              └─────┬──────┘ └─────┬──────┘ └─────┬──────┘
                                    │              │              │
                              Sonnet, 12 turns each, 10-call budget
                              search_all + inspect_segment (parallel)
                                    │              │              │
                                    └──────────────┼──────────────┘
                                                   │
                                            ┌──────▼──────────┐
                                            │video-synthesizer│ Sonnet
                                            │  search-only    │ 8 turns
                                            │  max 5 calls    │
                                            └──────┬──────────┘
                                                   │
                                            Final answer with
                                            timestamps, evidence,
                                            confidence levels
```

#### Agent Specifications

| Agent | Model | Max Turns | Tools | Role |
|-------|-------|:---------:|-------|------|
| **video-triage** | Haiku | 6 | `orient`, `search_all`, search-only + `Task(video-analyst)` | Fast entry point: answers from captions/transcript or escalates to Sonnet |
| **video-analyst** | Sonnet | 20 | All search + extract + compound + eval + `Task(decomposer, segment-analyst, synthesizer)` | Full analysis with frame inspection; orchestrates sub-agents for complex questions |
| **video-decomposer** | Haiku | 8 | Search-only (search_video, search_transcript, get_scene_list, get_transcript, discriminative_vqa) | Produces structured JSON decomposition plan with sub-questions and time ranges |
| **video-segment-analyst** | Sonnet | 12 | All search + extract + compound + all pixel tools + eval (no Task) | Analyzes one temporal region; runs in background for parallelism |
| **video-synthesizer** | Sonnet | 8 | Search-only (max 5 verification calls) | Aggregates per-segment results; resolves conflicts (visual > transcript) |

#### Search-First Strategy

Agents use compound tools to maximize parallel tool calls per turn:

1. **Turn 1 (parallel)** — `orient()` + `search_all(query, fields, transcript_query)` in the same response
2. **Turn 2 (parallel)** — `inspect_segment(start, end)` for all top hits simultaneously
3. **Turn 3** — Verify (screen content overrides transcript) and answer

Results from `orient`, `search_all`, `search_video`, `search_transcript`, `get_transcript`, `get_index_info`, and `get_scene_list` are **cached per video** — repeated calls across agents and turns return instantly without recomputation.

### Pixel Analysis Tools

For precise visual measurements (counting, motion detection, change tracking), KUAVi provides compositional pixel analysis via `kuavi_eval`:

```python
# Object counting via threshold + contours
frames = extract_frames(10.0, 12.0, fps=2)
result = threshold_frame(0, value=128)  # reference frame by index
print(f"Objects: {result['contour_count']}, areas: {result['contour_areas']}")

# Motion detection via frame differencing
diff = diff_frames(0, 1)  # compare first two extracted frames
print(f"Changed: {diff['changed_pct']:.1f}%")

# ROI tracking across time
for i in range(len(frames)):
    crop = crop_frame(i, x1_pct=0.2, y1_pct=0.3, x2_pct=0.8, y2_pct=0.7)
    info = frame_info(crop["image"])
    print(f"Frame {i}: brightness={info['brightness']['mean']:.1f}")

# Background extraction via blending
composite = blend_frames([0, 1, 2, 3, 4])  # average all frames

# Parallel LLM analysis of frames
prompts = [f"Describe frame {i}" for i in range(5)]
results = llm_query_batched(prompts, model="gemini-2.5-flash")
```

### Anti-Hallucination System

KUAVi enforces anti-hallucination through three layers:

| Layer | Mechanism | Description |
|-------|-----------|-------------|
| **Prompt-level** | `VIDEO_ANALYSIS_PROMPT` | 5 rules: never report unconfirmed numbers, never trust transcript numbers alone, be honest about inability to confirm, cross-reference, describe only consistent observations |
| **Hook-level** | `validate_transcript_claims.sh` | Fires on every transcript search; warns when response contains numbers or proper names that need visual confirmation |
| **Output-level** | `validate_visual_confirmation.sh` + `validate_analysis.sh` | Fires on stop; warns if final answer has numeric claims without frame evidence, or lacks timestamps/confidence markers |

All hooks are **advisory** (exit 0) — they add stderr warnings without blocking the agent, preserving autonomy while surfacing potential issues.

---

## Usage

### RLM Quick Start

```python
from rlm import RLM

rlm = RLM(
    backend="openai",
    backend_kwargs={"model_name": "gpt-5-nano"},
    verbose=True,
)

print(rlm.completion("Solve this step by step: what is 2^100?").response)
```

With custom tools:

```python
import math

rlm = RLM(backend="gemini", backend_kwargs={"model_name": "gemini-3.1-pro-preview"})

result = rlm.completion(
    "Calculate the area of a circle with radius 42",
    extra_tools={
        "circle_area": {"tool": lambda r: math.pi * r**2, "description": "Compute circle area"}
    },
)
```

### VideoRLM CLI

```bash
# Basic usage
uv run python run_video.py --video path/to/video.mp4 \
    --model gemini-3.1-pro-preview

# With specific question
uv run python run_video.py --video path/to/video.mp4 \
    --question "What happens in this video?"

# Auto-FPS + caching + high thinking
uv run python run_video.py --video path/to/video.mp4 \
    --auto-fps --cache-dir ./cache --thinking-level HIGH

# Minimal models (no V-JEPA, no EmbeddingGemma)
uv run python run_video.py --video path/to/video.mp4 \
    --no-scene-model --no-text-embedding
```

| Flag | Description |
|------|-------------|
| `--video` | Path to video file |
| `--model` | LLM model name (default: `gemini-3.1-pro-preview`) |
| `--backend` | LLM backend (default: `gemini`) |
| `--fps` | Manual FPS (default: 0.5) |
| `--auto-fps` | Auto-compute FPS from duration (targets 120 frames) |
| `--no-scene-model` | Disable V-JEPA 2; use SigLIP2 for scene detection |
| `--no-text-embedding` | Disable EmbeddingGemma; use SigLIP2 for text |
| `--cache-dir` | Cache indexes to disk |
| `--thinking-level` | Gemini thinking budget: `NONE`, `LOW`, `MEDIUM`, `HIGH` |

### KUAVi CLI

```bash
# Index a video
uv run python -m kuavi.cli index path/to/video.mp4 \
    --output ./my_index --auto-fps --cache-dir ./cache

# Search an index
uv run python -m kuavi.cli search "person walking" \
    --index-dir ./my_index --field visual --top-k 10

# Analyze with Claude Code agent
uv run python -m kuavi.cli analyze path/to/video.mp4 \
    -q "What are the key events in this video?"

# Batch analysis
uv run python -m kuavi.cli analyze --batch videos.txt \
    -q "Summarize the content" --max-parallel 4 --output-dir ./results
```

### KUAVi MCP Server

Start the server directly:

```bash
uv run python -m kuavi.mcp_server   # stdio transport
```

Or configure in `.mcp.json` for Claude Code:

```json
{
  "mcpServers": {
    "kuavi": {
      "command": "uv",
      "args": ["run", "python", "-m", "kuavi.mcp_server"]
    }
  }
}
```

---

## Trajectory Logging & Visualization

Both RLM and KUAVi support trajectory logging for debugging and analysis:

**RLM logging:**
```python
from rlm.logger import RLMLogger
from rlm import RLM

logger = RLMLogger(log_dir="./logs")  # JSONL per run
rlm = RLM(..., logger=logger)
```

**KUAVi logging:** Automatic dual-layer trace logging:
- **Server-side** (`_TraceLogger`): Logs every MCP tool call, LLM call, and eval execution
- **Hook-side** (`kuavi_trace_logger.sh`): Logs tool calls, agent start/stop, turn boundaries, final answers

Both write to `./logs/` as JSONL files with events: `session_start`, `turn_start`, `tool_call`, `llm_call`, `eval_execution`, `agent_start`, `agent_stop`, `final_answer`, `session_end`.

**Visualizer:**
```bash
cd visualizer/
npm run dev   # localhost:3001
```

Select `.jsonl` files to inspect code blocks, sub-LM calls, tool invocations, and full trajectories.

<p align="center">
  <img src="media/visualizer.png" alt="RLM Visualizer Example" width="800"/>
</p>

---

## Project Structure

```
rlm/                            # RLM core package
├── core/
│   ├── rlm.py                  # Main RLM class — REPL loop, stopping criteria
│   ├── lm_handler.py           # TCP server routing sub-LM requests
│   ├── comms_utils.py          # Wire protocol helpers
│   └── types.py                # REPLResult, RLMIteration, RLMChatCompletion
├── environments/
│   ├── local_repl.py           # Default: same-process exec
│   ├── docker_repl.py          # Docker container isolation
│   ├── modal_repl.py           # Modal cloud sandbox
│   ├── prime_repl.py           # Prime Intellect sandbox
│   ├── daytona_repl.py         # Daytona sandbox
│   └── e2b_repl.py             # E2B code interpreter
├── clients/
│   ├── openai.py, gemini.py, anthropic.py, portkey.py, litellm.py
│   └── base_lm.py              # Abstract base class
├── logger/                     # Trajectory capture
├── video/
│   ├── video_rlm.py            # VideoRLM — tool injection + sharding
│   ├── video_indexer.py         # Indexing pipeline (mirrors kuavi/indexer.py)
│   ├── video_search_tools.py   # Tool factories (mirrors kuavi/search.py)
│   └── probes.py               # AttentiveProbe (mirrors kuavi/probes.py)
└── utils/                      # Prompts, parsing

kuavi/                          # KUAVi package (standalone)
├── indexer.py                  # VideoIndexer, VideoIndex (8-stage pipeline)
├── search.py                   # Search tool factories (MMR, VQA, transcript, prediction)
├── loader.py                   # VideoLoader, LoadedVideo, VideoSegment
├── scene_detection.py          # V-JEPA 2 / SigLIP2 scene boundary detection
├── context.py                  # VideoContext, frame encoding
├── mcp_server.py               # FastMCP stdio server (30 tools, result caching)
├── captioners.py               # Pluggable captioner backends (Gemini, OpenAI, local)
├── probes.py                   # AttentiveProbe, ProbeRegistry (cross-attention classifiers)
├── corpus.py                   # CorpusIndex, CorpusIndexer (multi-video indexing)
├── prompts.py                  # VIDEO_ANALYSIS_PROMPT
├── types.py                    # KUAViConfig
└── cli.py                      # CLI: index, search, analyze, corpus

.claude/                        # Claude Code integration
├── agents/                     # 5 agent definitions
├── skills/                     # 8 skill definitions
├── hooks/                      # Validation + trace hooks
├── rules/                      # Architecture + development docs
└── settings.json               # Permissions + MCP config

run_video.py                    # VideoRLM entry point
visualizer/                     # Next.js trajectory viewer
```

---

## V-JEPA 2 + Action100M Integration

The unified integration plan merged insights from three research papers — [V-JEPA 2](https://arxiv.org/abs/2506.09985), [VL-JEPA](https://arxiv.org/abs/2410.07538), and [Action100M](https://arxiv.org/abs/2506.15686) — into KUAVi's indexing and search pipeline across 13 work items:

| Commit | Work Items | Description |
|--------|-----------|-------------|
| `3bfb9b3` | WI-0 | Fix temporal search dimension mismatch |
| `b9d3ba7` | WI-1, WI-2 | V-JEPA 2 model presets (fast/balanced/quality) + action-first indexing mode |
| `8f7bc5e` | WI-3, WI-4 | Self-Refine v2 (iterative annotation refinement) + multi-signal quality scoring |
| `f6d8c41` | WI-5, WI-7 | Pluggable captioner abstraction + spatial feature map storage |
| `5724cc6` | WI-6, WI-8 | Overlapping V-JEPA 2 windows (per-frame averaging) + semantic deduplication (K-means) |
| `2bfcc2e` | WI-9, WI-10 | Action anticipation (V-JEPA 2 predictor) + attentive probe classification (4-layer cross-attention) |
| `25e31b1` | WI-12 | Corpus-level multi-video indexing with cross-video search and action vocabulary |
| `a3b4595` | WI-11 | Predictive video understanding — temporal coherence verification and anomaly detection |

**Key capabilities added:**

- **Overlapping V-JEPA 2 windows** (WI-6): Configurable stride produces per-frame averaged embeddings for smoother, more accurate scene detection
- **Semantic deduplication** (WI-8): K-means clustering on caption embeddings to detect and filter near-duplicate segments
- **Action anticipation** (WI-9): Predict what happens next using V-JEPA 2's predictor or embedding similarity fallback
- **Attentive probes** (WI-10): 4-layer cross-attention classifiers on frozen V-JEPA 2 features for benchmark tasks (SSv2, K400, Diving48, etc.)
- **World model** (WI-11): Temporal coherence scoring and anomaly detection — segments where predicted ≠ actual are flagged as surprising
- **Corpus indexing** (WI-12): Parallel multi-video indexing with cross-video search via stacked embeddings

Test suite: 780 tests passing across 30 test files.

---

## Relevant Reading

* **[Dec '25]** [Recursive Language Models arXiv](https://arxiv.org/abs/2512.24601)
* **[Oct '25]** [Recursive Language Models Blogpost](https://alexzhang13.github.io/blog/2025/rlm/)

## Citation

```bibtex
@misc{zhang2025recursivelanguagemodels,
      title={Recursive Language Models},
      author={Alex L. Zhang and Tim Kraska and Omar Khattab},
      year={2025},
      eprint={2512.24601},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2512.24601},
}
```
