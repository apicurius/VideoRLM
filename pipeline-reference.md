# VideoRLM Pipeline — Complete Technical Reference

## System Architecture

```
Browser (http://localhost:4000)
         │
         │  HTTP / SSE
         ▼
Next.js Frontend  :4000
         │
         │  Proxy rewrite  /backend/* → localhost:7860/*
         ▼
FastAPI Backend   :7860   (web_app.py)
         │
         ├──► V-JEPA 2       (scene detection)
         ├──► Whisper         (speech recognition)
         ├──► VLM             (segment captioning)
         ├──► Gemma           (text embeddings)
         ├──► SigLIP2         (visual embeddings)
         ├──► FAISS           (search index)
         └──► OpenRouter API  (recursive agent loop)
```

---

## Setup & Running

### Environment Variables
```bash
# .env
OPENROUTER_API_KEY=sk-or-v1-...
```

### Start Everything
```bash
./run.sh          # starts backend on :7860 + frontend on :4000
./run_backend.sh  # FastAPI + uvicorn only
./run_frontend.sh # Next.js only
```

### What `run.sh` does
1. Loads `.env`
2. Warns if `OPENROUTER_API_KEY` is missing
3. Sets `PYTORCH_ALLOC_CONF=expandable_segments:True` for GPU memory
4. Installs missing Python deps (`openai`, `markdown`) via `uv`
5. Installs `frontend/node_modules` if absent
6. Starts FastAPI in the background (`--reload`)
7. Starts Next.js in the foreground
8. `Ctrl-C` kills both processes via `trap`

---

## Pipeline Stages

### Stage 01 — V-JEPA 2 Scene Detection

**Typical duration:** 10–30s depending on video length

**What it is:**
V-JEPA 2 is Meta's Video Joint Embedding Predictive Architecture — a self-supervised video encoder trained to predict future latent representations in embedding space (not pixel space).

**How it works:**
```
Raw video
    │
    ▼
Sliding temporal window (e.g. 16-frame clips)
    │
    ▼
V-JEPA 2 encoder → latent embedding per window
    │
    ▼
Cosine similarity between adjacent windows
    │
    ▼
Sharp similarity drop → scene boundary detected
    │
    ▼
Video cut into N semantically coherent segments
```

**Why V-JEPA 2 instead of frame-diff:**
- Frame-diff detects *pixel* changes (lighting, camera shake)
- V-JEPA 2 detects *semantic* changes ("car scene" → "building scene")
- Produces far fewer, more meaningful segments

**Output:**
```
N scenes detected
→ [seg_00, seg_01, ..., seg_N]
   each = { start_time, end_time, keyframes[] }
```

---

### Stage 02 — Speech Recognition

**Possible statuses:** `DONE` or `SKIP`

**What it is:**
An ASR (Automatic Speech Recognition) pass over the video's audio track using Whisper or Qwen3-ASR.

**Available models (selectable in UI):**
- `faster-whisper/base` — fastest
- `faster-whisper/small`
- `faster-whisper/medium`
- `faster-whisper/large-v3` — most accurate
- `Qwen/Qwen3-ASR-0.6B` — alternative

**Why it may be skipped:**
- The video has no audio track
- Audio contains no intelligible speech
- Common for: drone footage, silent surveillance, sports recordings

**If it ran:**
```
Audio track → Whisper → per-segment transcripts

seg_03: "The vehicle approaches the junction at high speed"
seg_07: "Target lost behind the building on the right"
```
These transcripts feed into Stage 03 (as VLM context) and Stage 04 (as additional text to embed), giving the system audio-grounded understanding of the video.

---

### Stage 03 — Segment Captioning

**Possible statuses:** `DONE`, `SKIP`

**What it is:**
A Vision-Language Model (VLM) is called on 1–3 keyframes per segment to generate a natural language description of what is visually happening.

**Two modes (selectable in UI):**

| Mode | Behaviour | Speed |
|------|-----------|-------|
| **Fast** | Skip captioning entirely | ~seconds total |
| **Captioned** | VLM called per segment | ~1–5s × N segments |

**If it ran (Captioned mode):**
```
seg_00 keyframe → VLM → "Aerial view of a highway interchange, light traffic"
seg_04 keyframe → VLM → "A red SUV merges onto the motorway from the left"
seg_09 keyframe → VLM → "Vehicle parks inside an underground structure"
```

**Caption model selection:**
- If `GEMINI_API_KEY` is set → uses `gemini-2.5-flash` for captioning
- Otherwise → falls back to the selected OpenRouter model (e.g. `openai/gpt-4o-mini`)

**Impact on answer quality:**
- **Fast mode:** Agent only has visual similarity vectors — can find *which segments look similar* to the query but cannot describe *what it sees*
- **Captioned mode:** Agent has rich text descriptions — can reason about colours, actions, objects, spatial relationships

---

### Stage 04 — Gemma Text Embeddings

**Typical duration:** 0ms (fast mode) to several seconds (captioned mode)

**What it is:**
Gemma (Google's lightweight open LLM) is used here purely as a **text encoder**, not a generator. It converts all available text signals into dense vectors.

**Inputs:**
- Segment captions (from Stage 03, if not skipped)
- ASR transcripts (from Stage 02, if not skipped)
- In Fast Mode: segment metadata only (timestamps, indices)

**How it works:**
```
"A red SUV merges onto the motorway"
    │
    ▼
Gemma encoder (last hidden state / mean pool)
    │
    ▼
[0.23, -0.11, 0.87, 0.04, ...]  ← 768-dim semantic vector
```

**Why 0ms in fast mode:**
No captions, no transcript → nothing to embed. Stage completes instantly with empty/placeholder embeddings.

---

### Stage 05 — SigLIP2 Visual Embeddings

**Typical duration:** 50–200ms

**What it is:**
SigLIP2 (Google's Sigmoid Loss for Image-Language Pretraining v2) embeds keyframes into a **shared vision-language embedding space** — meaning text and images can be directly compared.

**How it works:**
```
seg_00 keyframe (image)
    │
    ▼
SigLIP2 Vision Encoder (ViT-based)
    │
    ▼
[0.12, 0.88, -0.34, ...]  ← 1152-dim vector

User query (text)
    │
    ▼
SigLIP2 Text Encoder
    │
    ▼
[0.11, 0.85, -0.31, ...]  ← same 1152-dim space

dot_product(image_vec, text_vec) = similarity score
```

**Why SigLIP2 over CLIP:**
- Uses **sigmoid contrastive loss** instead of softmax
- More robust for open-vocabulary retrieval (no normalisation artifacts)
- Better at fine-grained visual matching

**Output:**
```
SigLIP2: N frame embeddings built
→ { seg_id → 1152-dim vector } × N
```

---

### Stage 06 — Search Index

**Typical duration:** near-instant

**What it is:**
All embeddings from Stage 04 (text) and Stage 05 (visual) are merged into a **FAISS index** for sub-millisecond nearest-neighbour retrieval.

**How it works:**
```
For each segment i:
    visual_vec[i]  = SigLIP2 embedding     (1152-dim)
    text_vec[i]    = Gemma embedding       (768-dim)

    combined[i] = concat(visual_vec[i], text_vec[i])
    OR
    combined[i] = α·visual_vec[i] + (1-α)·text_vec[i]

→ All combined vectors → FAISS FlatIP index (inner product)
```

**At query time:**
```
query → SigLIP2 text encoder → query_vector
FAISS.search(query_vector, k=5) → top-5 segment IDs + scores
```

**Caching:**
The index is cached in memory between queries on the same video. A second question on the same video skips Stages 01–06 entirely and jumps straight to Stage 07.

---

### Stage 07 — Recursive Agent Loop

**Typical duration:** 3–15s (LLM API latency dominated)

**What it is:**
The core reasoning engine. An LLM (via OpenRouter) runs a **multi-turn agentic loop** where it iteratively searches, inspects evidence, and decides whether it has enough information to answer.

**Why "Recursive":**
Mirrors the RLM (Recursive Language Model) architecture — the model can call tools recursively until confident, rather than doing a single-pass retrieval. This is crucial for complex multi-hop queries.

**Available tools the agent can call:**

| Tool | What it does |
|------|-------------|
| `search_video(query, field, top_k)` | FAISS search over visual or temporal embeddings |
| `search_transcript(query)` | Text search over ASR transcript |
| `extract_frames(start, end, fps)` | Extract raw frames from a time range |
| `get_scene_list()` | List all detected scenes with timestamps |
| `get_transcript(start, end)` | Get transcript text for a time range |
| `orient()` | Get video overview (index info + scene list) in one call |
| `search_all(query)` | Multi-field search in parallel (visual + temporal + transcript) |
| `inspect_segment(start, end)` | Extract frames + transcript for a range in one call |
| `crop_frame(image, x1, y1, x2, y2)` | Crop a frame to a region |
| `diff_frames(img_a, img_b)` | Pixel diff between two frames |
| `blend_frames(images)` | Average-blend multiple frames |
| `threshold_frame(image, value)` | Binary threshold a frame |
| `frame_info(image)` | Get width, height, brightness, colour stats |
| `discriminative_vqa(query, frames)` | Visual Q&A on specific frames |

**Full loop detail:**

#### Turn 1 — Query Embedding & Initial Retrieval
```
User: "How many times does a red vehicle appear, and where does it go?"
    │
    ▼
SigLIP2 text encoder → query_vector
    │
    ▼
FAISS.search(query_vector, k=3)
    │
    ▼
Returns: [seg_04 (score: 0.94), seg_07 (0.91), seg_09 (0.87)]
```

#### Turn 1 — LLM Reasoning
```
System: You are a video analysis agent with access to search and inspection tools.

Retrieved segments:
  seg_04 (~0:40): "Red SUV merges onto motorway"
  seg_07 (~1:10): "Same vehicle passes camera at junction"
  seg_09 (~1:50): "Vehicle enters underground parking"

Query: "How many times does a red vehicle appear, and where does it go?"

Decide: do you have sufficient evidence? If not, call a tool.
```

#### Turn 2 — Recursive Re-query (if needed)
```python
# LLM decides it needs more context
search_all("vehicle exit underground parking structure")
    │
    ▼
FAISS → [seg_10, seg_09]
```

#### Final Answer Generation
```
"The red vehicle (appears to be an SUV) appears 3 times:
  1. ~0:40 (seg_04) — merges onto the motorway from the left lane
  2. ~1:10 (seg_07) — passes the roadside camera at the junction
  3. ~1:50 (seg_09) — enters an underground parking structure and stops"
```

**Timing breakdown:**
```
5.2s total ≈
  ~100ms  FAISS search (×2 turns)
  ~4.8s   OpenRouter LLM API latency
  ~300ms  SSE streaming overhead
```

---

## Full Data Flow

```
┌─────────────────────────────────────────────────────────┐
│                       INPUT                             │
│           video.mp4  +  "user query text"               │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
                ┌─────────────────┐
                │   Stage 01      │
                │   V-JEPA 2      │
                │  Scene Detect   │
                └────────┬────────┘
                         │  N segments (time ranges + keyframes)
                         │
          ┌──────────────┼──────────────┐
          │              │              │
          ▼              ▼              ▼
 ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
 │   Stage 02   │ │   Stage 03   │ │   Stage 05   │
 │   Whisper    │ │  VLM Caption │ │   SigLIP2    │
 │     ASR      │ │  (optional)  │ │   Visual     │
 └──────┬───────┘ └──────┬───────┘ └──────┬───────┘
        │ transcripts    │ captions        │ visual vectors
        │                │                 │
        └────────────────▼─────────────────┘
                         │
                         ▼
                ┌─────────────────┐
                │   Stage 04      │
                │   Gemma Text    │
                │   Embeddings    │
                └────────┬────────┘
                         │  text vectors (768-dim)
                         │
                         ▼
                ┌─────────────────┐
                │   Stage 06      │
                │   FAISS Index   │
                │     Build       │
                └────────┬────────┘
                         │  indexed segments
                         │
                         ▼
                ┌─────────────────┐
                │   Stage 07      │
                │  Recursive      │◄──── OpenRouter LLM
                │  Agent Loop     │      (GPT-4o / Claude /
                └────────┬────────┘       Gemini / Llama)
                         │
                         ▼
                ┌─────────────────┐
                │   SSE Stream    │
                │  token by token │
                └────────┬────────┘
                         │
                         ▼
                Browser renders answer
```

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **V-JEPA 2 for segmentation** | Semantic boundaries, not pixel-diff boundaries |
| **Two embedding models (SigLIP2 + Gemma)** | SigLIP2 handles cross-modal vision↔text alignment; Gemma handles deeper textual semantics from captions/transcripts |
| **Fast vs Captioned mode** | VLM inference per segment costs ~1–5s each; 11 segments × 3s = 33s captioning alone. Fast mode skips this for quick demos |
| **FAISS FlatIP index** | Exact inner product search, no approximation error; fast enough for hundreds of segments |
| **Recursive agent loop** | Complex queries ("red car near a building") need multi-hop reasoning — find "red car" segments, then cross-reference "building" segments |
| **OpenRouter as LLM gateway** | Single API key unlocks 100+ models (GPT-4o, Claude Sonnet, Gemini 2.5, Llama 4) without per-provider keys |
| **Next.js proxy `/backend/*`** | Avoids CORS, hides backend port from browser |
| **Video uploads bypass proxy** | Large video files go direct to `localhost:7860` to avoid Next.js 10MB body cap |
| **SSE streaming** | User sees answer tokens as they arrive — no waiting for full LLM response |
| **In-memory index cache** | Second question on same video skips Stages 01–06, answer in ~5s instead of ~30s |

---

## RLM Core Architecture (rlm library)

The Stage 07 agent loop is built on the `rlm` library. Here is how the components relate:

### LM Client Interface
```python
# rlm/clients/base_lm.py
class BaseLM(ABC):
    def completion(self, prompt: str | list[dict], model: str | None = None) -> str: ...
    def acompletion(self, prompt: str | list[dict], model: str | None = None) -> str: ...
    def get_usage_summary(self) -> UsageSummary: ...
    def get_last_usage(self) -> ModelUsageSummary: ...
```

### Environment Interface
```python
# rlm/environments/base_env.py
class NonIsolatedEnv(ABC):
    def setup(self): ...          # init globals, helpers
    def load_context(self, payload): ...  # make context available as `context`
    def execute_code(self, code: str) -> REPLResult: ...  # run agent code
    def cleanup(self): ...        # release resources
```

### LMHandler Communication Protocol
```
Agent code calls llm_query("prompt")
    │
    ▼
send_lm_request(address, LMRequest)
    │
    │  TCP socket, 4-byte big-endian length prefix + UTF-8 JSON
    ▼
LMHandler (ThreadingTCPServer)
    │
    ▼
BaseLM.completion() → OpenRouter API
    │
    ▼
LMResponse → back through socket
```

### Globals available inside the agent execution environment
```python
context          # the loaded video index + segments
llm_query(prompt, model=None)          # single LLM call
llm_query_batched(prompts, model=None) # batched LLM calls
FINAL_VAR(variable_name)               # signal the final answer variable
```

### Naming & Style Conventions (from AGENTS.md)
- Methods: `snake_case`
- Classes: `PascalCase` (`OpenRouterClient`, `VideoEnvironment`)
- Constants: `UPPER_CASE` (`RLM_SYSTEM_PROMPT`, `SAFE_BUILTINS`)
- **Fail fast** — missing API key raises `ValueError` immediately, no silent fallback
- No `# type: ignore` without strong justification
- Formatting: strict `ruff` enforcement
