from __future__ import annotations

import asyncio
import base64
import concurrent.futures
import json
import logging
import os
import queue
import re
import shutil
import threading
import time
import uuid
from pathlib import Path

import markdown
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

load_dotenv(Path(__file__).parent / ".env")

app = FastAPI(title="VideoRLM + KUAVi Web")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("/tmp/rlm_web_uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)



def _seconds_to_label(seconds: float) -> str:
    total = int(seconds)
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    frac = seconds - int(seconds)
    frac_str = f".{int(frac * 10)}" if frac >= 0.05 else ""
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}{frac_str}"
    return f"{m}:{s:02d}{frac_str}"


def _parse_timestamps(text: str | None) -> list[dict]:
    if not text:
        return []
    found: list[float] = []

    for m in re.finditer(r"\[TS:\s*(\d+(?:\.\d+)?)\s*(?:s)?\]", text, re.IGNORECASE):
        found.append(float(m.group(1)))

    for m in re.finditer(r"\[TS:\s*(\d+):(\d{2})(?::(\d{2}))?\]", text, re.IGNORECASE):
        g = m.groups()
        if g[2] is not None:
            t = int(g[0]) * 3600 + int(g[1]) * 60 + int(g[2])
        else:
            t = int(g[0]) * 60 + int(g[1])
        found.append(float(t))

    seen: set[float] = set()
    unique: list[dict] = []
    for t in found:
        if t not in seen:
            seen.add(t)
            unique.append({"seconds": t, "label": _seconds_to_label(t)})

    return sorted(unique, key=lambda x: x["seconds"])


def _render_answer_html(text: str | None) -> str:
    if not text:
        return ""
    def replacer(m: re.Match) -> str:
        raw = m.group(0)
        try:
            if ":" in m.group(1):
                parts = m.group(1).split(":")
                if len(parts) == 3:
                    t = int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
                else:
                    t = int(parts[0]) * 60 + int(parts[1])
            else:
                t = float(m.group(1))
        except ValueError:
            return raw
        label = _seconds_to_label(float(t))
        return (
            f'<button class="ts-inline" data-t="{t}" '
            f'onclick="seekTo({t})">'
            f'⏱ {label}</button>'
        )

    pattern = r"\[TS:\s*([\d:.]+)\s*(?:s)?\]"
    processed_text = re.sub(pattern, replacer, text, flags=re.IGNORECASE)

    html_text = markdown.markdown(processed_text, extensions=['fenced_code', 'tables'])
    return html_text


SCENE_MODEL = "facebook/vjepa2-vitl-fpc64-256"
VISUAL_EMBED_MODEL = "google/siglip2-base-patch16-256"
TEXT_EMBED_MODEL = "google/embeddinggemma-300m"

PIPELINE_STEPS = [
    {"id": "vjepa",   "label": "V-JEPA 2 Scene Detection"},
    {"id": "whisper", "label": "Speech Recognition"},
    {"id": "caption", "label": "Segment Captioning"},
    {"id": "gemma",   "label": "Gemma Text Embeddings"},
    {"id": "siglip",  "label": "SigLIP2 Visual Embeddings"},
    {"id": "index",   "label": "Search Index"},
    {"id": "agent",   "label": "Recursive Agent Loop"},
]

_AGENT_TOOLS = [
    "search_video", "search_transcript", "extract_frames",
    "crop_frame", "diff_frames", "blend_frames", "threshold_frame", "frame_info",
    "get_scene_list", "get_transcript", "discriminative_vqa",
    "search_all", "inspect_segment", "orient",
]


@app.get("/api/arch")
async def arch_info():
    return JSONResponse({
        "scene_model": SCENE_MODEL,
        "visual_embed_model": VISUAL_EMBED_MODEL,
        "text_embed_model": TEXT_EMBED_MODEL,
        "tool_count": len(_TOOL_SCHEMAS),
        "tools": [s["function"]["name"] for s in _TOOL_SCHEMAS],
    })


@app.get("/api/tools")
async def list_tools():
    """Return full tool catalog with schemas."""
    return JSONResponse({
        "tools": _TOOL_SCHEMAS,
        "count": len(_TOOL_SCHEMAS),
    })


_log = logging.getLogger(__name__)


class _StepTimer:
    """Wraps an emit function to automatically track per-step elapsed time."""

    def __init__(self, raw_emit):
        self._raw_emit = raw_emit
        self._starts: dict[str, float] = {}
        self._completed_ms: dict[str, int] = {}

    def __call__(self, event: dict):
        if event.get("type") == "step":
            sid = event.get("id")
            status = event.get("status")
            if status == "running" and sid:
                # Only record start on the FIRST running event (don't reset on subsequent updates)
                if sid not in self._starts:
                    self._starts[sid] = time.time()
                # Always include elapsed_ms on running events so frontend can show live timer
                event = {**event, "elapsed_ms": int((time.time() - self._starts[sid]) * 1000)}
            elif status in ("done", "cached", "skip", "error") and sid:
                start = self._starts.pop(sid, None)
                if start is not None:
                    elapsed = int((time.time() - start) * 1000)
                    event = {**event, "elapsed_ms": elapsed}
                    self._completed_ms[sid] = elapsed
        self._raw_emit(event)

    def flush_summary(self) -> None:
        """Emit a timing_summary event with all completed step durations."""
        if not self._completed_ms:
            return
        step_timings = [{"id": sid, "elapsed_ms": ms} for sid, ms in self._completed_ms.items()]
        total_ms = sum(d["elapsed_ms"] for d in step_timings)
        self._raw_emit({"type": "timing_summary", "steps": step_timings, "total_ms": total_ms})


class _QueueLogHandler(logging.Handler):
    def __init__(self, emit, completed: set[str] | None = None):
        super().__init__()
        self._emit = emit
        self._completed = completed if completed is not None else set()

    def _emit_step(self, step_id: str, status: str, detail: str) -> None:
        self._emit({"type": "step", "id": step_id, "status": status, "detail": detail})
        if status in ("done", "cached", "skip"):
            self._completed.add(step_id)

    def emit(self, record):
        msg = record.getMessage()
        if "[pipeline] V-JEPA 2" in msg and "detecting scenes" in msg:
            self._emit_step("vjepa", "running", msg.split("[pipeline] ")[-1])
        elif "[pipeline] V-JEPA 2" in msg and "scenes detected" in msg:
            self._emit_step("vjepa", "done", msg.split("[pipeline] ")[-1])
        elif "[pipeline] SigLIP2: building" in msg:
            self._emit_step("siglip", "running", msg.split("[pipeline] ")[-1])
        elif "[pipeline] SigLIP2:" in msg:
            self._emit_step("siglip", "done", msg.split("[pipeline] ")[-1])
        elif "[pipeline] Gemma: embedding" in msg:
            self._emit_step("gemma", "running", msg.split("[pipeline] ")[-1])
        elif "[pipeline] Gemma:" in msg:
            self._emit_step("gemma", "done", msg.split("[pipeline] ")[-1])
        elif "[pipeline] Qwen3-ASR: loading" in msg or "[pipeline] Qwen3-ASR: starting" in msg:
            self._emit_step("whisper", "running", msg.split("[pipeline] ")[-1])
        elif "[pipeline] faster-whisper: loading" in msg or "[pipeline] faster-whisper: starting" in msg:
            self._emit_step("whisper", "running", msg.split("[pipeline] ")[-1])
        elif "qwen_asr not installed" in msg or "faster_whisper not installed" in msg:
            self._emit_step("whisper", "skip", msg.split("[pipeline] ")[-1] if "[pipeline] " in msg else msg)
        elif "[pipeline] Qwen3-ASR:" in msg:
            detail = msg.split("[pipeline] ")[-1]
            if "segments transcribed" in msg or "transcript segments" in msg:
                self._emit_step("whisper", "done", detail)
            else:
                self._emit_step("whisper", "running", detail)
        elif "[pipeline] faster-whisper:" in msg:
            detail = msg.split("[pipeline] ")[-1]
            if "segments transcribed" in msg or "transcript segments" in msg:
                self._emit_step("whisper", "done", detail)
            else:
                self._emit_step("whisper", "running", detail)
        elif "[pipeline] captioning: starting" in msg:
            self._emit_step("caption", "running", msg.split("[pipeline] ")[-1])
        elif "[pipeline] captioning:" in msg and "segments captioned" in msg:
            detail = msg.split("[pipeline] ")[-1]
            # "0 segments captioned" means no caption function was wired (fast mode) → skip
            if msg.strip().startswith("[pipeline] captioning: 0 "):
                self._emit_step("caption", "skip", "fast mode — embeddings only")
            else:
                self._emit_step("caption", "done", detail)
        elif "[pipeline] captioning:" in msg and "skipped" in msg:
            self._emit_step("caption", "skip", "fast mode — embeddings only")
        elif "[pipeline] captioning:" in msg:
            self._emit_step("caption", "running", msg.split("[pipeline] ")[-1])
        elif "Gemini caption" in msg or "caption_fn" in msg:
            if "failed" in msg:
                self._emit_step("caption", "running", "retrying...")
            else:
                self._emit_step("caption", "running", msg.split("] ")[-1] if "] " in msg else msg)
        elif "Re-captioned segment" in msg:
            self._emit_step("caption", "running", msg)
        elif "[pipeline] search index:" in msg:
            self._emit_step("index", "done", msg.split("[pipeline] ")[-1])
        elif "Returning in-memory cached index" in msg or "Loading cached index" in msg:
            for sid in ["vjepa", "whisper", "caption", "gemma", "siglip"]:
                self._emit_step(sid, "cached", "loaded from cache")
            self._emit_step("index", "done", "search index loaded from cache")


class _EventRLMLogger:
    def __init__(self, emit):
        self._emit = emit
        self._iter = 0
        self._iterations: list[dict] = []
        self._run_metadata: dict | None = None
        self._iteration_count = 0
        self._metadata_logged = False

    def log_metadata(self, metadata) -> None:
        if self._metadata_logged:
            return
        self._run_metadata = metadata.to_dict()
        self._metadata_logged = True

    def log_supplemental_metadata(self, **kwargs: object) -> None:
        if self._run_metadata is not None:
            self._run_metadata.update(kwargs)

    def log(self, iteration) -> None:
        if self._iter == 0:
            self._emit({"type": "step", "id": "agent", "status": "running"})
        self._iter += 1
        self._iteration_count = self._iter
        tools_used = []
        repl_errors = []
        for block in iteration.code_blocks:
            for tool in _AGENT_TOOLS:
                if tool in block.code and tool not in tools_used:
                    tools_used.append(tool)
            err = (block.result.stderr or "").strip()
            if err:
                repl_errors.append(err[:400])
        self._emit({"type": "iteration", "n": self._iter, "tools": tools_used, "errors": repl_errors})
        self._iterations.append({"type": "iteration", "iteration": self._iter, **iteration.to_dict()})

    def log_supplemental_metadata(self, **kwargs: object) -> None:
        if self._run_metadata is not None:
            self._run_metadata.update(kwargs)

    def clear_iterations(self) -> None:
        self._iterations = []
        self._iter = 0
        self._iteration_count = 0

    def get_trajectory(self) -> dict | None:
        if self._run_metadata is None:
            return None
        return {"metadata": self._run_metadata, "iterations": self._iterations}


def _use_gemini_captioning(backend: str, model: str) -> bool:
    """Check if we should use Gemini captioning based on backend/model."""
    return backend == "gemini" or "gemini" in model.lower()


# ---------------------------------------------------------------------------
# Re-export from kuavi.agent_runner — single source of truth
# ---------------------------------------------------------------------------

from kuavi.agent_runner import (  # noqa: E402
    AGENT_STRATEGY as _AGENT_STRATEGY,
    AGENT_SYSTEM as _AGENT_SYSTEM,
    SCENE_MODEL,
    TEXT_EMBED_MODEL,
    TOOL_SCHEMAS as _TOOL_SCHEMAS,
    VISUAL_EMBED_MODEL,
    make_compound_tools as _make_compound_tools,
    make_pixel_tools as _make_pixel_tools,
)


def _run_kuavi_agent(
    question: str,
    model: str,
    api_key: str,
    backend: str,
    tools_map: dict,
    emit,
    max_iterations: int = 12,
) -> str:
    """Thin wrapper around :mod:`kuavi.agent_runner` that converts the
    iterator-based interface into the emit-callback style used by the
    SSE pipeline.
    """
    from kuavi.agent_runner import _agent_gemini, _agent_openai

    is_gemini = backend == "gemini" or (backend != "openrouter" and "gemini" in model.lower())
    if is_gemini:
        agent_iter = _agent_gemini(question=question, model=model, api_key=api_key, tools_map=tools_map, max_iterations=max_iterations)
    else:
        agent_iter = _agent_openai(question=question, model=model, api_key=api_key, backend=backend, tools_map=tools_map, max_iterations=max_iterations)

    answer = ""
    for event in agent_iter:
        if event["type"] == "iteration":
            # Forward tool-call frames events for the frontend
            for name in event.get("tools", []):
                if name in ("extract_frames", "inspect_segment"):
                    pass  # frame events are emitted by the agent loop itself
            emit(event)
        elif event["type"] == "answer":
            answer = event.get("text", "")
    return answer


def _mark_pending_as_error(steps: list[dict], completed: set[str], emit, message: str) -> None:
    """Mark any still-pending pipeline steps as error when the pipeline fails."""
    for step in steps:
        if step["id"] not in completed:
            emit({"type": "step", "id": step["id"], "status": "error", "detail": message})


def _kuavi_pipeline(
    video_path: str,
    question: str,
    model: str,
    api_key: str,
    backend: str,
    emit,
    *,
    index_mode: str = "fast",
    asr_model: str = "faster-whisper/base",
) -> None:
    """KUAVi pipeline: VideoIndexer + search tools + tool-calling agent."""
    try:
        from kuavi.context import make_extract_frames
        from kuavi.indexer import VideoIndexer
        from kuavi.loader import VideoLoader
        from kuavi.search import (
            make_discriminative_vqa,
            make_get_scene_list,
            make_get_transcript,
            make_search_transcript,
            make_search_video,
        )
    except ImportError as exc:
        emit({"type": "error", "message": f"KUAVi not available: {exc}"})
        return

    completed: set[str] = set()

    def emit_step(step_id: str, status: str, detail: str | None = None) -> None:
        event: dict = {"type": "step", "id": step_id, "status": status}
        if detail:
            event["detail"] = detail
        if status in ("done", "cached", "skip"):
            completed.add(step_id)
        emit(event)

    # Attach log handler so pipeline stages emit real-time progress
    log_handler = _QueueLogHandler(emit, completed)
    log_handler.setLevel(logging.INFO)
    kuavi_logger = logging.getLogger("kuavi.indexer")
    kuavi_logger.setLevel(logging.INFO)
    kuavi_logger.addHandler(log_handler)

    try:
        emit_step("vjepa", "running", "loading video...")
        loader = VideoLoader(fps=0.5)
        loaded = loader.load(video_path)

        # Wire captioning only when user explicitly selects "captioned" mode
        caption_fn = None
        frame_caption_fn = None
        refine_fn = None
        use_captioning = index_mode == "captioned"

        if use_captioning:
            gemini_key = os.getenv("GEMINI_API_KEY") or (api_key if _use_gemini_captioning(backend, model) else None)
            if gemini_key:
                try:
                    from kuavi.captioning import (
                        make_gemini_caption_fn,
                        make_gemini_frame_caption_fn,
                        make_gemini_refine_fn,
                    )
                    caption_model = "gemini-2.5-flash"
                    caption_fn = make_gemini_caption_fn(model=caption_model, api_key=gemini_key)
                    frame_caption_fn = make_gemini_frame_caption_fn(model=caption_model, api_key=gemini_key)
                    refine_fn = make_gemini_refine_fn(model=caption_model, api_key=gemini_key)
                    emit_step("caption", "pending", f"using {caption_model}")
                except ImportError:
                    gemini_key = None

            if not gemini_key:
                try:
                    from rlm.clients.openai import OpenAIClient
                    cap_model = "openai/gpt-4o-mini" if backend == "openrouter" else model
                    cap_lm = OpenAIClient(
                        model_name=cap_model,
                        api_key=api_key,
                        base_url="https://openrouter.ai/api/v1" if backend == "openrouter" else None,
                    )

                    def caption_fn(frames):
                        parts: list = [
                            "Describe this video segment in 1-2 sentences. "
                            "Focus on what is shown visually, who/what is present, and any actions. "
                            "Be specific and concise."
                        ]
                        parts.extend(frames[:3])
                        try:
                            return cap_lm.completion(parts)
                        except Exception:
                            return ""

                    emit_step("caption", "pending", f"using {cap_model}")
                except ImportError:
                    emit_step("caption", "skip", "no captioning available")
        else:
            emit_step("caption", "skip", "fast mode — embeddings only")

        indexer = VideoIndexer(
            embedding_model=VISUAL_EMBED_MODEL,
            text_embedding_model=TEXT_EMBED_MODEL,
            scene_model=SCENE_MODEL,
        )
        index = indexer.index_video(
            loaded,
            asr_model=asr_model,
            caption_fn=caption_fn,
            frame_caption_fn=frame_caption_fn,
            refine_fn=refine_fn,
            mode="full" if index_mode == "captioned" else "fast",
        )

        n_scenes = len(index.scene_boundaries)
        n_segs = len(index.segments)

        # Emit done for any steps the log handler didn't already mark
        if "vjepa" not in completed:
            emit_step("vjepa", "done", f"{n_scenes} scene boundaries detected")

        if "whisper" not in completed:
            if index.transcript:
                emit_step("whisper", "done", f"{len(index.transcript)} transcript entries")
            else:
                emit_step("whisper", "skip", "no transcript")

        if "caption" not in completed:
            if use_captioning and caption_fn is not None:
                captioned = sum(1 for s in index.segments if s.get("caption"))
                emit_step("caption", "done", f"{captioned}/{n_segs} segments captioned")
            elif not use_captioning:
                pass  # already emitted "skip" above
            else:
                emit_step("caption", "skip", "no captioning available")

        if "gemma" not in completed:
            if use_captioning:
                emit_step("gemma", "done", "text embeddings ready")
            else:
                emit_step("gemma", "skip", "no captions to embed")
        if "siglip" not in completed:
            emit_step("siglip", "done", f"{n_segs} segments embedded")
        emit_step("index", "done", f"{n_segs} segments, {n_scenes} scenes")

        # Emit index stats for frontend
        emit({"type": "index_stats", "segments": n_segs, "scenes": n_scenes,
              "transcript_entries": len(index.transcript) if index.transcript else 0,
              "duration": index.segments[-1]["end_time"] if index.segments else 0})

        # Build basic tools map
        raw_extract = make_extract_frames(video_path)
        basic_tools = {
            "get_scene_list": make_get_scene_list(index)["tool"],
            "search_video": make_search_video(index)["tool"],
            "search_transcript": make_search_transcript(index)["tool"],
            "get_transcript": make_get_transcript(index)["tool"],
            "discriminative_vqa": make_discriminative_vqa(index)["tool"],
        }

        # Add pixel tools (wraps extract_frames with caching)
        pixel_tools = _make_pixel_tools(raw_extract)
        tools_map = {**basic_tools, **pixel_tools}

        # Add compound tools
        compound_tools = _make_compound_tools(index, tools_map)
        tools_map.update(compound_tools)

        emit_step("agent", "running")
        answer = _run_kuavi_agent(
            question=question,
            model=model,
            api_key=api_key,
            backend=backend,
            tools_map=tools_map,
            emit=emit,
        )

        emit_step("agent", "done")
        answer = answer or ""
        timestamps = _parse_timestamps(answer)
        answer_html = _render_answer_html(answer)
        emit({
            "type": "result",
            "answer": answer,
            "answer_html": answer_html,
            "timestamps": timestamps,
        })
    except Exception as exc:
        _log.exception("KUAVi pipeline error")
        short = str(exc)[:200]
        _mark_pending_as_error(PIPELINE_STEPS, completed, emit, short)
        emit({"type": "error", "message": str(exc)})
    finally:
        kuavi_logger.removeHandler(log_handler)

@app.post("/api/analyze")
async def analyze(
    video: UploadFile = File(...),  # noqa: B008
    question: str = Form(...),
    backend: str = Form(default="openrouter"),
    model: str = Form(default="openai/gpt-4o"),
    index_mode: str = Form(default="fast"),
    asr_model: str = Form(default="faster-whisper/base"),
    custom_api_key: str = Form(default=""),
):
    suffix = Path(video.filename or "upload.mp4").suffix or ".mp4"
    video_id = str(uuid.uuid4())
    video_path = UPLOAD_DIR / f"{video_id}{suffix}"

    with open(video_path, "wb") as f:
        shutil.copyfileobj(video.file, f)

    env_key = {
        "openai": os.getenv("OPENAI_API_KEY"),
        "openrouter": os.getenv("OPENROUTER_API_KEY"),
        "anthropic": os.getenv("ANTHROPIC_API_KEY"),
        "gemini": os.getenv("GEMINI_API_KEY"),
    }.get(backend) or os.getenv("OPENROUTER_API_KEY", "")

    api_key = custom_api_key.strip() or env_key

    event_q: queue.Queue = queue.Queue()

    def _raw_emit(event: dict) -> None:
        event_q.put(event)

    emit = _StepTimer(_raw_emit)

    def run() -> None:
        _kuavi_pipeline(str(video_path), question, model, api_key, backend, emit, index_mode=index_mode, asr_model=asr_model)
        emit.flush_summary()
        event_q.put(None)

    threading.Thread(target=run, daemon=True).start()
    loop = asyncio.get_event_loop()

    async def generator():
        yield f"data: {json.dumps({'type': 'init', 'steps': PIPELINE_STEPS})}\n\n"
        try:
            while True:
                event = await loop.run_in_executor(None, event_q.get)
                if event is None:
                    break
                yield f"data: {json.dumps(event)}\n\n"
        finally:
            video_path.unlink(missing_ok=True)

    return StreamingResponse(generator(), media_type="text/event-stream")


if __name__ == "__main__":
    uvicorn.run("web_app:app", host="0.0.0.0", port=8000, reload=True)
