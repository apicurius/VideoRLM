from __future__ import annotations

import asyncio
import json
import logging
import os
import queue
import re
import shutil
import threading
import uuid
from pathlib import Path

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

load_dotenv()

app = FastAPI(title="VideoRLM Web")

UPLOAD_DIR = Path("/tmp/rlm_web_uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

STATIC_DIR = Path(__file__).parent / "web_static"

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


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


def _parse_timestamps(text: str) -> list[dict]:
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


def _render_answer_html(text: str) -> str:
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
    return re.sub(pattern, replacer, text, flags=re.IGNORECASE)


SCENE_MODEL = "facebook/vjepa2-vitl-fpc64-256"
VISUAL_EMBED_MODEL = "google/siglip2-base-patch16-256"
TEXT_EMBED_MODEL = "google/embedding-gemma-300m"

PIPELINE_STEPS = [
    {"id": "vjepa",   "label": "V-JEPA 2 Scene Detection"},
    {"id": "siglip",  "label": "SigLIP2 Visual Embeddings"},
    {"id": "gemma",   "label": "Gemma Text Encoder"},
    {"id": "whisper", "label": "FastWhisper ASR"},
    {"id": "index",   "label": "Search Index"},
    {"id": "agent",   "label": "Recursive Agent Loop"},
]

_AGENT_TOOLS = [
    "search_video", "search_transcript", "extract_frames",
    "crop_frame", "diff_frames", "blend_frames",
    "get_scene_list", "get_transcript", "discriminative_vqa",
]


@app.get("/api/arch")
async def arch_info():
    return JSONResponse({
        "scene_model": SCENE_MODEL,
        "visual_embed_model": VISUAL_EMBED_MODEL,
        "text_embed_model": TEXT_EMBED_MODEL,
    })


_log = logging.getLogger(__name__)


class _QueueLogHandler(logging.Handler):
    def __init__(self, emit):
        super().__init__()
        self._emit = emit

    def emit(self, record):
        msg = record.getMessage()
        if "[pipeline] V-JEPA 2: detecting scenes" in msg:
            self._emit({"type": "step", "id": "vjepa", "status": "running", "detail": msg.split("[pipeline] ")[-1]})
        elif "[pipeline] V-JEPA 2:" in msg:
            self._emit({"type": "step", "id": "vjepa", "status": "done", "detail": msg.split("[pipeline] ")[-1]})
        elif "[pipeline] SigLIP2: building" in msg:
            self._emit({"type": "step", "id": "siglip", "status": "running", "detail": msg.split("[pipeline] ")[-1]})
        elif "[pipeline] SigLIP2:" in msg:
            self._emit({"type": "step", "id": "siglip", "status": "done", "detail": msg.split("[pipeline] ")[-1]})
        elif "[pipeline] Gemma: embedding" in msg:
            self._emit({"type": "step", "id": "gemma", "status": "running", "detail": msg.split("[pipeline] ")[-1]})
        elif "[pipeline] Gemma:" in msg:
            self._emit({"type": "step", "id": "gemma", "status": "done", "detail": msg.split("[pipeline] ")[-1]})
        elif "[pipeline] FastWhisper: starting" in msg:
            self._emit({"type": "step", "id": "whisper", "status": "running", "detail": msg.split("[pipeline] ")[-1]})
        elif "[pipeline] FastWhisper:" in msg or "faster_whisper not installed" in msg:
            status = "skip" if "not installed" in msg else "done"
            self._emit({"type": "step", "id": "whisper", "status": status, "detail": msg.split("[pipeline] ")[-1]})
        elif "[pipeline] search index:" in msg:
            self._emit({"type": "step", "id": "index", "status": "done", "detail": msg.split("[pipeline] ")[-1]})
        elif "Returning in-memory cached index" in msg or "Loading cached index" in msg:
            for sid in ["vjepa", "siglip", "gemma", "whisper"]:
                self._emit({"type": "step", "id": sid, "status": "cached", "detail": "loaded from cache"})
            self._emit({"type": "step", "id": "index", "status": "done", "detail": "search index loaded from cache"})


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

    def clear_iterations(self) -> None:
        self._iterations = []
        self._iter = 0
        self._iteration_count = 0

    def get_trajectory(self) -> dict | None:
        if self._run_metadata is None:
            return None
        return {"metadata": self._run_metadata, "iterations": self._iterations}


def _full_pipeline(
    video_path: str,
    question: str,
    model: str,
    api_key: str,
    backend: str,
    emit,
) -> None:
    from rlm.clients.openai import OpenAIClient
    from rlm.video.video_rlm import VideoRLM

    client_backend = "openai" if backend == "openrouter" else backend
    bkw: dict = {"model_name": model, "api_key": api_key}
    if backend == "openrouter":
        bkw["base_url"] = "https://openrouter.ai/api/v1"

    caption_model = "openai/gpt-4o-mini" if backend == "openrouter" else model
    caption_lm = OpenAIClient(
        model_name=caption_model,
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
            return caption_lm.completion(parts)
        except Exception:
            return ""

    log_handler = _QueueLogHandler(emit)
    log_handler.setLevel(logging.DEBUG)
    indexer_logger = logging.getLogger("rlm.video.video_indexer")
    indexer_logger.addHandler(log_handler)
    try:
        rlm_logger = _EventRLMLogger(emit)
        video_rlm = VideoRLM(
            backend=client_backend,
            backend_kwargs=bkw,
            enable_search=True,
            scene_model=SCENE_MODEL,
            embedding_model=VISUAL_EMBED_MODEL,
            text_embedding_model=TEXT_EMBED_MODEL,
            whisper_model="base",
            caption_fn=caption_fn,
            auto_fps=True,
            num_segments=8,
            max_frames_per_segment=4,
            max_iterations=15,
            token_budget=100_000,
            logger=rlm_logger,
        )
        augmented = (
            f"{question}\n\n"
            "ANALYSIS STRATEGY (follow this order):\n"
            "1. Call get_scene_list() to see all scenes and their timestamps.\n"
            "2. Use search_video(query, field='visual') to find visually relevant scenes.\n"
            "3. Call extract_frames(start, end, fps=2.0, max_frames=6) to zoom into promising scenes.\n"
            "4. For fine-grained detail, use crop_frame(frame, x1, y1, x2, y2) then llm_query().\n"
            "5. Use search_transcript(keyword) for any spoken/audio clues.\n"
            "6. Cite moments with [TS: X.X] (seconds) right after each claim."
        )
        result = video_rlm.completion(video_path, prompt=augmented)
        emit({"type": "step", "id": "agent", "status": "done"})
        timestamps = _parse_timestamps(result.response)
        answer_html = _render_answer_html(result.response)
        emit({
            "type": "result",
            "answer": result.response,
            "answer_html": answer_html,
            "timestamps": timestamps,
        })
    except Exception as exc:
        emit({"type": "error", "message": str(exc)})
    finally:
        indexer_logger.removeHandler(log_handler)


@app.post("/api/analyze")
async def analyze(
    video: UploadFile = File(...),
    question: str = Form(...),
    backend: str = Form(default="openrouter"),
    model: str = Form(default="openai/gpt-4o"),
):
    suffix = Path(video.filename or "upload.mp4").suffix or ".mp4"
    video_id = str(uuid.uuid4())
    video_path = UPLOAD_DIR / f"{video_id}{suffix}"

    with open(video_path, "wb") as f:
        shutil.copyfileobj(video.file, f)

    api_key = {
        "openai": os.getenv("OPENAI_API_KEY"),
        "openrouter": os.getenv("OPENROUTER_API_KEY"),
        "anthropic": os.getenv("ANTHROPIC_API_KEY"),
        "gemini": os.getenv("GEMINI_API_KEY"),
    }.get(backend) or os.getenv("OPENROUTER_API_KEY", "")

    event_q: queue.Queue = queue.Queue()

    def emit(event: dict) -> None:
        event_q.put(event)

    def run() -> None:
        _full_pipeline(str(video_path), question, model, api_key, backend, emit)
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
