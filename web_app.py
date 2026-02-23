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
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

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
    processed_text = re.sub(pattern, replacer, text, flags=re.IGNORECASE)

    html_text = markdown.markdown(processed_text, extensions=['fenced_code', 'tables'])
    return html_text


SCENE_MODEL = "facebook/vjepa2-vitl-fpc64-256"
VISUAL_EMBED_MODEL = "google/siglip2-base-patch16-256"
TEXT_EMBED_MODEL = "google/embeddinggemma-300m"

PIPELINE_STEPS = [
    {"id": "vjepa",   "label": "V-JEPA 2 Scene Detection"},
    {"id": "whisper", "label": "Qwen3-ASR"},
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
                    event = {**event, "elapsed_ms": int((time.time() - start) * 1000)}
        self._raw_emit(event)


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
        if "[pipeline] V-JEPA 2: detecting scenes" in msg:
            self._emit_step("vjepa", "running", msg.split("[pipeline] ")[-1])
        elif "[pipeline] V-JEPA 2:" in msg:
            self._emit_step("vjepa", "done", msg.split("[pipeline] ")[-1])
        elif "[pipeline] SigLIP2: building" in msg:
            self._emit_step("siglip", "running", msg.split("[pipeline] ")[-1])
        elif "[pipeline] SigLIP2:" in msg:
            self._emit_step("siglip", "done", msg.split("[pipeline] ")[-1])
        elif "[pipeline] Gemma: embedding" in msg:
            self._emit_step("gemma", "running", msg.split("[pipeline] ")[-1])
        elif "[pipeline] Gemma:" in msg:
            self._emit_step("gemma", "done", msg.split("[pipeline] ")[-1])
        elif "[pipeline] Qwen3-ASR: starting" in msg:
            self._emit_step("whisper", "running", msg.split("[pipeline] ")[-1])
        elif "[pipeline] Qwen3-ASR:" in msg or "qwen_asr not installed" in msg:
            status = "skip" if "not installed" in msg else "done"
            self._emit_step("whisper", status, msg.split("[pipeline] ")[-1])
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
# Pixel analysis tools (standalone, mirroring kuavi/mcp_server.py logic)
# ---------------------------------------------------------------------------

def _make_pixel_tools(extract_frames_fn):
    """Build pixel analysis tools that operate on extracted frame results.

    Returns a dict of tool_name -> callable, plus a shared frame cache
    so pixel tools can reference frames by index.
    """
    _frame_cache: list[dict] = []

    # Wrap extract_frames to populate the cache
    _orig_extract = extract_frames_fn

    def extract_frames_cached(**kwargs):
        result = _orig_extract(**kwargs)
        _frame_cache.clear()
        if isinstance(result, list):
            _frame_cache.extend(result)
        return result

    def _resolve(image):
        """Resolve an image — either a dict with data/mime_type, or an int index."""
        if isinstance(image, (int, float)):
            idx = int(image)
            if 0 <= idx < len(_frame_cache):
                return _frame_cache[idx]
            return {"error": f"Frame index {idx} out of range (0-{len(_frame_cache) - 1})"}
        return image

    def _decode(image):
        import cv2
        import numpy as np
        data = image.get("data", "")
        raw = base64.b64decode(data)
        arr = np.frombuffer(raw, np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)

    def _encode(frame):
        import cv2
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return {"data": base64.b64encode(buf.tobytes()).decode(), "mime_type": "image/jpeg"}

    def crop_frame(image, x1_pct, y1_pct, x2_pct, y2_pct):
        image = _resolve(image)
        if "error" in image:
            return image
        frame = _decode(image)
        h, w = frame.shape[:2]
        cropped = frame[int(y1_pct * h):int(y2_pct * h), int(x1_pct * w):int(x2_pct * w)]
        return {
            "image": _encode(cropped),
            "crop": {"x1_pct": x1_pct, "y1_pct": y1_pct, "x2_pct": x2_pct, "y2_pct": y2_pct,
                     "width": cropped.shape[1], "height": cropped.shape[0]},
        }

    def diff_frames(image_a, image_b):
        import cv2
        a = _resolve(image_a)
        b = _resolve(image_b)
        if isinstance(a, dict) and "error" in a:
            return a
        if isinstance(b, dict) and "error" in b:
            return b
        fa = _decode(a)
        fb = _decode(b)
        if fa.shape != fb.shape:
            fb = cv2.resize(fb, (fa.shape[1], fa.shape[0]))
        diff = cv2.absdiff(fa, fb)
        mean_diff = float(diff.mean())
        max_diff = int(diff.max())
        changed = (diff > 25).any(axis=2) if diff.ndim == 3 else (diff > 25)
        changed_pct = float(changed.sum() / changed.size * 100)
        return {
            "image": _encode(diff),
            "mean_diff": round(mean_diff, 2),
            "max_diff": max_diff,
            "changed_pct": round(changed_pct, 2),
        }

    def blend_frames(images):
        import cv2
        import numpy as np
        if not images:
            return {"error": "No images provided"}
        resolved = [_resolve(img) for img in images]
        frames = [_decode(r) for r in resolved if isinstance(r, dict) and "data" in r]
        if not frames:
            return {"error": "No valid frames"}
        target = frames[0].shape[:2]
        for i in range(1, len(frames)):
            if frames[i].shape[:2] != target:
                frames[i] = cv2.resize(frames[i], (target[1], target[0]))
        blended = np.mean(frames, axis=0).astype(np.uint8)
        return {"image": _encode(blended), "frame_count": len(frames)}

    def threshold_frame(image, value=128, invert=False):
        import cv2
        image = _resolve(image)
        if isinstance(image, dict) and "error" in image:
            return image
        frame = _decode(image)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        thresh_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
        _, mask = cv2.threshold(gray, value, 255, thresh_type)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        white_pct = float((mask == 255).sum() / mask.size * 100)
        contour_areas = sorted([float(cv2.contourArea(c)) for c in contours], reverse=True)
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        return {
            "image": _encode(mask_bgr),
            "white_pct": round(white_pct, 2),
            "contour_count": len(contours),
            "contour_areas": contour_areas[:20],
        }

    def frame_info(image):
        import cv2
        image = _resolve(image)
        if isinstance(image, dict) and "error" in image:
            return image
        frame = _decode(image)
        h, w = frame.shape[:2]
        channels = frame.shape[2] if frame.ndim == 3 else 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if channels == 3 else frame
        if channels == 3:
            b_mean, g_mean, r_mean = float(frame[:, :, 0].mean()), float(frame[:, :, 1].mean()), float(frame[:, :, 2].mean())
        else:
            b_mean = g_mean = r_mean = float(gray.mean())
        return {
            "width": w, "height": h, "channels": channels,
            "brightness": {"mean": round(float(gray.mean()), 2), "std": round(float(gray.std()), 2),
                           "min": int(gray.min()), "max": int(gray.max())},
            "color": {"b_mean": round(b_mean, 2), "g_mean": round(g_mean, 2), "r_mean": round(r_mean, 2)},
        }

    return {
        "extract_frames": extract_frames_cached,
        "crop_frame": crop_frame,
        "diff_frames": diff_frames,
        "blend_frames": blend_frames,
        "threshold_frame": threshold_frame,
        "frame_info": frame_info,
    }


# ---------------------------------------------------------------------------
# Compound tools (mirroring kuavi/mcp_server.py compound tools)
# ---------------------------------------------------------------------------

def _make_compound_tools(index, tools_map):
    """Build compound tools that combine multiple basic tool calls."""

    def orient():
        """Get video overview: index info + scene list in one call."""
        info = {
            "segments": len(index.segments),
            "duration": index.segments[-1]["end"] if index.segments else 0,
            "scene_boundaries": len(index.scene_boundaries),
            "has_transcript": bool(index.transcript),
            "transcript_entries": len(index.transcript) if index.transcript else 0,
        }
        scenes = tools_map["get_scene_list"]()
        return {"index_info": info, "scenes": scenes}

    def search_all(query, fields=None, top_k=5, transcript_query=None):
        """Multi-field search + transcript search in parallel."""
        if fields is None:
            fields = ["summary", "visual"]

        results = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = {}
            for field in fields:
                futures[executor.submit(tools_map["search_video"], query=query, field=field, top_k=top_k)] = f"search_{field}"
            tq = transcript_query or query
            futures[executor.submit(tools_map["search_transcript"], query=tq)] = "transcript"

            for future in concurrent.futures.as_completed(futures):
                key = futures[future]
                try:
                    results[key] = future.result()
                except Exception as exc:
                    results[key] = {"error": str(exc)}

        return results

    def inspect_segment(start_time, end_time, fps=2.0, max_frames=6):
        """Extract frames + get transcript for a time range in one call."""
        frames = tools_map["extract_frames"](start_time=start_time, end_time=end_time, fps=fps, max_frames=max_frames)
        transcript = tools_map["get_transcript"](start_time=start_time, end_time=end_time)
        return {"frames": frames, "transcript": transcript}

    return {
        "orient": orient,
        "search_all": search_all,
        "inspect_segment": inspect_segment,
    }


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

    try:
        emit_step("vjepa", "running", "loading video...")
        loader = VideoLoader(fps=0.5)
        loaded = loader.load(video_path)

        # Wire Gemini captioning when appropriate
        caption_fn = None
        frame_caption_fn = None
        refine_fn = None
        use_gemini = _use_gemini_captioning(backend, model)

        gemini_key = os.getenv("GEMINI_API_KEY") or (api_key if use_gemini else None)
        if gemini_key:
            try:
                from kuavi.captioning import (
                    make_gemini_caption_fn,
                    make_gemini_frame_caption_fn,
                    make_gemini_refine_fn,
                )
                caption_model = "gemini-3.1-pro-preview"
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

        indexer = VideoIndexer(
            embedding_model=VISUAL_EMBED_MODEL,
            text_embedding_model=TEXT_EMBED_MODEL,
            scene_model=SCENE_MODEL,
        )
        index = indexer.index_video(
            loaded,
            asr_model="Qwen/Qwen3-ASR-0.6B",
            caption_fn=caption_fn,
            frame_caption_fn=frame_caption_fn,
            refine_fn=refine_fn,
        )

        n_scenes = len(index.scene_boundaries)
        n_segs = len(index.segments)

        # Emit done in actual execution order: vjepa → whisper → caption → gemma → siglip → index
        emit_step("vjepa", "done", f"{n_scenes} scene boundaries detected")

        if index.transcript:
            emit_step("whisper", "done", f"{len(index.transcript)} transcript entries")
        else:
            emit_step("whisper", "skip", "no transcript")

        if use_gemini and caption_fn is not None:
            captioned = sum(1 for s in index.segments if s.get("caption"))
            emit_step("caption", "done", f"{captioned}/{n_segs} segments captioned")

        emit_step("gemma", "done", "text embeddings ready")
        emit_step("siglip", "done", f"{n_segs} segments embedded")
        emit_step("index", "done", f"{n_segs} segments, {n_scenes} scenes")

        # Emit index stats for frontend
        emit({"type": "index_stats", "segments": n_segs, "scenes": n_scenes,
              "transcript_entries": len(index.transcript) if index.transcript else 0,
              "duration": index.segments[-1]["end"] if index.segments else 0})

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
        timestamps = _parse_timestamps(answer)
        answer_html = _render_answer_html(answer)
        emit({
            "type": "result",
            "answer": answer,
            "answer_html": answer_html,
            "timestamps": timestamps,
        })
    except Exception as exc:
        short = str(exc)[:200]
        _mark_pending_as_error(PIPELINE_STEPS, completed, emit, short)
        emit({"type": "error", "message": str(exc)})


# ---------------------------------------------------------------------------
# Tool schemas for LLM function calling (OpenAI format)
# ---------------------------------------------------------------------------

_TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "orient",
            "description": "Get video overview: index metadata + full scene list in one call. Use this first to understand the video structure.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_all",
            "description": "Multi-field semantic search + transcript search in parallel. More efficient than separate search_video + search_transcript calls.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "fields": {
                        "type": "array",
                        "items": {"type": "string", "enum": ["summary", "action", "visual", "all"]},
                        "description": "Search fields to query (default: summary, visual)",
                    },
                    "top_k": {"type": "integer", "default": 5},
                    "transcript_query": {"type": "string", "description": "Optional different query for transcript search"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "inspect_segment",
            "description": "Extract frames + get transcript for a time range in one call. Use after search to inspect promising segments.",
            "parameters": {
                "type": "object",
                "properties": {
                    "start_time": {"type": "number", "description": "Start time in seconds"},
                    "end_time": {"type": "number", "description": "End time in seconds"},
                    "fps": {"type": "number", "default": 2.0},
                    "max_frames": {"type": "integer", "default": 6},
                },
                "required": ["start_time", "end_time"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_scene_list",
            "description": "List all detected scenes with start/end times and captions.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_video",
            "description": "Semantic search over video segments. Use field='summary' for caption-based, 'visual' for frame-based, 'action' for activity-based, 'all' for combined.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "top_k": {"type": "integer", "default": 5},
                    "field": {
                        "type": "string",
                        "enum": ["summary", "action", "visual", "all"],
                        "default": "summary",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_transcript",
            "description": "Keyword search over the ASR transcript.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_transcript",
            "description": "Get transcript text for a time range (seconds).",
            "parameters": {
                "type": "object",
                "properties": {
                    "start_time": {"type": "number"},
                    "end_time": {"type": "number"},
                },
                "required": ["start_time", "end_time"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "extract_frames",
            "description": "Extract video frames as base64 images from a time range.",
            "parameters": {
                "type": "object",
                "properties": {
                    "start_time": {"type": "number", "description": "Start time in seconds"},
                    "end_time": {"type": "number", "description": "End time in seconds"},
                    "fps": {"type": "number", "default": 2.0},
                    "max_frames": {"type": "integer", "default": 6},
                },
                "required": ["start_time", "end_time"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "discriminative_vqa",
            "description": "Embedding-based multiple-choice VQA. Ranks candidate answers by similarity.",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "candidates": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Candidate answers to rank",
                    },
                },
                "required": ["question", "candidates"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "crop_frame",
            "description": "Crop a region from an extracted frame using percentage coordinates (0.0-1.0).",
            "parameters": {
                "type": "object",
                "properties": {
                    "image": {"type": "integer", "description": "Frame index from last extract_frames result"},
                    "x1_pct": {"type": "number", "description": "Left edge (0.0-1.0)"},
                    "y1_pct": {"type": "number", "description": "Top edge (0.0-1.0)"},
                    "x2_pct": {"type": "number", "description": "Right edge (0.0-1.0)"},
                    "y2_pct": {"type": "number", "description": "Bottom edge (0.0-1.0)"},
                },
                "required": ["image", "x1_pct", "y1_pct", "x2_pct", "y2_pct"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "diff_frames",
            "description": "Compute absolute pixel difference between two frames. Returns mean_diff, max_diff, changed_pct.",
            "parameters": {
                "type": "object",
                "properties": {
                    "image_a": {"type": "integer", "description": "First frame index"},
                    "image_b": {"type": "integer", "description": "Second frame index"},
                },
                "required": ["image_a", "image_b"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "blend_frames",
            "description": "Average multiple frames into a composite image (background extraction / motion summary).",
            "parameters": {
                "type": "object",
                "properties": {
                    "images": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "List of frame indices to blend",
                    },
                },
                "required": ["images"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "threshold_frame",
            "description": "Apply binary threshold + contour detection. Returns white_pct, contour_count, contour_areas.",
            "parameters": {
                "type": "object",
                "properties": {
                    "image": {"type": "integer", "description": "Frame index"},
                    "value": {"type": "integer", "default": 128, "description": "Threshold value (0-255)"},
                    "invert": {"type": "boolean", "default": False},
                },
                "required": ["image"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "frame_info",
            "description": "Get image metadata: dimensions, brightness stats, color channel means.",
            "parameters": {
                "type": "object",
                "properties": {
                    "image": {"type": "integer", "description": "Frame index"},
                },
                "required": ["image"],
            },
        },
    },
]

_AGENT_SYSTEM = (
    "You are a video analysis assistant with access to a searchable video index.\n"
    "Use the tools to find relevant content, then answer the question.\n"
    "Always cite timestamps as [TS: X.X] (seconds) right after each factual claim.\n\n"
    "Available tools (use compound tools for efficiency):\n"
    "- orient(): Get video overview (metadata + scene list) in one call\n"
    "- search_all(query, fields, top_k, transcript_query): Multi-field search + transcript in parallel\n"
    "- inspect_segment(start_time, end_time): Extract frames + transcript for a time range\n"
    "- search_video(query, field, top_k): Semantic search (fields: summary, action, visual, all)\n"
    "- search_transcript(query): Keyword search over spoken words\n"
    "- get_transcript(start_time, end_time): Get transcript for a time range\n"
    "- extract_frames(start_time, end_time, fps, max_frames): Get video frames as images\n"
    "- discriminative_vqa(question, candidates): Multiple-choice VQA via embeddings\n"
    "- crop_frame(image, x1_pct, y1_pct, x2_pct, y2_pct): Crop region from frame\n"
    "- diff_frames(image_a, image_b): Pixel difference between frames\n"
    "- blend_frames(images): Average frames into composite\n"
    "- threshold_frame(image, value, invert): Binary threshold + contour detection\n"
    "- frame_info(image): Image dimensions, brightness, color stats\n"
)

_AGENT_STRATEGY = (
    "\n\nANALYSIS STRATEGY (follow this order for efficiency):\n"
    "1. Call orient() to see video structure, scenes, and timestamps.\n"
    "2. Use search_all(query, fields=['summary', 'visual']) for broad search.\n"
    "3. Use inspect_segment(start, end) to get frames + transcript for top hits.\n"
    "4. For fine-grained detail, use crop_frame, diff_frames, or frame_info.\n"
    "5. Use discriminative_vqa(question, candidates) for multiple-choice questions.\n"
    "6. Cite every fact with [TS: X.X].\n\n"
    "IMPORTANT: Prefer compound tools (orient, search_all, inspect_segment) over\n"
    "individual calls — they batch multiple operations into single calls."
)


def _run_kuavi_agent_gemini(
    question: str,
    model: str,
    api_key: str,
    tools_map: dict,
    emit,
    max_iterations: int = 12,
) -> str:
    """Tool-calling agent loop using native Gemini function calling."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key or os.getenv("GEMINI_API_KEY"))

    # Build Gemini-native tool declarations
    func_declarations = []
    for schema in _TOOL_SCHEMAS:
        fn = schema["function"]
        params = fn.get("parameters", {})
        props = params.get("properties", {})
        gemini_props = {}
        for k, v in props.items():
            ptype = v.get("type", "string").upper()
            if ptype == "INTEGER":
                ptype = "NUMBER"
            if ptype == "BOOLEAN":
                ptype = "BOOLEAN"
            if ptype == "ARRAY":
                item_type = v.get("items", {}).get("type", "STRING").upper()
                if item_type == "INTEGER":
                    item_type = "NUMBER"
                item_kwargs = {"type": item_type}
                if "enum" in v.get("items", {}):
                    item_kwargs["enum"] = v["items"]["enum"]
                gemini_props[k] = types.Schema(
                    type="ARRAY",
                    items=types.Schema(**item_kwargs),
                    description=v.get("description", ""),
                )
            else:
                schema_kwargs = {"type": ptype, "description": v.get("description", "")}
                if "enum" in v:
                    schema_kwargs["enum"] = v["enum"]
                gemini_props[k] = types.Schema(**schema_kwargs)
        func_declarations.append(types.FunctionDeclaration(
            name=fn["name"],
            description=fn["description"],
            parameters=types.Schema(
                type="OBJECT",
                properties=gemini_props,
                required=params.get("required", []),
            ) if gemini_props else None,
        ))

    tools_config = [types.Tool(function_declarations=func_declarations)]

    config = types.GenerateContentConfig(
        system_instruction=_AGENT_SYSTEM,
        tools=tools_config,
        temperature=0.3,
    )

    contents = [types.Content(
        role="user",
        parts=[types.Part(text=question + _AGENT_STRATEGY)],
    )]

    for i in range(max_iterations):
        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=config,
        )

        text_parts = []
        function_calls = []
        if response.candidates and response.candidates[0].content:
            for part in response.candidates[0].content.parts or []:
                if hasattr(part, "thought") and part.thought:
                    continue
                if hasattr(part, "text") and part.text:
                    text_parts.append(part.text)
                if hasattr(part, "function_call") and part.function_call:
                    function_calls.append(part.function_call)

        contents.append(response.candidates[0].content)

        if not function_calls:
            return " ".join(text_parts) or ""

        tools_used = []
        errors = []
        fc_response_parts = []
        for fc in function_calls:
            name = fc.name
            args = dict(fc.args) if fc.args else {}
            try:
                result = tools_map[name](**args)
                tools_used.append(name)
                # Emit frames event for frontend
                if name in ("extract_frames", "inspect_segment") and isinstance(result, (list, dict)):
                    _emit_frames(emit, name, result)
                content = json.dumps(result, default=str) if not isinstance(result, str) else result
                if len(content) > 8000:
                    content = content[:8000] + "\n... (truncated)"
            except Exception as exc:
                content = f"Error: {exc}"
                errors.append(str(exc)[:200])
            fc_response_parts.append(types.Part(function_response=types.FunctionResponse(
                name=name,
                response={"result": content},
            )))

        contents.append(types.Content(role="user", parts=fc_response_parts))
        emit({"type": "iteration", "n": i + 1, "tools": tools_used, "errors": errors})

    contents.append(types.Content(
        role="user",
        parts=[types.Part(text="Please provide your final answer now.")],
    ))
    response = client.models.generate_content(model=model, contents=contents, config=config)
    try:
        return response.text
    except (ValueError, AttributeError):
        return ""


def _emit_frames(emit, tool_name, result):
    """Emit frame data as SSE events for frontend display."""
    frames = []
    if tool_name == "inspect_segment" and isinstance(result, dict):
        raw_frames = result.get("frames", [])
    elif isinstance(result, list):
        raw_frames = result
    else:
        return

    for f in raw_frames:
        if isinstance(f, dict) and "data" in f:
            frames.append({
                "data": f["data"][:200] + "...",  # truncated preview
                "mime_type": f.get("mime_type", "image/jpeg"),
                "timestamp": f.get("timestamp"),
            })
    if frames:
        emit({"type": "frames", "tool": tool_name, "count": len(raw_frames),
              "timestamps": [f.get("timestamp") for f in raw_frames if isinstance(f, dict)]})


def _run_kuavi_agent(
    question: str,
    model: str,
    api_key: str,
    backend: str,
    tools_map: dict,
    emit,
    max_iterations: int = 12,
) -> str:
    """Tool-calling agent loop. Routes to Gemini or OpenAI-compatible backend."""

    if backend == "gemini" or (backend != "openrouter" and "gemini" in model.lower()):
        return _run_kuavi_agent_gemini(
            question=question, model=model, api_key=api_key,
            tools_map=tools_map, emit=emit, max_iterations=max_iterations,
        )

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(f"openai package required for KUAVi agent: {exc}") from exc

    if backend == "openrouter":
        client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
    elif backend == "anthropic":
        or_key = os.getenv("OPENROUTER_API_KEY", "")
        if or_key:
            client = OpenAI(api_key=or_key, base_url="https://openrouter.ai/api/v1")
            if not model.startswith("anthropic/"):
                model = f"anthropic/{model}"
        else:
            client = OpenAI(api_key=api_key, base_url="https://api.anthropic.com/v1")
    else:
        client = OpenAI(api_key=api_key)

    messages: list[dict] = [
        {"role": "system", "content": _AGENT_SYSTEM},
        {"role": "user", "content": question + _AGENT_STRATEGY},
    ]

    for i in range(max_iterations):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=_TOOL_SCHEMAS,
            tool_choice="auto",
            max_tokens=4000,
        )
        msg = response.choices[0].message

        assistant_msg: dict = {"role": "assistant", "content": msg.content or ""}
        if msg.tool_calls:
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in msg.tool_calls
            ]
        messages.append(assistant_msg)

        if not msg.tool_calls:
            return msg.content or ""

        tools_used: list[str] = []
        errors: list[str] = []
        for tc in msg.tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments)
                result = tools_map[name](**args)
                tools_used.append(name)
                # Emit frames event for frontend
                if name in ("extract_frames", "inspect_segment") and isinstance(result, (list, dict)):
                    _emit_frames(emit, name, result)
                content = json.dumps(result, default=str) if not isinstance(result, str) else result
                if len(content) > 8000:
                    content = content[:8000] + "\n... (truncated)"
            except Exception as exc:
                content = f"Error: {exc}"
                errors.append(str(exc)[:200])
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": content})

        emit({"type": "iteration", "n": i + 1, "tools": tools_used, "errors": errors})

    messages.append({"role": "user", "content": "Please provide your final answer now."})
    response = client.chat.completions.create(model=model, messages=messages, max_tokens=2000)
    return response.choices[0].message.content or ""


def _full_pipeline(
    video_path: str,
    question: str,
    model: str,
    api_key: str,
    backend: str,
    emit,
) -> None:
    from rlm.video.video_rlm import VideoRLM

    completed: set[str] = set()

    def emit_step(step_id: str, status: str, detail: str | None = None) -> None:
        event: dict = {"type": "step", "id": step_id, "status": status}
        if detail:
            event["detail"] = detail
        if status in ("done", "cached", "skip"):
            completed.add(step_id)
        emit(event)

    use_gemini = _use_gemini_captioning(backend, model)

    if use_gemini:
        client_backend = "gemini"
        bkw: dict = {
            "model_name": model,
            "api_key": api_key or os.getenv("GEMINI_API_KEY"),
            "thinking_level": "LOW",
        }
    else:
        client_backend = "openai" if backend == "openrouter" else backend
        bkw = {"model_name": model, "api_key": api_key}
        if backend == "openrouter":
            bkw["base_url"] = "https://openrouter.ai/api/v1"

    caption_fn = None
    frame_caption_fn = None

    gemini_key = os.getenv("GEMINI_API_KEY") or (api_key if use_gemini else None)
    if gemini_key:
        try:
            from kuavi.captioning import (
                make_gemini_caption_fn,
                make_gemini_frame_caption_fn,
            )
            caption_model_name = "gemini-3.1-pro-preview"
            caption_fn = make_gemini_caption_fn(model=caption_model_name, api_key=gemini_key)
            frame_caption_fn = make_gemini_frame_caption_fn(model=caption_model_name, api_key=gemini_key)
            emit_step("caption", "pending", f"using {caption_model_name}")
        except ImportError:
            gemini_key = None

    if not gemini_key:
        try:
            from rlm.clients.openai import OpenAIClient
            caption_model_name = "openai/gpt-4o-mini" if backend == "openrouter" else model
            caption_lm = OpenAIClient(
                model_name=caption_model_name,
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

            emit_step("caption", "pending", f"using {caption_model_name}")
        except ImportError:
            emit_step("caption", "skip", "no captioning available")

    log_handler = _QueueLogHandler(emit, completed)
    log_handler.setLevel(logging.INFO)
    indexer_logger = logging.getLogger("rlm.video.video_indexer")
    indexer_logger.setLevel(logging.INFO)
    kuavi_logger = logging.getLogger("kuavi.indexer")
    kuavi_logger.setLevel(logging.INFO)
    indexer_logger.addHandler(log_handler)
    kuavi_logger.addHandler(log_handler)

    try:
        emit_step("vjepa", "running", "loading video...")
        rlm_logger = _EventRLMLogger(emit)
        video_rlm = VideoRLM(
            backend=client_backend,
            backend_kwargs=bkw,
            enable_search=True,
            scene_model=SCENE_MODEL,
            embedding_model=VISUAL_EMBED_MODEL,
            text_embedding_model=TEXT_EMBED_MODEL,
            asr_model="Qwen/Qwen3-ASR-0.6B",
            caption_fn=caption_fn,
            frame_caption_fn=frame_caption_fn,
            refine_fn=None,  # Disabled to speed up Stage 3 flow
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
            "4. For fine-grained detail, use crop_frame(frame_index, x1, y1, x2, y2) or diff_frames.\n"
            "5. Use search_transcript(keyword) for any spoken/audio clues.\n"
            "6. Cite moments with [TS: X.X] (seconds) right after each claim."
        )
        result = video_rlm.completion(video_path, prompt=augmented)
        emit_step("caption", "done")
        emit_step("agent", "done")
        timestamps = _parse_timestamps(result.response)
        answer_html = _render_answer_html(result.response)
        emit({
            "type": "result",
            "answer": result.response,
            "answer_html": answer_html,
            "timestamps": timestamps,
        })
    except Exception as exc:
        short = str(exc)[:200]
        _mark_pending_as_error(PIPELINE_STEPS, completed, emit, short)
        emit({"type": "error", "message": str(exc)})
    finally:
        indexer_logger.removeHandler(log_handler)
        kuavi_logger.removeHandler(log_handler)


@app.post("/api/analyze")
async def analyze(
    video: UploadFile = File(...),  # noqa: B008
    question: str = Form(...),
    backend: str = Form(default="openrouter"),
    model: str = Form(default="openai/gpt-4o"),
    pipeline: str = Form(default="rlm"),
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
        if pipeline == "kuavi":
            _kuavi_pipeline(str(video_path), question, model, api_key, backend, emit)
        else:
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
