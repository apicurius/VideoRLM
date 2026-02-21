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

load_dotenv(Path(__file__).parent / ".env")

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
    {"id": "whisper", "label": "FastWhisper ASR"},
    {"id": "caption", "label": "Segment Captioning"},
    {"id": "gemma",   "label": "Gemma Text Embeddings"},
    {"id": "siglip",  "label": "SigLIP2 Visual Embeddings"},
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
        elif "Gemini caption" in msg or "caption_fn" in msg:
            if "failed" in msg:
                self._emit({"type": "step", "id": "caption", "status": "running", "detail": "retrying..."})
            else:
                self._emit({"type": "step", "id": "caption", "status": "running", "detail": msg.split("] ")[-1] if "] " in msg else msg})
        elif "Re-captioned segment" in msg:
            self._emit({"type": "step", "id": "caption", "status": "running", "detail": msg})
        elif "[pipeline] search index:" in msg:
            self._emit({"type": "step", "id": "index", "status": "done", "detail": msg.split("[pipeline] ")[-1]})
        elif "Returning in-memory cached index" in msg or "Loading cached index" in msg:
            for sid in ["vjepa", "whisper", "caption", "gemma", "siglip"]:
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


def _use_gemini_captioning(backend: str, model: str) -> bool:
    """Check if we should use Gemini captioning based on backend/model."""
    return backend == "gemini" or "gemini" in model.lower()


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
        from kuavi.loader import VideoLoader
        from kuavi.indexer import VideoIndexer
        from kuavi.context import make_extract_frames
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

    try:
        emit({"type": "step", "id": "vjepa", "status": "running", "detail": "loading video..."})
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
                caption_model = "gemini-2.5-flash"
                caption_fn = make_gemini_caption_fn(model=caption_model, api_key=gemini_key)
                frame_caption_fn = make_gemini_frame_caption_fn(model=caption_model, api_key=gemini_key)
                refine_fn = make_gemini_refine_fn(model=caption_model, api_key=gemini_key)
                emit({"type": "step", "id": "caption", "status": "running",
                      "detail": f"using {caption_model}"})
            except ImportError:
                gemini_key = None  # fall through to OpenAI fallback

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

                emit({"type": "step", "id": "caption", "status": "running",
                      "detail": f"using {cap_model}"})
            except ImportError:
                emit({"type": "step", "id": "caption", "status": "skip",
                      "detail": "no captioning available"})

        indexer = VideoIndexer(
            embedding_model=VISUAL_EMBED_MODEL,
            text_embedding_model=TEXT_EMBED_MODEL,
            scene_model=SCENE_MODEL,
        )
        index = indexer.index_video(
            loaded,
            whisper_model="base",
            caption_fn=caption_fn,
            frame_caption_fn=frame_caption_fn,
            refine_fn=refine_fn,
        )

        n_scenes = len(index.scene_boundaries)
        n_segs = len(index.segments)

        # Emit done in actual execution order: vjepa → whisper → caption → gemma → siglip → index
        emit({"type": "step", "id": "vjepa", "status": "done", "detail": f"{n_scenes} scene boundaries detected"})

        if index.transcript:
            emit({"type": "step", "id": "whisper", "status": "done",
                  "detail": f"{len(index.transcript)} transcript entries"})
        else:
            emit({"type": "step", "id": "whisper", "status": "skip", "detail": "no transcript"})

        if use_gemini and caption_fn is not None:
            captioned = sum(1 for s in index.segments if s.get("caption"))
            emit({"type": "step", "id": "caption", "status": "done",
                  "detail": f"{captioned}/{n_segs} segments captioned"})

        emit({"type": "step", "id": "gemma", "status": "done", "detail": "text embeddings ready"})
        emit({"type": "step", "id": "siglip", "status": "done", "detail": f"{n_segs} segments embedded"})
        emit({"type": "step", "id": "index", "status": "done",
              "detail": f"{n_segs} segments, {n_scenes} scenes"})

        # Build tools map with all available tools
        extract_frames = make_extract_frames(video_path)
        tools_map = {
            "get_scene_list": make_get_scene_list(index)["tool"],
            "search_video": make_search_video(index)["tool"],
            "search_transcript": make_search_transcript(index)["tool"],
            "get_transcript": make_get_transcript(index)["tool"],
            "discriminative_vqa": make_discriminative_vqa(index)["tool"],
            "extract_frames": extract_frames,
        }

        emit({"type": "step", "id": "agent", "status": "running"})
        answer = _run_kuavi_agent(
            question=question,
            model=model,
            api_key=api_key,
            backend=backend,
            tools_map=tools_map,
            emit=emit,
        )

        emit({"type": "step", "id": "agent", "status": "done"})
        timestamps = _parse_timestamps(answer)
        answer_html = _render_answer_html(answer)
        emit({
            "type": "result",
            "answer": answer,
            "answer_html": answer_html,
            "timestamps": timestamps,
        })
    except Exception as exc:
        emit({"type": "error", "message": str(exc)})


_TOOL_SCHEMAS = [
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
            "description": "Semantic search over video segments. Use field='summary' for caption-based search, 'visual' for frame-based, 'action' for activity-based, 'all' for combined.",
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
            "description": "Extract video frames as base64 images from a time range. Returns list of image dicts with data, mime_type, and timestamp.",
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
            "description": "Embedding-based multiple-choice VQA. Ranks candidate answers by similarity to video segments. No LLM generation needed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "candidates": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of candidate answers to rank",
                    },
                },
                "required": ["question", "candidates"],
            },
        },
    },
]

_AGENT_SYSTEM = (
    "You are a video analysis assistant with access to a searchable video index.\n"
    "Use the tools to find relevant content, then answer the question.\n"
    "Always cite timestamps as [TS: X.X] (seconds) right after each factual claim.\n\n"
    "Available tools:\n"
    "- get_scene_list(): Get all scenes with timestamps and captions\n"
    "- search_video(query, field, top_k): Semantic search (fields: summary, action, visual, all)\n"
    "- search_transcript(query): Keyword search over spoken words\n"
    "- get_transcript(start_time, end_time): Get transcript for a time range\n"
    "- extract_frames(start_time, end_time, fps, max_frames): Get video frames as images\n"
    "- discriminative_vqa(question, candidates): Multiple-choice VQA via embeddings\n"
)

_AGENT_STRATEGY = (
    "\n\nANALYSIS STRATEGY (follow this order):\n"
    "1. Call get_scene_list() to see all scenes and their timestamps.\n"
    "2. Use search_video(query, field='summary') for relevant scenes.\n"
    "3. Use search_video(query, field='visual') for visual details.\n"
    "4. Use search_transcript(keyword) for spoken clues.\n"
    "5. Use extract_frames(start, end) to inspect promising scenes visually.\n"
    "6. Use discriminative_vqa(question, candidates) for multiple-choice questions.\n"
    "7. Cite every fact with [TS: X.X]."
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
            if ptype == "ARRAY":
                gemini_props[k] = types.Schema(
                    type="ARRAY",
                    items=types.Schema(type="STRING"),
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

        # Collect text and function calls from response
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

        # Append model response to history
        contents.append(response.candidates[0].content)

        if not function_calls:
            return " ".join(text_parts) or ""

        # Execute function calls and build response parts
        tools_used = []
        errors = []
        fc_response_parts = []
        for fc in function_calls:
            name = fc.name
            args = dict(fc.args) if fc.args else {}
            try:
                result = tools_map[name](**args)
                tools_used.append(name)
                content = json.dumps(result, default=str) if not isinstance(result, str) else result
                # Truncate large results to avoid token overflow
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

    # Force final answer
    contents.append(types.Content(
        role="user",
        parts=[types.Part(text="Please provide your final answer now.")],
    ))
    response = client.models.generate_content(model=model, contents=contents, config=config)
    try:
        return response.text
    except (ValueError, AttributeError):
        return ""


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

    # Route to native Gemini function calling
    if backend == "gemini" or (backend != "openrouter" and "gemini" in model.lower()):
        return _run_kuavi_agent_gemini(
            question=question, model=model, api_key=api_key,
            tools_map=tools_map, emit=emit, max_iterations=max_iterations,
        )

    # OpenAI-compatible path
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
                content = json.dumps(result, default=str) if not isinstance(result, str) else result
                if len(content) > 8000:
                    content = content[:8000] + "\n... (truncated)"
            except Exception as exc:
                content = f"Error: {exc}"
                errors.append(str(exc)[:200])
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": content})

        emit({"type": "iteration", "n": i + 1, "tools": tools_used, "errors": errors})

    # Force final answer after max iterations
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

    use_gemini = _use_gemini_captioning(backend, model)

    # Backend routing
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

    # Captioning: always prefer Gemini Flash (cheap/fast), fall back to OpenAI
    caption_fn = None
    frame_caption_fn = None
    refine_fn = None

    gemini_key = os.getenv("GEMINI_API_KEY") or (api_key if use_gemini else None)
    if gemini_key:
        try:
            from kuavi.captioning import (
                make_gemini_caption_fn,
                make_gemini_frame_caption_fn,
                make_gemini_refine_fn,
            )
            caption_model_name = "gemini-2.5-flash"
            caption_fn = make_gemini_caption_fn(model=caption_model_name, api_key=gemini_key)
            frame_caption_fn = make_gemini_frame_caption_fn(model=caption_model_name, api_key=gemini_key)
            refine_fn = make_gemini_refine_fn(model=caption_model_name, api_key=gemini_key)
            emit({"type": "step", "id": "caption", "status": "running",
                  "detail": f"using {caption_model_name}"})
        except ImportError:
            gemini_key = None  # fall through to OpenAI fallback

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

            emit({"type": "step", "id": "caption", "status": "running",
                  "detail": f"using {caption_model_name}"})
        except ImportError:
            emit({"type": "step", "id": "caption", "status": "skip",
                  "detail": "no captioning available"})

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
            frame_caption_fn=frame_caption_fn,
            refine_fn=refine_fn,
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
        emit({"type": "step", "id": "caption", "status": "done"})
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
    pipeline: str = Form(default="rlm"),
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
