"""Standalone agent loop for video question-answering.

Extracts the tool-calling agent from ``web_app.py`` into a reusable module
that can be consumed from the CLI, the web backend, or any other driver.

Usage (iterator)::

    from kuavi.agent_runner import run_agent

    for event in run_agent(video_path, question, model=..., api_key=..., backend=...):
        if event["type"] == "result":
            print(event["answer"])

Usage (single-shot)::

    from kuavi.agent_runner import run_agent_sync

    answer = run_agent_sync(video_path, question, model=..., api_key=..., backend=...)
"""

from __future__ import annotations

import base64
import concurrent.futures
import json
import logging
import os
from collections.abc import Iterator
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants shared with web_app.py — kept in sync via single source
# ---------------------------------------------------------------------------

VISUAL_EMBED_MODEL = "google/siglip2-base-patch16-256"
TEXT_EMBED_MODEL = "google/embeddinggemma-300m"
SCENE_MODEL = "facebook/vjepa2-vitl-fpc64-256"

# ---------------------------------------------------------------------------
# Tool schemas (OpenAI function-calling format)
# ---------------------------------------------------------------------------

TOOL_SCHEMAS: list[dict] = [
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
            "description": "Multi-field semantic search + transcript search in parallel.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "fields": {
                        "type": "array",
                        "items": {"type": "string", "enum": ["summary", "action", "visual", "all"]},
                        "description": "Search fields (default: visual, temporal)",
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
            "description": "Extract frames + get transcript for a time range in one call.",
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
            "description": "Semantic search over video segments.",
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
                    "x1_pct": {"type": "number"}, "y1_pct": {"type": "number"},
                    "x2_pct": {"type": "number"}, "y2_pct": {"type": "number"},
                },
                "required": ["image", "x1_pct", "y1_pct", "x2_pct", "y2_pct"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "diff_frames",
            "description": "Compute absolute pixel difference between two frames.",
            "parameters": {
                "type": "object",
                "properties": {
                    "image_a": {"type": "integer"}, "image_b": {"type": "integer"},
                },
                "required": ["image_a", "image_b"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "blend_frames",
            "description": "Average multiple frames into a composite image.",
            "parameters": {
                "type": "object",
                "properties": {
                    "images": {"type": "array", "items": {"type": "integer"}},
                },
                "required": ["images"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "threshold_frame",
            "description": "Apply binary threshold + contour detection.",
            "parameters": {
                "type": "object",
                "properties": {
                    "image": {"type": "integer"},
                    "value": {"type": "integer", "default": 128},
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
                "properties": {"image": {"type": "integer"}},
                "required": ["image"],
            },
        },
    },
]

AGENT_SYSTEM = (
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

AGENT_STRATEGY = (
    "\n\nANALYSIS STRATEGY (follow this order for efficiency):\n"
    "1. Call orient() to see video structure, scenes, and timestamps.\n"
    "2. Use search_all(query, fields=['visual', 'temporal']) for broad search.\n"
    "3. Use inspect_segment(start, end) to get frames + transcript for top hits.\n"
    "4. For fine-grained detail, use crop_frame, diff_frames, or frame_info.\n"
    "5. Use discriminative_vqa(question, candidates) for multiple-choice questions.\n"
    "6. Cite every fact with [TS: X.X].\n\n"
    "IMPORTANT: Prefer compound tools (orient, search_all, inspect_segment) over\n"
    "individual calls — they batch multiple operations into single calls."
)


# ---------------------------------------------------------------------------
# Pixel tools builder
# ---------------------------------------------------------------------------

def make_pixel_tools(extract_frames_fn: Any) -> dict[str, Any]:
    """Build pixel analysis tools that operate on extracted frame results.

    Returns a dict of ``tool_name -> callable`` sharing an internal frame
    cache so pixel tools can reference frames by index.
    """
    _frame_cache: list[dict] = []
    _orig_extract = extract_frames_fn

    def extract_frames_cached(**kwargs: Any) -> Any:
        result = _orig_extract(**kwargs)
        _frame_cache.clear()
        if isinstance(result, list):
            _frame_cache.extend(result)
        return result

    def _resolve(image: Any) -> dict:
        if isinstance(image, (int, float)):
            idx = int(image)
            if 0 <= idx < len(_frame_cache):
                return _frame_cache[idx]
            return {"error": f"Frame index {idx} out of range (0-{len(_frame_cache) - 1})"}
        return image

    def _decode(image: dict) -> Any:
        import cv2
        import numpy as np
        raw = base64.b64decode(image.get("data", ""))
        arr = np.frombuffer(raw, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)

    def _encode(frame: Any) -> dict:
        import cv2
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return {"data": base64.b64encode(buf.tobytes()).decode(), "mime_type": "image/jpeg"}

    def crop_frame(image: Any, x1_pct: float, y1_pct: float, x2_pct: float, y2_pct: float) -> dict:
        image = _resolve(image)
        if "error" in image:
            return image
        frame = _decode(image)
        h, w = frame.shape[:2]
        cropped = frame[int(y1_pct * h):int(y2_pct * h), int(x1_pct * w):int(x2_pct * w)]
        return {"image": _encode(cropped), "crop": {"x1_pct": x1_pct, "y1_pct": y1_pct, "x2_pct": x2_pct, "y2_pct": y2_pct, "width": cropped.shape[1], "height": cropped.shape[0]}}

    def diff_frames(image_a: Any, image_b: Any) -> dict:
        import cv2
        a, b = _resolve(image_a), _resolve(image_b)
        if isinstance(a, dict) and "error" in a:
            return a
        if isinstance(b, dict) and "error" in b:
            return b
        fa, fb = _decode(a), _decode(b)
        if fa.shape != fb.shape:
            fb = cv2.resize(fb, (fa.shape[1], fa.shape[0]))
        diff = cv2.absdiff(fa, fb)
        changed = (diff > 25).any(axis=2) if diff.ndim == 3 else (diff > 25)
        return {"image": _encode(diff), "mean_diff": round(float(diff.mean()), 2), "max_diff": int(diff.max()), "changed_pct": round(float(changed.sum() / changed.size * 100), 2)}

    def blend_frames(images: list) -> dict:
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

    def threshold_frame(image: Any, value: int = 128, invert: bool = False) -> dict:
        import cv2
        image = _resolve(image)
        if isinstance(image, dict) and "error" in image:
            return image
        frame = _decode(image)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        thresh_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
        _, mask = cv2.threshold(gray, value, 255, thresh_type)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        return {"image": _encode(mask_bgr), "white_pct": round(float((mask == 255).sum() / mask.size * 100), 2), "contour_count": len(contours), "contour_areas": sorted([float(cv2.contourArea(c)) for c in contours], reverse=True)[:20]}

    def frame_info(image: Any) -> dict:
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
        return {"width": w, "height": h, "channels": channels, "brightness": {"mean": round(float(gray.mean()), 2), "std": round(float(gray.std()), 2), "min": int(gray.min()), "max": int(gray.max())}, "color": {"b_mean": round(b_mean, 2), "g_mean": round(g_mean, 2), "r_mean": round(r_mean, 2)}}

    return {
        "extract_frames": extract_frames_cached,
        "crop_frame": crop_frame,
        "diff_frames": diff_frames,
        "blend_frames": blend_frames,
        "threshold_frame": threshold_frame,
        "frame_info": frame_info,
    }


# ---------------------------------------------------------------------------
# Compound tools builder
# ---------------------------------------------------------------------------

def make_compound_tools(index: Any, tools_map: dict[str, Any]) -> dict[str, Any]:
    """Build compound tools that combine multiple basic tool calls."""

    def orient() -> dict:
        info = {
            "segments": len(index.segments),
            "duration": index.segments[-1]["end_time"] if index.segments else 0,
            "scene_boundaries": len(index.scene_boundaries),
            "has_transcript": bool(index.transcript),
            "transcript_entries": len(index.transcript) if index.transcript else 0,
        }
        scenes = tools_map["get_scene_list"]()
        return {"index_info": info, "scenes": scenes}

    def search_all(query: str, fields: list[str] | None = None, top_k: int = 5, transcript_query: str | None = None) -> dict:
        if fields is None:
            fields = ["visual", "temporal"]
        results: dict = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures: dict = {}
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

    def inspect_segment(start_time: float, end_time: float, fps: float = 2.0, max_frames: int = 6) -> dict:
        frames = tools_map["extract_frames"](start_time=start_time, end_time=end_time, fps=fps, max_frames=max_frames)
        transcript = tools_map["get_transcript"](start_time=start_time, end_time=end_time)
        return {"frames": frames, "transcript": transcript}

    return {"orient": orient, "search_all": search_all, "inspect_segment": inspect_segment}


# ---------------------------------------------------------------------------
# Agent backends
# ---------------------------------------------------------------------------

def _agent_gemini(
    question: str,
    model: str,
    api_key: str,
    tools_map: dict[str, Any],
    max_iterations: int = 12,
) -> Iterator[dict]:
    """Tool-calling agent loop using native Gemini function calling.

    Yields event dicts: ``{"type": "iteration", ...}``, ``{"type": "answer", ...}``.
    """
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key or os.getenv("GEMINI_API_KEY"))

    func_declarations = []
    for schema in TOOL_SCHEMAS:
        fn = schema["function"]
        params = fn.get("parameters", {})
        props = params.get("properties", {})
        gemini_props: dict = {}
        for k, v in props.items():
            ptype = v.get("type", "string").upper()
            if ptype == "INTEGER":
                ptype = "NUMBER"
            if ptype == "ARRAY":
                item_type = v.get("items", {}).get("type", "STRING").upper()
                if item_type == "INTEGER":
                    item_type = "NUMBER"
                item_kwargs: dict = {"type": item_type}
                if "enum" in v.get("items", {}):
                    item_kwargs["enum"] = v["items"]["enum"]
                gemini_props[k] = types.Schema(type="ARRAY", items=types.Schema(**item_kwargs), description=v.get("description", ""))
            else:
                schema_kwargs: dict = {"type": ptype, "description": v.get("description", "")}
                if "enum" in v:
                    schema_kwargs["enum"] = v["enum"]
                gemini_props[k] = types.Schema(**schema_kwargs)
        func_declarations.append(types.FunctionDeclaration(
            name=fn["name"],
            description=fn["description"],
            parameters=types.Schema(type="OBJECT", properties=gemini_props, required=params.get("required", [])) if gemini_props else None,
        ))

    tools_config = [types.Tool(function_declarations=func_declarations)]
    config = types.GenerateContentConfig(system_instruction=AGENT_SYSTEM, tools=tools_config, temperature=0.3)
    contents = [types.Content(role="user", parts=[types.Part(text=question + AGENT_STRATEGY)])]

    for i in range(max_iterations):
        response = client.models.generate_content(model=model, contents=contents, config=config)

        text_parts: list[str] = []
        function_calls: list = []
        if not response.candidates:
            yield {"type": "answer", "text": " ".join(text_parts) or "(No response from model)"}
            return
        candidate = response.candidates[0]
        if candidate.content and candidate.content.parts:
            for part in candidate.content.parts:
                if hasattr(part, "thought") and part.thought:
                    continue
                if hasattr(part, "text") and part.text:
                    text_parts.append(part.text)
                if hasattr(part, "function_call") and part.function_call:
                    function_calls.append(part.function_call)
            contents.append(candidate.content)
        else:
            yield {"type": "answer", "text": " ".join(text_parts) or "(Model returned empty response)"}
            return

        if not function_calls:
            yield {"type": "answer", "text": " ".join(text_parts) or ""}
            return

        tools_used: list[str] = []
        errors: list[str] = []
        fc_response_parts = []
        for fc in function_calls:
            name = fc.name
            args = dict(fc.args) if fc.args else {}
            try:
                result = tools_map[name](**args)
                tools_used.append(name)
                content = json.dumps(result, default=str) if not isinstance(result, str) else result
                if len(content) > 8000:
                    content = content[:8000] + "\n... (truncated)"
            except Exception as exc:
                content = f"Error: {exc}"
                errors.append(str(exc)[:200])
            fc_response_parts.append(types.Part(function_response=types.FunctionResponse(name=name, response={"result": content})))

        contents.append(types.Content(role="user", parts=fc_response_parts))
        yield {"type": "iteration", "n": i + 1, "tools": tools_used, "errors": errors}

    # Force final answer
    contents.append(types.Content(role="user", parts=[types.Part(text="Please provide your final answer now.")]))
    response = client.models.generate_content(model=model, contents=contents, config=config)
    try:
        yield {"type": "answer", "text": response.text}
    except (ValueError, AttributeError):
        yield {"type": "answer", "text": ""}


def _agent_openai(
    question: str,
    model: str,
    api_key: str,
    backend: str,
    tools_map: dict[str, Any],
    max_iterations: int = 12,
) -> Iterator[dict]:
    """Tool-calling agent loop for OpenAI-compatible backends.

    Yields event dicts: ``{"type": "iteration", ...}``, ``{"type": "answer", ...}``.
    """
    from openai import OpenAI

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
        {"role": "system", "content": AGENT_SYSTEM},
        {"role": "user", "content": question + AGENT_STRATEGY},
    ]

    for i in range(max_iterations):
        response = client.chat.completions.create(model=model, messages=messages, tools=TOOL_SCHEMAS, tool_choice="auto", max_tokens=4000)
        msg = response.choices[0].message

        assistant_msg: dict = {"role": "assistant", "content": msg.content or ""}
        if msg.tool_calls:
            assistant_msg["tool_calls"] = [
                {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in msg.tool_calls
            ]
        messages.append(assistant_msg)

        if not msg.tool_calls:
            yield {"type": "answer", "text": msg.content or ""}
            return

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

        yield {"type": "iteration", "n": i + 1, "tools": tools_used, "errors": errors}

    messages.append({"role": "user", "content": "Please provide your final answer now."})
    response = client.chat.completions.create(model=model, messages=messages, max_tokens=2000)
    yield {"type": "answer", "text": response.choices[0].message.content or ""}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_tools_map(
    index: Any,
    video_path: str,
) -> dict[str, Any]:
    """Build the complete tools map from a :class:`VideoIndex` and a video path.

    Returns a flat dict of ``tool_name -> callable`` ready for the agent loop.
    """
    from kuavi.context import make_extract_frames
    from kuavi.search import (
        make_discriminative_vqa,
        make_get_scene_list,
        make_get_transcript,
        make_search_transcript,
        make_search_video,
    )

    raw_extract = make_extract_frames(video_path)
    basic_tools = {
        "get_scene_list": make_get_scene_list(index)["tool"],
        "search_video": make_search_video(index)["tool"],
        "search_transcript": make_search_transcript(index)["tool"],
        "get_transcript": make_get_transcript(index)["tool"],
        "discriminative_vqa": make_discriminative_vqa(index)["tool"],
    }
    pixel_tools = make_pixel_tools(raw_extract)
    tools_map = {**basic_tools, **pixel_tools}
    compound_tools = make_compound_tools(index, tools_map)
    tools_map.update(compound_tools)
    return tools_map


def run_agent(
    video_path: str,
    question: str,
    *,
    model: str = "gemini-2.5-flash",
    api_key: str | None = None,
    backend: str = "gemini",
    index_mode: str = "fast",
    asr_model: str = "faster-whisper/base",
    max_iterations: int = 12,
    index: Any | None = None,
) -> Iterator[dict]:
    """Run the video-analysis agent, yielding event dicts.

    Event types:
        ``step``  — pipeline progress (indexing, embedding, …)
        ``iteration`` — one agent tool-call round
        ``answer`` — final answer text

    Args:
        video_path: Path to the video file.
        question: User question about the video.
        model: LLM model identifier.
        api_key: API key (falls back to env vars).
        backend: ``"gemini"``, ``"openrouter"``, ``"openai"``, ``"anthropic"``.
        index_mode: ``"fast"`` or ``"captioned"``.
        asr_model: ASR model for transcription.
        max_iterations: Maximum agent loop iterations.
        index: Pre-built :class:`VideoIndex` — skips indexing when provided.
    """
    from kuavi.indexer import VideoIndexer
    from kuavi.loader import VideoLoader

    # --- Load .env if python-dotenv is available ---
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    # --- Resolve API key from env vars if not provided ---
    if not api_key:
        _env_keys = {
            "gemini": "GEMINI_API_KEY",
            "openrouter": "OPENROUTER_API_KEY",
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
        }
        api_key = os.getenv(_env_keys.get(backend, ""), "") or os.getenv("OPENROUTER_API_KEY", "")

    # --- Indexing (skip if caller provides a pre-built index) ---
    if index is None:
        yield {"type": "step", "id": "index", "status": "running"}
        loader = VideoLoader(fps=0.5)
        loaded = loader.load(video_path)

        caption_fn = None
        frame_caption_fn = None
        refine_fn = None
        if index_mode == "captioned":
            gemini_key = api_key or os.getenv("GEMINI_API_KEY")
            if gemini_key:
                try:
                    from kuavi.captioning import (
                        make_gemini_caption_fn,
                        make_gemini_frame_caption_fn,
                        make_gemini_refine_fn,
                    )
                    caption_fn = make_gemini_caption_fn(api_key=gemini_key)
                    frame_caption_fn = make_gemini_frame_caption_fn(api_key=gemini_key)
                    refine_fn = make_gemini_refine_fn(api_key=gemini_key)
                except ImportError:
                    pass

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
        yield {"type": "step", "id": "index", "status": "done", "detail": f"{len(index.segments)} segments"}

    # --- Build tools ---
    tools_map = build_tools_map(index, video_path)

    # --- Agent loop ---
    yield {"type": "step", "id": "agent", "status": "running"}
    is_gemini = backend == "gemini" or (backend != "openrouter" and "gemini" in model.lower())

    if is_gemini:
        agent_iter = _agent_gemini(question=question, model=model, api_key=api_key or "", tools_map=tools_map, max_iterations=max_iterations)
    else:
        agent_iter = _agent_openai(question=question, model=model, api_key=api_key or "", backend=backend, tools_map=tools_map, max_iterations=max_iterations)

    answer_text = ""
    for event in agent_iter:
        if event["type"] == "answer":
            answer_text = event.get("text", "")
        yield event

    yield {"type": "step", "id": "agent", "status": "done"}
    yield {"type": "result", "answer": answer_text}


def run_agent_sync(
    video_path: str,
    question: str,
    **kwargs: Any,
) -> str:
    """Convenience wrapper that runs the agent and returns the final answer string."""
    answer = ""
    for event in run_agent(video_path, question, **kwargs):
        if event["type"] == "result":
            answer = event.get("answer", "")
    return answer
