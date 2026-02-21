"""Experiment runner: compare RLM vs KUAVi on the same video analysis task.

Usage:
    uv run python run_experiment.py --video test.mp4 --question "What happens?"
    uv run python run_experiment.py --video test.mp4 --pipeline both --output-dir experiments/
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import time
from datetime import datetime
from pathlib import Path

import uuid

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


class _KUAViTraceLogger:
    """Lightweight JSONL trace logger for KUAVi agent loop (matches visualizer event format)."""

    def __init__(self, log_dir: str = "./logs"):
        os.makedirs(log_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_id = str(uuid.uuid4())[:8]
        self._log_stem = f"kuavi_{ts}_{run_id}"
        self.log_file = os.path.join(log_dir, f"{self._log_stem}.jsonl")
        self._frames_dir = os.path.join(log_dir, f"{self._log_stem}.frames")
        self._frame_counter = 0
        self._last_tool_time: float | None = None
        self._turn = 0

    def _emit(self, event: dict) -> None:
        event.setdefault("timestamp", datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.000Z"))
        with open(self.log_file, "a") as f:
            json.dump(event, f, default=str)
            f.write("\n")

    def log_session_start(self, model: str | None = None) -> None:
        self._emit({"type": "session_start", "source": "experiment-runner", "model": model})

    def log_tool_call(
        self, tool_name: str, tool_input: dict, tool_response: object,
        duration_ms: float, has_error: bool = False,
    ) -> None:
        now = time.time()
        # Detect turn boundary (>3s gap)
        if self._last_tool_time is not None:
            gap = now - self._last_tool_time
            if gap > 3.0:
                self._turn += 1
                self._emit({"type": "turn_start", "turn": self._turn, "gap_seconds": round(gap, 1)})
        self._last_tool_time = now

        # Save frame images to sidecar directory, replace base64 with file references
        resp_summary = tool_response
        if isinstance(tool_response, list) and len(tool_response) > 0:
            if isinstance(tool_response[0], dict) and "data" in tool_response[0]:
                resp_summary = self._save_frames(tool_response)
        elif isinstance(tool_response, dict) and "data" in tool_response:
            resp_summary = self._save_frames([tool_response])
            if len(resp_summary) == 1:
                resp_summary = resp_summary[0]

        self._emit({
            "type": "tool_call",
            "tool_name": f"mcp__kuavi__kuavi_{tool_name}",
            "tool_input": tool_input,
            "tool_response": resp_summary,
            "duration_ms": round(duration_ms),
            "has_error": has_error,
        })

    def _save_frames(self, frames: list[dict]) -> list[dict]:
        """Save base64 frame images to sidecar directory, return list of references."""
        os.makedirs(self._frames_dir, exist_ok=True)
        refs = []
        for frame in frames:
            data = frame.get("data", "")
            mime = frame.get("mime_type", "image/jpeg")
            ext = "png" if "png" in mime else "jpg"
            fname = f"frame_{self._frame_counter:04d}.{ext}"
            self._frame_counter += 1
            # Write binary image file
            try:
                with open(os.path.join(self._frames_dir, fname), "wb") as fp:
                    fp.write(base64.b64decode(data))
            except Exception:
                pass
            # Build reference (keep all metadata except raw data)
            ref = {k: v for k, v in frame.items() if k != "data"}
            ref["_frame_file"] = fname
            refs.append(ref)
        return refs

    def log_metadata(self, video_path: str, duration: float, fps: float,
                     num_segments: int, num_scenes: int) -> None:
        self._emit({
            "type": "metadata",
            "video_path": video_path,
            "duration": duration,
            "fps": fps,
            "num_segments": num_segments,
            "num_scenes": num_scenes,
            "has_embeddings": True,
            "has_transcript": True,
        })

    def log_reasoning(self, iteration: int, text: str, token_usage: dict | None = None) -> None:
        """Log the model's text response (reasoning between tool calls)."""
        event: dict = {"type": "reasoning", "iteration": iteration, "text": text}
        if token_usage:
            event["token_usage"] = token_usage
        self._emit(event)

    def log_system_prompt(self, prompt: str) -> None:
        """Log the system prompt for display in visualizer."""
        self._emit({"type": "system_prompt", "text": prompt})

    def log_question(self, question: str) -> None:
        self._emit({"type": "question", "text": question})

    def log_final_answer(self, text: str) -> None:
        self._emit({"type": "final_answer", "text": text})


DEFAULT_QUESTION = (
    "Respond in English. Provide a comprehensive analysis of this video. "
    "First, search for all distinct scenes and topics covered. Then zoom into "
    "each key section to identify: (1) the main concepts being presented, "
    "(2) any diagrams, text, or visual aids shown on screen, (3) the speaker's "
    "key arguments and examples, and (4) how the sections connect to form the "
    "overall narrative. Finally, summarize the video's thesis and the evidence "
    "used to support it."
)


# ---------------------------------------------------------------------------
# RLM pipeline
# ---------------------------------------------------------------------------

def run_rlm(
    video_path: str,
    question: str,
    model: str,
    backend: str,
    cache_dir: str | None,
    max_iterations: int = 15,
    thinking_level: str = "LOW",
) -> dict:
    """Run VideoRLM pipeline and return result dict."""
    from rlm.logger import RLMLogger
    from rlm.video import VideoRLM

    TOOL_NAMES = [
        "search_video", "search_transcript", "extract_frames",
        "get_scene_list", "get_transcript", "discriminative_vqa",
        "crop_frame", "diff_frames", "blend_frames", "threshold_frame",
        "blend_frames", "frame_info",
    ]

    logger = RLMLogger(log_dir="./logs")

    t0 = time.monotonic()
    vrlm = VideoRLM(
        backend=backend,
        backend_kwargs={"model_name": model, "timeout": 300.0, "thinking_level": thinking_level},
        fps=0.5,
        num_segments=5,
        max_frames_per_segment=3,
        resize=(640, 480),
        max_iterations=max_iterations,
        logger=logger,
        verbose=True,
        enable_search=True,
        embedding_model="google/siglip2-base-patch16-256",
        cache_dir=cache_dir,
        auto_fps=False,
        scene_model="facebook/vjepa2-vitl-fpc64-256",
        text_embedding_model="google/embeddinggemma-300m",
    )
    result = vrlm.completion(video_path, prompt=question)
    elapsed = time.monotonic() - t0

    # Extract tool calls from logged iterations
    all_tools: list[str] = []
    trajectory = logger.get_trajectory()
    iteration_count = logger.iteration_count
    total_tokens = 0
    if trajectory:
        for it in trajectory.get("iterations", []):
            for block in it.get("code_blocks", []):
                code = block.get("code", "")
                for tool in TOOL_NAMES:
                    if tool + "(" in code and tool not in all_tools:
                        all_tools.append(tool)
            # Token usage from sub-calls
            for block in it.get("code_blocks", []):
                for sub in block.get("result", {}).get("rlm_calls", []):
                    usage = sub.get("usage_summary", {})
                    total_tokens += usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)

    # Also get usage from the completion itself
    if hasattr(result, "usage_summary") and result.usage_summary:
        us = result.usage_summary
        total_tokens += getattr(us, "prompt_tokens", 0) + getattr(us, "completion_tokens", 0)

    return {
        "pipeline": "rlm",
        "answer": result.response,
        "iteration_count": iteration_count,
        "tool_calls": all_tools,
        "wall_time_seconds": round(elapsed, 2),
        "approximate_tokens": total_tokens or None,
        "execution_time_seconds": round(result.execution_time, 2),
        "log_file": logger.log_file_path,
    }


# ---------------------------------------------------------------------------
# KUAVi pipeline
# ---------------------------------------------------------------------------

KUAVI_SYSTEM_PROMPT = """\
IMPORTANT: You MUST respond in English only.

You are a video analysis agent with access to tools for searching and inspecting a pre-indexed video. \
The video has already been indexed with scene detection, frame embeddings, captions, and ASR transcript.

## Available Tools

SEARCH TOOLS:
1. **get_scene_list()** — List all detected scene boundaries with structured annotations (summary, action, actor). Call this first to orient yourself.
2. **search_video(query, top_k=5, field="summary")** — Semantic search over indexed segments.
   - field="summary" (default): search visual descriptions
   - field="action": search by action/activity type
   - field="visual": search by frame embeddings — bypasses caption quality, best for finding specific numbers/text/charts on screen
   - field="all": search across all annotation fields
   Returns matches with start_time, end_time, score, caption, and structured annotation.
3. **search_transcript(query)** — Keyword search over ASR transcript. Use for dialogue, narration, names, numbers, quotes.
4. **get_transcript(start_time, end_time)** — Get full spoken transcript for a time range. Use after search hits to get complete dialogue context.
5. **discriminative_vqa(question, candidates)** — Fast embedding-based multiple-choice or yes/no answer selection without LLM generation. Ranks candidates by cosine similarity to video segment embeddings. Use for quick multiple-choice questions before committing to frame extraction.

FRAME EXTRACTION:
6. **extract_frames(start_time, end_time, fps=2.0, width=720, height=540, max_frames=5)** — Extract frames from the video as images you can see directly. \
ESSENTIAL for reading text, numbers, tables, charts, scoreboards, or any fine visual detail that text search cannot capture. \
Use after search to zoom into specific moments and visually inspect them. \
Adjust resolution: use width=1280, height=960 for high-res detail reading.

PIXEL MANIPULATION TOOLS (for code-based visual reasoning):
7. **crop_frame(image, x1_pct, y1_pct, x2_pct, y2_pct)** — Crop a region of interest using percentage coordinates (0.0-1.0). Example: crop_frame(frame, 0.1, 0.05, 0.9, 0.35) crops the top banner. Use to isolate text, faces, tables, or charts before detailed inspection.
8. **diff_frames(image_a, image_b)** — Compute absolute pixel difference between two frames. Bright areas in result = regions that changed. Use for motion/change detection.
9. **blend_frames(images)** — Average multiple frames into a single composite. Use for creating motion summary or detecting static elements across frames.
10. **threshold_frame(image, value=128)** — Convert frame to binary mask. Pixels > value become white, else black. Returns contour_count and contour_areas — use for object counting and segmentation.
11. **frame_info(image)** — Get frame dimensions and brightness/color statistics (width, height, mean_brightness, std_brightness, color_means). Use for detecting dark/bright scenes or color patterns.

## SEARCH-FIRST STRATEGY (always follow this)

1. **ORIENT**: Call get_scene_list() to see all scenes with their annotations.
   Review the time ranges, captions, and action descriptions to understand video structure.

2. **SEARCH**: Decompose your query into targeted searches:
   - "what happens" → search_video(query, field="action")
   - "what does it look like" → search_video(query, field="summary")
   - specific numbers/text/scores visible on screen → search_video(query, field="visual")
   - general queries → search_video(query, field="all")
   - spoken content (names, dialogue) → search_transcript(query)
   - multiple-choice or yes/no questions → discriminative_vqa(question, candidates)

3. **INSPECT**: For the most relevant search hits, extract frames and cross-reference:
   - extract_frames(start_time, end_time) to visually see what's happening
   - get_transcript(start_time, end_time) to hear what was said

4. **CROSS-REFERENCE**: Compare visual evidence with transcript. Use multiple search queries \
with different fields and phrasings to triangulate.

5. **VERIFY**: Before finalizing, verify key claims:
   - For names: search multiple transcript segments and check annotations
   - For numbers: search with field="visual", extract frames, and cross-check with transcript
   - ASR frequently misrecognizes proper names, numbers, and technical terms — \
screen content OVERRIDES transcript content

## CHOOSING YOUR STRATEGY

- For broad questions (summaries, themes, overall narrative): search with field="summary" and field="action", get transcripts, synthesize chronologically.
- For detailed visual questions (reading text, identifying small objects, examining specific moments): use the HIERARCHICAL ZOOM strategy below.
- For specific numbers, scores, or text shown on screen (tables, charts, scoreboards, benchmarks): use the PRECISE VALUE READING strategy below.
- For multiple-choice or yes/no questions: use discriminative_vqa() first for a quick answer, then verify visually.

## PRECISE VALUE READING (for scores, numbers, text on screen)

When the question asks for a specific number, score, or piece of text visible in the video:

Step 1: VISUAL SEARCH — use field="visual" to find frames with tables/charts/text.
Step 2: HIGH-RES EXTRACT — extract frames at high resolution: extract_frames(start, end, fps=4.0, width=1280, height=960, max_frames=10)
Step 3: IDENTIFY — examine the frames to locate the relevant region (top/bottom, left/right, which row/column).
Step 4: CROP — use crop_frame(frame, x1_pct, y1_pct, x2_pct, y2_pct) to isolate the specific region containing the value.
Step 5: CROSS-CHECK — read the same value from at least 2 different frames for confirmation.

## HIERARCHICAL ZOOM (3-pass approach)

When fine visual detail is needed:

Pass 1: COARSE SCAN — extract_frames at default resolution to find the relevant moment.
Pass 2: ZOOM — extract_frames at higher density (fps=2.0) for the relevant time range.
Pass 3: ULTRA-ZOOM — extract_frames at full resolution (width=1280, height=960, fps=4.0) for a narrow window, then crop_frame to isolate the region of interest.

## PIXEL TOOL PATTERNS

Use pixel tools when the question requires counting, measuring, comparing, or detecting changes:

- **Count objects**: extract frames → threshold_frame(frame, value=200) → check contour_count
- **Detect changes**: extract frames from two moments → diff_frames(frame_a, frame_b) → frame_info(diff) to measure change intensity
- **Isolate text/tables**: extract frames → crop_frame(frame, ...) to isolate the relevant area
- **Motion summary**: extract multiple frames → blend_frames(frames) to see static vs moving elements
- **Scene brightness**: extract frames → frame_info(frame) for brightness and color statistics

## Anti-Hallucination Rules

1. NEVER report information not supported by search results, transcript, or visual evidence from frames.
2. NEVER trust transcript numbers as ground truth — cross-reference with visual search and frame extraction.
3. NEVER output a number you haven't visually confirmed by examining an extracted frame.
4. If you cannot visually confirm a value after exhaustive search, say so honestly.
5. Always cite timestamps: [TS: X.X] after each factual claim.
6. When annotations conflict, describe only what is consistently observed.
7. Try ALL of the following before concluding content is absent:
   a. Search with DIFFERENT queries (e.g. "table", "results", "benchmark", "comparison")
   b. Search with DIFFERENT fields (field="visual", field="summary", field="action")
   c. Try ALL top-k hits, not just the first one
   d. Extract at multiple time ranges — content may appear at different points

Think step by step: orient with scenes, search with targeted queries, inspect with frame extraction, cross-reference visual and audio evidence, then synthesize a clear answer.\
"""


def _build_kuavi_tools_schema() -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": "get_scene_list",
                "description": "List all detected scenes with start/end times, captions, and structured annotations (summary, action, actor). Call this first to orient yourself.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "search_video",
                "description": "Semantic search over indexed video segments. Use field='summary' for visual descriptions, 'action' for activities, 'visual' for frame embeddings (best for finding numbers/text/charts), 'all' for everything.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                        "top_k": {"type": "integer", "default": 5, "description": "Number of results"},
                        "field": {
                            "type": "string",
                            "enum": ["summary", "action", "visual", "all"],
                            "default": "summary",
                            "description": "Which field to search: summary, action, visual, or all",
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
                "description": "Keyword search over the ASR transcript. Use for spoken content: names, dialogue, narration, numbers mentioned aloud.",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string", "description": "Keyword to search for"}},
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_transcript",
                "description": "Get the full spoken transcript text for a specific time range (in seconds). Use after finding relevant segments to get complete dialogue context.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "start_time": {"type": "number", "description": "Start time in seconds"},
                        "end_time": {"type": "number", "description": "End time in seconds"},
                    },
                    "required": ["start_time", "end_time"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "extract_frames",
                "description": "Extract frames from the video for a time range as images. Use to visually inspect specific moments — essential for reading text, numbers, tables, charts, or any fine visual detail. Adjust width/height for resolution: 720x540 (default), 1280x960 (high-res for value reading).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "start_time": {"type": "number", "description": "Start time in seconds"},
                        "end_time": {"type": "number", "description": "End time in seconds"},
                        "fps": {"type": "number", "default": 2.0, "description": "Frames per second to extract"},
                        "max_frames": {"type": "integer", "default": 5, "description": "Maximum number of frames"},
                        "width": {"type": "integer", "default": 720, "description": "Frame width in pixels (720 default, 1280 for high-res)"},
                        "height": {"type": "integer", "default": 540, "description": "Frame height in pixels (540 default, 960 for high-res)"},
                    },
                    "required": ["start_time", "end_time"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "discriminative_vqa",
                "description": "Fast embedding-based multiple-choice VQA without LLM generation. Ranks candidate answers by cosine similarity to video segment embeddings. Use for quick multiple-choice or yes/no questions.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "question": {"type": "string", "description": "The question to answer"},
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
        {
            "type": "function",
            "function": {
                "name": "crop_frame",
                "description": "Crop a region of interest from a frame using percentage coordinates (0.0-1.0). Use to isolate text, tables, charts, faces, or specific areas for detailed inspection.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "image": {"type": "object", "description": "Image dict from extract_frames"},
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
                "description": "Compute absolute pixel difference between two frames. Bright areas = changed regions. Use for motion/change detection between two moments.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "image_a": {"type": "object", "description": "First image dict"},
                        "image_b": {"type": "object", "description": "Second image dict"},
                    },
                    "required": ["image_a", "image_b"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "blend_frames",
                "description": "Average multiple frames into a single composite image. Use for motion summary or detecting static elements across frames.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "images": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of image dicts to blend (pass frame objects from extract_frames)",
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
                "description": "Convert frame to binary mask (pixels > value = white, else black). Returns contour_count and contour_areas for object counting and segmentation.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "image": {"type": "object", "description": "Image dict to threshold"},
                        "value": {"type": "integer", "default": 128, "description": "Threshold value (0-255)"},
                    },
                    "required": ["image"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "frame_info",
                "description": "Get frame dimensions and brightness/color statistics (width, height, mean_brightness, std_brightness, color_means). Use for scene analysis.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "image": {"type": "object", "description": "Image dict to analyze"},
                    },
                    "required": ["image"],
                },
            },
        },
    ]


def _kuavi_agent_loop_gemini(
    question: str,
    model: str,
    tools_map: dict,
    max_iterations: int = 12,
    thinking_level: str = "LOW",
    token_budget: int | None = None,
    tracer: _KUAViTraceLogger | None = None,
) -> tuple[str, list[str], int, int]:
    """Gemini-native tool-calling agent loop."""
    from google import genai
    from google.genai import types

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""
    client = genai.Client(api_key=api_key)

    _type_map = {
        "string": "STRING", "integer": "INTEGER", "number": "NUMBER",
        "boolean": "BOOLEAN", "object": "OBJECT", "array": "ARRAY",
    }

    def _to_gemini_schema(prop: dict) -> types.Schema:
        """Convert a JSON Schema property to a Gemini types.Schema."""
        ptype = prop.get("type", "string")
        gemini_type = _type_map.get(ptype, "STRING")
        kwargs: dict = {"type": gemini_type}
        if ptype == "array":
            items = prop.get("items", {})
            items_type = _type_map.get(items.get("type", "string"), "STRING")
            kwargs["items"] = types.Schema(type=items_type)
        return types.Schema(**kwargs)

    tool_declarations = []
    for schema in _build_kuavi_tools_schema():
        fn = schema["function"]
        props = fn["parameters"].get("properties", {})
        required = fn["parameters"].get("required", [])
        tool_declarations.append(types.FunctionDeclaration(
            name=fn["name"],
            description=fn["description"],
            parameters=types.Schema(
                type="OBJECT",
                properties={k: _to_gemini_schema(v) for k, v in props.items()},
                required=required,
            ) if props else None,
        ))

    tool = types.Tool(function_declarations=tool_declarations)

    # Build thinking config
    thinking_config = None
    if thinking_level and thinking_level != "NONE":
        thinking_config = types.ThinkingConfig(thinking_budget=-1)
        if thinking_level == "LOW":
            thinking_config = types.ThinkingConfig(thinking_budget=1024)
        elif thinking_level == "MEDIUM":
            thinking_config = types.ThinkingConfig(thinking_budget=4096)
        elif thinking_level == "HIGH":
            thinking_config = types.ThinkingConfig(thinking_budget=16384)

    system = KUAVI_SYSTEM_PROMPT
    augmented = question

    contents: list = [types.Content(role="user", parts=[types.Part(text=augmented)])]
    all_tool_calls: list[str] = []
    total_tokens = 0
    iteration_count = 0
    consecutive_no_tools = 0
    budget_warning_injected = False

    if tracer:
        tracer.log_system_prompt(system)

    for _ in range(max_iterations):
        config_kwargs: dict = {
            "system_instruction": system,
            "tools": [tool],
        }
        if thinking_config:
            config_kwargs["thinking_config"] = thinking_config

        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=types.GenerateContentConfig(**config_kwargs),
        )
        if response.usage_metadata:
            total_tokens += (response.usage_metadata.total_token_count or 0)

        # Check for function calls
        parts = response.candidates[0].content.parts or []
        fn_calls = [p for p in parts if p.function_call]
        text_parts = [p.text for p in parts if p.text]

        # Log reasoning text (model's explanation before/between tool calls)
        if tracer and text_parts:
            token_usage = None
            if response.usage_metadata:
                token_usage = {
                    "prompt_tokens": response.usage_metadata.prompt_token_count or 0,
                    "completion_tokens": response.usage_metadata.candidates_token_count or 0,
                    "total_tokens": response.usage_metadata.total_token_count or 0,
                }
            tracer.log_reasoning(iteration_count, "\n".join(text_parts), token_usage=token_usage)

        if not fn_calls:
            consecutive_no_tools += 1
            # Early stopping: 3 consecutive iterations with no tool calls
            if consecutive_no_tools >= 3:
                text = "\n".join(text_parts) if text_parts else ""
                return text, all_tool_calls, iteration_count, total_tokens
            # Add response and escalating nudge
            contents.append(response.candidates[0].content)
            if consecutive_no_tools == 1:
                nudge = (
                    "You haven't found the answer yet. Before giving up, try: "
                    "searching with different queries (e.g. 'table', 'results', 'benchmark', 'score'), "
                    "using different fields (field='visual', field='summary', field='action', field='all'), "
                    "extracting frames from other top-k hit time ranges, "
                    "and checking the scene list with get_scene_list()."
                )
            else:
                nudge = (
                    "You MUST try more search strategies before concluding content is absent. "
                    "Try at least 2 more searches with different query terms and fields, "
                    "and extract frames from at least 2 different time ranges."
                )
            contents.append(types.Content(role="user", parts=[types.Part(text=nudge)]))
            iteration_count += 1
            continue

        consecutive_no_tools = 0

        # Add assistant response to history
        contents.append(response.candidates[0].content)

        # Execute tool calls
        iteration_count += 1
        fn_responses = []
        image_parts = []
        for part in fn_calls:
            fc = part.function_call
            name = fc.name
            all_tool_calls.append(name)
            try:
                args = dict(fc.args) if fc.args else {}
                tc_start = time.monotonic()
                result = tools_map[name](**args)
                tc_elapsed = (time.monotonic() - tc_start) * 1000
                has_err = False
                # Tools that return images: extract_frames, crop_frame, diff_frames, blend_frames, threshold_frame
                is_image_result = name == "extract_frames" and isinstance(result, list)
                is_single_image = name in ("crop_frame", "diff_frames", "blend_frames", "threshold_frame") and isinstance(result, dict) and result.get("data")
                if is_image_result:
                    output = f"Extracted {len(result)} frames. Examine the images in the next message."
                    fn_responses.append(types.Part(function_response=types.FunctionResponse(
                        name=name, response={"result": output},
                    )))
                    import base64 as b64
                    for frame in result[:6]:
                        if isinstance(frame, dict) and "data" in frame:
                            img_bytes = b64.b64decode(frame["data"])
                            mime = frame.get("mime_type", "image/jpeg")
                            image_parts.append(types.Part(inline_data=types.Blob(
                                mime_type=mime, data=img_bytes,
                            )))
                    if tracer:
                        tracer.log_tool_call(name, args, result, tc_elapsed)
                    continue
                if is_single_image:
                    meta = {k: v for k, v in result.items() if k not in ("data", "mime_type", "__image__")}
                    output = json.dumps({"result": "Image produced. See next message.", **meta}, default=str)
                    fn_responses.append(types.Part(function_response=types.FunctionResponse(
                        name=name, response={"result": output},
                    )))
                    import base64 as b64
                    img_bytes = b64.b64decode(result["data"])
                    mime = result.get("mime_type", "image/jpeg")
                    image_parts.append(types.Part(inline_data=types.Blob(mime_type=mime, data=img_bytes)))
                    if tracer:
                        tracer.log_tool_call(name, args, result, tc_elapsed)
                    continue
                output = json.dumps(result, default=str) if not isinstance(result, str) else result
            except Exception as exc:
                tc_elapsed = (time.monotonic() - tc_start) * 1000 if 'tc_start' in dir() else 0
                output = f"Error: {exc}"
                has_err = True
            if tracer:
                tracer.log_tool_call(name, args, output, tc_elapsed, has_error=has_err)
            fn_responses.append(types.Part(function_response=types.FunctionResponse(
                name=name, response={"result": output},
            )))
        contents.append(types.Content(role="user", parts=fn_responses))
        # Send extracted frames/images as a separate user message
        if image_parts:
            image_parts.insert(0, types.Part(text="Here are the extracted frames/images. Examine them carefully:"))
            contents.append(types.Content(role="user", parts=image_parts))

        # Token budget check
        if token_budget and total_tokens >= token_budget and not budget_warning_injected:
            budget_warning_injected = True
            contents.append(types.Content(role="user", parts=[types.Part(
                text="TOKEN BUDGET REACHED. You must provide your final answer NOW based on evidence gathered so far. Do not call any more tools.")]))

    # Force final answer
    contents.append(types.Content(role="user", parts=[types.Part(text="Please provide your final answer now. Do not call any more tools.")]))
    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=types.GenerateContentConfig(system_instruction=system),
    )
    if response.usage_metadata:
        total_tokens += (response.usage_metadata.total_token_count or 0)
    final_parts = response.candidates[0].content.parts or []
    final_text = "\n".join(p.text for p in final_parts if p.text) or ""
    return final_text, all_tool_calls, iteration_count, total_tokens


def _kuavi_agent_loop_openai(
    question: str,
    model: str,
    api_key: str,
    backend: str,
    tools_map: dict,
    max_iterations: int = 12,
    token_budget: int | None = None,
    tracer: _KUAViTraceLogger | None = None,
) -> tuple[str, list[str], int, int]:
    """OpenAI-compatible tool-calling agent loop (for openai/openrouter/anthropic)."""
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

    tools_schema = _build_kuavi_tools_schema()

    system = KUAVI_SYSTEM_PROMPT
    augmented = question

    messages: list[dict] = [
        {"role": "system", "content": system},
        {"role": "user", "content": augmented},
    ]

    all_tool_calls: list[str] = []
    total_tokens = 0
    iteration_count = 0
    consecutive_no_tools = 0
    budget_warning_injected = False

    if tracer:
        tracer.log_system_prompt(system)

    for i in range(max_iterations):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools_schema,
            tool_choice="auto",
            max_tokens=4000,
        )
        if response.usage:
            total_tokens += response.usage.total_tokens or 0

        msg = response.choices[0].message

        # Log reasoning text
        if tracer and msg.content:
            token_usage = None
            if response.usage:
                token_usage = {
                    "prompt_tokens": response.usage.prompt_tokens or 0,
                    "completion_tokens": response.usage.completion_tokens or 0,
                    "total_tokens": response.usage.total_tokens or 0,
                }
            tracer.log_reasoning(iteration_count, msg.content, token_usage=token_usage)

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
            consecutive_no_tools += 1
            # Early stopping: 3 consecutive iterations with no tool calls
            if consecutive_no_tools >= 3:
                return msg.content or "", all_tool_calls, iteration_count, total_tokens
            if consecutive_no_tools == 1:
                nudge = (
                    "You haven't found the answer yet. Before giving up, try: "
                    "searching with different queries (e.g. 'table', 'results', 'benchmark', 'score'), "
                    "using different fields (field='visual', field='summary', field='action', field='all'), "
                    "extracting frames from other top-k hit time ranges, "
                    "and checking the scene list with get_scene_list()."
                )
            else:
                nudge = (
                    "You MUST try more search strategies before concluding content is absent. "
                    "Try at least 2 more searches with different query terms and fields, "
                    "and extract frames from at least 2 different time ranges."
                )
            messages.append({"role": "user", "content": nudge})
            iteration_count += 1
            continue

        consecutive_no_tools = 0
        iteration_count += 1
        image_parts_to_send: list[dict] = []
        for tc in msg.tool_calls:
            name = tc.function.name
            all_tool_calls.append(name)
            try:
                args = json.loads(tc.function.arguments)
                tc_start = time.monotonic()
                result = tools_map[name](**args)
                tc_elapsed = (time.monotonic() - tc_start) * 1000
                has_err = False
                is_image_result = name == "extract_frames" and isinstance(result, list)
                is_single_image = name in ("crop_frame", "diff_frames", "blend_frames", "threshold_frame") and isinstance(result, dict) and result.get("data")
                if is_image_result:
                    content = f"Extracted {len(result)} frames. The images are included in the next message."
                    messages.append({"role": "tool", "tool_call_id": tc.id, "content": content})
                    for frame in result[:8]:
                        if isinstance(frame, dict) and "data" in frame:
                            mime = frame.get("mime_type", "image/jpeg")
                            image_parts_to_send.append({
                                "type": "image_url",
                                "image_url": {"url": f"data:{mime};base64,{frame['data']}"},
                            })
                    if tracer:
                        tracer.log_tool_call(name, args, result, tc_elapsed)
                    continue
                if is_single_image:
                    meta = {k: v for k, v in result.items() if k not in ("data", "mime_type", "__image__")}
                    content = json.dumps({"result": "Image produced. See next message.", **meta}, default=str)
                    messages.append({"role": "tool", "tool_call_id": tc.id, "content": content})
                    mime = result.get("mime_type", "image/jpeg")
                    image_parts_to_send.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{result['data']}"},
                    })
                    if tracer:
                        tracer.log_tool_call(name, args, result, tc_elapsed)
                    continue
                content = json.dumps(result, default=str) if not isinstance(result, str) else result
            except Exception as exc:
                tc_elapsed = (time.monotonic() - tc_start) * 1000 if 'tc_start' in dir() else 0
                content = f"Error: {exc}"
                has_err = True
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": content})
            if tracer:
                tracer.log_tool_call(name, args, content, tc_elapsed, has_error=has_err)

        # Send images as a separate multimodal user message
        if image_parts_to_send:
            parts: list[dict] = [{"type": "text", "text": "Here are the extracted frames/images. Examine them carefully:"}]
            parts.extend(image_parts_to_send)
            messages.append({"role": "user", "content": parts})

        # Token budget check
        if token_budget and total_tokens >= token_budget and not budget_warning_injected:
            budget_warning_injected = True
            messages.append({"role": "user", "content": "TOKEN BUDGET REACHED. You must provide your final answer NOW based on evidence gathered so far. Do not call any more tools."})

    # Force final answer
    messages.append({"role": "user", "content": "Please provide your final answer now."})
    response = client.chat.completions.create(model=model, messages=messages, max_tokens=2000)
    if response.usage:
        total_tokens += response.usage.total_tokens or 0
    return response.choices[0].message.content or "", all_tool_calls, iteration_count, total_tokens


def run_kuavi(
    video_path: str,
    question: str,
    model: str,
    backend: str,
    cache_dir: str | None,
    max_iterations: int = 12,
    thinking_level: str = "LOW",
    token_budget: int | None = None,
) -> dict:
    """Run KUAVi pipeline and return result dict."""
    from kuavi.indexer import VideoIndexer
    from kuavi.loader import VideoLoader
    from kuavi.search import (
        make_get_scene_list,
        make_get_transcript,
        make_search_transcript,
        make_search_video,
    )

    api_key = {
        "openai": os.getenv("OPENAI_API_KEY"),
        "openrouter": os.getenv("OPENROUTER_API_KEY"),
        "anthropic": os.getenv("ANTHROPIC_API_KEY"),
        "gemini": os.getenv("GEMINI_API_KEY"),
    }.get(backend) or os.getenv("OPENROUTER_API_KEY", "")

    t0 = time.monotonic()

    from kuavi.verbose import KUAViPrinter

    printer = KUAViPrinter()
    printer.print_header("KUAVi Agent", {
        "Video": Path(video_path).name,
        "Model": model,
        "Backend": backend,
        "Max Iterations": max_iterations,
    })

    printer.print_step("Loading video", video_path)
    loader = VideoLoader(fps=0.5)
    loaded = loader.load(video_path)
    printer.print_step_done(
        "Loaded",
        f"{loaded.metadata.duration:.1f}s, {loaded.metadata.extracted_frame_count} frames",
    )

    printer.print_step("Building index")
    indexer = VideoIndexer(
        embedding_model="google/siglip2-base-patch16-256",
        text_embedding_model="google/embeddinggemma-300m",
        scene_model="facebook/vjepa2-vitl-fpc64-256",
        cache_dir=cache_dir,
    )
    index = indexer.index_video(loaded, whisper_model="base")
    index_time = time.monotonic() - t0
    printer.print_step_done(
        "Index built",
        f"{len(index.segments)} segments, {len(index.scene_boundaries)} scenes",
        elapsed=index_time,
    )

    from kuavi.context import make_extract_frames
    from kuavi.search import make_discriminative_vqa

    extract_fn = make_extract_frames(str(Path(video_path).resolve()))

    def _extract_frames_wrapper(
        start_time: float, end_time: float, fps: float = 2.0, max_frames: int = 5,
        width: int = 720, height: int = 540,
    ):
        return extract_fn(start_time, end_time, fps=fps, resize=(width, height), max_frames=max_frames)

    # Pixel tools (matching RLM's _make_pixel_tools)
    import base64 as _b64
    import cv2 as _cv2
    import numpy as _np

    def _decode_image_dict(image_dict: dict) -> _np.ndarray:
        data = _b64.b64decode(image_dict["data"])
        arr = _np.frombuffer(data, dtype=_np.uint8)
        return _cv2.imdecode(arr, _cv2.IMREAD_COLOR)

    def _to_image_dict(frame: _np.ndarray) -> dict:
        _, buf = _cv2.imencode(".jpg", frame, [_cv2.IMWRITE_JPEG_QUALITY, 85])
        return {"__image__": True, "data": _b64.b64encode(buf.tobytes()).decode(), "mime_type": "image/jpeg"}

    def _crop_frame(image: dict, x1_pct: float, y1_pct: float, x2_pct: float, y2_pct: float) -> dict:
        frame = _decode_image_dict(image)
        h, w = frame.shape[:2]
        x1, y1 = max(0, int(x1_pct * w)), max(0, int(y1_pct * h))
        x2, y2 = min(w, int(x2_pct * w)), min(h, int(y2_pct * h))
        return _to_image_dict(frame[y1:y2, x1:x2])

    def _diff_frames(image_a: dict, image_b: dict) -> dict:
        a, b = _decode_image_dict(image_a), _decode_image_dict(image_b)
        if a.shape != b.shape:
            b = _cv2.resize(b, (a.shape[1], a.shape[0]))
        diff = _cv2.absdiff(a, b)
        result = _to_image_dict(diff)
        gray = _cv2.cvtColor(diff, _cv2.COLOR_BGR2GRAY)
        result["mean_diff"] = float(gray.mean())
        result["changed_pct"] = float((gray > 25).sum() / gray.size * 100)
        return result

    def _blend_frames(images: list) -> dict:
        frames = [_decode_image_dict(d) for d in images]
        if not frames:
            return {"error": "No frames to blend"}
        target = frames[0].shape[:2]
        resized = [_cv2.resize(f, (target[1], target[0])) if f.shape[:2] != target else f for f in frames]
        composite = _np.mean(_np.stack(resized), axis=0).astype(_np.uint8)
        return _to_image_dict(composite)

    def _threshold_frame(image: dict, value: int = 128) -> dict:
        frame = _decode_image_dict(image)
        gray = _cv2.cvtColor(frame, _cv2.COLOR_BGR2GRAY)
        _, mask = _cv2.threshold(gray, value, 255, _cv2.THRESH_BINARY)
        contours, _ = _cv2.findContours(mask, _cv2.RETR_EXTERNAL, _cv2.CHAIN_APPROX_SIMPLE)
        result = _to_image_dict(_cv2.cvtColor(mask, _cv2.COLOR_GRAY2BGR))
        result["contour_count"] = len(contours)
        result["contour_areas"] = sorted([_cv2.contourArea(c) for c in contours], reverse=True)[:20]
        result["white_pct"] = float((mask > 0).sum() / mask.size * 100)
        return result

    def _frame_info(image: dict) -> dict:
        frame = _decode_image_dict(image)
        gray = _cv2.cvtColor(frame, _cv2.COLOR_BGR2GRAY)
        h, w = frame.shape[:2]
        return {
            "width": w, "height": h,
            "channels": frame.shape[2] if len(frame.shape) > 2 else 1,
            "mean_brightness": float(gray.mean()),
            "std_brightness": float(gray.std()),
            "min_brightness": int(gray.min()),
            "max_brightness": int(gray.max()),
            "color_means": {"b": float(frame[:, :, 0].mean()), "g": float(frame[:, :, 1].mean()), "r": float(frame[:, :, 2].mean())},
        }

    tools_map = {
        "get_scene_list": make_get_scene_list(index)["tool"],
        "search_video": make_search_video(index)["tool"],
        "search_transcript": make_search_transcript(index)["tool"],
        "get_transcript": make_get_transcript(index)["tool"],
        "discriminative_vqa": make_discriminative_vqa(index)["tool"],
        "extract_frames": _extract_frames_wrapper,
        "crop_frame": _crop_frame,
        "diff_frames": _diff_frames,
        "blend_frames": _blend_frames,
        "threshold_frame": _threshold_frame,
        "frame_info": _frame_info,
    }

    # Initialize trace logger for visualizer compatibility
    tracer = _KUAViTraceLogger(log_dir="./logs")
    tracer.log_session_start(model=model)
    tracer.log_question(question)
    tracer.log_metadata(
        video_path=video_path,
        duration=loaded.metadata.duration,
        fps=loaded.metadata.extraction_fps,
        num_segments=len(index.segments),
        num_scenes=len(index.scene_boundaries),
    )

    printer.print_step("Running agent loop")
    if backend == "gemini":
        answer, tool_calls, iteration_count, total_tokens = _kuavi_agent_loop_gemini(
            question=question,
            model=model,
            tools_map=tools_map,
            max_iterations=max_iterations,
            thinking_level=thinking_level,
            token_budget=token_budget,
            tracer=tracer,
        )
    else:
        answer, tool_calls, iteration_count, total_tokens = _kuavi_agent_loop_openai(
            question=question,
            model=model,
            api_key=api_key,
            backend=backend,
            tools_map=tools_map,
            max_iterations=max_iterations,
            token_budget=token_budget,
            tracer=tracer,
        )

    tracer.log_final_answer(answer)
    elapsed = time.monotonic() - t0

    return {
        "pipeline": "kuavi",
        "answer": answer,
        "iteration_count": iteration_count,
        "tool_calls": tool_calls,
        "wall_time_seconds": round(elapsed, 2),
        "approximate_tokens": total_tokens or None,
        "index_time_seconds": round(index_time, 2),
        "num_segments": len(index.segments),
        "num_scenes": len(index.scene_boundaries),
        "log_file": tracer.log_file,
    }


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary(results: list[dict]) -> None:
    from kuavi.verbose import (
        COLORS,
        STYLE_ACCENT,
        STYLE_MUTED,
        STYLE_PRIMARY,
        STYLE_TEXT,
        STYLE_WARNING,
    )
    from rich.console import Console
    from rich.panel import Panel
    from rich.rule import Rule
    from rich.style import Style
    from rich.table import Table
    from rich.text import Text

    console = Console()

    title = Text()
    title.append("★ ", style=STYLE_WARNING)
    title.append("Experiment Summary", style=Style(color=COLORS["warning"], bold=True))

    table = Table(
        show_edge=False,
        padding=(0, 1),
        expand=True,
        border_style=COLORS["border"],
    )
    table.add_column("Pipeline", style=STYLE_PRIMARY, width=10)
    table.add_column("Wall Time", style=STYLE_ACCENT, width=12, justify="right")
    table.add_column("Iterations", style=STYLE_ACCENT, width=12, justify="right")
    table.add_column("Tool Calls", style=STYLE_ACCENT, width=12, justify="right")
    table.add_column("~Tokens", style=STYLE_ACCENT, width=12, justify="right")
    table.add_column("Index Time", style=STYLE_ACCENT, width=12, justify="right")

    for r in results:
        tokens = f"{r['approximate_tokens']:,}" if r.get("approximate_tokens") else "-"
        idx_time = f"{r['index_time_seconds']}s" if r.get("index_time_seconds") else "-"
        tools_str = str(len(r["tool_calls"]))
        table.add_row(
            r["pipeline"].upper(),
            f"{r['wall_time_seconds']}s",
            str(r["iteration_count"]),
            tools_str,
            tokens,
            idx_time,
        )

    panel = Panel(
        table,
        title=title,
        title_align="left",
        border_style=COLORS["border"],
        padding=(1, 2),
    )
    console.print()
    console.print(panel)

    # Print answer previews
    for r in results:
        pipeline = r["pipeline"].upper()
        preview = r["answer"][:200].strip()
        if preview:
            answer_title = Text()
            answer_title.append(f"  {pipeline} Answer Preview", style=STYLE_MUTED)
            console.print(answer_title)
            console.print(Text(f"  {preview}...", style=STYLE_TEXT))
    console.print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Compare RLM vs KUAVi on video analysis")
    parser.add_argument("--video", required=True, help="Path to the video file")
    parser.add_argument("--question", default=DEFAULT_QUESTION, help="Question to ask about the video")
    parser.add_argument("--model", default="gemini-3.1-pro-preview", help="LLM model name")
    parser.add_argument("--backend", default="gemini", help="LLM backend (gemini, openai, openrouter, anthropic)")
    parser.add_argument(
        "--pipeline", default="both", choices=["rlm", "kuavi", "both"],
        help="Which pipeline(s) to run (default: both)"
    )
    parser.add_argument("--cache-dir", default=None, help="Shared cache directory for video indexes")
    parser.add_argument("--output-dir", default="experiments", help="Directory to save JSON results (default: experiments/)")
    parser.add_argument("--max-iterations", type=int, default=15, help="Max agent iterations (default: 15)")
    parser.add_argument("--thinking-level", default="LOW", choices=["NONE", "LOW", "MEDIUM", "HIGH"],
                        help="Gemini thinking level for RLM (default: LOW)")
    args = parser.parse_args()

    video_path = args.video
    if not Path(video_path).exists():
        from kuavi.verbose import KUAViPrinter
        KUAViPrinter().print_error(f"Video file not found: {video_path}")
        raise SystemExit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    video_stem = Path(video_path).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pipelines = ["rlm", "kuavi"] if args.pipeline == "both" else [args.pipeline]

    from kuavi.verbose import COLORS, STYLE_PRIMARY
    from rich.console import Console
    from rich.rule import Rule
    from rich.text import Text

    exp_console = Console()

    results: list[dict] = []
    for pipeline in pipelines:
        exp_console.print()
        exp_console.print(Rule(
            Text(f" {pipeline.upper()} ", style=STYLE_PRIMARY),
            style=COLORS["border"],
            characters="═",
        ))
        try:
            if pipeline == "rlm":
                result = run_rlm(
                    video_path=video_path,
                    question=args.question,
                    model=args.model,
                    backend=args.backend,
                    cache_dir=args.cache_dir,
                    max_iterations=args.max_iterations,
                    thinking_level=args.thinking_level,
                )
            else:
                result = run_kuavi(
                    video_path=video_path,
                    question=args.question,
                    model=args.model,
                    backend=args.backend,
                    cache_dir=args.cache_dir,
                    max_iterations=args.max_iterations,
                    thinking_level=args.thinking_level,
                )
        except Exception as exc:
            from kuavi.verbose import KUAViPrinter
            KUAViPrinter().print_error(f"[{pipeline}] {exc}")
            result = {
                "pipeline": pipeline,
                "error": str(exc),
                "answer": "",
                "iteration_count": 0,
                "tool_calls": [],
                "wall_time_seconds": 0.0,
                "approximate_tokens": None,
            }

        result["video"] = video_path
        result["question"] = args.question
        result["model"] = args.model
        result["backend"] = args.backend
        result["timestamp"] = timestamp
        results.append(result)

        # Save per-pipeline result
        out_path = output_dir / f"{video_stem}_{pipeline}_{timestamp}.json"
        out_path.write_text(json.dumps(result, indent=2))
        from kuavi.verbose import STYLE_MUTED, STYLE_SUCCESS
        exp_console.print(Text(f"  Result saved to: {out_path}", style=STYLE_MUTED))

    if args.pipeline == "both":
        print_summary(results)
        combined_path = output_dir / f"{video_stem}_both_{timestamp}.json"
        combined_path.write_text(json.dumps(results, indent=2))
        exp_console.print(Text(f"Combined results saved to: {combined_path}", style=STYLE_MUTED))
    else:
        from kuavi.verbose import KUAViPrinter as _KP
        _KP().print_final_summary({"Answer": results[0].get("answer", results[0].get("error", ""))[:500]})


if __name__ == "__main__":
    main()
