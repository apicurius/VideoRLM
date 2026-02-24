"""KUAVi vs RLM benchmark with full token tracking.

Runs both RLM (Gemini) and KUAVi (Anthropic) on the same video question,
capturing real API-reported input/output token counts for fair comparison.

Usage:
    uv run python benchmark.py --video test_video.mp4 \
        --question "What is OOLONG score of RLM?" --systems rlm kuavi

    # KUAVi only
    uv run python benchmark.py --video test_video.mp4 \
        --question "What is OOLONG score of RLM?" --systems kuavi

    # RLM only
    uv run python benchmark.py --video test_video.mp4 \
        --question "What is OOLONG score of RLM?" --systems rlm
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Tool schema definitions for the Anthropic API
# ---------------------------------------------------------------------------

KUAVI_TOOLS: list[dict[str, Any]] = [
    {
        "name": "kuavi_orient",
        "description": "Get video overview: metadata + scene list in one call.",
        "input_schema": {
            "type": "object",
            "properties": {
                "video_id": {"type": "string", "description": "Video ID (optional)."},
            },
        },
    },
    {
        "name": "kuavi_search_all",
        "description": (
            "Multi-field semantic search + transcript keyword search in parallel. "
            "Fields: summary, action, visual. Returns results grouped by field."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Semantic search query."},
                "fields": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Fields to search. Default: [summary, action, visual].",
                },
                "transcript_query": {
                    "type": "string",
                    "description": "Transcript keyword query (optional).",
                },
                "top_k": {"type": "integer", "description": "Results per field. Default: 5."},
                "level": {"type": "integer", "description": "Hierarchy level. Default: 0."},
            },
            "required": ["query"],
        },
    },
    {
        "name": "kuavi_quick_answer",
        "description": (
            "One-shot search + inspect: find relevant segments and extract frames/transcript. "
            "Combines search_all + inspect_segment for top hits in one call."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "question": {"type": "string", "description": "Search query / question."},
                "fields": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Search fields. Default: [summary, action, visual].",
                },
                "transcript_query": {
                    "type": "string",
                    "description": "Transcript keyword query (optional).",
                },
                "top_k": {"type": "integer", "description": "Results per field. Default: 3."},
                "inspect_top": {
                    "type": "integer",
                    "description": "How many top hits to auto-inspect. Default: 3.",
                },
                "zoom_level": {
                    "type": "integer",
                    "description": "Frame zoom preset (1-3). Default: 2.",
                },
                "max_frames_per_hit": {
                    "type": "integer",
                    "description": "Max frames per inspected hit. Default: 5.",
                },
            },
            "required": ["question"],
        },
    },
    {
        "name": "kuavi_inspect_segment",
        "description": "Extract frames + get transcript for a time range in one call.",
        "input_schema": {
            "type": "object",
            "properties": {
                "start_time": {"type": "number", "description": "Start time in seconds."},
                "end_time": {"type": "number", "description": "End time in seconds."},
                "include_transcript": {
                    "type": "boolean",
                    "description": "Include transcript. Default: true.",
                },
                "include_frames": {
                    "type": "boolean",
                    "description": "Extract frames. Default: true.",
                },
                "zoom_level": {
                    "type": "integer",
                    "description": "Frame zoom preset (1-3). Default: 2.",
                },
                "max_frames": {
                    "type": "integer",
                    "description": "Max frames to extract. Default: 5.",
                },
            },
            "required": ["start_time", "end_time"],
        },
    },
    {
        "name": "kuavi_search_video",
        "description": (
            "Semantic search over indexed video segments. "
            'Fields: "summary", "action", "visual", "temporal", "all".'
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query."},
                "top_k": {"type": "integer", "description": "Number of results. Default: 5."},
                "field": {
                    "type": "string",
                    "description": "Search field. Default: summary.",
                },
                "level": {"type": "integer", "description": "Hierarchy level. Default: 0."},
            },
            "required": ["query"],
        },
    },
    {
        "name": "kuavi_search_transcript",
        "description": "Keyword search over ASR transcript (case-insensitive).",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Keyword query."},
            },
            "required": ["query"],
        },
    },
    {
        "name": "kuavi_get_transcript",
        "description": "Get transcript text for a specific time range (seconds).",
        "input_schema": {
            "type": "object",
            "properties": {
                "start_time": {"type": "number", "description": "Start time in seconds."},
                "end_time": {"type": "number", "description": "End time in seconds."},
            },
            "required": ["start_time", "end_time"],
        },
    },
    {
        "name": "kuavi_get_scene_list",
        "description": "List all detected scenes with annotations.",
        "input_schema": {
            "type": "object",
            "properties": {
                "video_id": {"type": "string", "description": "Video ID (optional)."},
            },
        },
    },
    {
        "name": "kuavi_get_index_info",
        "description": "Get metadata about the current video index.",
        "input_schema": {
            "type": "object",
            "properties": {
                "video_id": {"type": "string", "description": "Video ID (optional)."},
            },
        },
    },
    {
        "name": "kuavi_extract_frames",
        "description": "Extract frames from the indexed video as base64 JPEG images.",
        "input_schema": {
            "type": "object",
            "properties": {
                "start_time": {"type": "number", "description": "Start time in seconds."},
                "end_time": {"type": "number", "description": "End time in seconds."},
                "fps": {"type": "number", "description": "Frames per second. Default: 2."},
                "width": {"type": "integer", "description": "Frame width. Default: 720."},
                "height": {"type": "integer", "description": "Frame height. Default: 540."},
                "max_frames": {"type": "integer", "description": "Max frames. Default: 10."},
            },
            "required": ["start_time", "end_time"],
        },
    },
    {
        "name": "kuavi_zoom_frames",
        "description": (
            "Extract frames at preset zoom levels. "
            "L1: 480x360 overview. L2: 720x540 detail. L3: 1280x960 high-res."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "start_time": {"type": "number", "description": "Start time in seconds."},
                "end_time": {"type": "number", "description": "End time in seconds."},
                "level": {"type": "integer", "description": "Zoom level (1-3). Default: 1."},
            },
            "required": ["start_time", "end_time"],
        },
    },
    {
        "name": "kuavi_discriminative_vqa",
        "description": (
            "Embedding-based multiple-choice VQA. "
            "Ranks candidates by cosine similarity — no LLM generation."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "question": {"type": "string", "description": "The question."},
                "candidates": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Candidate answers.",
                },
                "start_time": {"type": "number", "description": "Optional start time."},
                "end_time": {"type": "number", "description": "Optional end time."},
            },
            "required": ["question", "candidates"],
        },
    },
    {
        "name": "kuavi_eval",
        "description": (
            "Execute Python code in a persistent namespace with KUAVi tools available. "
            "Pre-populated with np, cv2, and all kuavi tool functions as short names. "
            "Use for programmatic analysis, counting, iteration, and chaining tool calls."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Python code to execute."},
            },
            "required": ["code"],
        },
    },
]

# Maximum characters for tool results sent back to the model
_MAX_TOOL_RESULT_CHARS = 80_000


# ---------------------------------------------------------------------------
# RLM Benchmark
# ---------------------------------------------------------------------------


def run_rlm_benchmark(
    video_path: str,
    question: str,
    *,
    backend: str = "gemini",
    model: str = "gemini-3.1-pro-preview",
    cache_dir: str | None = None,
    auto_fps: bool = True,
    max_iterations: int = 15,
    thinking_level: str = "LOW",
) -> dict[str, Any]:
    """Run RLM benchmark, return metrics dict with real Gemini API token counts."""
    from rlm.logger import RLMLogger
    from rlm.video import VideoRLM

    logger = RLMLogger(log_dir=None)  # in-memory only

    vrlm = VideoRLM(
        backend=backend,
        backend_kwargs={
            "model_name": model,
            "timeout": 300.0,
            "thinking_level": thinking_level,
        },
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
        auto_fps=auto_fps,
        scene_model="facebook/vjepa2-vitl-fpc64-256",
        text_embedding_model="google/embeddinggemma-300m",
    )

    start = time.time()
    result = vrlm.completion(video_path, prompt=question)
    elapsed = time.time() - start

    # Extract token usage from all models
    usage = result.usage_summary
    total_input = 0
    total_output = 0
    for _model_name, model_usage in usage.model_usage_summaries.items():
        total_input += model_usage.total_input_tokens
        total_output += model_usage.total_output_tokens

    iterations = logger.iteration_count

    return {
        "system": "rlm",
        "model": model,
        "answer": result.response,
        "elapsed_s": round(elapsed, 2),
        "iterations": iterations,
        "input_tokens": total_input,
        "output_tokens": total_output,
        "total_tokens": total_input + total_output,
        "per_model_usage": {
            m: {"input": u.total_input_tokens, "output": u.total_output_tokens}
            for m, u in usage.model_usage_summaries.items()
        },
    }


# ---------------------------------------------------------------------------
# KUAVi Benchmark
# ---------------------------------------------------------------------------


def _index_video_in_process(
    video_path: str,
    *,
    cache_dir: str | None = None,
    auto_fps: bool = True,
) -> None:
    """Index a video and register it in the MCP server's global state."""
    import cv2

    from kuavi.indexer import VideoIndexer
    from kuavi.loader import VideoLoader
    from kuavi.mcp_server import _precompute_orient, _state

    # Auto-FPS
    fps = 1.0
    if auto_fps:
        cap = cv2.VideoCapture(video_path)
        try:
            video_fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            if video_fps > 0 and frame_count > 0:
                duration = frame_count / video_fps
                fps = max(0.1, min(5.0, 120 / duration))
        finally:
            cap.release()

    loader = VideoLoader(fps=fps)
    loaded_video = loader.load(video_path)

    indexer = VideoIndexer(
        embedding_model="google/siglip2-base-patch16-256",
        cache_dir=cache_dir,
        text_embedding_model=None,
        scene_model="facebook/vjepa2-vitl-fpc64-256",
    )

    # Build captioning functions
    import os

    caption_fn = None
    frame_caption_fn = None
    refine_fn = None
    try:
        gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        from kuavi.captioning import (
            make_gemini_caption_fn,
            make_gemini_frame_caption_fn,
            make_gemini_refine_fn,
        )

        caption_fn = make_gemini_caption_fn(api_key=gemini_key)
        frame_caption_fn = make_gemini_frame_caption_fn(api_key=gemini_key)
        refine_fn = make_gemini_refine_fn(api_key=gemini_key)
    except Exception:
        pass

    index = indexer.index_video(
        loaded_video,
        caption_fn=caption_fn,
        frame_caption_fn=frame_caption_fn,
        refine_fn=refine_fn,
        asr_model="Qwen/Qwen3-ASR-0.6B",
        mode="full",
    )

    video_id = Path(video_path).stem
    _state["videos"][video_id] = {
        "index": index,
        "indexer": indexer,
        "loaded_video": loaded_video,
        "video_path": video_path,
    }
    _state["active_video"] = video_id
    _state["result_cache"].clear()
    _precompute_orient(video_id)


def _build_tool_dispatch() -> dict[str, Any]:
    """Build a dispatch map from tool name -> callable."""
    from kuavi.mcp_server import (
        kuavi_discriminative_vqa,
        kuavi_eval,
        kuavi_extract_frames,
        kuavi_get_index_info,
        kuavi_get_scene_list,
        kuavi_get_transcript,
        kuavi_inspect_segment,
        kuavi_orient,
        kuavi_quick_answer,
        kuavi_search_all,
        kuavi_search_transcript,
        kuavi_search_video,
        kuavi_zoom_frames,
    )

    return {
        "kuavi_orient": kuavi_orient,
        "kuavi_search_all": kuavi_search_all,
        "kuavi_quick_answer": kuavi_quick_answer,
        "kuavi_inspect_segment": kuavi_inspect_segment,
        "kuavi_search_video": kuavi_search_video,
        "kuavi_search_transcript": kuavi_search_transcript,
        "kuavi_get_transcript": kuavi_get_transcript,
        "kuavi_get_scene_list": kuavi_get_scene_list,
        "kuavi_get_index_info": kuavi_get_index_info,
        "kuavi_extract_frames": kuavi_extract_frames,
        "kuavi_zoom_frames": kuavi_zoom_frames,
        "kuavi_discriminative_vqa": kuavi_discriminative_vqa,
        "kuavi_eval": kuavi_eval,
    }


def _extract_images_from_result(obj: Any) -> list[dict[str, Any]]:
    """Extract base64 image dicts from a tool result (recursively).

    Returns a list of {"data": ..., "mime_type": ...} dicts found in the result.
    """
    images: list[dict[str, Any]] = []
    if isinstance(obj, dict):
        if "data" in obj and "mime_type" in obj and isinstance(obj["data"], str) and len(obj["data"]) > 500:
            images.append(obj)
        else:
            for v in obj.values():
                images.extend(_extract_images_from_result(v))
    elif isinstance(obj, list):
        for item in obj:
            images.extend(_extract_images_from_result(item))
    return images


def _strip_base64_from_result(obj: Any) -> Any:
    """Replace base64 image data with short placeholders for the JSON portion."""
    if isinstance(obj, dict):
        result = {}
        for k, v in obj.items():
            if k == "data" and isinstance(v, str) and len(v) > 500:
                result[k] = "<see image below>"
            else:
                result[k] = _strip_base64_from_result(v)
        return result
    if isinstance(obj, list):
        return [_strip_base64_from_result(item) for item in obj]
    return obj


def _build_tool_result_content(result: Any) -> list[dict[str, Any]]:
    """Build Anthropic tool_result content blocks: JSON text + image blocks.

    Extracts base64 images from the result, replaces them with placeholders
    in the JSON text, and appends proper image content blocks so the model
    can actually see the frames.
    """
    # Extract images before stripping
    images = _extract_images_from_result(result)

    # Strip base64 from the JSON representation
    stripped = _strip_base64_from_result(result)
    result_str = json.dumps(stripped, default=str)
    if len(result_str) > _MAX_TOOL_RESULT_CHARS:
        result_str = result_str[:_MAX_TOOL_RESULT_CHARS] + "... [truncated]"

    content: list[dict[str, Any]] = [{"type": "text", "text": result_str}]

    # Append images as proper image content blocks (cap to avoid huge payloads)
    for img in images[:10]:
        media_type = img.get("mime_type", "image/jpeg")
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": media_type,
                "data": img["data"],
            },
        })

    return content


def run_kuavi_benchmark(
    video_path: str,
    question: str,
    *,
    model: str = "claude-sonnet-4-20250514",
    max_turns: int = 20,
    cache_dir: str | None = None,
    auto_fps: bool = True,
) -> dict[str, Any]:
    """Run KUAVi benchmark via direct Anthropic API calls with real token tracking."""
    import anthropic

    # 1. Index video in-process (reuses MCP server state)
    print("[KUAVi] Indexing video...")
    index_start = time.time()
    _index_video_in_process(video_path, cache_dir=cache_dir, auto_fps=auto_fps)
    index_elapsed = time.time() - index_start
    print(f"[KUAVi] Indexing complete in {index_elapsed:.1f}s")

    # 2. Build tool dispatch
    tool_dispatch = _build_tool_dispatch()

    # 3. Load agent prompt
    agent_prompt_path = Path(".claude/agents/video-analyst.md")
    if agent_prompt_path.exists():
        raw = agent_prompt_path.read_text()
        # Strip YAML frontmatter
        if raw.startswith("---"):
            end = raw.find("---", 3)
            if end != -1:
                raw = raw[end + 3 :].strip()
        agent_prompt = raw
    else:
        agent_prompt = (
            "You are a video analysis agent. Use the available KUAVi tools to answer "
            "questions about the indexed video. Follow a search-first strategy: orient, "
            "search, inspect frames, then answer with evidence."
        )

    # 4. Agent loop via Anthropic API
    client = anthropic.Anthropic()
    messages: list[dict[str, Any]] = [{"role": "user", "content": question}]

    total_input = 0
    total_output = 0
    total_cache_read = 0
    total_cache_create = 0
    tool_call_count = 0
    answer = ""
    turn = 0

    print(f"[KUAVi] Starting agent loop (model={model}, max_turns={max_turns})")
    start = time.time()

    for turn in range(max_turns):
        response = client.messages.create(
            model=model,
            max_tokens=4096,
            system=agent_prompt,
            tools=KUAVI_TOOLS,
            messages=messages,
        )

        # Track tokens
        total_input += response.usage.input_tokens
        total_output += response.usage.output_tokens
        if hasattr(response.usage, "cache_read_input_tokens"):
            total_cache_read += response.usage.cache_read_input_tokens or 0
        if hasattr(response.usage, "cache_creation_input_tokens"):
            total_cache_create += response.usage.cache_creation_input_tokens or 0

        # Check for tool use
        tool_uses = [b for b in response.content if b.type == "tool_use"]
        if not tool_uses:
            # Final answer — extract text
            answer = "".join(b.text for b in response.content if b.type == "text")
            print(f"[KUAVi] Final answer at turn {turn + 1}")
            break

        # Log tool calls
        tool_names = [t.name for t in tool_uses]
        print(f"[KUAVi] Turn {turn + 1}: {len(tool_uses)} tool call(s): {tool_names}")

        # Execute tools
        messages.append({"role": "assistant", "content": response.content})
        tool_results = []
        for tool_use in tool_uses:
            tool_call_count += 1
            fn = tool_dispatch.get(tool_use.name)
            if fn:
                try:
                    result = fn(**tool_use.input)
                except Exception as e:
                    result = {"error": str(e)}
            else:
                result = {"error": f"Unknown tool: {tool_use.name}"}

            # Build tool result with proper image content blocks
            content = _build_tool_result_content(result)

            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": tool_use.id,
                    "content": content,
                }
            )
        messages.append({"role": "user", "content": tool_results})
    else:
        # Exhausted turns without a final answer
        answer = "[max turns reached without final answer]"

    elapsed = time.time() - start

    return {
        "system": "kuavi",
        "model": model,
        "answer": answer,
        "elapsed_s": round(elapsed, 2),
        "index_elapsed_s": round(index_elapsed, 2),
        "turns": turn + 1,
        "tool_calls": tool_call_count,
        "input_tokens": total_input,
        "output_tokens": total_output,
        "total_tokens": total_input + total_output,
        "cache_read_tokens": total_cache_read,
        "cache_creation_tokens": total_cache_create,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="KUAVi vs RLM benchmark with full token tracking"
    )
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--question", required=True, help="Question to ask about the video")
    parser.add_argument(
        "--systems",
        nargs="+",
        default=["rlm", "kuavi"],
        choices=["rlm", "kuavi"],
        help="Systems to benchmark (default: both)",
    )
    parser.add_argument("--cache-dir", default="./cache", help="Index cache directory")
    parser.add_argument(
        "--kuavi-model",
        default="claude-sonnet-4-20250514",
        help="Anthropic model for KUAVi agent",
    )
    parser.add_argument(
        "--rlm-model",
        default="gemini-3.1-pro-preview",
        help="Gemini model for RLM",
    )
    parser.add_argument(
        "--rlm-backend",
        default="gemini",
        help="RLM backend (default: gemini)",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=20,
        help="Max agent turns for KUAVi (default: 20)",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=15,
        help="Max REPL iterations for RLM (default: 15)",
    )
    parser.add_argument(
        "--output",
        default="benchmark_results.json",
        help="Output JSON file (default: benchmark_results.json)",
    )
    args = parser.parse_args()

    results: dict[str, Any] = {
        "question": args.question,
        "video": args.video,
    }

    for system in args.systems:
        print(f"\n{'=' * 60}")
        print(f"Running {system.upper()} benchmark...")
        print(f"{'=' * 60}\n")

        if system == "rlm":
            results["rlm"] = run_rlm_benchmark(
                args.video,
                args.question,
                backend=args.rlm_backend,
                model=args.rlm_model,
                cache_dir=args.cache_dir,
                max_iterations=args.max_iterations,
            )
        elif system == "kuavi":
            results["kuavi"] = run_kuavi_benchmark(
                args.video,
                args.question,
                model=args.kuavi_model,
                max_turns=args.max_turns,
                cache_dir=args.cache_dir,
            )

    # Print comparison table
    print(f"\n{'=' * 70}")
    print(f"{'BENCHMARK RESULTS':^70}")
    print(f"{'=' * 70}")
    print(f"Question: {args.question}")
    print(f"Video: {args.video}")
    print(f"{'-' * 70}")

    headers = ["Metric"]
    for s in args.systems:
        headers.append(s.upper())
    header_fmt = "{:<25}" + "{:>20}" * len(args.systems)
    print(header_fmt.format(*headers))
    print("-" * 70)

    metrics = [
        ("elapsed_s", "Wall time (s)"),
        ("input_tokens", "Input tokens"),
        ("output_tokens", "Output tokens"),
        ("total_tokens", "Total tokens"),
        ("tool_calls", "Tool calls"),
        ("iterations", "REPL iterations"),
        ("turns", "Agent turns"),
        ("cache_read_tokens", "Cache read tokens"),
        ("cache_creation_tokens", "Cache creation tokens"),
        ("index_elapsed_s", "Index time (s)"),
    ]

    for key, label in metrics:
        vals = []
        for s in args.systems:
            v = results.get(s, {}).get(key, "-")
            if isinstance(v, float):
                v = f"{v:.1f}"
            elif v == 0 and key in ("cache_read_tokens", "cache_creation_tokens"):
                v = "-"
            vals.append(str(v))
        # Skip rows where all values are "-"
        if all(v == "-" for v in vals):
            continue
        print(header_fmt.format(label, *vals))

    # Print answers
    for s in args.systems:
        r = results.get(s, {})
        answer = r.get("answer", "")
        if answer:
            print(f"\n{'-' * 70}")
            print(f"{s.upper()} Answer:")
            print(answer[:500])
            if len(answer) > 500:
                print(f"... [{len(answer) - 500} more chars]")

    # Save JSON
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
