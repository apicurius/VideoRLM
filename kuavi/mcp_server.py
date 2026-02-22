"""KUAVi MCP server — exposes video analysis tools over stdio.

Usage:
    uv run python -m kuavi.mcp_server

Register with Claude Code:
    claude mcp add --transport stdio kuavi -- uv run python -m kuavi.mcp_server
"""

from __future__ import annotations

import base64
import json
import logging
import os
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

# Load .env from the project root (or CLAUDE_PROJECT_DIR) so API keys are available
_project_dir = os.environ.get("CLAUDE_PROJECT_DIR", "")
_dotenv_path = os.path.join(_project_dir, ".env") if _project_dir else ".env"
load_dotenv(_dotenv_path, override=False)

logger = logging.getLogger(__name__)

mcp = FastMCP("kuavi", json_response=True)


# ---------------------------------------------------------------------------
# Server-side trace logger
# ---------------------------------------------------------------------------


_TURN_BOUNDARY_SECONDS = 3.0  # Gap between tool calls that signals a new agent turn


class _TraceLogger:
    """Writes KUAVi JSONL trace files matching the visualizer's KUAViEvent schema.

    Every MCP tool call is logged regardless of caller (parent, subagent, teammate).
    A new trace file is created on each ``kuavi_index_video`` call.
    """

    # Tool names whose responses contain frame data eligible for sidecar saving
    _FRAME_TOOLS = frozenset({
        "extract_frames", "zoom_frames", "crop_frame", "blend_frames", "diff_frames",
    })

    def __init__(self) -> None:
        # Use absolute path so subagent MCP servers write to the same logs dir.
        # Resolve relative to CLAUDE_PROJECT_DIR (set by Claude Code) if available.
        default_log_dir = os.environ.get("KUAVI_LOG_DIR", "")
        if not default_log_dir:
            project_dir = os.environ.get("CLAUDE_PROJECT_DIR", "")
            if project_dir:
                default_log_dir = os.path.join(project_dir, "logs")
            else:
                default_log_dir = "./logs"
        self._log_dir = Path(default_log_dir).resolve()
        self._log_file: Path | None = None
        self._session_started = False
        self._run_counter = 0
        self._last_tool_call_time: float = 0.0
        self._turn_counter: int = 0
        self._frames_dir: Path | None = None
        self._frame_counter: int = 0
        self._current_eval_id: str | None = None

    def _ensure_log_dir(self) -> None:
        self._log_dir.mkdir(parents=True, exist_ok=True)

    def _ensure_frames_dir(self) -> Path:
        """Create and return the frames sidecar directory for the current log file."""
        if self._frames_dir is None:
            log_path = self._current_file()
            self._frames_dir = log_path.with_suffix("").with_name(log_path.stem + ".frames")
        self._frames_dir.mkdir(parents=True, exist_ok=True)
        return self._frames_dir

    def _save_frames(self, obj: Any) -> Any:
        """Save base64 frame data to sidecar files, returning references.

        Walks the response structure looking for dicts with 'data' + 'mime_type'
        keys (frame dicts). Each frame's base64 data is decoded and written to
        ``{log_stem}.frames/frame_NNNN.jpg``. The dict is replaced with a
        ``_frame_file`` reference that the visualizer's ``VideoFrameViewer``
        already knows how to resolve via the ``/api/frames/`` route.
        """
        if isinstance(obj, list):
            return [self._save_frames(item) for item in obj]
        if isinstance(obj, dict):
            # Check if this dict is a frame (has base64 'data' and 'mime_type')
            if "data" in obj and "mime_type" in obj and isinstance(obj["data"], str) and len(obj["data"]) > 200:
                try:
                    frames_dir = self._ensure_frames_dir()
                    self._frame_counter += 1
                    ext = ".jpg" if "jpeg" in obj["mime_type"] else ".png"
                    fname = f"frame_{self._frame_counter:04d}{ext}"
                    fpath = frames_dir / fname
                    fpath.write_bytes(base64.b64decode(obj["data"]))
                    # Return a reference dict instead of inline base64
                    result = {k: v for k, v in obj.items() if k != "data"}
                    result["_frame_file"] = fname
                    return result
                except Exception:
                    # Fall back to stripping base64 if save fails
                    return self._strip_base64(obj)
            # Recurse into nested dicts (e.g. crop_frame returns {"image": {...}, "crop": {...}})
            return {k: self._save_frames(v) for k, v in obj.items()}
        return obj

    def _new_trace_file(self) -> Path:
        """Create a new trace file with a timestamp-based name.

        Also publishes the path to ``logs/.kuavi_mcp_trace`` so the Claude Code
        hook logger can append conversation events (question, final_answer,
        agent lifecycle) to the **same** file instead of creating a separate one.
        """
        self._ensure_log_dir()
        self._run_counter += 1
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        suffix = f"_r{self._run_counter}" if self._run_counter > 1 else ""
        path = self._log_dir / f"kuavi_{ts}{suffix}_mcp.jsonl"
        self._log_file = path
        self._session_started = False
        self._last_tool_call_time = 0.0
        self._turn_counter = 0
        self._frames_dir = None
        self._frame_counter = 0
        # Publish trace path for hook coordination
        self._publish_trace_path(path)
        return path

    def _publish_trace_path(self, path: Path) -> None:
        """Write the current trace file path to a well-known state file.

        The hook logger (``kuavi_trace_logger.sh``) reads this to append
        conversation-level events (question, agent lifecycle, final_answer)
        to the same trace file, achieving a single unified trace.
        """
        try:
            state_file = self._log_dir / ".kuavi_mcp_trace"
            state_file.write_text(str(path))
        except OSError:
            logger.warning("Failed to publish trace path to .kuavi_mcp_trace")

    def _current_file(self) -> Path:
        """Get the current trace file, creating one if needed."""
        if self._log_file is None:
            return self._new_trace_file()
        return self._log_file

    def _write_event(self, event: dict[str, Any]) -> None:
        """Append a JSON line to the current trace file."""
        path = self._current_file()
        try:
            with open(path, "a") as f:
                f.write(json.dumps(event, default=str) + "\n")
        except OSError:
            logger.warning("Failed to write trace event to %s", path)

    def _emit_session_start(self) -> None:
        """Write a session_start event on first tool call."""
        if not self._session_started:
            self._write_event({
                "type": "session_start",
                "timestamp": datetime.now(UTC).isoformat(timespec="milliseconds"),
                "source": "mcp-server",
            })
            self._session_started = True

    @staticmethod
    def _strip_base64(obj: Any, max_str_len: int = 200) -> Any:
        """Recursively strip large base64 data fields to keep log files small."""
        if isinstance(obj, dict):
            result = {}
            for k, v in obj.items():
                if k == "data" and isinstance(v, str) and len(v) > max_str_len:
                    result[k] = f"<base64:{len(v)} chars>"
                else:
                    result[k] = _TraceLogger._strip_base64(v, max_str_len)
            return result
        if isinstance(obj, list):
            return [_TraceLogger._strip_base64(item, max_str_len) for item in obj]
        return obj

    @staticmethod
    def _approx_tokens(obj: Any) -> int:
        """Approximate token count from object size (1 token ≈ 4 chars of JSON)."""
        try:
            return max(1, len(json.dumps(obj, default=str)) // 4)
        except Exception:
            return 0

    @staticmethod
    def _summarize_response(tool_name: str, tool_response: Any, has_error: bool) -> str:
        """Generate a concise human-readable summary of a tool response."""
        if has_error:
            err_str = (
                tool_response if isinstance(tool_response, str) else json.dumps(tool_response, default=str)
            )
            return f"Error: {err_str[:80]}"

        # Unwrap MCP result wrapper
        resp = tool_response
        if isinstance(resp, dict) and "result" in resp:
            resp = resp["result"]

        if "index_video" in tool_name or "load_index" in tool_name:
            if isinstance(resp, dict):
                segs = resp.get("segments", "?")
                scenes = resp.get("scenes", resp.get("scene_boundaries", "?"))
                dur = resp.get("duration", "?")
                dur_str = f"{dur:.1f}s" if isinstance(dur, float) else f"{dur}s"
                return f"indexed: {segs} segs, {scenes} scenes, {dur_str}"

        if "search_video" in tool_name or "search_transcript" in tool_name or "get_scene_list" in tool_name:
            if isinstance(resp, list):
                n = len(resp)
                return f"{n} result{'s' if n != 1 else ''}"

        if "discriminative_vqa" in tool_name:
            if isinstance(resp, list) and resp:
                top = resp[0]
                if isinstance(top, dict) and "answer" in top:
                    return f"top: {str(top['answer'])[:40]}"
            if isinstance(resp, list):
                n = len(resp)
                return f"{n} candidate{'s' if n != 1 else ''}"

        if "extract_frames" in tool_name or "zoom_frames" in tool_name:
            if isinstance(resp, list):
                n = len(resp)
                return f"{n} frame{'s' if n != 1 else ''}"

        if "get_transcript" in tool_name:
            if isinstance(resp, str):
                preview = resp[:60].replace("\n", " ")
                return preview + ("…" if len(resp) > 60 else "")

        if "analyze_shards" in tool_name:
            if isinstance(resp, dict):
                count = resp.get("shard_count", "?")
                return f"{count} shards analyzed"

        if "get_session_stats" in tool_name:
            if isinstance(resp, dict):
                calls = resp.get("tool_calls", "?")
                return f"{calls} tool calls total"

        if "eval" in tool_name:
            if isinstance(resp, dict):
                stdout = str(resp.get("stdout", "")).strip()
                if stdout:
                    return stdout.split("\n")[0][:60]
                return "executed"

        if isinstance(resp, dict):
            status = resp.get("status")
            if status:
                return str(status)

        resp_str = json.dumps(resp, default=str)
        return resp_str[:60] + ("…" if len(resp_str) > 60 else "")

    def log_tool_call(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        tool_response: Any,
        duration_ms: int,
        has_error: bool,
    ) -> None:
        """Log a tool_call event."""
        # Start a new trace file for each index_video call
        if "index_video" in tool_name:
            self._new_trace_file()

        self._emit_session_start()

        # Detect agent turn boundary: gap > _TURN_BOUNDARY_SECONDS since last tool call
        now = time.time()
        if self._last_tool_call_time > 0:
            gap = now - self._last_tool_call_time
            if gap > _TURN_BOUNDARY_SECONDS:
                self._turn_counter += 1
                self._write_event({
                    "type": "turn_start",
                    "timestamp": datetime.now(UTC).isoformat(timespec="milliseconds"),
                    "turn": self._turn_counter,
                    "gap_seconds": round(gap, 1),
                })
        self._last_tool_call_time = now

        # Approximate token usage from response size
        response_tokens = self._approx_tokens(tool_response)
        input_tokens = self._approx_tokens(tool_input)

        # Compute a human-readable response summary
        response_summary = self._summarize_response(tool_name, tool_response, has_error)

        # For frame-producing tools, save frames to sidecar files instead of stripping base64
        short_name = (
            tool_name
            .replace("mcp__kuavi__kuavi_", "")
            .replace("mcp__kuavi__", "")
            .replace("kuavi_", "")
        )
        if short_name in self._FRAME_TOOLS and not has_error:
            logged_response = self._save_frames(tool_response)
        else:
            logged_response = self._strip_base64(tool_response)

        self._write_event({
            "type": "tool_call",
            "timestamp": datetime.now(UTC).isoformat(timespec="milliseconds"),
            "tool_name": tool_name,
            "tool_input": self._strip_base64(tool_input),
            "tool_response": logged_response,
            "response_summary": response_summary,
            "duration_ms": duration_ms,
            "has_error": has_error,
            "token_usage": {
                "input_tokens_approx": input_tokens,
                "output_tokens_approx": response_tokens,
            },
        })

        # After index_video tool_call: extract and emit metadata from the response.
        # This must happen here (not in kuavi_index_video) because _new_trace_file()
        # is called at the top of this method, so the new trace file is ready.
        if "index_video" in tool_name and not has_error:
            resp = tool_response
            # MCP responses can be [text_content, dict] or {result: dict} or plain dict
            if isinstance(resp, list):
                for item in resp:
                    if isinstance(item, dict) and "status" in item:
                        resp = item
                        break
            if isinstance(resp, dict) and "result" in resp:
                resp = resp["result"]
            if isinstance(resp, dict):
                self.log_video_metadata(
                    video_path=str(resp.get("video_path", "")),
                    fps=float(resp.get("fps", 0)),
                    duration=float(resp.get("duration", 0)),
                    num_segments=int(resp.get("segments", 0)),
                    num_scenes=int(resp.get("scenes", resp.get("scene_boundaries", 0))),
                    has_embeddings=bool(resp.get("has_embeddings", False)),
                    has_transcript=bool(resp.get("transcript_entries", 0) > 0),
                )

    def log_video_metadata(
        self,
        video_path: str,
        fps: float,
        duration: float,
        num_segments: int,
        num_scenes: int,
        has_embeddings: bool,
        has_transcript: bool,
    ) -> None:
        """Log a metadata event after video indexing (matches RLM's metadata format style)."""
        self._write_event({
            "type": "metadata",
            "timestamp": datetime.now(UTC).isoformat(timespec="milliseconds"),
            "video_path": video_path,
            "fps": fps,
            "duration": duration,
            "num_segments": num_segments,
            "num_scenes": num_scenes,
            "has_embeddings": has_embeddings,
            "has_transcript": has_transcript,
        })

    def log_llm_call(
        self,
        prompt: str,
        model: str,
        backend: str,
        response: str,
        duration_ms: int,
        has_error: bool = False,
        context: str | None = None,
        num_frames: int = 0,
    ) -> None:
        """Log an llm_call event (emitted when kuavi_eval or kuavi_analyze_shards calls an LLM)."""
        self._emit_session_start()

        # Summarise long prompts to keep log files small (prompts can be huge with multimodal)
        prompt_summary = prompt[:300] + "..." if len(prompt) > 300 else prompt
        # Keep both a truncated summary for quick display and the full response for expand
        response_summary = response[:300] + "..." if len(response) > 300 else response

        event: dict[str, Any] = {
            "type": "llm_call",
            "timestamp": datetime.now(UTC).isoformat(timespec="milliseconds"),
            "model": model,
            "backend": backend,
            "prompt_summary": prompt_summary,
            "prompt_tokens_approx": max(1, len(prompt) // 4),
            "response_summary": response_summary,
            "response_tokens_approx": max(1, len(response) // 4),
            "duration_ms": duration_ms,
            "has_error": has_error,
            "context": context,
        }
        # Include full response when it was truncated
        if len(response) > 300:
            event["response_full"] = response
        # Include frame count for multimodal prompts
        if num_frames > 0:
            event["num_frames"] = num_frames
        # Link to parent eval execution if one is active
        if self._current_eval_id:
            event["eval_id"] = self._current_eval_id
        self._write_event(event)

    def log_eval_execution(
        self,
        code: str,
        stdout: str,
        execution_time_ms: int,
        has_error: bool,
        result_type: str | None = None,
        eval_id: str | None = None,
    ) -> None:
        """Log an eval_execution event for kuavi_eval calls."""
        self._emit_session_start()

        event: dict[str, Any] = {
            "type": "eval_execution",
            "timestamp": datetime.now(UTC).isoformat(timespec="milliseconds"),
            "code": code,
            "stdout": stdout,
            "execution_time_ms": execution_time_ms,
            "has_error": has_error,
            "result_type": result_type,
        }
        if eval_id:
            event["eval_id"] = eval_id
        self._write_event(event)

    def log_session_end(self, reason: str = "shutdown") -> None:
        """Write a session_end event."""
        if self._log_file is not None and self._session_started:
            self._write_event({
                "type": "session_end",
                "timestamp": datetime.now(UTC).isoformat(timespec="milliseconds"),
                "reason": reason,
            })


_trace_logger = _TraceLogger()


def _install_trace_logging() -> None:
    """Monkey-patch the ToolManager.call_tool to log every tool call."""
    tm = mcp._tool_manager  # noqa: SLF001
    _orig_call_tool = tm.call_tool

    async def _logged_call_tool(
        name: str,
        arguments: dict[str, Any],
        context: Any = None,
        convert_result: bool = False,
    ) -> Any:
        t0 = time.time()
        result: Any = None
        has_error = False
        try:
            result = await _orig_call_tool(
                name, arguments, context=context, convert_result=convert_result
            )
            return result
        except Exception as exc:
            has_error = True
            result = {"error": f"{type(exc).__name__}: {exc}"}
            raise
        finally:
            duration_ms = int((time.time() - t0) * 1000)
            _trace_logger.log_tool_call(name, arguments, result, duration_ms, has_error)

    tm.call_tool = _logged_call_tool


_install_trace_logging()

# ---------------------------------------------------------------------------
# Global state: multi-video session
# ---------------------------------------------------------------------------
_state: dict[str, Any] = {
    "videos": {},  # video_id -> {"index", "indexer", "loaded_video", "video_path"}
    "active_video": None,  # current active video_id
    "eval_namespace": None,  # reserved for eval tool
    "llm_config": None,  # dict with primary/secondary backend+model routing, or None
    "stats": {
        "tool_calls": 0,
        "frames_extracted": 0,
        "searches_performed": 0,
        "session_start": None,
        "tokens_used": 0,
    },
    "last_frames": [],  # cache of last extract_frames/zoom_frames result for reference by index
    "llm_clients": {},  # cached LLM clients per backend (avoids per-call reconnection)
    "budget": {
        "max_tool_calls": 50,
        "warn_tool_calls": 35,
        "max_elapsed_seconds": 300,
        "warn_elapsed_seconds": 200,
        "exceeded": False,
        "max_tokens": None,   # None = no token limit
        "warn_tokens": None,  # None = no token warning
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_video_entry(video_id: str | None = None) -> dict[str, Any] | None:
    """Get video entry by ID, or active video if None."""
    vid = video_id or _state["active_video"]
    if vid is None:
        return None
    return _state["videos"].get(vid)


def _get_active_index(video_id: str | None = None):
    """Get the VideoIndex for a video, or the active video."""
    entry = _get_video_entry(video_id)
    return entry["index"] if entry else None


def _get_active_video_path(video_id: str | None = None) -> str | None:
    """Get the video file path for a video, or the active video."""
    entry = _get_video_entry(video_id)
    return entry["video_path"] if entry else None


def _resolve_image(image: Any) -> dict[str, Any]:
    """Resolve an image argument that may be a dict, index, or string index.

    Models often pass ``"0"`` or ``0`` to reference a frame from the last
    extract_frames / zoom_frames call instead of the full base64 dict.
    This helper transparently resolves such references.
    """
    if isinstance(image, dict) and "data" in image:
        return image  # already a proper image dict

    # Try to resolve as an index into last_frames
    idx: int | None = None
    if isinstance(image, int):
        idx = image
    elif isinstance(image, str):
        try:
            idx = int(image)
        except ValueError:
            pass

    if idx is not None:
        frames = _state["last_frames"]
        if 0 <= idx < len(frames):
            return frames[idx]
        raise ValueError(
            f"Frame index {idx} out of range. Last extract_frames returned "
            f"{len(frames)} frame(s) (indices 0–{len(frames) - 1})."
        )

    raise ValueError(
        f"Invalid image argument: expected an image dict with 'data' and 'mime_type' keys, "
        f"or a frame index (0–N) referencing the last extract_frames result. Got: {type(image).__name__}"
    )


def _track_tool_call(tool_type: str) -> None:
    """Increment usage counters for session stats."""
    import time

    _state["stats"]["tool_calls"] += 1
    if _state["stats"]["session_start"] is None:
        _state["stats"]["session_start"] = time.time()
    if tool_type == "extract_frames":
        _state["stats"]["frames_extracted"] += 1
    elif tool_type == "search":
        _state["stats"]["searches_performed"] += 1


def _estimate_tokens(content: Any) -> int:
    """Estimate token count from content (~4 chars per token)."""
    try:
        text = json.dumps(content) if not isinstance(content, str) else content
        return max(1, len(text) // 4)
    except Exception:
        return 0


def _track_response_tokens(response: Any) -> None:
    """Add estimated token count from a tool response to session stats."""
    tokens = _estimate_tokens(response)
    _state["stats"]["tokens_used"] = _state["stats"].get("tokens_used", 0) + tokens


def _check_budget_gate() -> tuple[dict[str, str] | None, str | None]:
    """Check budget limits and return (error_or_none, warning_or_none).

    Returns a tuple:
      - First element: an error dict if the hard limit is exceeded, else None.
      - Second element: a warning string if in the warning zone, else None.

    Once exceeded, ``_state["budget"]["exceeded"]`` is set to True so all
    subsequent gated tool calls are immediately blocked.
    """
    import time

    stats = _state["stats"]
    budget = _state["budget"]

    # Already exceeded — fast path
    if budget["exceeded"]:
        elapsed = 0.0
        if stats["session_start"] is not None:
            elapsed = time.time() - stats["session_start"]
        return (
            {
                "error": "BUDGET EXCEEDED",
                "message": (
                    f"You have used {stats['tool_calls']}/{budget['max_tool_calls']} tool calls "
                    f"over {elapsed:.0f}s. Synthesize your answer NOW from evidence gathered so "
                    f"far. Call kuavi_get_session_stats for details."
                ),
            },
            None,
        )

    # Check hard limits
    over_calls = stats["tool_calls"] >= budget["max_tool_calls"]
    elapsed = 0.0
    over_time = False
    if stats["session_start"] is not None:
        elapsed = time.time() - stats["session_start"]
        over_time = elapsed >= budget["max_elapsed_seconds"]

    tokens_used = stats.get("tokens_used", 0)
    max_tokens = budget.get("max_tokens")
    over_tokens = max_tokens is not None and tokens_used >= max_tokens

    if over_calls or over_time or over_tokens:
        budget["exceeded"] = True
        reason = "token budget" if over_tokens else "tool call/time budget"
        tokens_msg = (
            f" Token usage: {tokens_used}/{max_tokens}." if max_tokens is not None else ""
        )
        return (
            {
                "error": "BUDGET EXCEEDED",
                "message": (
                    f"You have exceeded the {reason}. "
                    f"Used {stats['tool_calls']}/{budget['max_tool_calls']} tool calls "
                    f"over {elapsed:.0f}s.{tokens_msg} Synthesize your answer NOW from evidence "
                    f"gathered so far. Call kuavi_get_session_stats for details."
                ),
            },
            None,
        )

    # Check warning zone
    warn_calls = stats["tool_calls"] >= budget["warn_tool_calls"]
    warn_time = False
    if stats["session_start"] is not None:
        warn_time = elapsed >= budget["warn_elapsed_seconds"]
    warn_tokens_val = budget.get("warn_tokens")
    warn_tokens = warn_tokens_val is not None and tokens_used >= warn_tokens_val

    if warn_calls or warn_time or warn_tokens:
        remaining_calls = budget["max_tool_calls"] - stats["tool_calls"]
        remaining_time = budget["max_elapsed_seconds"] - elapsed
        token_info = ""
        if max_tokens is not None:
            remaining_tokens = max_tokens - tokens_used
            token_info = f" Tokens: {tokens_used}/{max_tokens} used, {remaining_tokens} remaining."
        return (
            None,
            (
                f"Approaching budget limit: {stats['tool_calls']}/{budget['max_tool_calls']} "
                f"tool calls used, {remaining_calls} remaining. "
                f"{elapsed:.0f}s/{budget['max_elapsed_seconds']:.0f}s elapsed, "
                f"{remaining_time:.0f}s remaining.{token_info} Start wrapping up."
            ),
        )

    return (None, None)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@mcp.tool()
def kuavi_index_video(
    video_path: str,
    fps: float = 1.0,
    auto_fps: bool = False,
    target_frames: int = 120,
    embedding_model: str = "google/siglip2-base-patch16-256",
    scene_model: str | None = "facebook/vjepa2-vitl-fpc64-256",
    text_embedding_model: str | None = None,
    asr_model: str = "Qwen/Qwen3-ASR-1.7B",
    cache_dir: str | None = None,
    no_scene_model: bool = False,
    no_text_embedding: bool = False,
    no_caption: bool = False,
    caption_resize_width: int | None = None,
    caption_resize_height: int | None = None,
    transcript_path: str | None = None,
    auto_shard_segments: int | None = None,
    num_segments: int | None = None,
    scene_model_preset: str | None = None,
    mode: str = "full",
    caption_preset: str | None = None,
    store_feature_maps: bool = False,
    overlapping_vjepa: bool = False,
    semantic_dedup: bool = False,
) -> dict[str, Any]:
    """Index a video file for search and analysis.

    Must be called before using search or analysis tools.
    Performs scene detection, frame embedding, and optional ASR transcription.

    When num_segments is set, the video is split into that many equal-duration
    temporal segments instead of using V-JEPA 2 scene detection.

    When mode is set to "fast", the indexer skips Tree-of-Captions and
    Self-Refine to produce a quickly searchable index using only midpoint
    frame captions. Use mode="full" (default) for the richest annotations.
    """
    import cv2

    from kuavi.indexer import VideoIndexer
    from kuavi.loader import VideoLoader

    if no_scene_model:
        scene_model_resolved = None
    elif scene_model_preset is not None:
        scene_model_resolved = None  # preset overrides scene_model directly in VideoIndexer
    else:
        scene_model_resolved = scene_model

    if no_text_embedding:
        text_embedding_model_resolved = None
    else:
        text_embedding_model_resolved = text_embedding_model

    # Auto-FPS computation
    actual_fps = fps
    if auto_fps:
        cap = cv2.VideoCapture(video_path)
        try:
            video_fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            if video_fps > 0 and frame_count > 0:
                duration = frame_count / video_fps
                optimal_fps = target_frames / duration
                actual_fps = max(0.1, min(5.0, optimal_fps))
        finally:
            cap.release()

    loader = VideoLoader(fps=actual_fps)
    if num_segments is not None:
        loaded_video = loader.load_and_segment(video_path, num_segments=num_segments)
    else:
        loaded_video = loader.load(video_path)

    # Resolve caption_resize from width/height params
    caption_resize: tuple[int, int] | None = None
    if caption_resize_width is not None and caption_resize_height is not None:
        caption_resize = (caption_resize_width, caption_resize_height)

    indexer = VideoIndexer(
        embedding_model=embedding_model,
        cache_dir=cache_dir,
        text_embedding_model=text_embedding_model_resolved,
        scene_model=scene_model_resolved,
        caption_resize=caption_resize,
        scene_model_preset=scene_model_preset,
    )

    # Build captioning functions when captioning is enabled
    caption_fn = None
    frame_caption_fn = None
    refine_fn = None
    if not no_caption:
        try:
            gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            if caption_preset is not None:
                from kuavi.captioners import create_captioner

                captioner, aggregator = create_captioner(caption_preset, api_key=gemini_key)
                caption_fn = captioner.caption_segment
                frame_caption_fn = captioner.caption_frame
                if aggregator is not None:
                    refine_fn = aggregator.refine
            else:
                from kuavi.captioning import (
                    make_gemini_caption_fn,
                    make_gemini_frame_caption_fn,
                    make_gemini_refine_fn,
                )

                caption_fn = make_gemini_caption_fn(api_key=gemini_key)
                frame_caption_fn = make_gemini_frame_caption_fn(api_key=gemini_key)
                refine_fn = make_gemini_refine_fn(api_key=gemini_key)
        except Exception:
            logger.warning("Failed to initialize captioning; indexing without captions.")

    index = indexer.index_video(
        loaded_video,
        caption_fn=caption_fn,
        frame_caption_fn=frame_caption_fn,
        refine_fn=refine_fn,
        asr_model=asr_model,
        transcript_path=transcript_path,
        mode=mode,
        store_feature_maps=store_feature_maps,
        overlapping_vjepa=overlapping_vjepa,
        semantic_dedup=semantic_dedup,
    )

    video_id = Path(video_path).stem
    _state["videos"][video_id] = {
        "index": index,
        "indexer": indexer,
        "loaded_video": loaded_video,
        "video_path": video_path,
    }
    _state["active_video"] = video_id
    _track_tool_call("index")

    num_segments = len(index.segments)

    result: dict[str, Any] = {
        "status": "indexed",
        "video_id": video_id,
        "video_path": video_path,
        "duration": round(loaded_video.metadata.duration, 2),
        "fps": round(actual_fps, 3),
        "frames_extracted": loaded_video.metadata.extracted_frame_count,
        "segments": num_segments,
        "scenes": len(index.scene_boundaries),
        "transcript_entries": len(index.transcript),
        "has_embeddings": index.embeddings is not None,
        "has_frame_embeddings": index.frame_embeddings is not None,
        "has_temporal_embeddings": index.temporal_embeddings is not None,
    }

    # Auto-sharding: if segment count exceeds threshold, recommend shard analysis
    shard_threshold = auto_shard_segments or 30
    if num_segments > shard_threshold:
        result["auto_shard_recommended"] = True
        result["auto_shard_reason"] = (
            f"Video has {num_segments} segments (threshold: {shard_threshold}). "
            f"Consider calling kuavi_analyze_shards() for efficient long-video analysis."
        )

    # Note: metadata event is now emitted by log_tool_call() after the tool_call event,
    # ensuring it goes into the correct (new) trace file.

    return result


@mcp.tool()
def kuavi_search_video(
    query: str,
    top_k: int = 5,
    field: str = "summary",
    diverse: bool = True,
    cluster_diverse: bool = False,
    exclude_non_action: bool = True,
    level: int = 0,
    video_id: str | None = None,
) -> list[dict[str, Any]]:
    """Semantic search over indexed video segments.

    Fields: "summary" (visual descriptions), "action" (activities),
    "visual" (frame embeddings), "temporal" (V-JEPA 2 motion dynamics),
    "all" (weighted composite of all fields).
    Level 0 = fine-grained, level 1+ = coarser hierarchy.
    """
    index = _get_active_index(video_id)
    if index is None:
        return [{"error": "No video indexed. Call kuavi_index_video first."}]

    _track_tool_call("search")
    gate, warning = _check_budget_gate()
    if gate is not None:
        return [gate]

    from kuavi.search import make_search_video

    tool = make_search_video(index)
    results = tool["tool"](
        query=query,
        top_k=top_k,
        field=field,
        diverse=diverse,
        cluster_diverse=cluster_diverse,
        exclude_non_action=exclude_non_action,
        level=level,
    )
    _track_response_tokens(results)
    if warning and results:
        results[-1]["_budget_warning"] = warning
    return results


@mcp.tool()
def kuavi_search_transcript(
    query: str,
    video_id: str | None = None,
) -> list[dict[str, Any]]:
    """Keyword search over ASR transcript (case-insensitive)."""
    index = _get_active_index(video_id)
    if index is None:
        return [{"error": "No video indexed. Call kuavi_index_video first."}]

    _track_tool_call("search")
    gate, warning = _check_budget_gate()
    if gate is not None:
        return [gate]

    from kuavi.search import make_search_transcript

    tool = make_search_transcript(index)
    results = tool["tool"](query=query)
    _track_response_tokens(results)
    if warning and results:
        results[-1]["_budget_warning"] = warning
    return results


@mcp.tool()
def kuavi_get_transcript(
    start_time: float,
    end_time: float,
    video_id: str | None = None,
) -> str:
    """Get transcript text for a specific time range (seconds)."""
    index = _get_active_index(video_id)
    if index is None:
        return "No video indexed. Call kuavi_index_video first."

    _track_tool_call("transcript")
    gate, warning = _check_budget_gate()
    if gate is not None:
        return gate["message"]

    from kuavi.search import make_get_transcript

    tool = make_get_transcript(index)
    result = tool["tool"](start_time=start_time, end_time=end_time)
    _track_response_tokens(result)
    if warning:
        result = result + f"\n\n[WARNING: {warning}]"
    return result


@mcp.tool()
def kuavi_get_scene_list(video_id: str | None = None) -> list[dict[str, Any]]:
    """List all detected scenes with annotations."""
    index = _get_active_index(video_id)
    if index is None:
        return [{"error": "No video indexed. Call kuavi_index_video first."}]

    _track_tool_call("scene_list")
    gate, warning = _check_budget_gate()
    if gate is not None:
        return [gate]

    from kuavi.search import make_get_scene_list

    tool = make_get_scene_list(index)
    results = tool["tool"]()
    _track_response_tokens(results)
    if warning and results:
        results[-1]["_budget_warning"] = warning
    return results


@mcp.tool()
def kuavi_discriminative_vqa(
    question: str,
    candidates: list[str],
    start_time: float | None = None,
    end_time: float | None = None,
    video_id: str | None = None,
) -> list[dict[str, Any]]:
    """Embedding-based multiple-choice VQA without LLM generation.

    Ranks candidate answers by cosine similarity to video segment embeddings.
    Optionally filter by time range.
    """
    index = _get_active_index(video_id)
    if index is None:
        return [{"error": "No video indexed. Call kuavi_index_video first."}]

    _track_tool_call("vqa")
    gate, warning = _check_budget_gate()
    if gate is not None:
        return [gate]

    from kuavi.search import make_discriminative_vqa

    tool = make_discriminative_vqa(index)
    time_range = (start_time, end_time) if start_time is not None and end_time is not None else None
    results = tool["tool"](question=question, candidates=candidates, time_range=time_range)
    _track_response_tokens(results)
    if warning and results:
        results[-1]["_budget_warning"] = warning
    return results


@mcp.tool()
def kuavi_anticipate_action(
    time_point: float,
    top_k: int = 3,
    candidates: list[str] | None = None,
    video_id: str | None = None,
) -> dict[str, Any]:
    """Predict what happens next after a given time point.

    Uses V-JEPA 2 predictor when available, falls back to embedding
    similarity for anticipation.
    """
    entry = _get_video_entry(video_id)
    if entry is None:
        return {"error": "No video indexed. Call kuavi_index_video first."}

    _track_tool_call("anticipate")
    gate, warning = _check_budget_gate()
    if gate is not None:
        return gate

    idx = entry["index"]
    tools = entry.get("tools", {})

    if "anticipate_action" not in tools:
        from kuavi.search import make_anticipate_action

        tools["anticipate_action"] = make_anticipate_action(idx)
        entry["tools"] = tools

    result = tools["anticipate_action"]["tool"](
        time_point=time_point,
        top_k=top_k,
        candidates=candidates,
    )
    _track_response_tokens(result)
    if warning and isinstance(result, dict):
        result["_budget_warning"] = warning
    return result


@mcp.tool()
def kuavi_classify_segment(
    task: str = "k400",
    start_time: float | None = None,
    end_time: float | None = None,
    segment_index: int | None = None,
    top_k: int = 5,
    video_id: str | None = None,
) -> dict[str, Any]:
    """Classify a video segment using attentive probes on V-JEPA 2 features.

    Requires store_feature_maps=True during indexing.
    task options: ssv2, k400, diving48, jester, coin, imagenet.
    Provide either segment_index or (start_time, end_time).
    """
    entry = _get_video_entry(video_id)
    if entry is None:
        return {"error": "No video indexed. Call kuavi_index_video first."}

    _track_tool_call("classify")
    gate, warning = _check_budget_gate()
    if gate is not None:
        return gate

    idx = entry["index"]
    tools = entry.get("tools", {})

    if "classify_segment" not in tools:
        from kuavi.search import make_classify_segment

        tools["classify_segment"] = make_classify_segment(idx)
        entry["tools"] = tools

    result = tools["classify_segment"]["tool"](
        start_time=start_time,
        end_time=end_time,
        segment_index=segment_index,
        task=task,
        top_k=top_k,
    )
    _track_response_tokens(result)
    if warning and isinstance(result, dict):
        result["_budget_warning"] = warning
    return result


@mcp.tool()
def kuavi_extract_frames(
    start_time: float,
    end_time: float,
    fps: float = 2.0,
    width: int = 720,
    height: int = 540,
    max_frames: int = 10,
    video_id: str | None = None,
) -> list[dict[str, str]]:
    """Extract frames from the indexed video as base64 JPEG images.

    Returns a list of dicts with 'data' (base64) and 'mime_type' keys.
    """
    video_path = _get_active_video_path(video_id)
    if video_path is None:
        return [{"error": "No video indexed. Call kuavi_index_video first."}]

    _track_tool_call("extract_frames")
    gate, warning = _check_budget_gate()
    if gate is not None:
        return [gate]

    import cv2

    from kuavi.context import _encode_frame
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return [{"error": f"Cannot open video: {video_path}"}]

    try:
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / original_fps if original_fps > 0 else 0.0

        start_time = max(0.0, start_time)
        end_time = min(end_time, duration)
        if end_time <= start_time:
            return []

        interval = 1.0 / fps
        times = []
        t = start_time
        while t < end_time:
            times.append(t)
            t += interval

        if len(times) > max_frames:
            step = len(times) / max_frames
            times = [times[int(i * step)] for i in range(max_frames)]

        frames = []
        for t in times:
            frame_idx = int(t * original_fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.resize(frame, (width, height))
            encoded = _encode_frame(frame, format=".jpg", quality=85)
            frames.append({
                "data": encoded["data"],
                "mime_type": encoded["mime_type"],
                "timestamp": str(round(t, 2)),
            })

        _track_response_tokens(frames)
        if warning and frames:
            frames[-1]["_budget_warning"] = warning
        _state["last_frames"] = frames  # cache for pixel tool reference by index
        return frames
    finally:
        cap.release()


@mcp.tool()
def kuavi_get_index_info(video_id: str | None = None) -> dict[str, Any]:
    """Get metadata about the current video index."""
    index = _get_active_index(video_id)
    if index is None:
        return {"error": "No video indexed. Call kuavi_index_video first."}

    _track_tool_call("info")

    entry = _get_video_entry(video_id)
    loaded = entry["loaded_video"] if entry else None

    info: dict[str, Any] = {
        "video_path": _get_active_video_path(video_id),
        "segments": len(index.segments),
        "transcript_entries": len(index.transcript),
        "scene_boundaries": len(index.scene_boundaries),
        "has_embeddings": index.embeddings is not None,
        "has_action_embeddings": index.action_embeddings is not None,
        "has_frame_embeddings": index.frame_embeddings is not None,
        "has_temporal_embeddings": index.temporal_embeddings is not None,
        "hierarchy_levels": len(index.segment_hierarchy),
    }

    if loaded:
        info["duration"] = round(loaded.metadata.duration, 2)
        info["original_fps"] = round(loaded.metadata.original_fps, 2)
        info["extraction_fps"] = round(loaded.metadata.extraction_fps, 3)
        info["frames_extracted"] = loaded.metadata.extracted_frame_count
        info["resolution"] = f"{loaded.metadata.width}x{loaded.metadata.height}"

    if index.embedding_quality:
        info["embedding_quality"] = index.embedding_quality

    return info


@mcp.tool()
def kuavi_get_session_stats() -> dict[str, Any]:
    """Get usage statistics for the current MCP session."""
    import time

    _track_tool_call("stats")

    stats = _state["stats"]
    elapsed = None
    if stats["session_start"] is not None:
        elapsed = round(time.time() - stats["session_start"], 2)

    budget = _state["budget"]
    tokens_used = stats.get("tokens_used", 0)
    max_tokens = budget.get("max_tokens")
    budget_info: dict[str, Any] = {
        "max_tool_calls": budget["max_tool_calls"],
        "remaining_tool_calls": max(0, budget["max_tool_calls"] - stats["tool_calls"]),
        "max_elapsed_seconds": budget["max_elapsed_seconds"],
        "exceeded": budget["exceeded"],
        "tokens_used": tokens_used,
    }
    if max_tokens is not None:
        budget_info["max_tokens"] = max_tokens
        budget_info["remaining_tokens"] = max(0, max_tokens - tokens_used)
    return {
        "tool_calls": stats["tool_calls"],
        "frames_extracted": stats["frames_extracted"],
        "searches_performed": stats["searches_performed"],
        "elapsed_seconds": elapsed,
        "videos_loaded": len(_state["videos"]),
        "budget": budget_info,
    }


@mcp.tool()
def kuavi_set_budget(
    max_tool_calls: int = 50,
    warn_tool_calls: int = 35,
    max_elapsed_seconds: float = 300,
    warn_elapsed_seconds: float = 200,
    max_tokens: int | None = None,
    warn_tokens: int | None = None,
) -> dict[str, Any]:
    """Configure budget limits for the current session.

    The budget enforces tool-call, time, and token limits. Once exceeded, all
    gated tools return an error instead of results, forcing the agent to
    synthesize an answer from evidence already gathered.

    Args:
        max_tool_calls: Hard limit on total tool calls before blocking.
        warn_tool_calls: Soft warning threshold (results include a warning).
        max_elapsed_seconds: Hard time limit in seconds.
        warn_elapsed_seconds: Soft time warning threshold in seconds.
        max_tokens: Hard token limit (estimated from response sizes). None = no limit.
        warn_tokens: Soft token warning threshold. None = no warning.
    """
    budget = _state["budget"]
    budget["max_tool_calls"] = max_tool_calls
    budget["warn_tool_calls"] = warn_tool_calls
    budget["max_elapsed_seconds"] = max_elapsed_seconds
    budget["warn_elapsed_seconds"] = warn_elapsed_seconds
    budget["max_tokens"] = max_tokens
    budget["warn_tokens"] = warn_tokens
    # Reset exceeded flag when budget is reconfigured
    budget["exceeded"] = False

    stats = _state["stats"]
    tokens_used = stats.get("tokens_used", 0)
    result: dict[str, Any] = {
        "status": "budget_configured",
        "max_tool_calls": max_tool_calls,
        "warn_tool_calls": warn_tool_calls,
        "max_elapsed_seconds": max_elapsed_seconds,
        "warn_elapsed_seconds": warn_elapsed_seconds,
        "current_tool_calls": stats["tool_calls"],
        "remaining_tool_calls": max(0, max_tool_calls - stats["tool_calls"]),
    }
    if max_tokens is not None:
        result["max_tokens"] = max_tokens
        result["tokens_used"] = tokens_used
        result["remaining_tokens"] = max(0, max_tokens - tokens_used)
    return result


@mcp.tool()
def kuavi_zoom_frames(
    start_time: float,
    end_time: float,
    level: int = 1,
    video_id: str | None = None,
) -> list[dict[str, str]]:
    """Extract frames at preset zoom levels (1=overview, 2=detail, 3=high-res).

    Level 1: fps=1, 480x360, max 5 frames.
    Level 2: fps=2, 720x540, max 10 frames.
    Level 3: fps=4, 1280x960, max 10 frames.
    """
    _track_tool_call("extract_frames")
    gate, _warning = _check_budget_gate()
    if gate is not None:
        return [gate]

    presets = {
        1: {"fps": 1.0, "width": 480, "height": 360, "max_frames": 5},
        2: {"fps": 2.0, "width": 720, "height": 540, "max_frames": 10},
        3: {"fps": 4.0, "width": 1280, "height": 960, "max_frames": 10},
    }
    params = presets.get(level, presets[1])
    return kuavi_extract_frames(
        start_time=start_time,
        end_time=end_time,
        fps=params["fps"],
        width=params["width"],
        height=params["height"],
        max_frames=params["max_frames"],
        video_id=video_id,
    )


@mcp.tool()
def kuavi_load_index(
    index_dir: str,
    video_id: str | None = None,
    embedding_model: str = "google/siglip2-base-patch16-256",
) -> dict[str, Any]:
    """Load a previously saved .kuavi index directory.

    Re-attaches embedding functions from a fresh VideoIndexer so search works.
    """
    from kuavi.indexer import VideoIndex, VideoIndexer

    _track_tool_call("load_index")

    directory = Path(index_dir)
    if not directory.exists():
        return {"error": f"Index directory not found: {index_dir}"}

    index = VideoIndex.load(index_dir)

    # Re-attach embed_fn from a fresh indexer
    indexer = VideoIndexer(embedding_model=embedding_model)
    indexer._ensure_model()
    index.embed_fn = indexer._encode_query
    index.visual_embed_fn = indexer._encode_query_siglip

    vid = video_id or directory.stem
    _state["videos"][vid] = {
        "index": index,
        "indexer": indexer,
        "loaded_video": None,
        "video_path": None,
    }
    _state["active_video"] = vid

    return {
        "status": "loaded",
        "video_id": vid,
        "index_dir": str(directory),
        "segments": len(index.segments),
        "transcript_entries": len(index.transcript),
        "scene_boundaries": len(index.scene_boundaries),
        "has_embeddings": index.embeddings is not None,
        "has_frame_embeddings": index.frame_embeddings is not None,
        "has_temporal_embeddings": index.temporal_embeddings is not None,
    }


def _build_shard_prompts(
    shard_duration: float = 30.0,
    max_frames_per_segment: int = 3,
    frame_width: int = 480,
    frame_height: int = 360,
) -> list[list]:
    """Build multimodal shard prompts with text captions and actual keyframes.

    Each shard prompt is a list of strings and image dicts suitable for
    ``llm_query()`` or ``llm_query_batched()``.  This mirrors RLM's
    ``shard_prompts`` variable.
    """
    index = _get_active_index()
    if index is None:
        return []

    segments = index.segments
    if not segments:
        return []

    video_path = _get_active_video_path()
    all_starts = [s["start_time"] for s in segments]
    all_ends = [s["end_time"] for s in segments]
    video_start = min(all_starts)
    video_end = max(all_ends)

    # Build shards
    shards: list[list[dict]] = []
    t = video_start
    while t < video_end:
        shard_end = min(t + shard_duration, video_end)
        shard_segs = [
            s for s in segments if s["start_time"] >= t and s["start_time"] < shard_end
        ]
        if shard_segs:
            shards.append(shard_segs)
        t = shard_end

    total = len(shards)

    # Extract frames if video path is available
    cap = None
    original_fps = 0.0
    if video_path:
        try:
            import cv2

            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                original_fps = cap.get(cv2.CAP_PROP_FPS)
            else:
                cap = None
        except Exception:
            cap = None

    shard_prompts: list[list] = []
    try:
        for shard_idx, shard_segs in enumerate(shards):
            parts: list = [
                f"You are analyzing shard {shard_idx + 1}/{total} of a video "
                f"(time range {shard_segs[0]['start_time']:.1f}s - "
                f"{shard_segs[-1]['end_time']:.1f}s).\n"
            ]
            for s in shard_segs:
                ann = s.get("annotation", {})
                summary = ann.get("summary", {}).get("brief", s.get("caption", "No caption"))
                parts.append(
                    f"\n--- Segment [{s['start_time']:.1f}s - {s['end_time']:.1f}s]: "
                    f"{summary}"
                )
                # Extract keyframes for this segment
                if cap is not None and original_fps > 0:
                    from kuavi.context import _encode_frame

                    seg_start = s["start_time"]
                    seg_end = s["end_time"]
                    seg_dur = seg_end - seg_start
                    n_frames = min(max_frames_per_segment, max(1, int(seg_dur * 0.5)))
                    if n_frames == 1:
                        times = [(seg_start + seg_end) / 2]
                    else:
                        step = seg_dur / (n_frames + 1)
                        times = [seg_start + step * (j + 1) for j in range(n_frames)]
                    for t_frame in times:
                        frame_idx = int(t_frame * original_fps)
                        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                        ret, frame = cap.read()
                        if ret:
                            frame = cv2.resize(frame, (frame_width, frame_height))
                            encoded = _encode_frame(frame, format=".jpg", quality=75)
                            parts.append({
                                "data": encoded["data"],
                                "mime_type": encoded.get("mime_type", "image/jpeg"),
                            })
            shard_prompts.append(parts)
    finally:
        if cap is not None:
            cap.release()

    return shard_prompts


def _build_shard_info(shard_duration: float = 30.0) -> list[str]:
    """Build shard descriptions from the current index's segments."""
    index = _get_active_index()
    if index is None:
        return ["No video indexed."]

    segments = index.segments
    if not segments:
        return ["No segments in index."]

    all_starts = [s["start_time"] for s in segments]
    all_ends = [s["end_time"] for s in segments]
    video_start = min(all_starts)
    video_end = max(all_ends)

    descriptions: list[str] = []
    t = video_start
    shard_idx = 0
    while t < video_end:
        shard_end = min(t + shard_duration, video_end)
        shard_segs = [
            s for s in segments if s["start_time"] >= t and s["start_time"] < shard_end
        ]
        if shard_segs:
            seg_summaries = []
            for s in shard_segs:
                ann = s.get("annotation", {})
                brief = ann.get("summary", {}).get("brief", s.get("caption", "No caption"))
                seg_summaries.append(
                    f"  [{s['start_time']:.1f}s-{s['end_time']:.1f}s] {brief}"
                )
            desc = (
                f"Shard {shard_idx} ({t:.1f}s-{shard_end:.1f}s): "
                f"{len(shard_segs)} segments\n" + "\n".join(seg_summaries)
            )
            descriptions.append(desc)
        t = shard_end
        shard_idx += 1

    return descriptions


@mcp.tool()
def kuavi_eval(code: str) -> dict[str, Any]:
    """Execute Python code in a persistent namespace with KUAVi tools available.

    The namespace persists across calls, so variables set in one call are
    available in subsequent calls. Pre-populated with np, cv2, and all
    kuavi tool functions.
    """
    import contextlib
    import io

    _track_tool_call("eval")
    gate, warning = _check_budget_gate()
    if gate is not None:
        return gate

    # Lazily initialize the persistent namespace
    if _state["eval_namespace"] is None:
        import cv2
        import numpy as np

        _state["eval_namespace"] = {
            "np": np,
            "cv2": cv2,
            # All kuavi tool functions as short-name callables
            "search_video": kuavi_search_video,
            "search_transcript": kuavi_search_transcript,
            "get_transcript": kuavi_get_transcript,
            "get_scene_list": kuavi_get_scene_list,
            "discriminative_vqa": kuavi_discriminative_vqa,
            "extract_frames": kuavi_extract_frames,
            "get_index_info": kuavi_get_index_info,
            "crop_frame": kuavi_crop_frame,
            "diff_frames": kuavi_diff_frames,
            "blend_frames": kuavi_blend_frames,
            "threshold_frame": kuavi_threshold_frame,
            "frame_info": kuavi_frame_info,
            "zoom_frames": kuavi_zoom_frames,
            "get_session_stats": kuavi_get_session_stats,
            "set_budget": kuavi_set_budget,
            "llm_query": lambda prompt, backend="gemini", model="gemini-2.5-flash": _call_llm(
                prompt, backend, model, role="secondary", _log_context="kuavi_eval:llm_query"
            ),
            "llm_query_batched": _llm_query_batched,
            "SHOW_VARS": lambda: {
                k: type(v).__name__
                for k, v in _state["eval_namespace"].items()
                if not k.startswith("_")
            },
            "get_shard_info": lambda shard_duration=30.0: _build_shard_info(shard_duration),
            "build_shard_prompts": _build_shard_prompts,
        }

    # Snapshot protected tool refs for namespace protection
    _protected_keys = {
        k for k in _state["eval_namespace"] if not k.startswith("_")
    }
    _protected_refs = {k: _state["eval_namespace"][k] for k in _protected_keys}

    ns = _state["eval_namespace"]
    stdout_buf = io.StringIO()
    t0_eval = time.time()
    has_eval_error = False
    result = None

    # Generate a unique eval_id so LLM calls within this eval can be linked
    eval_id = uuid.uuid4().hex[:12]
    _trace_logger._current_eval_id = eval_id

    try:
        with contextlib.redirect_stdout(stdout_buf):
            try:
                result = eval(code, ns)
            except SyntaxError:
                exec(code, ns)
                result = None
        # Namespace protection: restore any overwritten tool functions
        for k, ref in _protected_refs.items():
            if ns.get(k) is not ref:
                ns[k] = ref
        ret = {"result": result, "stdout": stdout_buf.getvalue()}
        _track_response_tokens(ret)
        if warning:
            ret["_budget_warning"] = warning
        return ret
    except Exception as e:
        has_eval_error = True
        return {"error": f"{type(e).__name__}: {e}", "stdout": stdout_buf.getvalue()}
    finally:
        _trace_logger._current_eval_id = None
        exec_ms = int((time.time() - t0_eval) * 1000)
        _trace_logger.log_eval_execution(
            code=code,
            stdout=stdout_buf.getvalue(),
            execution_time_ms=exec_ms,
            has_error=has_eval_error,
            result_type=type(result).__name__ if result is not None else None,
            eval_id=eval_id,
        )


# ---------------------------------------------------------------------------
# Pixel manipulation tools
# ---------------------------------------------------------------------------


@mcp.tool()
def kuavi_crop_frame(
    image: dict[str, Any],
    x1_pct: float,
    y1_pct: float,
    x2_pct: float,
    y2_pct: float,
) -> dict[str, Any]:
    """Crop a region from an image using percentage coordinates (0.0-1.0)."""
    _track_tool_call("pixel")
    gate, warning = _check_budget_gate()
    if gate is not None:
        return gate

    from kuavi.context import _decode_frame, _encode_frame

    image = _resolve_image(image)
    frame = _decode_frame(image)
    h, w = frame.shape[:2]
    x1 = int(x1_pct * w)
    y1 = int(y1_pct * h)
    x2 = int(x2_pct * w)
    y2 = int(y2_pct * h)
    cropped = frame[y1:y2, x1:x2]
    return {
        "image": _encode_frame(cropped),
        "crop": {
            "x1_pct": x1_pct,
            "y1_pct": y1_pct,
            "x2_pct": x2_pct,
            "y2_pct": y2_pct,
            "width": cropped.shape[1],
            "height": cropped.shape[0],
        },
    }


@mcp.tool()
def kuavi_diff_frames(
    image_a: dict[str, Any],
    image_b: dict[str, Any],
) -> dict[str, Any]:
    """Compute absolute pixel difference between two images."""
    _track_tool_call("pixel")
    gate, warning = _check_budget_gate()
    if gate is not None:
        return gate

    import cv2

    from kuavi.context import _decode_frame, _encode_frame

    image_a = _resolve_image(image_a)
    image_b = _resolve_image(image_b)
    frame_a = _decode_frame(image_a)
    frame_b = _decode_frame(image_b)
    # Resize b to match a if needed
    if frame_a.shape != frame_b.shape:
        frame_b = cv2.resize(frame_b, (frame_a.shape[1], frame_a.shape[0]))
    diff = cv2.absdiff(frame_a, frame_b)
    mean_diff = float(diff.mean())
    max_diff = int(diff.max())
    # Changed pixels: any channel diff > 25
    changed = (diff > 25).any(axis=2) if diff.ndim == 3 else (diff > 25)
    changed_pct = float(changed.sum() / changed.size * 100)
    return {
        "image": _encode_frame(diff),
        "mean_diff": round(mean_diff, 2),
        "max_diff": max_diff,
        "changed_pct": round(changed_pct, 2),
    }


@mcp.tool()
def kuavi_blend_frames(images: list[dict[str, Any]]) -> dict[str, Any]:
    """Average multiple frames into a composite image."""
    _track_tool_call("pixel")
    gate, warning = _check_budget_gate()
    if gate is not None:
        return gate

    import cv2
    import numpy as np

    from kuavi.context import _decode_frame, _encode_frame

    if not images:
        return {"error": "No images provided"}
    images = [_resolve_image(img) for img in images]
    frames = [_decode_frame(img) for img in images]
    # Resize all to first frame's size
    target_shape = frames[0].shape[:2]
    for i in range(1, len(frames)):
        if frames[i].shape[:2] != target_shape:
            frames[i] = cv2.resize(frames[i], (target_shape[1], target_shape[0]))
    blended = np.mean(frames, axis=0).astype(np.uint8)
    return {
        "image": _encode_frame(blended),
        "frame_count": len(frames),
    }


@mcp.tool()
def kuavi_threshold_frame(
    image: dict[str, Any],
    value: int = 128,
    invert: bool = False,
) -> dict[str, Any]:
    """Apply binary threshold to produce a mask, then find contours."""
    _track_tool_call("pixel")
    gate, warning = _check_budget_gate()
    if gate is not None:
        return gate

    import cv2

    from kuavi.context import _decode_frame, _encode_frame

    image = _resolve_image(image)
    frame = _decode_frame(image)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    thresh_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    _, mask = cv2.threshold(gray, value, 255, thresh_type)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    total_pixels = mask.size
    white_pct = float((mask == 255).sum() / total_pixels * 100)
    contour_areas = sorted([float(cv2.contourArea(c)) for c in contours], reverse=True)
    # Encode mask as 3-channel for consistency
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    return {
        "image": _encode_frame(mask_bgr),
        "white_pct": round(white_pct, 2),
        "contour_count": len(contours),
        "contour_areas": contour_areas[:20],  # top 20
    }


@mcp.tool()
def kuavi_frame_info(image: dict[str, Any]) -> dict[str, Any]:
    """Get metadata and statistics for an image."""
    _track_tool_call("pixel")
    gate, warning = _check_budget_gate()
    if gate is not None:
        return gate

    import cv2

    from kuavi.context import _decode_frame

    image = _resolve_image(image)
    frame = _decode_frame(image)
    h, w = frame.shape[:2]
    channels = frame.shape[2] if frame.ndim == 3 else 1
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if channels == 3 else frame
    b_mean, g_mean, r_mean = (
        (float(frame[:, :, 0].mean()), float(frame[:, :, 1].mean()), float(frame[:, :, 2].mean()))
        if channels == 3
        else (float(gray.mean()), float(gray.mean()), float(gray.mean()))
    )
    return {
        "width": w,
        "height": h,
        "channels": channels,
        "brightness": {
            "mean": round(float(gray.mean()), 2),
            "std": round(float(gray.std()), 2),
            "min": int(gray.min()),
            "max": int(gray.max()),
        },
        "color_means": {
            "b": round(b_mean, 2),
            "g": round(g_mean, 2),
            "r": round(r_mean, 2),
        },
    }


# ---------------------------------------------------------------------------
# LLM config tool
# ---------------------------------------------------------------------------


@mcp.tool()
def kuavi_set_llm_config(
    primary_backend: str | None = None,
    primary_model: str | None = None,
    secondary_backend: str | None = None,
    secondary_model: str | None = None,
) -> dict[str, Any]:
    """Configure LLM routing for primary and secondary roles.

    - primary: used by kuavi_analyze_shards
    - secondary: used by kuavi_eval's llm_query / llm_query_batched

    Pass None (or omit) for any field to leave it unchanged / use the caller-supplied default.
    Call with no arguments to clear the routing config and restore default behaviour.
    """
    if all(v is None for v in (primary_backend, primary_model, secondary_backend, secondary_model)):
        _state["llm_config"] = None
        return {"status": "llm_config_cleared"}

    config: dict[str, str | None] = {
        "primary_backend": primary_backend,
        "primary_model": primary_model,
        "secondary_backend": secondary_backend,
        "secondary_model": secondary_model,
    }
    _state["llm_config"] = config
    return {
        "status": "llm_config_set",
        **{k: v for k, v in config.items() if v is not None},
    }


# ---------------------------------------------------------------------------
# LLM helper (private, not an MCP tool)
# ---------------------------------------------------------------------------


def _get_llm_client(backend: str) -> Any:
    """Get or create a cached LLM client for the given backend."""
    clients = _state["llm_clients"]
    if backend not in clients:
        if backend == "gemini":
            from google import genai
            from google.genai import types

            http_options = types.HttpOptions(timeout=300_000)  # 5 min
            clients[backend] = genai.Client(http_options=http_options)
        elif backend == "anthropic":
            import anthropic

            clients[backend] = anthropic.Anthropic()
        elif backend == "openai":
            import openai

            clients[backend] = openai.OpenAI()
        else:
            raise ValueError(f"Unsupported backend: {backend}")
    return clients[backend]


_LLM_MAX_RETRIES = 3
_LLM_RETRY_BASE_DELAY = 2.0


def _call_llm(
    prompt: str | list,
    backend: str,
    model: str,
    *,
    role: str = "default",
    _log_context: str | None = None,
) -> str:
    """Call an LLM with a text or multimodal prompt. Supports gemini, anthropic, openai.

    When ``prompt`` is a list, items can be:
      - ``str``: text content
      - ``dict`` with ``"data"`` and ``"mime_type"`` keys: base64-encoded image

    If ``role`` is "primary" or "secondary" and ``_state["llm_config"]`` is set,
    the configured backend/model for that role overrides the passed values.
    """
    # Route based on role and llm_config
    llm_config = _state.get("llm_config")
    if llm_config is not None:
        if role == "primary":
            backend = llm_config.get("primary_backend") or backend
            model = llm_config.get("primary_model") or model
        elif role == "secondary":
            backend = llm_config.get("secondary_backend") or backend
            model = llm_config.get("secondary_model") or model

    t0 = time.time()
    response_text = ""
    has_error = False
    prompt_log = prompt if isinstance(prompt, str) else f"[multimodal: {len(prompt)} parts]"
    # Count frames in multimodal prompts for logging
    _num_frames = 0
    if isinstance(prompt, list):
        _num_frames = sum(
            1 for item in prompt if isinstance(item, dict) and "data" in item
        )
    try:
        if backend == "gemini":
            from google.genai import types

            client = _get_llm_client("gemini")
            if isinstance(prompt, list):
                parts = []
                for item in prompt:
                    if isinstance(item, str):
                        parts.append(types.Part(text=item))
                    elif isinstance(item, dict) and "data" in item:
                        parts.append(types.Part(inline_data=types.Blob(
                            mime_type=item.get("mime_type", "image/jpeg"),
                            data=base64.b64decode(item["data"]),
                        )))
                contents = [parts]
            else:
                contents = prompt
            # Add thinking config for Gemini 3 models (matches RLM's approach)
            config = None
            if "gemini-3" in model:
                config = types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(thinking_level="LOW"),
                )
            # Retry on transient server errors
            for attempt in range(_LLM_MAX_RETRIES):
                try:
                    response = client.models.generate_content(
                        model=model, contents=contents, config=config,
                    )
                    response_text = response.text
                    break
                except Exception as e:
                    status = getattr(e, "status_code", None) or getattr(e, "code", 0)
                    if status not in (500, 504) or attempt == _LLM_MAX_RETRIES - 1:
                        raise
                    delay = min(_LLM_RETRY_BASE_DELAY * (2 ** attempt), 15.0)
                    logger.warning("Gemini %s on attempt %d, retrying in %.1fs", status, attempt + 1, delay)
                    time.sleep(delay)
        elif backend == "anthropic":
            client = _get_llm_client("anthropic")
            if isinstance(prompt, list):
                content_parts = []
                for item in prompt:
                    if isinstance(item, str):
                        content_parts.append({"type": "text", "text": item})
                    elif isinstance(item, dict) and "data" in item:
                        content_parts.append({
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": item.get("mime_type", "image/jpeg"),
                                "data": item["data"],
                            },
                        })
                messages = [{"role": "user", "content": content_parts}]
            else:
                messages = [{"role": "user", "content": prompt}]
            message = client.messages.create(model=model, max_tokens=1024, messages=messages)
            response_text = message.content[0].text
        elif backend == "openai":
            client = _get_llm_client("openai")
            if isinstance(prompt, list):
                content_parts = []
                for item in prompt:
                    if isinstance(item, str):
                        content_parts.append({"type": "text", "text": item})
                    elif isinstance(item, dict) and "data" in item:
                        mime = item.get("mime_type", "image/jpeg")
                        content_parts.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime};base64,{item['data']}"},
                        })
                messages = [{"role": "user", "content": content_parts}]
            else:
                messages = [{"role": "user", "content": prompt}]
            response = client.chat.completions.create(model=model, messages=messages)
            response_text = response.choices[0].message.content
        else:
            raise ValueError(f"Unsupported backend: {backend}")
        return response_text
    except Exception:
        has_error = True
        raise
    finally:
        duration_ms = int((time.time() - t0) * 1000)
        _trace_logger.log_llm_call(
            prompt=prompt_log,
            model=model,
            backend=backend,
            response=response_text,
            duration_ms=duration_ms,
            has_error=has_error,
            context=_log_context,
            num_frames=_num_frames,
        )


def _llm_query_batched(
    prompts: list[str | list],
    backend: str = "gemini",
    model: str = "gemini-2.5-flash",
    max_workers: int = 4,
) -> list[str]:
    """Call an LLM with multiple prompts in parallel. Returns list of response texts.

    Each prompt can be a string or a multimodal list (see ``_call_llm``).
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    results: list[str | None] = [None] * len(prompts)
    with ThreadPoolExecutor(max_workers=min(len(prompts), max_workers)) as executor:
        future_to_idx = {
            executor.submit(_call_llm, p, backend, model): i for i, p in enumerate(prompts)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                results[idx] = f"ERROR: {e}"
    return results


# ---------------------------------------------------------------------------
# Shard analysis tool
# ---------------------------------------------------------------------------


@mcp.tool()
def kuavi_analyze_shards(
    question: str,
    shard_duration: float = 30.0,
    max_shards: int = 20,
    include_frames: bool = True,
    frames_per_shard: int = 2,
    backend: str = "gemini",
    model: str = "gemini-2.5-flash",
    video_id: str | None = None,
) -> dict[str, Any]:
    """Analyze video in parallel temporal shards using an LLM.

    Splits the indexed video into time-based shards, sends each shard's
    segment captions to an LLM in parallel, and returns structured results.

    When include_frames=True (default), keyframes are extracted and included
    in each shard prompt for multimodal analysis.
    """
    import concurrent.futures

    index = _get_active_index(video_id)
    if index is None:
        return {"error": "No video indexed. Call kuavi_index_video first."}

    _track_tool_call("analyze_shards")
    gate, warning = _check_budget_gate()
    if gate is not None:
        return gate

    segments = index.segments
    if not segments:
        return {"error": "No segments found in index."}

    # Determine total time range
    all_starts = [s["start_time"] for s in segments]
    all_ends = [s["end_time"] for s in segments]
    video_start = min(all_starts)
    video_end = max(all_ends)

    # Build temporal shards
    shards: list[dict[str, Any]] = []
    t = video_start
    while t < video_end:
        shard_end = t + shard_duration
        shard_segments = [
            s for s in segments if s["start_time"] >= t and s["start_time"] < shard_end
        ]
        if shard_segments:
            shards.append({
                "start_time": t,
                "end_time": min(shard_end, video_end),
                "segments": shard_segments,
            })
        t = shard_end

    # Limit shards
    if len(shards) > max_shards:
        shards = shards[:max_shards]

    total = len(shards)

    # Pre-extract keyframes for all shards if multimodal mode is enabled
    shard_frames: list[list[dict[str, str]]] = [[] for _ in range(total)]
    if include_frames:
        video_path = _get_active_video_path(video_id)
        if video_path:
            try:
                import cv2

                from kuavi.context import _encode_frame

                cap = cv2.VideoCapture(video_path)
                if cap.isOpened():
                    original_fps = cap.get(cv2.CAP_PROP_FPS)
                    try:
                        for si, shard in enumerate(shards):
                            shard_start = shard["start_time"]
                            shard_end = shard["end_time"]
                            shard_dur = shard_end - shard_start
                            # Pick evenly spaced timestamps within the shard
                            n_frames = min(frames_per_shard, max(1, int(shard_dur)))
                            if n_frames == 1:
                                times = [(shard_start + shard_end) / 2]
                            else:
                                step = shard_dur / (n_frames + 1)
                                times = [shard_start + step * (j + 1) for j in range(n_frames)]
                            for t_frame in times:
                                frame_idx = int(t_frame * original_fps)
                                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                                ret, frame = cap.read()
                                if ret:
                                    frame = cv2.resize(frame, (480, 360))
                                    encoded = _encode_frame(frame, format=".jpg", quality=75)
                                    shard_frames[si].append(encoded)
                    finally:
                        cap.release()
            except Exception:
                logger.warning("Failed to extract shard keyframes; falling back to text-only.")

    def _analyze_shard(i: int, shard: dict[str, Any]) -> dict[str, Any]:
        seg_lines = []
        for s in shard["segments"]:
            ann = s.get("annotation", {})
            summary = ann.get("summary", {}).get("brief", "No caption")
            seg_lines.append(
                f"- Segment {s.get('segment_index', '?')} "
                f"[{s['start_time']:.1f}s - {s['end_time']:.1f}s]: {summary}"
            )
        text_prompt = (
            f"You are analyzing shard {i + 1}/{total} of a video "
            f"(time range {shard['start_time']:.1f}s - {shard['end_time']:.1f}s).\n\n"
            f"Segments in this shard:\n"
            + "\n".join(seg_lines)
            + f"\n\nQuestion: {question}\n\n"
            f"Provide a concise answer based only on the content in this shard."
        )

        # Build multimodal prompt if frames are available
        frames = shard_frames[i]
        if frames:
            prompt: str | list = [text_prompt] + [
                {"data": f["data"], "mime_type": f.get("mime_type", "image/jpeg")}
                for f in frames
            ]
        else:
            prompt = text_prompt

        try:
            answer = _call_llm(
                prompt, backend, model,
                role="primary",
                _log_context=f"kuavi_analyze_shards shard {i + 1}/{total}",
            )
            return {
                "shard_index": i,
                "start_time": shard["start_time"],
                "end_time": shard["end_time"],
                "answer": answer,
                "has_frames": bool(frames),
            }
        except Exception as e:
            return {
                "shard_index": i,
                "start_time": shard["start_time"],
                "end_time": shard["end_time"],
                "error": str(e),
            }

    # Execute in parallel
    max_workers = min(len(shards), 4)
    results = [None] * total
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_analyze_shard, i, shard): i for i, shard in enumerate(shards)}
        for future in concurrent.futures.as_completed(futures):
            idx = futures[future]
            results[idx] = future.result()

    result: dict[str, Any] = {
        "question": question,
        "shard_count": total,
        "multimodal": include_frames and any(f for f in shard_frames),
        "results": results,
    }
    if warning:
        result["_budget_warning"] = warning
    return result


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import atexit

    atexit.register(_trace_logger.log_session_end, "shutdown")
    mcp.run(transport="stdio")
