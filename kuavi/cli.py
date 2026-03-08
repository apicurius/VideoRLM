"""KUAVi CLI — direct video indexing, search, analysis, and all MCP tool equivalents.

Subcommands (original):
    kuavi index <video>           — Index a video (no Claude Code needed)
    kuavi search <query>          — Search an indexed video
    kuavi analyze <video> -q "?"  — Headless analysis via `claude -p`
    kuavi analyze --batch <file>  — Batch analyze multiple videos

MCP-equivalent subcommands (all output JSON to stdout):
    kuavi load-index              — Load a saved .kuavi index
    kuavi batch-index             — Index multiple videos
    kuavi video-info              — Index metadata
    kuavi session-stats           — Session statistics
    kuavi set-budget              — Configure budget limits
    kuavi set-llm                 — Configure LLM routing
    kuavi run-code                — Execute Python code
    kuavi search-transcript       — Transcript search
    kuavi get-transcript          — Get transcript for time range
    kuavi list-scenes             — List detected scenes
    kuavi extract-frames          — Extract frames from time range
    kuavi zoom                    — Multi-level zoom into time range
    kuavi vqa                     — Multiple-choice VQA
    kuavi predict-action          — Predict next action
    kuavi classify-segment        — Classify segment with probes
    kuavi predict-future-tokens   — Predict future V-JEPA tokens
    kuavi verify-coherence        — Verify temporal coherence
    kuavi shard-analysis          — Shard-based LLM analysis
    kuavi crop-frame              — Crop frame region
    kuavi diff-frames             — Pixel difference of frames
    kuavi blend-frames            — Blend multiple frames
    kuavi threshold-frame         — Binary threshold + contours
    kuavi frame-info              — Frame dimensions and stats
    kuavi corpus-search           — Cross-video semantic search
    kuavi corpus-stats            — Corpus statistics
    kuavi orient                  — Video overview (info + scenes)
    kuavi search-all              — Multi-field search + transcript
    kuavi inspect-segment         — Frames + transcript for range
    kuavi quick-answer            — One-shot search + inspect
    kuavi agent                   — Run agent loop (CHANGE 4)
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from kuavi.verbose import KUAViPrinter

# ---------------------------------------------------------------------------
# Shared state — mirrors mcp_server.py's module-level _state dict
# ---------------------------------------------------------------------------

_state: dict[str, Any] = {
    "videos": {},
    "active_video": None,
    "corpus": None,
    "eval_namespace": None,
    "llm_config": None,
    "stats": {
        "tool_calls": 0,
        "frames_extracted": 0,
        "searches_performed": 0,
        "session_start": None,
        "tokens_used": 0,
    },
    "last_frames": [],
    "llm_clients": {},
    "result_cache": {},
    "budget": {
        "max_tool_calls": 50,
        "warn_tool_calls": 35,
        "max_elapsed_seconds": 300,
        "warn_elapsed_seconds": 200,
        "exceeded": False,
        "max_tokens": None,
        "warn_tokens": None,
    },
}


def _get_video_path(args: argparse.Namespace) -> str:
    """Resolve video path from --video-path arg or KUAVI_VIDEO_PATH env var."""
    import os

    path = getattr(args, "video_path", None) or os.environ.get("KUAVI_VIDEO_PATH")
    if not path:
        print("Error: --video-path required or set KUAVI_VIDEO_PATH", file=sys.stderr)
        sys.exit(1)
    return path


def _active_index():
    """Get the active video index or exit with error."""
    vid = _state["active_video"]
    if vid is None or vid not in _state["videos"]:
        print("Error: No video indexed. Run 'kuavi index <video>' first.", file=sys.stderr)
        sys.exit(1)
    return _state["videos"][vid]["index"]


def _ensure_index_loaded(args: argparse.Namespace) -> None:
    """Ensure an index is loaded — either by indexing or loading from disk."""
    if _state["active_video"] is not None:
        return

    idx_dir = getattr(args, "index_dir", None)
    if idx_dir and Path(idx_dir).exists():
        _do_load_index(idx_dir)
        return

    vp = getattr(args, "video_path", None)
    if vp:
        derived = str(Path(vp).with_suffix(".kuavi"))
        if Path(derived).exists():
            _do_load_index(derived)
            return

    print("Error: No index available. Run 'kuavi index' or pass --index-dir.", file=sys.stderr)
    sys.exit(1)


def _do_load_index(
    index_dir: str,
    video_id: str | None = None,
    embedding_model: str = "google/siglip2-base-patch16-256",
) -> dict:
    """Load a saved .kuavi index into _state."""
    from kuavi.indexer import VideoIndex, VideoIndexer

    idx = VideoIndex.load(index_dir)
    indexer = VideoIndexer(embedding_model=embedding_model)
    indexer._ensure_model()
    idx.embed_fn = indexer._encode_query
    idx.visual_embed_fn = getattr(indexer, "_encode_query_siglip", idx.embed_fn)

    vid = video_id or Path(index_dir).stem
    _state["videos"][vid] = {
        "index": idx,
        "indexer": indexer,
        "loaded_video": None,
        "video_path": None,
    }
    _state["active_video"] = vid
    return {
        "video_id": vid,
        "segments": len(idx.segments),
        "scenes": len(idx.scene_boundaries),
        "transcript_entries": len(idx.transcript),
    }


def _output(result: Any, fmt: str = "json") -> None:
    """Print result to stdout in the requested format."""
    if fmt == "jsonl":
        if isinstance(result, list):
            for item in result:
                print(json.dumps(item, default=str))
        else:
            print(json.dumps(result, default=str))
    else:
        print(json.dumps(result, indent=2, default=str))


# ---------------------------------------------------------------------------
# Original CLI commands (preserved)
# ---------------------------------------------------------------------------


def cmd_index(args: argparse.Namespace) -> None:
    """Index a video file and save the index to disk."""
    from kuavi.indexer import VideoIndexer
    from kuavi.loader import VideoLoader

    printer = KUAViPrinter()
    video_path = args.video

    if not Path(video_path).exists():
        printer.print_error(f"Video file not found: {video_path}")
        sys.exit(1)

    # Auto-FPS
    fps = args.fps
    if args.auto_fps:
        import cv2

        cap = cv2.VideoCapture(video_path)
        try:
            video_fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            if video_fps > 0 and frame_count > 0:
                duration = frame_count / video_fps
                optimal = args.target_frames / duration
                fps = max(0.1, min(5.0, optimal))
        finally:
            cap.release()

    scene_model = None if args.no_scene_model else args.scene_model
    text_model = None if args.no_text_embedding else args.text_embedding_model

    printer.print_header("Index Video", {
        "Video": Path(video_path).name,
        "FPS": f"{fps:.2f}" + (" (auto)" if args.auto_fps else ""),
        "Embedding Model": args.embedding_model.split("/")[-1],
        "Scene Model": scene_model.split("/")[-1] if scene_model else "disabled",
        "ASR": args.asr_model,
        "Cache": args.cache_dir or "none",
    })

    # Load video
    printer.print_step("Loading video")
    t0 = time.time()
    loader = VideoLoader(fps=fps)
    loaded_video = loader.load(video_path)
    printer.print_step_done(
        "Loaded",
        f"{loaded_video.metadata.duration:.1f}s, "
        f"{loaded_video.metadata.extracted_frame_count} frames",
        elapsed=time.time() - t0,
    )

    # Index
    printer.print_step("Indexing video")
    t0 = time.time()

    force_reindex = getattr(args, "force_reindex", False)
    stages = None
    if hasattr(args, "stages") and args.stages:
        stages = [s.strip() for s in args.stages.split(",")]
    mode = getattr(args, "mode", "full")

    indexer = VideoIndexer(
        embedding_model=args.embedding_model,
        cache_dir=args.cache_dir,
        scene_model=scene_model,
        text_embedding_model=text_model,
    )
    index = indexer.index_video(
        loaded_video,
        asr_model=args.asr_model,
        mode=mode,
        force_reindex=force_reindex,
        stages=stages,
    )
    index_time = time.time() - t0
    printer.print_step_done("Indexing complete", elapsed=index_time)

    # Save
    output_dir = args.output or str(Path(video_path).with_suffix(".kuavi"))
    printer.print_step("Saving index", output_dir)
    index.save(output_dir)
    printer.print_step_done("Saved", output_dir)

    # Store in state for subsequent commands
    vid = Path(video_path).stem
    _state["videos"][vid] = {
        "index": index,
        "indexer": indexer,
        "loaded_video": loaded_video,
        "video_path": video_path,
    }
    _state["active_video"] = vid

    printer.print_final_summary({
        "Segments": len(index.segments),
        "Scenes": len(index.scene_boundaries),
        "Transcript entries": len(index.transcript),
        "Output": output_dir,
        "Index time": f"{index_time:.2f}s",
    })


def cmd_search(args: argparse.Namespace) -> None:
    """Search an indexed video."""
    from kuavi.indexer import VideoIndex, VideoIndexer
    from kuavi.search import make_search_transcript, make_search_video

    printer = KUAViPrinter()
    index_dir = args.index_dir

    if not Path(index_dir).exists():
        printer.print_error(f"Index directory not found: {index_dir}\nRun 'kuavi index <video>' first.")
        sys.exit(1)

    printer.print_header("Search Video", {
        "Query": args.query,
        "Index": index_dir,
        "Field": args.field,
        "Top-K": args.top_k,
    })

    printer.print_step("Loading index")
    t0 = time.time()
    index = VideoIndex.load(index_dir)

    # Re-attach embed_fn from a fresh indexer (not serializable)
    if index.embed_fn is None:
        embedding_model = getattr(args, "embedding_model", "google/siglip2-base-patch16-256")
        indexer = VideoIndexer(embedding_model=embedding_model)
        indexer._ensure_model()
        index.embed_fn = indexer._encode_query
        index.visual_embed_fn = indexer._encode_query_siglip
    printer.print_step_done("Index loaded", elapsed=time.time() - t0)

    query = args.query

    # Semantic search
    printer.print_step("Semantic search", f'"{query}"')
    t0 = time.time()
    tool = make_search_video(index)
    results = tool["tool"](query=query, top_k=args.top_k, field=args.field)
    printer.print_step_done("Search complete", f"{len(results)} results", elapsed=time.time() - t0)

    printer.print_search_results(results, args.field)

    # Transcript search
    if index.transcript:
        printer.print_step("Transcript search")
        t_tool = make_search_transcript(index)
        t_results = t_tool["tool"](query=query)
        printer.print_transcript_results(t_results)


def _build_analyze_prompt(
    video_path: str,
    question: str,
    mode: str = "fast",
    asr_model: str = "faster-whisper/base",
    no_scene_model: bool = False,
) -> str:
    """Build the analysis prompt for a single video."""
    index_args = f'video_path="{video_path}", mode="{mode}", asr_model="{asr_model}"'
    if no_scene_model:
        index_args += ", no_scene_model=True"
    return (
        f"Use the KUAVi MCP tools to analyze this video: {video_path}\n\n"
        f"Question: {question}\n\n"
        "Steps:\n"
        f"1. Call kuavi_index_video({index_args})\n"
        "2. Call kuavi_get_scene_list to understand the structure\n"
        "3. Use kuavi_search_video and kuavi_search_transcript to find relevant content\n"
        "4. Use kuavi_extract_frames for visual evidence\n"
        "5. Provide a clear, evidence-based answer"
    )


def _analyze_single_video(
    video_path: str, question: str, output_format: str
) -> dict:
    """Run Claude Code analysis on a single video and return the result."""
    prompt = _build_analyze_prompt(video_path, question)
    cmd = ["claude", "-p", prompt]
    if output_format == "json":
        cmd.extend(["--output-format", "json"])

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        return {
            "video": video_path,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    except FileNotFoundError:
        return {
            "video": video_path,
            "returncode": 1,
            "stdout": "",
            "stderr": (
                "Error: 'claude' CLI not found. Install Claude Code: "
                "https://claude.ai/code"
            ),
        }


def cmd_analyze(args: argparse.Namespace) -> None:
    """Analyze a video using headless Claude Code."""
    printer = KUAViPrinter()
    video_path = args.video
    question = args.question
    batch_file = getattr(args, "batch", None)
    output_format = getattr(args, "output_format", "text")
    output_dir = getattr(args, "output_dir", None)
    max_parallel = getattr(args, "max_parallel", 1)

    # Determine video list
    if batch_file:
        batch_path = Path(batch_file)
        if not batch_path.exists():
            printer.print_error(f"Batch file not found: {batch_file}")
            sys.exit(1)
        video_paths = []
        for line in batch_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                video_paths.append(line)
        if not video_paths:
            printer.print_error("Batch file contains no video paths.")
            sys.exit(1)
        # Validate all paths exist
        missing = [v for v in video_paths if not Path(v).exists()]
        if missing:
            for m in missing:
                printer.print_error(f"Video file not found: {m}")
            sys.exit(1)
    elif video_path:
        if not Path(video_path).exists():
            printer.print_error(f"Video file not found: {video_path}")
            sys.exit(1)
        # Single-video mode: preserve original behavior (no capture, stream output)
        prompt = _build_analyze_prompt(
            video_path, question,
            mode=getattr(args, "mode", "fast"),
            asr_model=getattr(args, "asr_model", "faster-whisper/base"),
            no_scene_model=getattr(args, "no_scene_model", False),
        )
        cmd = ["claude", "-p", prompt]
        printer.print_header("Analyze Video", {
            "Video": Path(video_path).name,
            "Question": question,
        })
        try:
            result = subprocess.run(cmd, capture_output=False, text=True)
            sys.exit(result.returncode)
        except FileNotFoundError:
            printer.print_error(
                "'claude' CLI not found. Install Claude Code: https://claude.ai/code"
            )
            sys.exit(1)
    else:
        printer.print_error("Provide a video path or --batch file.")
        sys.exit(1)

    # Batch mode execution
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    printer.print_header("Batch Analyze", {
        "Videos": len(video_paths),
        "Question": question,
        "Parallel": max_parallel,
        "Format": output_format,
    })

    if max_parallel > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_parallel) as executor:
            futures = {
                executor.submit(_analyze_single_video, vp, question, output_format): vp
                for vp in video_paths
            }
            results = []
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())
        # Sort by original order
        order = {vp: i for i, vp in enumerate(video_paths)}
        results.sort(key=lambda r: order[r["video"]])
    else:
        results = []
        for vp in video_paths:
            printer.print_step("Analyzing", Path(vp).name)
            results.append(_analyze_single_video(vp, question, output_format))

    # Write per-video output files
    if output_dir:
        for r in results:
            stem = Path(r["video"]).stem
            out_path = Path(output_dir) / f"{stem}.json"
            out_path.write_text(json.dumps(r, indent=2))

    # Print collected results
    if output_format == "json":
        print(json.dumps(results, indent=2))
    else:
        for r in results:
            print(f"\n=== {r['video']} (exit {r['returncode']}) ===")
            if r["stdout"]:
                print(r["stdout"])
            if r["stderr"]:
                print(r["stderr"], file=sys.stderr)

    succeeded = sum(1 for r in results if r["returncode"] == 0)
    failed = len(results) - succeeded

    printer.print_final_summary({
        "Total": len(results),
        "Succeeded": succeeded,
        "Failed": failed,
    })

    # Exit with non-zero if any failed
    if any(r["returncode"] != 0 for r in results):
        sys.exit(1)


def cmd_corpus_index(args: argparse.Namespace) -> None:
    """Index a directory of videos into a corpus."""
    from kuavi.corpus import CorpusIndexer, corpus_stats, discover_videos

    printer = KUAViPrinter()
    directory = Path(args.directory)

    if not directory.is_dir():
        printer.print_error(f"Directory not found: {args.directory}")
        sys.exit(1)

    video_paths = discover_videos(directory)
    if not video_paths:
        printer.print_error(f"No video files found in {args.directory}")
        sys.exit(1)

    printer.print_header("Index Corpus", {
        "Directory": str(directory),
        "Videos found": len(video_paths),
        "Mode": args.mode,
        "Max workers": args.max_workers,
        "Output": args.output or "(none)",
    })

    indexer = CorpusIndexer(max_workers=args.max_workers)

    completed = []

    def _progress(path: str, status: str, elapsed: float) -> None:
        completed.append(path)
        printer.print_step_done(
            f"[{len(completed)}/{len(video_paths)}] {Path(path).name}",
            status,
            elapsed=elapsed,
        )

    t0 = time.time()
    corpus = indexer.index_corpus(
        video_paths,
        mode=args.mode,
        progress_callback=_progress,
    )
    total_time = time.time() - t0

    if args.output:
        printer.print_step("Saving corpus index", args.output)
        corpus.save(args.output)
        printer.print_step_done("Saved", args.output)

    stats = corpus_stats(corpus)
    printer.print_final_summary({
        "Videos indexed": stats["num_videos"],
        "Total segments": stats["total_segments"],
        "Total duration": f"{stats['total_duration_seconds']:.1f}s",
        "Action vocabulary": stats["action_vocabulary_size"],
        "Index time": f"{total_time:.2f}s",
    })


def cmd_corpus_search(args: argparse.Namespace) -> None:
    """Search a saved corpus index."""
    from kuavi.corpus import CorpusIndex, search_corpus

    printer = KUAViPrinter()
    index_dir = args.index_dir

    if not Path(index_dir).exists():
        printer.print_error(f"Corpus index not found: {index_dir}\nRun 'kuavi corpus index' first.")
        sys.exit(1)

    printer.print_header("Search Corpus", {
        "Query": args.query,
        "Index": index_dir,
        "Top-K": args.top_k,
    })

    printer.print_step("Loading corpus index")
    t0 = time.time()
    corpus = CorpusIndex.load(index_dir)
    printer.print_step_done(
        "Loaded",
        f"{corpus.num_videos} videos, {corpus.total_segments} segments",
        elapsed=time.time() - t0,
    )

    # Re-attach embed_fn to each video index (not serialized)
    from kuavi.indexer import VideoIndexer

    embedding_model = getattr(args, "embedding_model", "google/siglip2-base-patch16-256")
    reindexer = VideoIndexer(embedding_model=embedding_model)
    reindexer._ensure_model()
    for idx in corpus.video_indices.values():
        if idx.embed_fn is None:
            idx.embed_fn = reindexer._encode_query

    printer.print_step("Searching corpus", f'"{args.query}"')
    t0 = time.time()
    results = search_corpus(corpus, args.query, top_k=args.top_k)
    printer.print_step_done("Search complete", f"{len(results)} results", elapsed=time.time() - t0)

    for i, r in enumerate(results, 1):
        vid = r.get("video_id", "?")
        start = r.get("start_time", 0)
        end = r.get("end_time", 0)
        score = r.get("score", 0)
        caption = r.get("caption", "")[:80]
        print(f"  {i:2}. [{vid}] {start:.1f}s-{end:.1f}s  score={score:.3f}  {caption}")


# ---------------------------------------------------------------------------
# New MCP-equivalent CLI commands
# ---------------------------------------------------------------------------


def cmd_load_index(args: argparse.Namespace) -> None:
    """Load a previously saved .kuavi index directory."""
    result = _do_load_index(
        args.index_dir,
        video_id=getattr(args, "video_id", None),
        embedding_model=getattr(args, "embedding_model", "google/siglip2-base-patch16-256"),
    )
    _output(result, args.output_format)


def cmd_batch_index(args: argparse.Namespace) -> None:
    """Index multiple videos in batch."""
    from kuavi.corpus import CorpusIndexer, discover_videos

    video_dir = getattr(args, "video_dir", None)
    video_paths_str = getattr(args, "video_paths", None)

    if video_dir:
        paths = discover_videos(Path(video_dir))
    elif video_paths_str:
        paths = [p.strip() for p in video_paths_str.split(",") if p.strip()]
    else:
        print("Error: --video-dir or --video-paths required", file=sys.stderr)
        sys.exit(1)

    indexer = CorpusIndexer(max_workers=getattr(args, "max_workers", 4))
    corpus = indexer.index_corpus(paths, mode=getattr(args, "mode", "fast"))

    if getattr(args, "output", None):
        corpus.save(args.output)

    _state["corpus"] = {"index": corpus}
    _output({"videos_indexed": len(paths), "output": getattr(args, "output", None)}, args.output_format)


def cmd_video_info(args: argparse.Namespace) -> None:
    """Get metadata about the current video index."""
    _ensure_index_loaded(args)
    index = _active_index()
    result = {
        "segments": len(index.segments),
        "scenes": len(index.scene_boundaries),
        "has_transcript": bool(index.transcript),
        "transcript_entries": len(index.transcript) if index.transcript else 0,
        "duration": index.segments[-1]["end_time"] if index.segments else 0,
        "has_temporal_embeddings": index.temporal_embeddings is not None,
    }
    _output(result, args.output_format)


def cmd_session_stats(args: argparse.Namespace) -> None:
    """Get usage statistics for the current CLI session."""
    _output(_state["stats"], args.output_format)


def cmd_set_budget(args: argparse.Namespace) -> None:
    """Configure budget limits for the current session."""
    budget = _state["budget"]
    budget["max_tool_calls"] = args.max_tool_calls
    budget["warn_tool_calls"] = args.warn_tool_calls
    budget["max_elapsed_seconds"] = args.max_elapsed_seconds
    budget["warn_elapsed_seconds"] = args.warn_elapsed_seconds
    if args.max_tokens is not None:
        budget["max_tokens"] = args.max_tokens
    if args.warn_tokens is not None:
        budget["warn_tokens"] = args.warn_tokens
    _output(budget, args.output_format)


def cmd_set_llm(args: argparse.Namespace) -> None:
    """Configure LLM routing for primary and secondary roles."""
    config = _state.get("llm_config") or {}
    if args.primary_backend:
        config["primary_backend"] = args.primary_backend
    if args.primary_model:
        config["primary_model"] = args.primary_model
    if args.secondary_backend:
        config["secondary_backend"] = args.secondary_backend
    if args.secondary_model:
        config["secondary_model"] = args.secondary_model
    _state["llm_config"] = config
    _output(config, args.output_format)


def cmd_run_code(args: argparse.Namespace) -> None:
    """Execute Python code in a persistent namespace with KUAVi tools available."""
    import contextlib
    import io

    ns = _state.get("eval_namespace")
    if ns is None:
        ns = {"__builtins__": __builtins__}
        _state["eval_namespace"] = ns

    if _state["active_video"]:
        ns["index"] = _active_index()

    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()
    result_val = None
    try:
        with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
            exec(args.code, ns)  # noqa: S102
        result_val = ns.get("result")
    except Exception as exc:
        stderr_buf.write(str(exc))

    _output({
        "stdout": stdout_buf.getvalue(),
        "stderr": stderr_buf.getvalue(),
        "result": str(result_val) if result_val is not None else None,
    }, args.output_format)


def cmd_search_video(args: argparse.Namespace) -> None:
    """Semantic search over indexed video segments."""
    _ensure_index_loaded(args)
    from kuavi.search import make_search_video

    index = _active_index()
    tool = make_search_video(index)
    results = tool["tool"](
        query=args.query,
        top_k=getattr(args, "top_k", 5),
        field=getattr(args, "field", "summary"),
    )
    _output(results, args.output_format)


def cmd_search_transcript(args: argparse.Namespace) -> None:
    """Keyword search over ASR transcript."""
    _ensure_index_loaded(args)
    from kuavi.search import make_search_transcript

    index = _active_index()
    tool = make_search_transcript(index)
    results = tool["tool"](query=args.query)
    _output(results, args.output_format)


def cmd_get_transcript(args: argparse.Namespace) -> None:
    """Get transcript text for a specific time range (seconds)."""
    _ensure_index_loaded(args)
    from kuavi.search import make_get_transcript

    index = _active_index()
    tool = make_get_transcript(index)
    result = tool["tool"](start_time=args.start, end_time=args.end)
    _output({"transcript": result}, args.output_format)


def cmd_list_scenes(args: argparse.Namespace) -> None:
    """List all detected scenes with annotations."""
    _ensure_index_loaded(args)
    from kuavi.search import make_get_scene_list

    index = _active_index()
    tool = make_get_scene_list(index)
    result = tool["tool"]()
    _output(result, args.output_format)


def cmd_extract_frames(args: argparse.Namespace) -> None:
    """Extract frames from the video as base64 JPEG images."""
    video_path = _get_video_path(args)
    from kuavi.context import make_extract_frames

    fn = make_extract_frames(video_path)
    frames = fn(
        start_time=args.start,
        end_time=args.end,
        fps=getattr(args, "fps", 2.0),
        max_frames=getattr(args, "max_frames", 10),
    )
    _state["last_frames"] = frames if isinstance(frames, list) else []

    summary = []
    for f in (frames if isinstance(frames, list) else []):
        entry = {k: v for k, v in f.items() if k != "data"}
        entry["data_length"] = len(f.get("data", ""))
        summary.append(entry)
    _output({"frame_count": len(summary), "frames": summary}, args.output_format)


def cmd_zoom(args: argparse.Namespace) -> None:
    """Extract frames at preset zoom levels (1=overview, 2=detail, 3=high-res)."""
    video_path = _get_video_path(args)
    from kuavi.context import make_extract_frames

    level = getattr(args, "level", 1)
    level_params = {
        1: {"fps": 0.5, "max_frames": 4},
        2: {"fps": 2.0, "max_frames": 8},
        3: {"fps": 4.0, "max_frames": 16},
    }
    params = level_params.get(level, level_params[2])

    fn = make_extract_frames(video_path)
    frames = fn(start_time=args.start, end_time=args.end, fps=params["fps"], max_frames=params["max_frames"])
    _state["last_frames"] = frames if isinstance(frames, list) else []
    _output({"level": level, "frame_count": len(frames) if isinstance(frames, list) else 0}, args.output_format)


def cmd_vqa(args: argparse.Namespace) -> None:
    """Embedding-based multiple-choice VQA without LLM generation."""
    _ensure_index_loaded(args)
    from kuavi.search import make_discriminative_vqa

    index = _active_index()
    tool = make_discriminative_vqa(index)
    candidates = [c.strip() for c in args.choices.split(",")]
    kwargs: dict[str, Any] = {"question": args.question, "candidates": candidates}
    if args.start is not None:
        kwargs["start_time"] = args.start
    if args.end is not None:
        kwargs["end_time"] = args.end
    result = tool["tool"](**kwargs)
    _output(result, args.output_format)


def cmd_predict_action(args: argparse.Namespace) -> None:
    """Predict what happens next after a given time point."""
    _ensure_index_loaded(args)
    index = _active_index()
    t = args.time_point
    best_seg = None
    for seg in index.segments:
        if seg["start_time"] <= t <= seg["end_time"]:
            best_seg = seg
            break
    if best_seg is None and index.segments:
        best_seg = min(index.segments, key=lambda s: abs(s["start_time"] - t))
    _output({
        "time_point": t,
        "segment": {
            "start_time": best_seg["start_time"] if best_seg else 0,
            "end_time": best_seg["end_time"] if best_seg else 0,
            "caption": best_seg.get("caption", "") if best_seg else "",
        },
        "note": "Full action prediction requires V-JEPA 2 predictor. Use MCP server for full functionality.",
    }, args.output_format)


def cmd_classify_segment(args: argparse.Namespace) -> None:
    """Classify a video segment using attentive probes on V-JEPA 2 features."""
    _ensure_index_loaded(args)
    index = _active_index()
    if index.temporal_embeddings is None:
        print("Error: No temporal embeddings. Index with V-JEPA 2.", file=sys.stderr)
        sys.exit(1)
    try:
        from kuavi.probes import classify_segment_with_probes
        seg_idx = args.segment_index if args.segment_index is not None else 0
        result = classify_segment_with_probes(index, task=args.task, segment_index=seg_idx, top_k=args.top_k)
        _output(result, args.output_format)
    except ImportError:
        _output({"error": "kuavi.probes not available"}, args.output_format)
        sys.exit(1)


def cmd_predict_future_tokens(args: argparse.Namespace) -> None:
    """Predict future V-JEPA feature tokens for a time range."""
    _ensure_index_loaded(args)
    _output({
        "start": args.start, "end": args.end,
        "n_future_tokens": args.n_future_tokens,
        "note": "Full prediction requires V-JEPA 2 predictor. Use MCP server for full functionality.",
    }, args.output_format)


def cmd_verify_coherence(args: argparse.Namespace) -> None:
    """Verify temporal coherence between segments and detect anomalies."""
    _ensure_index_loaded(args)
    import numpy as np

    index = _active_index()
    if index.temporal_embeddings is None:
        _output({"error": "No temporal embeddings. Index with V-JEPA 2."}, args.output_format)
        sys.exit(1)

    segs_in_range = [
        (i, s) for i, s in enumerate(index.segments)
        if s["end_time"] >= args.start and s["start_time"] <= args.end
    ]
    if len(segs_in_range) < 2:
        _output({"coherent": True, "segments_checked": len(segs_in_range)}, args.output_format)
        return

    similarities = []
    for j in range(len(segs_in_range) - 1):
        i1, i2 = segs_in_range[j][0], segs_in_range[j + 1][0]
        e1, e2 = index.temporal_embeddings[i1], index.temporal_embeddings[i2]
        sim = float(np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2) + 1e-8))
        similarities.append(sim)

    anomalies = [
        {"between_segments": [segs_in_range[j][0], segs_in_range[j + 1][0]], "similarity": round(similarities[j], 4)}
        for j in range(len(similarities)) if similarities[j] < args.threshold
    ]
    _output({
        "coherent": len(anomalies) == 0,
        "segments_checked": len(segs_in_range),
        "mean_similarity": round(float(np.mean(similarities)), 4) if similarities else None,
        "anomalies": anomalies,
    }, args.output_format)


def cmd_shard_analysis(args: argparse.Namespace) -> None:
    """Analyze video in parallel temporal shards using an LLM."""
    _ensure_index_loaded(args)
    _output({
        "question": args.question, "shard_duration": args.shard_duration,
        "max_shards": args.max_shards, "backend": args.backend, "model": args.model,
        "note": "Shard analysis requires LLM clients. Use MCP server or web_app for full functionality.",
    }, args.output_format)


def cmd_crop_frame(args: argparse.Namespace) -> None:
    """Crop a region from a frame using percentage coordinates."""
    import base64

    import cv2
    import numpy as np

    frames = _state.get("last_frames", [])
    idx = int(args.frame_index) if args.frame_index is not None else 0
    if idx >= len(frames):
        print(f"Error: Frame index {idx} out of range (0-{len(frames) - 1})", file=sys.stderr)
        sys.exit(1)
    raw = base64.b64decode(frames[idx].get("data", ""))
    frame = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    h, w = frame.shape[:2]
    cropped = frame[int(args.y1 * h):int(args.y2 * h), int(args.x1 * w):int(args.x2 * w)]
    _, buf = cv2.imencode(".jpg", cropped, [cv2.IMWRITE_JPEG_QUALITY, 85])
    _output({
        "crop": {"x1": args.x1, "y1": args.y1, "x2": args.x2, "y2": args.y2},
        "width": cropped.shape[1], "height": cropped.shape[0], "data_length": len(buf),
    }, args.output_format)


def cmd_diff_frames(args: argparse.Namespace) -> None:
    """Compute absolute pixel difference between two frames."""
    import base64

    import cv2
    import numpy as np

    frames = _state.get("last_frames", [])
    for idx in (args.frame_a, args.frame_b):
        if idx >= len(frames):
            print(f"Error: Frame index {idx} out of range", file=sys.stderr)
            sys.exit(1)

    def _dec(i):
        raw = base64.b64decode(frames[i].get("data", ""))
        return cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)

    fa, fb = _dec(args.frame_a), _dec(args.frame_b)
    if fa.shape != fb.shape:
        fb = cv2.resize(fb, (fa.shape[1], fa.shape[0]))
    diff = cv2.absdiff(fa, fb)
    changed = (diff > 25).any(axis=2) if diff.ndim == 3 else (diff > 25)
    _output({
        "mean_diff": round(float(diff.mean()), 2),
        "max_diff": int(diff.max()),
        "changed_pct": round(float(changed.sum() / changed.size * 100), 2),
    }, args.output_format)


def cmd_blend_frames(args: argparse.Namespace) -> None:
    """Average multiple frames into a composite image."""
    import base64

    import cv2
    import numpy as np

    frames = _state.get("last_frames", [])
    indices = [int(i) for i in args.frame_indices.split(",")]
    decoded = []
    for idx in indices:
        if idx >= len(frames):
            print(f"Error: Frame index {idx} out of range", file=sys.stderr)
            sys.exit(1)
        raw = base64.b64decode(frames[idx].get("data", ""))
        decoded.append(cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR))
    if not decoded:
        _output({"error": "No valid frames"}, args.output_format)
        sys.exit(1)
    target = decoded[0].shape[:2]
    for i in range(1, len(decoded)):
        if decoded[i].shape[:2] != target:
            decoded[i] = cv2.resize(decoded[i], (target[1], target[0]))
    blended = np.mean(decoded, axis=0).astype(np.uint8)
    _output({"frame_count": len(decoded), "shape": list(blended.shape)}, args.output_format)


def cmd_threshold_frame(args: argparse.Namespace) -> None:
    """Apply binary threshold + contour detection to a frame."""
    import base64

    import cv2
    import numpy as np

    frames = _state.get("last_frames", [])
    idx = int(args.frame_index) if args.frame_index is not None else 0
    if idx >= len(frames):
        print(f"Error: Frame index {idx} out of range", file=sys.stderr)
        sys.exit(1)
    raw = base64.b64decode(frames[idx].get("data", ""))
    frame = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    thresh_type = cv2.THRESH_BINARY_INV if args.invert else cv2.THRESH_BINARY
    _, mask = cv2.threshold(gray, args.value, 255, thresh_type)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    _output({
        "white_pct": round(float((mask == 255).sum() / mask.size * 100), 2),
        "contour_count": len(contours),
        "contour_areas": sorted([float(cv2.contourArea(c)) for c in contours], reverse=True)[:20],
    }, args.output_format)


def cmd_frame_info(args: argparse.Namespace) -> None:
    """Get image metadata: dimensions, brightness stats, color channel means."""
    import base64

    import cv2
    import numpy as np

    frames = _state.get("last_frames", [])
    idx = int(args.frame_index) if args.frame_index is not None else 0
    if idx >= len(frames):
        print(f"Error: Frame index {idx} out of range", file=sys.stderr)
        sys.exit(1)
    raw = base64.b64decode(frames[idx].get("data", ""))
    frame = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _output({
        "width": w, "height": h, "channels": frame.shape[2] if frame.ndim == 3 else 1,
        "brightness": {
            "mean": round(float(gray.mean()), 2), "std": round(float(gray.std()), 2),
            "min": int(gray.min()), "max": int(gray.max()),
        },
        "color": {
            "b_mean": round(float(frame[:, :, 0].mean()), 2),
            "g_mean": round(float(frame[:, :, 1].mean()), 2),
            "r_mean": round(float(frame[:, :, 2].mean()), 2),
        },
    }, args.output_format)


def cmd_corpus_search_cli(args: argparse.Namespace) -> None:
    """Search across all videos in the corpus."""
    corpus_data = _state.get("corpus")
    if corpus_data is None:
        print("Error: No corpus indexed. Run 'kuavi batch-index' first.", file=sys.stderr)
        sys.exit(1)
    from kuavi.corpus import search_corpus

    results = search_corpus(corpus_data["index"], args.query, top_k=args.top_k)
    _output(results, args.output_format)


def cmd_corpus_stats_cli(args: argparse.Namespace) -> None:
    """Get statistics about the indexed corpus."""
    corpus_data = _state.get("corpus")
    if corpus_data is None:
        print("Error: No corpus indexed. Run 'kuavi batch-index' first.", file=sys.stderr)
        sys.exit(1)
    from kuavi.corpus import corpus_stats

    _output(corpus_stats(corpus_data["index"]), args.output_format)


def cmd_orient(args: argparse.Namespace) -> None:
    """Get video overview: index metadata + scene list in one call."""
    _ensure_index_loaded(args)
    from kuavi.search import make_get_scene_list

    index = _active_index()
    info = {
        "segments": len(index.segments),
        "duration": index.segments[-1]["end_time"] if index.segments else 0,
        "scene_boundaries": len(index.scene_boundaries),
        "has_transcript": bool(index.transcript),
        "transcript_entries": len(index.transcript) if index.transcript else 0,
    }
    scene_tool = make_get_scene_list(index)
    scenes = scene_tool["tool"]()
    _output({"index_info": info, "scenes": scenes}, args.output_format)


def cmd_search_all(args: argparse.Namespace) -> None:
    """Multi-field semantic search + transcript search in one call."""
    _ensure_index_loaded(args)
    from kuavi.search import make_search_transcript, make_search_video

    index = _active_index()
    search_tool = make_search_video(index)
    transcript_tool = make_search_transcript(index)

    fields = args.fields.split(",") if args.fields else ["visual", "temporal"]
    results: dict[str, Any] = {}
    for field in fields:
        results[f"search_{field}"] = search_tool["tool"](query=args.query, field=field, top_k=getattr(args, "top_k", 5))
    tq = args.transcript_query or args.query
    results["transcript"] = transcript_tool["tool"](query=tq)
    _output(results, args.output_format)


def cmd_inspect_segment(args: argparse.Namespace) -> None:
    """Inspect a video segment: extract frames and transcript in one call."""
    _ensure_index_loaded(args)
    video_path = _get_video_path(args)
    from kuavi.context import make_extract_frames
    from kuavi.search import make_get_transcript

    index = _active_index()
    result: dict[str, Any] = {}

    if not args.no_frames:
        fn = make_extract_frames(video_path)
        frames = fn(start_time=args.start, end_time=args.end, fps=getattr(args, "fps", 2.0), max_frames=getattr(args, "max_frames", 5))
        result["frame_count"] = len(frames) if isinstance(frames, list) else 0

    if not args.no_transcript:
        t_tool = make_get_transcript(index)
        result["transcript"] = t_tool["tool"](start_time=args.start, end_time=args.end)

    _output(result, args.output_format)


def cmd_quick_answer(args: argparse.Namespace) -> None:
    """One-shot search + inspect: find relevant segments and extract frames/transcript."""
    _ensure_index_loaded(args)
    from kuavi.search import make_search_video

    index = _active_index()
    search_tool = make_search_video(index)
    hits = search_tool["tool"](query=args.question, top_k=args.top_k, field="visual")
    _output({
        "question": args.question,
        "top_hits": hits,
        "note": "Use inspect-segment on top hits for detailed analysis.",
    }, args.output_format)


def cmd_agent(args: argparse.Namespace) -> None:
    """Run the agent loop on a video and stream events as JSONL."""
    from kuavi.agent_runner import run_agent

    fmt = getattr(args, "output_format", "jsonl")
    api_key = getattr(args, "custom_api_key", None)
    for event in run_agent(
        video_path=args.video,
        question=args.question,
        model=args.model,
        api_key=api_key,
        backend=args.backend,
        index_mode=getattr(args, "index_mode", "fast"),
        asr_model=getattr(args, "asr_model", "faster-whisper/base"),
        max_iterations=getattr(args, "max_iterations", 10),
    ):
        if fmt == "jsonl":
            print(json.dumps(event, default=str), flush=True)
        else:
            _output(event, fmt)


# ---------------------------------------------------------------------------
# Argument parser helpers
# ---------------------------------------------------------------------------


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add --output-format shared across subcommands."""
    parser.add_argument("--output-format", choices=["json", "jsonl"], default="json", help="Output format (default: json)")


def _add_video_args(parser: argparse.ArgumentParser) -> None:
    """Add --video-path and --index-dir arguments for commands that need a video."""
    parser.add_argument("--video-path", default=None, help="Path to video file (or set KUAVI_VIDEO_PATH)")
    parser.add_argument("--index-dir", default=None, help="Path to .kuavi index directory")


def main() -> None:
    """CLI entry point."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    parser = argparse.ArgumentParser(prog="kuavi", description="KUAVi: Agentic Vision Intelligence")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- index ---
    p_index = subparsers.add_parser("index", help="Index a video file")
    p_index.add_argument("video", help="Path to the video file")
    p_index.add_argument("--output", "-o", help="Output directory for index")
    p_index.add_argument("--fps", type=float, default=1.0, help="Extraction FPS")
    p_index.add_argument("--auto-fps", action="store_true", help="Auto-compute FPS")
    p_index.add_argument("--target-frames", type=int, default=120, help="Target frames for auto-fps")
    p_index.add_argument("--embedding-model", default="google/siglip2-base-patch16-256")
    p_index.add_argument("--scene-model", default="facebook/vjepa2-vitl-fpc64-256")
    p_index.add_argument("--no-scene-model", action="store_true")
    p_index.add_argument("--text-embedding-model", default=None)
    p_index.add_argument("--no-text-embedding", action="store_true")
    p_index.add_argument("--asr-model", default="Qwen/Qwen3-ASR-0.6B")
    p_index.add_argument("--cache-dir", default=None)
    p_index.add_argument("--mode", choices=["fast", "full"], default="full", help="Indexing mode")
    p_index.add_argument("--force-reindex", action="store_true", help="Ignore cached stages")
    p_index.add_argument("--stages", default=None, help="Comma-separated stages to run")
    _add_common_args(p_index)

    # --- search ---
    p_search = subparsers.add_parser("search", help="Search an indexed video")
    p_search.add_argument("query", help="Search query")
    p_search.add_argument("--index-dir", required=True)
    p_search.add_argument("--top-k", type=int, default=5)
    p_search.add_argument("--field", default="summary", choices=["summary", "action", "visual", "all"])
    p_search.add_argument("--embedding-model", default="google/siglip2-base-patch16-256")
    _add_common_args(p_search)

    # --- analyze ---
    p_analyze = subparsers.add_parser("analyze", help="Analyze a video with Claude Code")
    p_analyze.add_argument("video", nargs="?", default=None)
    p_analyze.add_argument("-q", "--question", required=True)
    p_analyze.add_argument("--batch", metavar="FILE")
    p_analyze.add_argument("--output-format", choices=["text", "json"], default="text")
    p_analyze.add_argument("--output-dir", metavar="DIR")
    p_analyze.add_argument("--max-parallel", type=int, default=1)
    p_analyze.add_argument("--mode", choices=["fast", "full"], default="fast")
    p_analyze.add_argument("--asr-model", default="faster-whisper/base")
    p_analyze.add_argument("--no-scene-model", action="store_true")

    # --- corpus ---
    p_corpus = subparsers.add_parser("corpus", help="Multi-video corpus operations")
    corpus_sub = p_corpus.add_subparsers(dest="corpus_command")
    p_corpus_index = corpus_sub.add_parser("index", help="Index a directory of videos")
    p_corpus_index.add_argument("directory")
    p_corpus_index.add_argument("--output", "-o")
    p_corpus_index.add_argument("--mode", default="fast", choices=["fast", "full"])
    p_corpus_index.add_argument("--max-workers", type=int, default=4)
    p_corpus_search_p = corpus_sub.add_parser("search", help="Search a corpus index")
    p_corpus_search_p.add_argument("query")
    p_corpus_search_p.add_argument("--index-dir", required=True)
    p_corpus_search_p.add_argument("--top-k", type=int, default=10)
    p_corpus_search_p.add_argument("--embedding-model", default="google/siglip2-base-patch16-256")

    # ===================================================================
    # New MCP-equivalent subcommands
    # ===================================================================

    p_li = subparsers.add_parser("load-index", help="Load a saved .kuavi index directory")
    p_li.add_argument("index_dir")
    p_li.add_argument("--video-id", default=None)
    p_li.add_argument("--embedding-model", default="google/siglip2-base-patch16-256")
    _add_common_args(p_li)

    p_bi = subparsers.add_parser("batch-index", help="Index multiple videos in batch")
    p_bi.add_argument("--video-dir", default=None)
    p_bi.add_argument("--video-paths", default=None, help="Comma-separated video paths")
    p_bi.add_argument("--mode", default="fast", choices=["fast", "full"])
    p_bi.add_argument("--max-workers", type=int, default=4)
    p_bi.add_argument("--output", "-o", default=None)
    _add_common_args(p_bi)

    p_vi = subparsers.add_parser("video-info", help="Get metadata about the current video index")
    _add_video_args(p_vi)
    _add_common_args(p_vi)

    p_ss = subparsers.add_parser("session-stats", help="Get session usage statistics")
    _add_common_args(p_ss)

    p_sb = subparsers.add_parser("set-budget", help="Configure budget limits")
    p_sb.add_argument("--max-tool-calls", type=int, default=50)
    p_sb.add_argument("--warn-tool-calls", type=int, default=35)
    p_sb.add_argument("--max-elapsed-seconds", type=float, default=300)
    p_sb.add_argument("--warn-elapsed-seconds", type=float, default=200)
    p_sb.add_argument("--max-tokens", type=int, default=None)
    p_sb.add_argument("--warn-tokens", type=int, default=None)
    _add_common_args(p_sb)

    p_sl = subparsers.add_parser("set-llm", help="Configure LLM routing")
    p_sl.add_argument("--primary-backend", default=None)
    p_sl.add_argument("--primary-model", default=None)
    p_sl.add_argument("--secondary-backend", default=None)
    p_sl.add_argument("--secondary-model", default=None)
    _add_common_args(p_sl)

    p_rc = subparsers.add_parser("run-code", help="Execute Python code in persistent namespace")
    p_rc.add_argument("code")
    _add_common_args(p_rc)

    p_st = subparsers.add_parser("search-transcript", help="Keyword search over ASR transcript")
    p_st.add_argument("query")
    _add_video_args(p_st)
    _add_common_args(p_st)

    p_gt = subparsers.add_parser("get-transcript", help="Get transcript for a time range")
    p_gt.add_argument("--start", type=float, required=True)
    p_gt.add_argument("--end", type=float, required=True)
    _add_video_args(p_gt)
    _add_common_args(p_gt)

    p_ls = subparsers.add_parser("list-scenes", help="List all detected scenes")
    _add_video_args(p_ls)
    _add_common_args(p_ls)

    p_ef = subparsers.add_parser("extract-frames", help="Extract frames from time range")
    p_ef.add_argument("--start", type=float, required=True)
    p_ef.add_argument("--end", type=float, required=True)
    p_ef.add_argument("--fps", type=float, default=2.0)
    p_ef.add_argument("--max-frames", type=int, default=10)
    _add_video_args(p_ef)
    _add_common_args(p_ef)

    p_zm = subparsers.add_parser("zoom", help="Multi-level zoom into time range")
    p_zm.add_argument("--start", type=float, required=True)
    p_zm.add_argument("--end", type=float, required=True)
    p_zm.add_argument("--level", type=int, default=1, choices=[1, 2, 3])
    _add_video_args(p_zm)
    _add_common_args(p_zm)

    p_vq = subparsers.add_parser("vqa", help="Multiple-choice VQA via embeddings")
    p_vq.add_argument("question")
    p_vq.add_argument("--choices", required=True, help="Comma-separated candidate answers")
    p_vq.add_argument("--start", type=float, default=None)
    p_vq.add_argument("--end", type=float, default=None)
    _add_video_args(p_vq)
    _add_common_args(p_vq)

    p_pa = subparsers.add_parser("predict-action", help="Predict next action from V-JEPA features")
    p_pa.add_argument("--time-point", type=float, required=True)
    p_pa.add_argument("--top-k", type=int, default=3)
    _add_video_args(p_pa)
    _add_common_args(p_pa)

    p_cs = subparsers.add_parser("classify-segment", help="Classify segment with attentive probes")
    p_cs.add_argument("--task", default="k400")
    p_cs.add_argument("--segment-index", type=int, default=None)
    p_cs.add_argument("--top-k", type=int, default=5)
    _add_video_args(p_cs)
    _add_common_args(p_cs)

    p_pf = subparsers.add_parser("predict-future-tokens", help="Predict future V-JEPA feature tokens")
    p_pf.add_argument("--start", type=float, required=True)
    p_pf.add_argument("--end", type=float, required=True)
    p_pf.add_argument("--n-future-tokens", type=int, default=16)
    _add_video_args(p_pf)
    _add_common_args(p_pf)

    p_vc = subparsers.add_parser("verify-coherence", help="Verify temporal coherence of a segment")
    p_vc.add_argument("--start", type=float, required=True)
    p_vc.add_argument("--end", type=float, required=True)
    p_vc.add_argument("--threshold", type=float, default=0.3)
    _add_video_args(p_vc)
    _add_common_args(p_vc)

    p_sa = subparsers.add_parser("shard-analysis", help="Shard-based LLM analysis")
    p_sa.add_argument("question")
    p_sa.add_argument("--shard-duration", type=float, default=30.0)
    p_sa.add_argument("--max-shards", type=int, default=20)
    p_sa.add_argument("--backend", default="gemini")
    p_sa.add_argument("--model", default="gemini-2.5-flash")
    _add_video_args(p_sa)
    _add_common_args(p_sa)

    p_cf = subparsers.add_parser("crop-frame", help="Crop region from a frame")
    p_cf.add_argument("--frame-index", type=int, default=0)
    p_cf.add_argument("--x1", type=float, required=True)
    p_cf.add_argument("--y1", type=float, required=True)
    p_cf.add_argument("--x2", type=float, required=True)
    p_cf.add_argument("--y2", type=float, required=True)
    _add_common_args(p_cf)

    p_df = subparsers.add_parser("diff-frames", help="Pixel difference between two frames")
    p_df.add_argument("--frame-a", type=int, required=True)
    p_df.add_argument("--frame-b", type=int, required=True)
    _add_common_args(p_df)

    p_bf = subparsers.add_parser("blend-frames", help="Blend multiple frames")
    p_bf.add_argument("--frame-indices", required=True, help="Comma-separated frame indices")
    _add_common_args(p_bf)

    p_tf = subparsers.add_parser("threshold-frame", help="Binary threshold + contour detection")
    p_tf.add_argument("--frame-index", type=int, default=0)
    p_tf.add_argument("--value", type=int, default=128)
    p_tf.add_argument("--invert", action="store_true")
    _add_common_args(p_tf)

    p_fi = subparsers.add_parser("frame-info", help="Frame dimensions, brightness, color stats")
    p_fi.add_argument("--frame-index", type=int, default=0)
    _add_common_args(p_fi)

    p_csr = subparsers.add_parser("corpus-search", help="Cross-video semantic search")
    p_csr.add_argument("query")
    p_csr.add_argument("--top-k", type=int, default=10)
    _add_common_args(p_csr)

    p_cst = subparsers.add_parser("corpus-stats", help="Corpus statistics")
    _add_common_args(p_cst)

    p_or = subparsers.add_parser("orient", help="Video overview (info + scenes)")
    _add_video_args(p_or)
    _add_common_args(p_or)

    p_sal = subparsers.add_parser("search-all", help="Multi-field search + transcript")
    p_sal.add_argument("query")
    p_sal.add_argument("--fields", default=None, help="Comma-separated fields")
    p_sal.add_argument("--transcript-query", default=None)
    p_sal.add_argument("--top-k", type=int, default=5)
    _add_video_args(p_sal)
    _add_common_args(p_sal)

    p_is = subparsers.add_parser("inspect-segment", help="Frames + transcript for a time range")
    p_is.add_argument("--start", type=float, required=True)
    p_is.add_argument("--end", type=float, required=True)
    p_is.add_argument("--fps", type=float, default=2.0)
    p_is.add_argument("--max-frames", type=int, default=5)
    p_is.add_argument("--no-transcript", action="store_true")
    p_is.add_argument("--no-frames", action="store_true")
    _add_video_args(p_is)
    _add_common_args(p_is)

    p_qa = subparsers.add_parser("quick-answer", help="One-shot search + inspect")
    p_qa.add_argument("question")
    p_qa.add_argument("--top-k", type=int, default=3)
    _add_video_args(p_qa)
    _add_common_args(p_qa)

    p_ag = subparsers.add_parser("agent", help="Run agent loop on a video")
    p_ag.add_argument("--video", required=True)
    p_ag.add_argument("--question", "-q", required=True)
    p_ag.add_argument("--model", default="openai/gpt-4o")
    p_ag.add_argument("--backend", default="openrouter")
    p_ag.add_argument("--pipeline", default="kuavi", choices=["kuavi", "rlm"])
    p_ag.add_argument("--index-mode", default="fast", choices=["fast", "full", "captioned"])
    p_ag.add_argument("--asr-model", default="faster-whisper/base")
    p_ag.add_argument("--max-iterations", type=int, default=10)
    p_ag.add_argument("--custom-api-key", default=None)
    _add_common_args(p_ag)

    # ===================================================================
    # Dispatch
    # ===================================================================

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    dispatch = {
        "index": cmd_index,
        "search": cmd_search,
        "analyze": cmd_analyze,
        "load-index": cmd_load_index,
        "batch-index": cmd_batch_index,
        "video-info": cmd_video_info,
        "session-stats": cmd_session_stats,
        "set-budget": cmd_set_budget,
        "set-llm": cmd_set_llm,
        "run-code": cmd_run_code,
        "search-transcript": cmd_search_transcript,
        "get-transcript": cmd_get_transcript,
        "list-scenes": cmd_list_scenes,
        "extract-frames": cmd_extract_frames,
        "zoom": cmd_zoom,
        "vqa": cmd_vqa,
        "predict-action": cmd_predict_action,
        "classify-segment": cmd_classify_segment,
        "predict-future-tokens": cmd_predict_future_tokens,
        "verify-coherence": cmd_verify_coherence,
        "shard-analysis": cmd_shard_analysis,
        "crop-frame": cmd_crop_frame,
        "diff-frames": cmd_diff_frames,
        "blend-frames": cmd_blend_frames,
        "threshold-frame": cmd_threshold_frame,
        "frame-info": cmd_frame_info,
        "corpus-search": cmd_corpus_search_cli,
        "corpus-stats": cmd_corpus_stats_cli,
        "orient": cmd_orient,
        "search-all": cmd_search_all,
        "inspect-segment": cmd_inspect_segment,
        "quick-answer": cmd_quick_answer,
        "agent": cmd_agent,
    }

    try:
        if args.command == "corpus":
            if not hasattr(args, "corpus_command") or args.corpus_command is None:
                p_corpus.print_help()
                sys.exit(1)
            if args.corpus_command == "index":
                cmd_corpus_index(args)
            elif args.corpus_command == "search":
                cmd_corpus_search(args)
        elif args.command in dispatch:
            dispatch[args.command](args)
        else:
            parser.print_help()
            sys.exit(1)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
