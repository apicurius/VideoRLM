"""KUAVi CLI — direct video indexing, search, and headless Claude Code analysis.

Subcommands:
    kuavi index <video>           — Index a video (no Claude Code needed)
    kuavi search <query>          — Search an indexed video
    kuavi analyze <video> -q "?"  — Headless analysis via `claude -p`
    kuavi analyze --batch <file>  — Batch analyze multiple videos
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import subprocess
import sys
import time
from pathlib import Path

from kuavi.verbose import KUAViPrinter


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
    indexer = VideoIndexer(
        embedding_model=args.embedding_model,
        cache_dir=args.cache_dir,
        scene_model=scene_model,
        text_embedding_model=text_model,
    )
    index = indexer.index_video(
        loaded_video,
        asr_model=args.asr_model,
    )
    index_time = time.time() - t0
    printer.print_step_done("Indexing complete", elapsed=index_time)

    # Save
    output_dir = args.output or str(Path(video_path).with_suffix(".kuavi"))
    printer.print_step("Saving index", output_dir)
    index.save(output_dir)
    printer.print_step_done("Saved", output_dir)

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


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="kuavi",
        description="KUAVi: Agentic Vision Intelligence",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- index ---
    p_index = subparsers.add_parser("index", help="Index a video file")
    p_index.add_argument("video", help="Path to the video file")
    p_index.add_argument("--output", "-o", help="Output directory for index")
    p_index.add_argument("--fps", type=float, default=1.0, help="Extraction FPS")
    p_index.add_argument("--auto-fps", action="store_true", help="Auto-compute FPS")
    p_index.add_argument("--target-frames", type=int, default=120, help="Target frames for auto-fps")
    p_index.add_argument(
        "--embedding-model", default="google/siglip2-base-patch16-256",
        help="Embedding model"
    )
    p_index.add_argument(
        "--scene-model", default="facebook/vjepa2-vitl-fpc64-256",
        help="Scene detection model"
    )
    p_index.add_argument("--no-scene-model", action="store_true", help="Disable scene model")
    p_index.add_argument(
        "--text-embedding-model", default=None,
        help="Separate text embedding model"
    )
    p_index.add_argument("--no-text-embedding", action="store_true")
    p_index.add_argument("--asr-model", default="Qwen/Qwen3-ASR-0.6B", help="Qwen3-ASR model name")
    p_index.add_argument("--cache-dir", default=None, help="Cache directory")

    # --- search ---
    p_search = subparsers.add_parser("search", help="Search an indexed video")
    p_search.add_argument("query", help="Search query")
    p_search.add_argument("--index-dir", required=True, help="Path to index directory")
    p_search.add_argument("--top-k", type=int, default=5, help="Number of results")
    p_search.add_argument("--field", default="summary", choices=["summary", "action", "visual", "all"])
    p_search.add_argument(
        "--embedding-model", default="google/siglip2-base-patch16-256",
        help="Embedding model for query encoding"
    )

    # --- analyze ---
    p_analyze = subparsers.add_parser("analyze", help="Analyze a video with Claude Code")
    p_analyze.add_argument("video", nargs="?", default=None, help="Path to the video file")
    p_analyze.add_argument("-q", "--question", required=True, help="Question about the video")
    p_analyze.add_argument("--batch", metavar="FILE", help="Text file with one video path per line")
    p_analyze.add_argument(
        "--output-format", choices=["text", "json"], default="text",
        help="Output format (default: text)"
    )
    p_analyze.add_argument("--output-dir", metavar="DIR", help="Directory for per-video result files")
    p_analyze.add_argument(
        "--max-parallel", type=int, default=1, metavar="N",
        help="Max parallel analyses (default: 1)"
    )
    p_analyze.add_argument(
        "--mode", choices=["fast", "full"], default="fast",
        help="Indexing mode: fast (embeddings only) or full (with captioning)"
    )
    p_analyze.add_argument(
        "--asr-model", default="faster-whisper/base",
        help="ASR model (default: faster-whisper/base)"
    )
    p_analyze.add_argument("--no-scene-model", action="store_true", help="Disable V-JEPA 2 scene model")

    # --- corpus ---
    p_corpus = subparsers.add_parser("corpus", help="Multi-video corpus operations")
    corpus_sub = p_corpus.add_subparsers(dest="corpus_command", help="Corpus subcommands")

    p_corpus_index = corpus_sub.add_parser("index", help="Index a directory of videos")
    p_corpus_index.add_argument("directory", help="Directory containing video files")
    p_corpus_index.add_argument("--output", "-o", help="Output directory for corpus index")
    p_corpus_index.add_argument(
        "--mode", default="fast", choices=["fast", "full"],
        help="Indexing mode (default: fast)"
    )
    p_corpus_index.add_argument(
        "--max-workers", type=int, default=4, metavar="N",
        help="Max parallel indexing workers (default: 4)"
    )

    p_corpus_search = corpus_sub.add_parser("search", help="Search a corpus index")
    p_corpus_search.add_argument("query", help="Search query")
    p_corpus_search.add_argument(
        "--index-dir", required=True, help="Path to corpus index directory"
    )
    p_corpus_search.add_argument("--top-k", type=int, default=10, help="Number of results")
    p_corpus_search.add_argument(
        "--embedding-model", default="google/siglip2-base-patch16-256",
        help="Embedding model for query encoding"
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "index":
        cmd_index(args)
    elif args.command == "search":
        cmd_search(args)
    elif args.command == "analyze":
        cmd_analyze(args)
    elif args.command == "corpus":
        if not hasattr(args, "corpus_command") or args.corpus_command is None:
            p_corpus.print_help()
            sys.exit(1)
        if args.corpus_command == "index":
            cmd_corpus_index(args)
        elif args.corpus_command == "search":
            cmd_corpus_search(args)


if __name__ == "__main__":
    main()
