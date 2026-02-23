"""Run VideoRLM on a local video file with Gemini.

Usage:
    uv run python run_video.py --video test_video.mp4
    uv run python run_video.py --video test_video.mp4 \
        --question "What is the OOLONG score of RLM shown in this video?"
    uv run python run_video.py --video test_video.mp4 --no-search --fps 1.0
"""

import argparse

from rlm.logger import RLMLogger
from rlm.video import VideoRLM

DEFAULT_PROMPT = (
    "Respond in English. Provide a comprehensive analysis of this video. "
    "First, search for all distinct scenes and topics covered. Then zoom into "
    "each key section to identify: (1) the main concepts being presented, "
    "(2) any diagrams, text, or visual aids shown on screen, (3) the speaker's "
    "key arguments and examples, and (4) how the sections connect to form the "
    "overall narrative. Finally, summarize the video's thesis and the evidence "
    "used to support it."
)


def main():
    parser = argparse.ArgumentParser(description="Run VideoRLM on a local video file")
    parser.add_argument("--video", required=True, help="Path to the video file")
    parser.add_argument("--question", default=DEFAULT_PROMPT, help="Question to ask about the video")
    parser.add_argument("--backend", default="gemini", help="LLM backend (default: gemini)")
    parser.add_argument("--model", default="gemini-3.1-pro-preview", help="Model name")
    parser.add_argument("--fps", type=float, default=0.5, help="Frames per second (default: 0.5)")
    parser.add_argument("--num-segments", type=int, default=5, help="Number of segments (default: 5)")
    parser.add_argument("--max-frames-per-segment", type=int, default=3, help="Max frames per segment (default: 3)")
    parser.add_argument("--max-iterations", type=int, default=15, help="Max REPL iterations (default: 15)")
    parser.add_argument("--cost-budget", type=float, default=None, help="Max estimated cost in USD (default: no limit)")
    parser.add_argument("--no-search", action="store_true", help="Disable semantic search tools")
    parser.add_argument("--no-scene-model", action="store_true", help="Disable V-JEPA 2 scene detection")
    parser.add_argument("--no-text-embedding", action="store_true", help="Disable separate text embedding model")
    parser.add_argument("--embedding-model", default="google/siglip2-base-patch16-256", help="Vision-text embedding model (default: google/siglip2-base-patch16-256)")
    parser.add_argument("--cache-dir", default=None, help="Directory to cache video indexes")
    parser.add_argument("--auto-fps", action="store_true", help="Auto-compute FPS based on video duration")
    parser.add_argument("--thinking-level", default="LOW", choices=["NONE", "LOW", "MEDIUM", "HIGH"], help="Gemini thinking level (default: LOW)")
    args = parser.parse_args()

    logger = RLMLogger(log_dir="./logs")

    vrlm = VideoRLM(
        backend=args.backend,
        backend_kwargs={"model_name": args.model, "timeout": 300.0, "thinking_level": args.thinking_level},
        fps=args.fps,
        num_segments=args.num_segments,
        max_frames_per_segment=args.max_frames_per_segment,
        resize=(640, 480),
        max_iterations=args.max_iterations,
        cost_budget=args.cost_budget,
        logger=logger,
        verbose=True,
        enable_search=not args.no_search,
        embedding_model=args.embedding_model,
        cache_dir=args.cache_dir,
        auto_fps=args.auto_fps,
        scene_model=None if args.no_scene_model else "facebook/vjepa2-vitl-fpc64-256",
        text_embedding_model=None if args.no_text_embedding else "google/embeddinggemma-300m",
    )

    result = vrlm.completion(args.video, prompt=args.question)

    print("\n" + "=" * 60)
    print("FINAL ANSWER:")
    print("=" * 60)
    print(result.response)


if __name__ == "__main__":
    main()
