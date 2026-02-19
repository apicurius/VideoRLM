"""Test: Ask a fine-grained detail question - OOLONG score of RLM.

Usage:
    uv run python run_test_oolong.py --video test_video.mp4
    uv run python run_test_oolong.py --video test_video.mp4 \
        --question "What is the OOLONG score of RLM shown in this video?"
"""

import argparse

from rlm.logger import RLMLogger
from rlm.video import VideoRLM

DEFAULT_PROMPT = "What is the OOLONG score of RLM shown in this video?"


def main():
    parser = argparse.ArgumentParser(description="Test fine-grained detail question")
    parser.add_argument("--video", required=True, help="Path to the video file")
    parser.add_argument("--question", default=DEFAULT_PROMPT, help="Question to ask about the video")
    parser.add_argument("--backend", default="gemini", help="LLM backend (default: gemini)")
    parser.add_argument("--model", default="gemini-3.1-pro-preview", help="Model name")
    parser.add_argument("--fps", type=float, default=0.5, help="Frames per second (default: 0.5)")
    parser.add_argument("--num-segments", type=int, default=5, help="Number of segments (default: 5)")
    parser.add_argument("--max-frames-per-segment", type=int, default=3, help="Max frames per segment (default: 3)")
    parser.add_argument("--max-iterations", type=int, default=15, help="Max REPL iterations (default: 15)")
    parser.add_argument("--no-search", action="store_true", help="Disable semantic search tools")
    parser.add_argument("--no-scene-model", action="store_true", help="Disable V-JEPA 2 scene detection")
    parser.add_argument("--no-text-embedding", action="store_true", help="Disable separate text embedding model")
    parser.add_argument("--embedding-model", default="google/siglip2-base-patch16-256", help="Vision-text embedding model (default: google/siglip2-base-patch16-256)")
    parser.add_argument("--cache-dir", default=None, help="Directory to cache video indexes")
    parser.add_argument("--auto-fps", action="store_true", help="Auto-compute FPS based on video duration")
    args = parser.parse_args()

    logger = RLMLogger(log_dir="./logs")

    vrlm = VideoRLM(
        backend=args.backend,
        backend_kwargs={"model_name": args.model, "timeout": 300.0},
        fps=args.fps,
        num_segments=args.num_segments,
        max_frames_per_segment=args.max_frames_per_segment,
        resize=(640, 480),
        max_iterations=args.max_iterations,
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
