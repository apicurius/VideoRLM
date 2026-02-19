"""
Example: Long Video Understanding (LVU) with RLM.

This demonstrates how to use VideoRLM to analyse a video file end-to-end:
frame extraction, scene detection, captioning, semantic search, and
recursive LLM querying — all handled by a single high-level class.

Usage:
    uv run python examples/video_understanding.py --video /path/to/video.mp4
    uv run python examples/video_understanding.py --video /path/to/video.mp4 \
        --question "What actions happen in this video?" --auto-fps

Requirements:
    - opencv-python (cv2)
    - numpy
    - A configured LLM backend (e.g. Portkey with PORTKEY_API_KEY)
"""

import argparse
import os

from dotenv import load_dotenv

from rlm.logger import RLMLogger
from rlm.video import VideoRLM

load_dotenv()


def main():
    parser = argparse.ArgumentParser(description="Video understanding with RLM")
    parser.add_argument("--video", required=True, help="Path to the video file")
    parser.add_argument(
        "--question",
        default="Describe what happens in this video.",
        help="Question to ask about the video",
    )
    parser.add_argument("--fps", type=float, default=1.0, help="Frames per second to extract")
    parser.add_argument("--num-segments", type=int, default=None, help="Number of segments")
    parser.add_argument(
        "--max-frames-per-segment", type=int, default=5, help="Max frames per segment"
    )
    parser.add_argument(
        "--auto-fps",
        action="store_true",
        help="Automatically compute FPS based on video duration (targets ~120 frames)",
    )
    parser.add_argument(
        "--no-search", action="store_true", help="Disable semantic search tools"
    )
    parser.add_argument("--cache-dir", default=None, help="Directory to cache video indexes")
    args = parser.parse_args()

    logger = RLMLogger(log_dir="./logs")

    video_rlm = VideoRLM(
        backend="portkey",
        backend_kwargs={
            "model_name": "@openai/gpt-5-nano",
            "api_key": os.getenv("PORTKEY_API_KEY"),
        },
        fps=args.fps,
        num_segments=args.num_segments,
        max_frames_per_segment=args.max_frames_per_segment,
        auto_fps=args.auto_fps,
        enable_search=not args.no_search,
        cache_dir=args.cache_dir,
        logger=logger,
        verbose=True,
    )

    print(f"Analyzing video: {args.video}")
    print(f"Question: {args.question}\n")

    result = video_rlm.completion(args.video, prompt=args.question)
    print(f"\nAnswer: {result}")


if __name__ == "__main__":
    main()
