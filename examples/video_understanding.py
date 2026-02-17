"""
Example: Long Video Understanding (LVU) with RLM.

This demonstrates how to use the RLM video modules to analyse a video file
by extracting frames, building context, and querying an LLM recursively.

Usage:
    uv run python examples/video_understanding.py --video /path/to/video.mp4

Requirements:
    - opencv-python (cv2)
    - numpy
    - A configured LLM backend (e.g. Portkey with PORTKEY_API_KEY)
"""

import argparse
import os

from dotenv import load_dotenv

from rlm import RLM
from rlm.logger import RLMLogger
from rlm.video import VideoContext, VideoLoader

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
    parser.add_argument("--num-segments", type=int, default=4, help="Number of segments")
    parser.add_argument(
        "--max-frames-per-segment", type=int, default=5, help="Max frames per segment"
    )
    args = parser.parse_args()

    # 1. Load and segment the video
    print(f"Loading video: {args.video}")
    loader = VideoLoader(fps=args.fps, resize=(320, 240))
    loaded_video = loader.load_and_segment(args.video, num_segments=args.num_segments)

    print(f"  Duration: {loaded_video.metadata.duration:.1f}s")
    print(f"  Extracted frames: {loaded_video.metadata.extracted_frame_count}")
    print(f"  Segments: {len(loaded_video.segments)}")

    # 2. Build context payload from the video
    video_ctx = VideoContext(
        format=".jpg",
        quality=80,
        max_frames_per_segment=args.max_frames_per_segment,
    )
    context_payload = video_ctx.build_context(loaded_video)

    print(f"  Context segments: {context_payload.get('num_segments', 'N/A')}")

    # 3. Set up RLM with the video context
    logger = RLMLogger(log_dir="./logs")

    rlm = RLM(
        backend="portkey",
        backend_kwargs={
            "model_name": "@openai/gpt-5-nano",
            "api_key": os.getenv("PORTKEY_API_KEY"),
        },
        environment="local",
        environment_kwargs={"context_payload": context_payload},
        max_depth=1,
        logger=logger,
        verbose=True,
    )

    # 4. Ask a question about the video
    print(f"\nQuestion: {args.question}\n")
    result = rlm.completion(args.question)
    print(f"\nAnswer: {result}")


if __name__ == "__main__":
    main()
