"""Run VideoRLM on a local video file with Gemini."""

from rlm.logger import RLMLogger
from rlm.video import VideoRLM

logger = RLMLogger(log_dir="./logs")

vrlm = VideoRLM(
    backend="gemini",
    backend_kwargs={"model_name": "gemini-2.5-flash"},
    fps=0.5,
    num_segments=5,
    max_frames_per_segment=3,
    resize=(640, 480),
    max_iterations=15,
    logger=logger,
    verbose=True,
    enable_search=True,
)

result = vrlm.completion(
    "/Users/oerdogan/LVU/test_video.mp4",
    prompt="What is the most striking or visually memorable moment in this video? First scan all segments to find it, then use extract_frames() to zoom in and describe the moment in vivid detail - colors, expressions, objects, text on screen, everything you can see.",
)

print("\n" + "=" * 60)
print("FINAL ANSWER:")
print("=" * 60)
print(result.response)
