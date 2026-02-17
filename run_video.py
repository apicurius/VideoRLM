"""Run VideoRLM on a local video file with Gemini."""

from rlm.logger import RLMLogger
from rlm.video import VideoRLM

logger = RLMLogger(log_dir="./logs")

vrlm = VideoRLM(
    backend="gemini",
    backend_kwargs={"model_name": "gemini-3-flash-preview"},
    fps=0.5,
    num_segments=5,
    max_frames_per_segment=3,
    resize=(320, 240),
    max_iterations=15,
    logger=logger,
    verbose=True,
)

result = vrlm.completion(
    "/Users/oerdogan/LVU/test_video.mp4",
    prompt="Describe what happens in this video. What are the key events and topics discussed?",
)

print("\n" + "=" * 60)
print("FINAL ANSWER:")
print("=" * 60)
print(result.response)
