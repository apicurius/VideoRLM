"""Run VideoRLM on a local video file with Gemini."""

from rlm.logger import RLMLogger
from rlm.video import VideoRLM

logger = RLMLogger(log_dir="./logs")

vrlm = VideoRLM(
    backend="gemini",
    backend_kwargs={"model_name": "gemini-3-flash-preview", "timeout": 300.0},
    fps=0.5,
    num_segments=5,
    max_frames_per_segment=3,
    resize=(640, 480),
    max_iterations=15,
    logger=logger,
    verbose=True,
    enable_search=True,
    scene_model="facebook/vjepa2-vitl-fpc64-256",
    text_embedding_model="google/embeddinggemma-300m",
)

result = vrlm.completion(
    "/Users/oerdogan/LVU/test_video.mp4",
    prompt="What is the most important concept explained in this video? Use the search tools to find the key moments, then zoom in to describe them in detail.",
)

print("\n" + "=" * 60)
print("FINAL ANSWER:")
print("=" * 60)
print(result.response)
