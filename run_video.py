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
    prompt="Respond in English. Provide a comprehensive analysis of this video. First, search for all distinct scenes and topics covered. Then zoom into each key section to identify: (1) the main concepts being presented, (2) any diagrams, text, or visual aids shown on screen, (3) the speaker's key arguments and examples, and (4) how the sections connect to form the overall narrative. Finally, summarize the video's thesis and the evidence used to support it.",
)

print("\n" + "=" * 60)
print("FINAL ANSWER:")
print("=" * 60)
print(result.response)
