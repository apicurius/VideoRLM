"""Video indexing: scene detection, captioning, embedding, and ASR transcript."""

from __future__ import annotations

import logging
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from rlm.video.scene_detection import detect_scenes
from rlm.video.video_loader import LoadedVideo

logger = logging.getLogger(__name__)


@dataclass
class VideoIndex:
    """Pre-computed searchable index for a video."""

    segments: list[dict] = field(default_factory=list)
    embeddings: np.ndarray | None = None
    transcript: list[dict] = field(default_factory=list)
    scene_boundaries: list[float] = field(default_factory=list)
    embed_fn: Any = None


class VideoIndexer:
    """Build a searchable :class:`VideoIndex` from a loaded video.

    Handles scene detection, optional captioning, sentence-transformer embedding,
    and Whisper-based ASR transcription.

    Args:
        embedding_model: HuggingFace model id for sentence-transformers.
        device: Torch device string (``"auto"`` lets sentence-transformers choose).
    """

    def __init__(
        self,
        embedding_model: str = "jinaai/jina-clip-v2",
        device: str = "auto",
    ):
        self._embedding_model_name = embedding_model
        self._device = device
        self._model = None

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------

    def _ensure_model(self) -> None:
        """Lazily load the sentence-transformers model on first use."""
        if self._model is not None:
            return
        from sentence_transformers import SentenceTransformer

        device = self._device if self._device != "auto" else None
        self._model = SentenceTransformer(self._embedding_model_name, device=device)
        logger.info("Loaded embedding model %s", self._embedding_model_name)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def index_video(
        self,
        loaded_video: LoadedVideo,
        *,
        caption_fn: Callable | None = None,
        whisper_model: str = "base",
        transcript_path: str | None = None,
    ) -> VideoIndex:
        """Build a full searchable index from a loaded video.

        Args:
            loaded_video: A :class:`LoadedVideo` returned by :class:`VideoLoader`.
            caption_fn: Optional function ``(list[np.ndarray]) -> str`` that
                produces a text caption for a list of frames.  If *None*, captions
                are left empty.
            whisper_model: Faster-whisper model size to use for ASR.
            transcript_path: Path to a pre-existing transcript JSON/SRT file.
                When provided, Whisper ASR is skipped.

        Returns:
            A :class:`VideoIndex` ready for use with the search-tool factories in
            :mod:`rlm.video.video_search_tools`.
        """
        fps = loaded_video.metadata.extraction_fps
        frames = loaded_video.frames

        # 1. Compute per-frame timestamps
        timestamps = [i / fps for i in range(len(frames))]

        # 2. Detect scene boundaries (uses the same embedding model)
        self._ensure_model()
        scenes = detect_scenes(frames, timestamps, embed_fn=self._encode_frames)
        scene_boundaries = [start for start, _end in scenes]

        # 3. Build segment dicts -— prefer existing segments, fall back to scenes
        if loaded_video.segments:
            segment_infos = self._segments_from_loaded(loaded_video)
        else:
            segment_infos = self._segments_from_scenes(scenes, frames, timestamps)

        # 4. Caption each segment (if a caption function was provided)
        if caption_fn is not None:
            for seg in segment_infos:
                seg["caption"] = caption_fn(seg.pop("_frames"))
        else:
            for seg in segment_infos:
                seg.pop("_frames", None)

        # 5. Embed captions
        embeddings = self._embed_captions(segment_infos)

        # 6. Transcript (Whisper ASR or pre-existing file)
        transcript = self._get_transcript(
            loaded_video.metadata.path,
            whisper_model=whisper_model,
            transcript_path=transcript_path,
        )

        return VideoIndex(
            segments=segment_infos,
            embeddings=embeddings,
            transcript=transcript,
            scene_boundaries=scene_boundaries,
            embed_fn=self._encode_query,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _segments_from_loaded(self, loaded_video: LoadedVideo) -> list[dict]:
        """Convert :class:`VideoSegment` objects to plain dicts."""
        results: list[dict] = []
        for seg in loaded_video.segments:
            results.append(
                {
                    "start_time": seg.start_time,
                    "end_time": seg.end_time,
                    "caption": "",
                    "_frames": seg.frames,
                }
            )
        return results

    def _segments_from_scenes(
        self,
        scenes: list[tuple[float, float]],
        frames: list[np.ndarray],
        timestamps: list[float],
    ) -> list[dict]:
        """Create segment dicts from detected scene boundaries."""
        results: list[dict] = []
        for start, end in scenes:
            seg_frames = [
                f for f, t in zip(frames, timestamps) if start <= t < end or t == end
            ]
            results.append(
                {
                    "start_time": start,
                    "end_time": end,
                    "caption": "",
                    "_frames": seg_frames,
                }
            )
        return results

    def _embed_captions(self, segments: list[dict]) -> np.ndarray | None:
        """Encode segment captions into an (N, D) embedding matrix."""
        captions = [seg.get("caption", "") for seg in segments]
        if not any(captions):
            return None

        self._ensure_model()
        embeddings = self._model.encode(captions, show_progress_bar=False)
        return np.asarray(embeddings)

    def _encode_frames(self, frames: list[np.ndarray]) -> np.ndarray:
        """Encode a batch of BGR frames into an (N, D) embedding matrix.

        Used by :func:`detect_scenes` for semantic scene boundary detection.
        Converts frames from BGR to RGB before encoding, as sentence-transformers
        image encoders expect RGB input.
        """
        from PIL import Image

        images = [Image.fromarray(f[:, :, ::-1]) for f in frames]  # BGR → RGB
        return np.asarray(self._model.encode(images, show_progress_bar=False))

    def _encode_query(self, text: str) -> np.ndarray:
        """Encode a single query string — stored as ``embed_fn`` on the index."""
        self._ensure_model()
        return np.asarray(self._model.encode(text, show_progress_bar=False))

    def _get_transcript(
        self,
        video_path: str,
        *,
        whisper_model: str = "base",
        transcript_path: str | None = None,
    ) -> list[dict]:
        """Return ASR transcript as a list of ``{start_time, end_time, text}`` dicts."""
        if transcript_path is not None:
            return self._load_transcript_file(transcript_path)

        return self._run_whisper(video_path, whisper_model)

    @staticmethod
    def _load_transcript_file(path: str) -> list[dict]:
        """Load a transcript from a JSON file."""
        import json

        try:
            data = json.loads(Path(path).read_text())
            if isinstance(data, list):
                return data
            logger.warning("Transcript file %s is not a JSON list; ignoring.", path)
        except Exception:
            logger.warning("Failed to load transcript from %s", path, exc_info=True)
        return []

    @staticmethod
    def _extract_audio(video_path: str, out_wav: str) -> bool:
        """Extract audio track to a WAV file using ffmpeg."""
        try:
            subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-i", video_path,
                    "-vn",
                    "-acodec", "pcm_s16le",
                    "-ar", "16000",
                    "-ac", "1",
                    out_wav,
                ],
                check=True,
                capture_output=True,
            )
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.warning("Audio extraction failed for %s (ffmpeg may be missing).", video_path)
            return False

    def _run_whisper(self, video_path: str, model_size: str) -> list[dict]:
        """Transcribe audio using faster-whisper."""
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            logger.info("faster_whisper not installed; skipping ASR.")
            return []

        with tempfile.TemporaryDirectory() as tmp:
            wav_path = str(Path(tmp) / "audio.wav")
            if not self._extract_audio(video_path, wav_path):
                return []

            try:
                model = WhisperModel(model_size, device="auto", compute_type="int8")
                segments_iter, _info = model.transcribe(wav_path)

                transcript: list[dict] = []
                for seg in segments_iter:
                    transcript.append(
                        {
                            "start_time": round(seg.start, 2),
                            "end_time": round(seg.end, 2),
                            "text": seg.text.strip(),
                        }
                    )
                return transcript
            except Exception:
                logger.warning("Whisper transcription failed.", exc_info=True)
                return []
