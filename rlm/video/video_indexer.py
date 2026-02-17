"""Video indexing: scene detection, captioning, embedding, and ASR transcript."""

from __future__ import annotations

import json
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
    action_embeddings: np.ndarray | None = None
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
        embedding_model: str = "google/siglip2-base-patch16-256",
        device: str = "auto",
    ):
        self._embedding_model_name = embedding_model
        self._device = device
        self._model = None
        self._processor = None

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------

    def _ensure_model(self) -> None:
        """Lazily load the SigLIP2 model on first use."""
        if self._model is not None:
            return
        import torch
        from transformers import AutoModel, GemmaTokenizerFast, SiglipImageProcessor

        device = self._device
        if device == "auto":
            device = "mps" if torch.backends.mps.is_available() else (
                "cuda" if torch.cuda.is_available() else "cpu"
            )

        # AutoProcessor/AutoTokenizer crash with SigLIP2 on transformers >=5.2
        # due to a tokenizer registration bug. Load components explicitly.
        self._image_processor = SiglipImageProcessor.from_pretrained(self._embedding_model_name)
        self._tokenizer = GemmaTokenizerFast.from_pretrained(self._embedding_model_name)
        self._model = AutoModel.from_pretrained(self._embedding_model_name).eval().to(device)
        self._torch_device = device
        logger.info("Loaded embedding model %s on %s", self._embedding_model_name, device)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def index_video(
        self,
        loaded_video: LoadedVideo,
        *,
        caption_fn: Callable | None = None,
        refine_fn: Callable | None = None,
        whisper_model: str = "base",
        transcript_path: str | None = None,
    ) -> VideoIndex:
        """Build a full searchable index from a loaded video.

        Args:
            loaded_video: A :class:`LoadedVideo` returned by :class:`VideoLoader`.
            caption_fn: Optional function that produces a caption for a list of
                frames.  May return a plain string (backward-compatible) or a
                structured annotation dict with ``summary`` and ``action`` keys.
            refine_fn: Optional function ``(draft: str, context: str) -> str``
                used for Self-Refine.  When provided, annotations are iteratively
                refined for 3 rounds using neighbor and transcript context.
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

        # 2. Detect scene boundaries (uses the CLIP embedding model)
        self._ensure_model()
        scenes = detect_scenes(frames, timestamps, embed_fn=self._encode_frames)
        scene_boundaries = [start for start, _end in scenes]

        # 3. Build segment dicts — prefer existing segments, fall back to scenes
        if loaded_video.segments:
            segment_infos = self._segments_from_loaded(loaded_video)
        else:
            segment_infos = self._segments_from_scenes(scenes, frames, timestamps)

        # 4. Transcript (Whisper ASR or pre-existing file) — run before captioning
        #    so ASR context can be injected into caption prompts
        transcript = self._get_transcript(
            loaded_video.metadata.path,
            whisper_model=whisper_model,
            transcript_path=transcript_path,
        )

        # 5. Caption each segment (if a caption function was provided)
        if caption_fn is not None:
            for seg in segment_infos:
                seg_frames = seg.pop("_frames")
                # ASR context injection: prepend transcript text for this segment
                transcript_text = self._transcript_for_range(
                    transcript, seg["start_time"], seg["end_time"],
                )
                if transcript_text:
                    seg_frames = [f"[transcript] {transcript_text}"] + seg_frames

                result = caption_fn(seg_frames)
                # Backward compat: wrap plain strings into structured annotation
                if isinstance(result, str):
                    annotation = {
                        "summary": {"brief": result, "detailed": result},
                        "action": {"brief": "", "detailed": "", "actor": None},
                    }
                else:
                    annotation = result
                seg["annotation"] = annotation
                seg["caption"] = annotation.get("summary", {}).get("brief", "")
        else:
            for seg in segment_infos:
                seg.pop("_frames", None)

        # 6. Self-Refine annotations
        self._refine_annotations(segment_infos, transcript, refine_fn)

        # 7. Embed captions
        embeddings, action_embeddings = self._embed_captions(segment_infos)

        return VideoIndex(
            segments=segment_infos,
            embeddings=embeddings,
            action_embeddings=action_embeddings,
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

    def _embed_captions(
        self, segments: list[dict],
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Encode segment captions and action briefs into embedding matrices."""
        captions = [seg.get("caption", "") for seg in segments]
        actions = [
            seg.get("annotation", {}).get("action", {}).get("brief", "")
            for seg in segments
        ]

        self._ensure_model()

        embeddings = None
        if any(captions):
            embeddings = self._encode_texts(captions)

        action_embeddings = None
        if any(actions):
            action_embeddings = self._encode_texts(actions)

        return embeddings, action_embeddings

    @staticmethod
    def _transcript_for_range(
        transcript: list[dict], start: float, end: float,
    ) -> str:
        """Return concatenated transcript text overlapping a time range."""
        return " ".join(
            e["text"]
            for e in transcript
            if e["end_time"] >= start and e["start_time"] <= end
        )

    def _refine_annotations(
        self,
        segments: list[dict],
        transcript: list[dict],
        refine_fn: Callable | None,
    ) -> None:
        """Iteratively refine segment annotations using the Self-Refine pattern.

        ``refine_fn`` signature: ``(draft: str, context: str) -> str``.
        Runs 3 rounds of refinement, passing the prior annotation together with
        neighbor context and overlapping transcript text.
        """
        if refine_fn is None:
            return
        for _round in range(3):
            for i, seg in enumerate(segments):
                neighbors = segments[max(0, i - 1) : i + 2]
                neighbor_text = " | ".join(
                    n.get("caption", "") for n in neighbors if n is not seg
                )
                transcript_text = self._transcript_for_range(
                    transcript, seg["start_time"], seg["end_time"],
                )
                context = f"Neighbors: {neighbor_text}\nTranscript: {transcript_text}"
                draft = json.dumps(seg.get("annotation", {}))
                refined = refine_fn(draft, context)
                try:
                    seg["annotation"] = json.loads(refined)
                    seg["caption"] = (
                        seg["annotation"]
                        .get("summary", {})
                        .get("brief", seg.get("caption", ""))
                    )
                except (json.JSONDecodeError, TypeError):
                    pass  # keep existing annotation if refinement fails

    def _encode_frames(self, frames: list[np.ndarray]) -> np.ndarray:
        """Encode a batch of BGR frames into an (N, D) embedding matrix.

        Used by :func:`detect_scenes` for semantic scene boundary detection.
        Converts frames from BGR to RGB PIL Images before encoding.
        """
        import torch
        from PIL import Image

        images = [Image.fromarray(f[:, :, ::-1]) for f in frames]  # BGR → RGB

        # Process in batches of 32 to avoid OOM
        all_embs = []
        batch_size = 32
        for i in range(0, len(images), batch_size):
            batch = images[i : i + batch_size]
            inputs = self._image_processor(images=batch, return_tensors="pt").to(self._torch_device)
            with torch.no_grad():
                out = self._model.get_image_features(**inputs)
                emb = out.pooler_output if hasattr(out, "pooler_output") else out
                emb = emb / emb.norm(p=2, dim=-1, keepdim=True)
            all_embs.append(emb.cpu().numpy())

        return np.concatenate(all_embs, axis=0)

    def _encode_texts(self, texts: list[str]) -> np.ndarray:
        """Encode a list of text strings into an (N, D) embedding matrix."""
        import torch

        inputs = self._tokenizer(
            texts, padding="max_length", max_length=64, return_tensors="pt",
        ).to(self._torch_device)
        with torch.no_grad():
            out = self._model.get_text_features(**inputs)
            emb = out.pooler_output if hasattr(out, "pooler_output") else out
            emb = emb / emb.norm(p=2, dim=-1, keepdim=True)
        return emb.cpu().numpy()

    def _encode_query(self, text: str) -> np.ndarray:
        """Encode a single query string — stored as ``embed_fn`` on the index."""
        self._ensure_model()
        return self._encode_texts([text])[0]

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
