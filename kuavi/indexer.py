"""Video indexing: scene detection, captioning, embedding, and ASR transcript.

This module is the orchestration layer. The actual work is delegated to:

- :mod:`kuavi.encoding` — frame/text encoding (SigLIP2, EmbeddingGemma, V-JEPA 2)
- :mod:`kuavi.caption_pipeline` — selective decode, Tree-of-Captions, Self-Refine, quality scoring
- :mod:`kuavi.dedup` — pre-caption, adjacent, global, and semantic deduplication
- :mod:`kuavi.embedding` — caption embedding, smoothing, quality checks, coarse levels, prediction
- :mod:`kuavi.transcript` — ASR pipeline (Qwen3-ASR, faster-whisper, audio extraction)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from kuavi import caption_pipeline, dedup, encoding, transcript
from kuavi import embedding as emb_mod
from kuavi.loader import LoadedVideo
from kuavi.scene_detection import detect_scenes, detect_scenes_hierarchical

logger = logging.getLogger(__name__)


@dataclass
class VideoIndex:
    """Pre-computed searchable index for a video."""

    segments: list[dict] = field(default_factory=list)
    embeddings: np.ndarray | None = None
    action_embeddings: np.ndarray | None = None
    transcript: list[dict] = field(default_factory=list)
    scene_boundaries: list[float] = field(default_factory=list)
    embedding_quality: dict = field(default_factory=dict)
    embed_fn: Any = None
    frame_embeddings: np.ndarray | None = None
    visual_embed_fn: Any = None
    temporal_embeddings: np.ndarray | None = None  # (N_segments, 1024) from V-JEPA 2
    temporal_feature_maps: np.ndarray | None = None  # (N_segments, num_patches, D) from V-JEPA 2
    segment_hierarchy: list[list[dict]] = field(default_factory=list)
    hierarchy_embeddings: list[np.ndarray | None] = field(default_factory=list)

    def save(self, path: str | Path) -> None:
        """Persist index to *path* (a directory).

        Embeddings are stored as a ``.npz`` file; metadata (segments,
        transcript, scene_boundaries) as ``metadata.json``.  The callable
        ``embed_fn`` is **not** serialized.
        """
        directory = Path(path)
        directory.mkdir(parents=True, exist_ok=True)

        # Save embeddings
        arrays: dict[str, np.ndarray] = {}
        if self.embeddings is not None:
            arrays["embeddings"] = self.embeddings
        if self.action_embeddings is not None:
            arrays["action_embeddings"] = self.action_embeddings
        if self.frame_embeddings is not None:
            arrays["frame_embeddings"] = self.frame_embeddings
        if self.temporal_embeddings is not None:
            arrays["temporal_embeddings"] = self.temporal_embeddings
        if self.temporal_feature_maps is not None:
            arrays["temporal_feature_maps"] = self.temporal_feature_maps
        for lvl_idx, h_emb in enumerate(self.hierarchy_embeddings):
            if h_emb is not None:
                arrays[f"hierarchy_emb_L{lvl_idx}"] = h_emb
        np.savez(directory / "embeddings.npz", **arrays)

        # Save metadata
        metadata = {
            "segments": self.segments,
            "transcript": self.transcript,
            "scene_boundaries": self.scene_boundaries,
            "embedding_quality": self.embedding_quality,
            "segment_hierarchy": self.segment_hierarchy,
        }
        (directory / "metadata.json").write_text(json.dumps(metadata))

    @classmethod
    def load(cls, path: str | Path) -> VideoIndex:
        """Load a previously saved index from *path*.

        ``embed_fn`` will be ``None`` on the returned object — the caller
        is responsible for re-attaching it if needed.
        """
        directory = Path(path)
        metadata = json.loads((directory / "metadata.json").read_text())

        npz = np.load(directory / "embeddings.npz")
        embeddings = npz["embeddings"] if "embeddings" in npz else None
        action_embeddings = npz["action_embeddings"] if "action_embeddings" in npz else None
        frame_embeddings = npz["frame_embeddings"] if "frame_embeddings" in npz else None
        temporal_embeddings = npz["temporal_embeddings"] if "temporal_embeddings" in npz else None
        temporal_feature_maps = (
            npz["temporal_feature_maps"] if "temporal_feature_maps" in npz else None
        )

        # Load hierarchy embeddings
        hierarchy_embeddings: list[np.ndarray | None] = []
        lvl = 0
        while f"hierarchy_emb_L{lvl}" in npz:
            hierarchy_embeddings.append(npz[f"hierarchy_emb_L{lvl}"])
            lvl += 1

        return cls(
            segments=metadata["segments"],
            embeddings=embeddings,
            action_embeddings=action_embeddings,
            frame_embeddings=frame_embeddings,
            temporal_embeddings=temporal_embeddings,
            temporal_feature_maps=temporal_feature_maps,
            transcript=metadata["transcript"],
            scene_boundaries=metadata["scene_boundaries"],
            embedding_quality=metadata.get("embedding_quality", {}),
            segment_hierarchy=metadata.get("segment_hierarchy", []),
            hierarchy_embeddings=hierarchy_embeddings,
        )


def _cache_key(video_path: str) -> str:
    """Compute a deterministic cache key from video path, size, and mtime."""
    p = Path(video_path).resolve()
    stat = os.stat(p)
    raw = f"{p}|{stat.st_size}|{stat.st_mtime}"
    return hashlib.md5(raw.encode()).hexdigest()


class _StageCache:
    """Sidecar cache that persists individual pipeline stage outputs.

    Stages are stored under ``<video_path>.kuavi/<cache_key>/``:
    - JSON-serialisable stages (``scenes``, ``transcript``, ``segments``,
      ``captions``) as ``.json`` files.
    - Numeric arrays (``embeddings``) as ``.npz`` files.

    This is **additive** — a missing sidecar file simply means the stage
    hasn't been cached yet, and the pipeline falls through to compute it.
    """

    # Canonical ordered list of stage names.
    ALL_STAGES: list[str] = [
        "scenes",
        "transcript",
        "segments",
        "captions",
        "embeddings",
    ]

    def __init__(self, video_path: str, cache_key: str) -> None:
        self.dir = Path(video_path).with_suffix(".kuavi") / cache_key
        self.dir.mkdir(parents=True, exist_ok=True)

    # -- JSON helpers --
    def has_json(self, stage: str) -> bool:
        return (self.dir / f"{stage}.json").exists()

    def load_json(self, stage: str) -> Any:
        return json.loads((self.dir / f"{stage}.json").read_text())

    def save_json(self, stage: str, data: Any) -> None:
        (self.dir / f"{stage}.json").write_text(json.dumps(data))

    # -- NumPy helpers --
    def has_npz(self, stage: str) -> bool:
        return (self.dir / f"{stage}.npz").exists()

    def load_npz(self, stage: str) -> dict[str, np.ndarray]:
        return dict(np.load(self.dir / f"{stage}.npz"))

    def save_npz(self, stage: str, arrays: dict[str, np.ndarray]) -> None:
        np.savez(self.dir / f"{stage}.npz", **arrays)


def _should_run(stage: str, stages: list[str] | None) -> bool:
    """Return True if *stage* should be executed given an optional allow-list."""
    return stages is None or stage in stages


# Keep backward-compat alias
_is_faster_whisper_model = transcript.is_faster_whisper_model


class VideoIndexer:
    """Build a searchable :class:`VideoIndex` from a loaded video.

    Handles scene detection, optional captioning, sentence-transformer embedding,
    and Qwen3-ASR-based speech transcription.

    Args:
        embedding_model: HuggingFace model id for sentence-transformers.
        device: Torch device string (``"auto"`` lets sentence-transformers choose).
    """

    def __init__(
        self,
        embedding_model: str = "google/siglip2-base-patch16-256",
        device: str = "auto",
        temporal_window: int = 4,
        max_frames_per_segment: int = 32,
        cache_dir: str | Path | None = None,
        caption_resize: tuple[int, int] | None = None,
        embedding_stride: int | None = None,
        text_embedding_model: str | None = None,
        hierarchical: bool = False,
        scene_model: str | None = None,
        scene_clip_size: int = 16,
        scene_stride: int = 8,
        scene_model_preset: str | None = None,
    ):
        from kuavi.types import VJEPA2_PRESETS

        self._embedding_model_name = embedding_model
        self._device = device
        self._temporal_window = temporal_window
        self._max_frames_per_segment = max_frames_per_segment
        self._cache_dir = Path(cache_dir) if cache_dir is not None else None
        self._caption_resize = caption_resize
        self._embedding_stride = embedding_stride
        self._hierarchical = hierarchical
        self._model = None
        self._processor = None
        self._text_embedding_model_name = text_embedding_model
        self._text_model = None
        self._text_tokenizer = None
        self._memory_cache: dict[str, VideoIndex] = {}
        self._scene_model = None
        self._scene_processor = None
        self._scene_predictor = None
        self._asr_model = None
        self._asr_model_name: str | None = None

        if scene_model_preset is not None:
            if scene_model_preset not in VJEPA2_PRESETS:
                raise ValueError(
                    f"Unknown scene_model_preset {scene_model_preset!r}. "
                    f"Valid presets: {list(VJEPA2_PRESETS)}"
                )
            preset = VJEPA2_PRESETS[scene_model_preset]
            self._scene_model_name = preset["model"]
            self._scene_clip_size = preset["clip_size"]
            self._scene_embed_dim = preset["embed_dim"]
        else:
            self._scene_model_name = scene_model
            self._scene_clip_size = scene_clip_size
            self._scene_embed_dim = 1024  # default ViT-L
        self._scene_stride = scene_stride

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------

    def _ensure_text_model(self) -> None:
        """Lazily load a separate text embedding model if configured."""
        if self._text_embedding_model_name is None:
            return
        if self._text_model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer

            # SentenceTransformer doesn't accept "auto" — resolve device first
            device = self._device
            if device == "auto":
                import torch

                device = (
                    "mps"
                    if torch.backends.mps.is_available()
                    else ("cuda" if torch.cuda.is_available() else "cpu")
                )

            self._text_model = SentenceTransformer(
                self._text_embedding_model_name,
                device=device,
            )
            self._text_model_type = "sentence_transformers"
        except ImportError:
            from transformers import AutoModel, AutoTokenizer

            self._text_tokenizer = AutoTokenizer.from_pretrained(
                self._text_embedding_model_name,
            )
            self._text_model = AutoModel.from_pretrained(
                self._text_embedding_model_name,
            ).eval()
            self._text_model_type = "transformers"
        logger.info(
            "Loaded text embedding model %s (type=%s)",
            self._text_embedding_model_name,
            self._text_model_type,
        )

    def _free_scene_model(self) -> None:
        """Unload V-JEPA 2 from GPU to free VRAM before loading the ASR model."""
        if self._scene_model is None:
            return
        import torch

        self._scene_model = None
        self._scene_processor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("[pipeline] V-JEPA 2: unloaded from GPU (freeing VRAM for ASR)")

    def _ensure_scene_model(self) -> None:
        """Lazily load V-JEPA 2 for scene detection."""
        if self._scene_model is not None:
            return
        import torch

        device = self._device
        if device == "auto":
            device = (
                "mps"
                if torch.backends.mps.is_available()
                else ("cuda" if torch.cuda.is_available() else "cpu")
            )

        from transformers import AutoModel, AutoVideoProcessor

        self._scene_processor = AutoVideoProcessor.from_pretrained(self._scene_model_name)
        self._scene_model = (
            AutoModel.from_pretrained(self._scene_model_name, torch_dtype=torch.float16)
            .eval()
            .to(device)
        )
        self._scene_torch_device = device
        logger.info("Loaded scene model %s on %s", self._scene_model_name, device)

        # Try to access predictor from model (may not be available in HF checkpoint)
        predictor = getattr(self._scene_model, "predictor", None)
        if predictor is not None:
            self._scene_predictor = predictor
            logger.info("V-JEPA 2 predictor loaded from model checkpoint")
        else:
            self._scene_predictor = None
            logger.warning(
                "V-JEPA 2 predictor not found in HF checkpoint %s. "
                "Action anticipation will not be available. "
                "The predictor may need to be loaded from the original "
                "facebookresearch/vjepa2 repository weights.",
                self._scene_model_name,
            )

    def _ensure_model(self) -> None:
        """Lazily load the SigLIP2 model on first use."""
        if self._model is not None:
            return
        import torch
        from transformers import AutoModel, GemmaTokenizerFast, SiglipImageProcessor

        device = self._device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        elif device == "mps":
            # SigLIP2 produces degenerate (identical) embeddings on MPS;
            # fall back to CPU for correctness.
            device = "cpu"

        # AutoProcessor/AutoTokenizer crash with SigLIP2 on transformers >=5.2
        # due to a tokenizer registration bug. Load components explicitly.
        self._image_processor = SiglipImageProcessor.from_pretrained(self._embedding_model_name)
        self._tokenizer = GemmaTokenizerFast.from_pretrained(self._embedding_model_name)
        self._model = AutoModel.from_pretrained(self._embedding_model_name).eval().to(device)
        self._torch_device = device
        logger.info("Loaded embedding model %s on %s", self._embedding_model_name, device)

    # ------------------------------------------------------------------
    # Delegated encoding methods (thin wrappers preserving original API)
    # ------------------------------------------------------------------

    def _predict_future_embedding(
        self,
        context_features: np.ndarray,
        n_future_tokens: int = 16,
    ) -> np.ndarray | None:
        return emb_mod.predict_future_embedding(
            self._scene_predictor,
            getattr(self, "_scene_torch_device", "cpu"),
            context_features,
            n_future_tokens,
        )

    def _encode_frames(
        self, frames: list[np.ndarray], temporal_window: int = 1, stride: int | None = None
    ) -> np.ndarray:
        return encoding.encode_frames(
            self._model,
            self._image_processor,
            self._torch_device,
            frames,
            temporal_window,
            stride,
        )

    def _encode_texts(self, texts: list[str]) -> np.ndarray:
        self._ensure_text_model()
        return encoding.encode_texts(
            texts,
            text_embedding_model_name=self._text_embedding_model_name,
            text_model=self._text_model,
            text_model_type=getattr(self, "_text_model_type", None),
            text_tokenizer=self._text_tokenizer,
            siglip_model=self._model,
            siglip_tokenizer=self._tokenizer,
            siglip_device=self._torch_device,
        )

    def _encode_texts_siglip(self, texts: list[str]) -> np.ndarray:
        return encoding.encode_texts_siglip(
            self._model,
            self._tokenizer,
            self._torch_device,
            texts,
        )

    def _encode_query_siglip(self, text: str) -> np.ndarray:
        self._ensure_model()
        return self._encode_texts_siglip([text])[0]

    def _encode_query(self, text: str) -> np.ndarray:
        self._ensure_model()
        return self._encode_texts([text])[0]

    def _encode_clips_vjepa(
        self,
        clips: list[list[np.ndarray]],
        return_full: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, list[np.ndarray]]:
        return encoding.encode_clips_vjepa(
            self._scene_model,
            clips,
            self._scene_clip_size,
            self._scene_processor,
            getattr(self, "_scene_torch_device", "cpu"),
            return_full=return_full,
        )

    def _group_frames_into_clips(
        self,
        frames: list[np.ndarray],
        timestamps: list[float],
        clip_size: int,
    ) -> tuple[list[list[np.ndarray]], list[float]]:
        return encoding.group_frames_into_clips(frames, timestamps, clip_size)

    def _encode_frames_overlapping_vjepa(
        self,
        frames: list[np.ndarray],
        timestamps: list[float],
        clip_size: int | None = None,
        stride: int | None = None,
        store_feature_maps: bool = False,
    ):
        return encoding.encode_frames_overlapping_vjepa(
            self._encode_clips_vjepa,
            frames,
            timestamps,
            clip_size=clip_size or self._scene_clip_size,
            stride=stride or self._scene_stride,
            scene_embed_dim=self._scene_embed_dim,
            store_feature_maps=store_feature_maps,
        )

    # ------------------------------------------------------------------
    # Delegated transcript methods
    # ------------------------------------------------------------------

    def _get_transcript(
        self,
        video_path: str,
        *,
        asr_model: str = "Qwen/Qwen3-ASR-0.6B",
        transcript_path: str | None = None,
    ) -> list[dict]:
        result, fw_model, fw_size = transcript.get_transcript(
            video_path,
            asr_model_name=asr_model,
            transcript_path=transcript_path,
            asr_model=self._asr_model,
            asr_batch_size=getattr(self, "_asr_batch_size", None),
            _faster_whisper_model=getattr(self, "_faster_whisper_model", None),
            _faster_whisper_model_size=getattr(self, "_faster_whisper_model_size", None),
        )
        # Cache faster-whisper model if used
        if fw_model is not None:
            self._faster_whisper_model = fw_model
            self._faster_whisper_model_size = fw_size
        return result

    def _ensure_asr_model(self, model_name: str) -> None:
        model, name, batch_size = transcript.ensure_asr_model(
            self._asr_model,
            self._asr_model_name,
            model_name,
            self._device,
        )
        self._asr_model = model
        self._asr_model_name = name
        self._asr_batch_size = batch_size

    @staticmethod
    def _load_transcript_file(path: str) -> list[dict]:
        return transcript.load_transcript_file(path)

    @staticmethod
    def _extract_audio(video_path: str, out_wav: str) -> bool:
        return transcript.extract_audio(video_path, out_wav)

    # Batch size constants (kept for backward compat)
    _ASR_BATCH_CPU = transcript.ASR_BATCH_CPU
    _ASR_BATCH_CUDA = transcript.ASR_BATCH_CUDA
    _ASR_CHUNK_SEC = transcript.ASR_CHUNK_SEC
    _ASR_OVERLAP_SEC = transcript.ASR_OVERLAP_SEC

    @staticmethod
    def _split_audio_chunks(
        wav_path: str,
        chunk_sec: int,
        tmp_dir: str,
        overlap_sec: float = 0.0,
    ) -> list[tuple[str, float]]:
        return transcript.split_audio_chunks(wav_path, chunk_sec, tmp_dir, overlap_sec)

    def _run_faster_whisper(self, video_path: str, model_name: str) -> list[dict]:
        result, model, size = transcript.run_faster_whisper(
            video_path,
            model_name,
            _cached_model=getattr(self, "_faster_whisper_model", None),
            _cached_model_size=getattr(self, "_faster_whisper_model_size", None),
        )
        if model is not None:
            self._faster_whisper_model = model
            self._faster_whisper_model_size = size
        return result

    def _run_asr(self, video_path: str, model_name: str) -> list[dict]:
        result, fw_model, fw_size = transcript.run_asr(
            video_path,
            model_name,
            asr_model=self._asr_model,
            asr_batch_size=getattr(self, "_asr_batch_size", None),
            _faster_whisper_model=getattr(self, "_faster_whisper_model", None),
            _faster_whisper_model_size=getattr(self, "_faster_whisper_model_size", None),
        )
        if fw_model is not None:
            self._faster_whisper_model = fw_model
            self._faster_whisper_model_size = fw_size
        return result

    @staticmethod
    def _collect_transcript_segments(
        asr_result,
        offset: float,
        transcript_list: list[dict],
        *,
        skip_before: float = 0.0,
    ) -> None:
        transcript.collect_transcript_segments(
            asr_result,
            offset,
            transcript_list,
            skip_before=skip_before,
        )

    # ------------------------------------------------------------------
    # Delegated captioning / dedup / embedding methods
    # ------------------------------------------------------------------

    def _pre_caption_dedup(self, segments: list[dict], threshold: float = 0.90) -> None:
        self._ensure_model()
        dedup.pre_caption_dedup(segments, self._encode_frames, threshold)

    def _semantic_deduplicate(
        self,
        segment_infos: list[dict],
        embeddings: np.ndarray | None,
        action_embeddings: np.ndarray | None = None,
        n_clusters: int | None = None,
        similarity_threshold: float = 0.92,
    ) -> np.ndarray | None:
        return dedup.semantic_deduplicate(
            segment_infos,
            embeddings,
            action_embeddings,
            n_clusters,
            similarity_threshold,
        )

    def _filter_edge_frames(self, seg_frames: list, threshold: float = 0.5) -> list:
        return caption_pipeline.filter_edge_frames(seg_frames, self._encode_frames, threshold)

    def _check_embedding_quality(self, embeddings: np.ndarray, label: str = "caption") -> dict:
        return emb_mod.check_embedding_quality(embeddings, label)

    @staticmethod
    def _smooth_embeddings(embs: np.ndarray, window: int = 3) -> np.ndarray:
        return emb_mod.smooth_embeddings(embs, window)

    def _embed_captions(
        self,
        segments: list[dict],
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        self._ensure_model()
        return emb_mod.embed_captions(segments, self._encode_texts)

    def _deduplicate_segments(self, segments: list[dict], threshold: float = 0.95) -> None:
        self._ensure_model()
        dedup.deduplicate_segments(segments, self._encode_texts, threshold)

    def _global_deduplicate(self, segments: list[dict], threshold: float = 0.90) -> None:
        self._ensure_model()
        dedup.global_deduplicate(segments, self._encode_texts, threshold)

    @staticmethod
    def _transcript_for_range(
        transcript_list: list[dict],
        start: float,
        end: float,
    ) -> str:
        return caption_pipeline.transcript_for_range(transcript_list, start, end)

    def _refine_annotations(
        self,
        segments: list[dict],
        transcript_list: list[dict],
        refine_fn: Callable | None,
        video_metadata=None,
        rounds: int = 3,
    ) -> None:
        caption_pipeline.refine_annotations(
            segments,
            transcript_list,
            refine_fn,
            video_metadata,
            rounds,
        )

    @staticmethod
    def _score_format_compliance(seg: dict) -> float:
        return caption_pipeline.score_format_compliance(seg)

    @staticmethod
    def _score_action_frequency(segments: list[dict]) -> None:
        caption_pipeline.score_action_frequency(segments)

    def _score_annotations(
        self,
        segments: list[dict],
        loaded_video_frames: list[np.ndarray],
        timestamps: list[float],
        min_similarity: float = 0.3,
    ) -> None:
        self._ensure_model()
        caption_pipeline.score_annotations(
            segments,
            loaded_video_frames,
            timestamps,
            self._encode_frames,
            self._encode_texts,
            self._text_embedding_model_name,
            min_similarity,
        )

    def _fix_low_quality_annotations(
        self,
        segments: list[dict],
        loaded_video_frames: list[np.ndarray],
        timestamps: list[float],
        caption_fn: Callable | None = None,
        threshold: float = 0.3,
        num_retries: int = 3,
    ) -> None:
        caption_pipeline.fix_low_quality_annotations(
            segments,
            loaded_video_frames,
            timestamps,
            caption_fn,
            threshold,
            num_retries,
        )

    def _selective_decode(
        self,
        segments: list[dict],
        frames: list[np.ndarray],
        timestamps: list[float],
        similarity_threshold: float = 0.98,
        temporal_clip_embeddings: np.ndarray | None = None,
        temporal_clip_timestamps: list[float] | None = None,
    ) -> None:
        self._ensure_model()
        caption_pipeline.selective_decode(
            segments,
            frames,
            timestamps,
            self._encode_frames,
            similarity_threshold,
            temporal_clip_embeddings,
            temporal_clip_timestamps,
        )

    def _action_first_pass(
        self,
        segment_infos: list[dict],
        frame_caption_fn: Callable | None,
    ) -> None:
        caption_pipeline.action_first_pass(segment_infos, frame_caption_fn)

    def _build_coarse_level(
        self,
        segments: list[dict],
        embeddings: np.ndarray,
        target_duration: float = 30.0,
    ) -> tuple[list[dict], np.ndarray | None]:
        return emb_mod.build_coarse_level(segments, embeddings, target_duration)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def index_video(
        self,
        loaded_video: LoadedVideo,
        *,
        caption_fn: Callable | None = None,
        frame_caption_fn: Callable | None = None,
        refine_fn: Callable | None = None,
        asr_model: str = "Qwen/Qwen3-ASR-0.6B",
        transcript_path: str | None = None,
        refine_rounds: int = 0,
        mode: str = "full",
        store_feature_maps: bool = False,
        overlapping_vjepa: bool = False,
        semantic_dedup: bool = False,
        force_reindex: bool = False,
        stages: list[str] | None = None,
    ) -> VideoIndex:
        """Build a full searchable index from a loaded video.

        Args:
            loaded_video: A :class:`LoadedVideo` returned by :class:`VideoLoader`.
            caption_fn: Optional function that produces a caption for a list of
                frames.  May return a plain string (backward-compatible) or a
                structured annotation dict with ``summary`` and ``action`` keys.
            frame_caption_fn: Optional function that captions a single keyframe.
            refine_fn: Optional function ``(draft: str, context: str) -> str``
                used for Self-Refine.
            asr_model: Qwen3-ASR model name for speech transcription.
            transcript_path: Path to a pre-existing transcript JSON/SRT file.
            refine_rounds: Number of Self-Refine iterations. Default 0 (single-pass
                captioning). Set to 3 for the original multi-round refinement.
            mode: Indexing mode — ``"full"`` (default) runs the captioning
                pipeline; ``"fast"`` skips segment captioning, using only
                midpoint frame captions to produce a quickly searchable index.
            force_reindex: When True, ignore sidecar stage caches and recompute
                every stage from scratch.
            stages: Optional allow-list of stage names to run.  Valid names are
                ``scenes``, ``transcript``, ``segments``, ``captions``,
                ``embeddings``.  When ``None`` (default) all stages run.

        Returns:
            A :class:`VideoIndex` ready for use with the search functions in
            :mod:`kuavi.search`.
        """
        # --- In-memory / disk cache lookup ---
        mem_key: str | None = None
        try:
            mem_key = _cache_key(loaded_video.metadata.path)
        except (FileNotFoundError, OSError):
            pass

        if not force_reindex and stages is None:
            if mem_key is not None and mem_key in self._memory_cache:
                logger.info("Returning in-memory cached index for %s", loaded_video.metadata.path)
                return self._memory_cache[mem_key]

        cache_path: Path | None = None
        if mem_key is not None and self._cache_dir is not None:
            cache_path = self._cache_dir / mem_key
            if not force_reindex and stages is None and (cache_path / "metadata.json").exists():
                logger.info("Loading cached index from %s", cache_path)
                idx = VideoIndex.load(cache_path)
                idx.embed_fn = self._encode_query
                self._memory_cache[mem_key] = idx
                return idx

        # --- Stage-level sidecar cache ---
        sc: _StageCache | None = None
        if mem_key is not None:
            sc = _StageCache(loaded_video.metadata.path, mem_key)

        fps = loaded_video.metadata.extraction_fps
        frames = loaded_video.frames

        # 1. Compute per-frame timestamps
        timestamps = [i / fps for i in range(len(frames))]

        # 2. Detect scene boundaries
        hierarchy_result: dict | None = None
        vjepa_clip_embeddings: np.ndarray | None = None
        vjepa_clip_timestamps: list[float] | None = None
        vjepa_clip_feature_maps: list[np.ndarray] | None = None

        _scenes_cached = False
        if (
            not force_reindex
            and _should_run("scenes", stages)
            and sc is not None
            and sc.has_json("scenes")
        ):
            logger.info("[stage-cache] loading scenes from sidecar")
            _sc_data = sc.load_json("scenes")
            scenes = [tuple(s) for s in _sc_data["scenes"]]
            scene_boundaries = _sc_data["scene_boundaries"]
            hierarchy_result = _sc_data.get("hierarchy_result")
            _scenes_cached = True
        elif _should_run("scenes", stages):
            if self._scene_model_name and overlapping_vjepa:
                # Overlapping V-JEPA 2 windows with per-frame averaging
                from kuavi.scene_detection import detect_scenes_perframe

                self._ensure_scene_model()
                logger.info(
                    "[pipeline] V-JEPA 2: detecting scenes (overlapping windows, stride=%d)",
                    self._scene_stride,
                )
                ovl_result = self._encode_frames_overlapping_vjepa(
                    frames,
                    timestamps,
                    clip_size=self._scene_clip_size,
                    stride=self._scene_stride,
                    store_feature_maps=store_feature_maps,
                )
                if store_feature_maps and len(ovl_result) == 3:
                    per_frame_embs, _, ovl_feature_maps = ovl_result
                    # Feature maps are per-window; compute window midpoint timestamps
                    # so downstream segment aggregation can index correctly.
                    ovl_window_ts: list[float] = []
                    for start in range(0, len(frames), self._scene_stride):
                        end = min(start + self._scene_clip_size, len(frames))
                        if end - start < 2:
                            continue
                        mid = min(start + (end - start) // 2, len(frames) - 1)
                        ovl_window_ts.append(timestamps[mid])
                    # Store window-level feature maps with their own timestamps;
                    # these will be used for per-segment aggregation after scene
                    # detection (overrides vjepa_clip_feature_maps).
                    vjepa_clip_feature_maps = ovl_feature_maps
                    # We need a separate timestamps array for feature maps that
                    # aligns with vjepa_clip_feature_maps (per-window, not per-frame).
                    # Store it so the aggregation code below can use it.
                    _fmap_timestamps = ovl_window_ts
                else:
                    per_frame_embs = ovl_result[0]
                    _fmap_timestamps = None

                if self._hierarchical:
                    levels = []
                    for thresh, min_dur in zip((0.10, 0.20, 0.35), (0.5, 2.0, 4.0), strict=False):
                        scenes_level = detect_scenes_perframe(
                            per_frame_embs, timestamps, threshold=thresh, min_duration=min_dur
                        )
                        levels.append(scenes_level)
                    hierarchy_result = {"levels": levels}
                    scenes = hierarchy_result["levels"][0]
                else:
                    scenes = detect_scenes_perframe(per_frame_embs, timestamps, threshold=0.20)

                # Store per-frame embeddings as temporal embeddings (segment-averaged later)
                vjepa_clip_embeddings = per_frame_embs
                vjepa_clip_timestamps = timestamps
                logger.info("[pipeline] V-JEPA 2 (overlapping): %d scenes detected", len(scenes))
            elif self._scene_model_name:
                # V-JEPA 2 clip-level scene detection (non-overlapping, default)
                self._ensure_scene_model()
                logger.info("[pipeline] V-JEPA 2: detecting scenes")
                clips, clip_timestamps = self._group_frames_into_clips(
                    frames, timestamps, self._scene_clip_size
                )

                # Compute clip embeddings once and cache for reuse
                if store_feature_maps:
                    vjepa_clip_embeddings, vjepa_clip_feature_maps = self._encode_clips_vjepa(
                        clips, return_full=True
                    )
                else:
                    vjepa_clip_embeddings = self._encode_clips_vjepa(clips)
                vjepa_clip_timestamps = clip_timestamps

                def _vjepa_embed_fn(_frames):
                    return vjepa_clip_embeddings

                clip_representatives = [c[len(c) // 2] for c in clips]

                if self._hierarchical:
                    hierarchy_result = detect_scenes_hierarchical(
                        clip_representatives,
                        clip_timestamps,
                        embed_fn=_vjepa_embed_fn,
                    )
                    scenes = hierarchy_result["levels"][0]
                else:
                    scenes = detect_scenes(
                        clip_representatives, clip_timestamps, embed_fn=_vjepa_embed_fn
                    )
                logger.info("[pipeline] V-JEPA 2: %d scenes detected", len(scenes))
            else:
                # Existing SigLIP2 path
                self._ensure_model()

                def _scene_embed_fn(f):
                    return self._encode_frames(
                        f, temporal_window=self._temporal_window, stride=self._embedding_stride
                    )

                if self._hierarchical:
                    hierarchy_result = detect_scenes_hierarchical(
                        frames,
                        timestamps,
                        embed_fn=_scene_embed_fn,
                    )
                    scenes = hierarchy_result["levels"][0]
                else:
                    scenes = detect_scenes(frames, timestamps, embed_fn=_scene_embed_fn)
            scene_boundaries = [start for start, _end in scenes]

            # Save scenes to sidecar cache
            if sc is not None:
                sc.save_json(
                    "scenes",
                    {
                        "scenes": [list(s) for s in scenes],
                        "scene_boundaries": scene_boundaries,
                        "hierarchy_result": hierarchy_result,
                    },
                )
                logger.info("[stage-cache] saved scenes to sidecar")
        else:
            # Stage skipped — initialise with empty defaults
            scenes = []
            scene_boundaries = []

        # 3. Build segment dicts — prefer existing segments, fall back to scenes
        if loaded_video.segments:
            segment_infos = self._segments_from_loaded(loaded_video)
        else:
            segment_infos = self._segments_from_scenes(scenes, frames, timestamps)

        # 4. Transcript (Qwen3-ASR or pre-existing file) — run before captioning
        #    so ASR context can be injected into caption prompts.
        #    Free V-JEPA 2 from GPU first — it stays loaded after scene detection
        #    and leaves too little VRAM for the ASR model on 11-12 GiB GPUs.
        if (
            not force_reindex
            and _should_run("transcript", stages)
            and sc is not None
            and sc.has_json("transcript")
        ):
            logger.info("[stage-cache] loading transcript from sidecar")
            transcript_data = sc.load_json("transcript")
        elif _should_run("transcript", stages):
            self._free_scene_model()
            transcript_data = self._get_transcript(
                loaded_video.metadata.path,
                asr_model=asr_model,
                transcript_path=transcript_path,
            )
            if sc is not None:
                sc.save_json("transcript", transcript_data)
                logger.info("[stage-cache] saved transcript to sidecar")
        else:
            self._free_scene_model()
            transcript_data = []

        # 4b. Pre-captioning dedup: identify visually similar segments
        #     and only caption representatives, propagating results afterward.
        if (
            not force_reindex
            and _should_run("captions", stages)
            and sc is not None
            and sc.has_json("captions")
        ):
            logger.info("[stage-cache] loading captions from sidecar")
            segment_infos = sc.load_json("captions")
        elif _should_run("captions", stages):
            self._pre_caption_dedup(segment_infos)

            # 4c. Selective decoding: 3-tier (dead / static-informative / dynamic)
            self._selective_decode(
                segment_infos,
                frames,
                timestamps,
                temporal_clip_embeddings=vjepa_clip_embeddings,
                temporal_clip_timestamps=vjepa_clip_timestamps,
            )

            if mode == "fast":
                # Fast mode: use midpoint frame captions only — skip Tree-of-Captions and Self-Refine.
                if frame_caption_fn is not None:
                    logger.info("[pipeline] captioning: starting fast-mode frame captioning")
                    logger.info(
                        "[pipeline] captioning: starting fast-mode for %d segments",
                        len(segment_infos),
                    )
                self._action_first_pass(segment_infos, frame_caption_fn)

                # 5c (fast). Propagate captions from representatives to skipped duplicates
                for seg in segment_infos:
                    src_idx = seg.get("_caption_source")
                    if src_idx is not None:
                        src = segment_infos[src_idx]
                        for key in ("caption", "annotation", "frame_caption", "is_non_action"):
                            if key in src:
                                seg[key] = src[key]

                captioned_count = sum(
                    1 for s in segment_infos if s.get("caption") or s.get("frame_caption")
                )
                if frame_caption_fn is not None:
                    logger.info("[pipeline] captioning: %d segments captioned", captioned_count)
                else:
                    logger.info("[pipeline] captioning: skipped (fast mode, no caption model)")

                # Clean up temporary dedup keys
                for seg in segment_infos:
                    seg.pop("_skip_caption", None)
                    seg.pop("_caption_source", None)
            else:
                # Full mode: Tree-of-Captions + Self-Refine (original behavior)

                # 5. Caption each segment (if a caption function was provided)
                logger.info("[pipeline] captioning: starting segment captioning")
                if caption_fn is not None or frame_caption_fn is not None:
                    logger.info(
                        "[pipeline] captioning: starting for %d segments", len(segment_infos)
                    )
                    # Prepare all segments first (skip near-duplicates)
                    caption_tasks = []
                    for seg in segment_infos:
                        seg_frames = seg.pop("_frames")
                        if seg.get("_skip_caption"):
                            continue
                        # ASR context injection: prepend transcript text for this segment
                        transcript_text = self._transcript_for_range(
                            transcript_data,
                            seg["start_time"],
                            seg["end_time"],
                        )
                        if transcript_text:
                            seg_frames = [f"[transcript] {transcript_text}"] + seg_frames
                        caption_tasks.append((seg, seg_frames))

                    # 5b. Segment-level captioning
                    if caption_fn is not None:

                        def _caption_segment(args):
                            seg, seg_frames = args
                            # Filter visually dissimilar edge frames
                            if self._model is not None:
                                real_frames = [f for f in seg_frames if not isinstance(f, str)]
                                if len(real_frames) >= 5:
                                    filtered_real = self._filter_edge_frames(
                                        real_frames, threshold=0.5
                                    )
                                    str_tokens = [f for f in seg_frames if isinstance(f, str)]
                                    seg_frames = str_tokens + filtered_real
                            # Resize real frames for captioning if caption_resize is set
                            if self._caption_resize:
                                import cv2

                                resized = []
                                for f in seg_frames:
                                    if isinstance(f, str):
                                        resized.append(f)
                                    else:
                                        resized.append(cv2.resize(f, self._caption_resize))
                                seg_frames = resized
                            result = caption_fn(seg_frames)
                            # Backward compat: wrap plain strings into structured annotation
                            if isinstance(result, str):
                                annotation = {
                                    "summary": {"brief": result, "detailed": result},
                                    "action": {"brief": "", "detailed": "", "actor": None},
                                }
                            else:
                                annotation = result
                            return seg, annotation

                        with ThreadPoolExecutor(max_workers=8) as pool:
                            futures = [
                                pool.submit(_caption_segment, task) for task in caption_tasks
                            ]
                            for future in as_completed(futures):
                                try:
                                    seg, annotation = future.result()
                                    seg["annotation"] = annotation
                                    seg["caption"] = annotation.get("summary", {}).get("brief", "")
                                    action_brief = (
                                        annotation.get("action", {}).get("brief", "").strip()
                                    )
                                    if not action_brief or action_brief.upper() == "N/A":
                                        seg["is_non_action"] = True
                                except Exception:
                                    logger.warning(
                                        "Caption future raised an exception", exc_info=True
                                    )
                else:
                    for seg in segment_infos:
                        seg.pop("_frames", None)

                captioned = sum(1 for s in segment_infos if s.get("caption"))
                logger.info("[pipeline] captioning: %d segments captioned", captioned)

                # 5c. Propagate captions from representatives to skipped duplicates
                for seg in segment_infos:
                    src_idx = seg.get("_caption_source")
                    if src_idx is not None:
                        src = segment_infos[src_idx]
                        for key in ("caption", "annotation", "frame_caption", "is_non_action"):
                            if key in src:
                                seg[key] = src[key]

                # Clean up temporary dedup keys
                for seg in segment_infos:
                    seg.pop("_skip_caption", None)
                    seg.pop("_caption_source", None)

                captioned_count = sum(1 for s in segment_infos if s.get("caption"))
                logger.info("[pipeline] captioning: %d segments captioned", captioned_count)

                # 6. Self-Refine annotations
                self._refine_annotations(
                    segment_infos,
                    transcript_data,
                    refine_fn,
                    video_metadata=loaded_video.metadata,
                    rounds=refine_rounds,
                )

                # 6b. Mark near-duplicate adjacent segments before embedding
                self._deduplicate_segments(segment_infos)

                # 6c. Global dedup: find duplicates anywhere (non-adjacent)
                self._global_deduplicate(segment_infos)

                # 6.5 Score annotation quality
                self._score_annotations(
                    segment_infos,
                    loaded_video_frames=frames,
                    timestamps=timestamps,
                )

            # Save captioned segments to sidecar cache
            if sc is not None:
                sc.save_json("captions", segment_infos)
                logger.info("[stage-cache] saved captions to sidecar")

        # 7. Embed captions
        if (
            not force_reindex
            and _should_run("embeddings", stages)
            and sc is not None
            and sc.has_npz("embeddings")
        ):
            logger.info("[stage-cache] loading embeddings from sidecar")
            _emb_data = sc.load_npz("embeddings")
            embeddings = _emb_data.get("embeddings")
            action_embeddings = _emb_data.get("action_embeddings")
            frame_embeddings = _emb_data.get("frame_embeddings")
            temporal_embeddings = _emb_data.get("temporal_embeddings")
            temporal_feature_maps = _emb_data.get("temporal_feature_maps")
            _emb_meta = sc.load_json("embeddings_meta") if sc.has_json("embeddings_meta") else {}
            quality = _emb_meta.get("quality", {})
            segment_hierarchy = _emb_meta.get("segment_hierarchy", [])
            hierarchy_embeddings_raw: list[np.ndarray | None] = []
            lvl = 0
            while f"hierarchy_emb_L{lvl}" in _emb_data:
                hierarchy_embeddings_raw.append(_emb_data[f"hierarchy_emb_L{lvl}"])
                lvl += 1
            hierarchy_embeddings = hierarchy_embeddings_raw
        elif _should_run("embeddings", stages):
            if self._text_embedding_model_name is not None:
                logger.info(
                    "[pipeline] Gemma: embedding captions for %d segments", len(segment_infos)
                )
            embeddings, action_embeddings = self._embed_captions(segment_infos)

            # 7b. Smooth embeddings to reduce noise across adjacent segments
            if embeddings is not None:
                embeddings = self._smooth_embeddings(embeddings, window=3)
            if action_embeddings is not None:
                action_embeddings = self._smooth_embeddings(action_embeddings, window=3)

            quality = self._check_embedding_quality(embeddings, label="caption")
            if self._text_embedding_model_name is not None:
                logger.info("[pipeline] Gemma: caption embeddings complete")

            # 7b2. Semantic deduplication via k-means clustering (optional)
            if semantic_dedup:
                self._semantic_deduplicate(
                    segment_infos,
                    embeddings,
                    action_embeddings=action_embeddings,
                )

            # 7c. Embed representative frame per segment for visual search
            rep_frames = []
            for seg in segment_infos:
                seg_frames_list = [
                    f
                    for f, t in zip(frames, timestamps, strict=False)
                    if seg["start_time"] <= t <= seg["end_time"]
                ]
                if seg_frames_list:
                    rep_frames.append(seg_frames_list[len(seg_frames_list) // 2])
                else:
                    rep_frames.append(frames[0])  # fallback

            logger.info(
                "[pipeline] SigLIP2: building frame embeddings for %d segments", len(rep_frames)
            )
            self._ensure_model()
            frame_embeddings = self._encode_frames(rep_frames)
            frame_embeddings = self._smooth_embeddings(frame_embeddings, window=3)
            self._check_embedding_quality(frame_embeddings, label="frame")
            logger.info("[pipeline] SigLIP2: %d frame embeddings built", len(rep_frames))

            # 7d. Aggregate V-JEPA 2 temporal embeddings per segment
            temporal_embeddings: np.ndarray | None = None
            temporal_feature_maps: np.ndarray | None = None
            # Feature maps may use different timestamps than clip embeddings
            # (e.g. overlapping path: embeddings are per-frame, feature maps per-window).
            fmap_ts = locals().get("_fmap_timestamps") or vjepa_clip_timestamps
            if vjepa_clip_embeddings is not None and vjepa_clip_timestamps is not None:
                temporal_per_seg: list[np.ndarray] = []
                feature_maps_per_seg: list[np.ndarray] = []
                for seg in segment_infos:
                    clip_indices = [
                        i
                        for i, ct in enumerate(vjepa_clip_timestamps)
                        if seg["start_time"] <= ct <= seg["end_time"]
                    ]
                    if clip_indices:
                        seg_emb = vjepa_clip_embeddings[clip_indices].mean(axis=0)
                        norm = np.linalg.norm(seg_emb)
                        if norm > 1e-10:
                            seg_emb = seg_emb / norm
                        temporal_per_seg.append(seg_emb)
                        if vjepa_clip_feature_maps is not None:
                            # Use fmap_ts (may differ from vjepa_clip_timestamps)
                            fmap_indices = [
                                i
                                for i, ct in enumerate(fmap_ts)
                                if seg["start_time"] <= ct <= seg["end_time"]
                            ]
                            if fmap_indices:
                                maps = [vjepa_clip_feature_maps[i] for i in fmap_indices]
                                shapes = {m.shape for m in maps}
                                if len(shapes) == 1:
                                    # All same shape — average across clips
                                    feature_maps_per_seg.append(np.stack(maps).mean(axis=0))
                                else:
                                    # Variable shapes (different clip lengths) —
                                    # pick the clip closest to segment midpoint
                                    seg_mid = (seg["start_time"] + seg["end_time"]) / 2
                                    best_idx = min(
                                        fmap_indices,
                                        key=lambda i: abs(fmap_ts[i] - seg_mid),
                                    )
                                    feature_maps_per_seg.append(vjepa_clip_feature_maps[best_idx])
                            else:
                                num_patches = vjepa_clip_feature_maps[0].shape[0]
                                patch_dim = vjepa_clip_feature_maps[0].shape[1]
                                feature_maps_per_seg.append(np.zeros((num_patches, patch_dim)))
                    else:
                        temporal_per_seg.append(np.zeros(vjepa_clip_embeddings.shape[1]))
                        if vjepa_clip_feature_maps is not None:
                            num_patches = vjepa_clip_feature_maps[0].shape[0]
                            patch_dim = vjepa_clip_feature_maps[0].shape[1]
                            feature_maps_per_seg.append(np.zeros((num_patches, patch_dim)))
                temporal_embeddings = np.stack(temporal_per_seg)
                temporal_embeddings = self._smooth_embeddings(temporal_embeddings, window=3)
                self._check_embedding_quality(temporal_embeddings, label="temporal")
                if vjepa_clip_feature_maps is not None and feature_maps_per_seg:
                    shapes = {m.shape for m in feature_maps_per_seg}
                    if len(shapes) == 1:
                        temporal_feature_maps = np.stack(feature_maps_per_seg)
                    else:
                        # Variable patch counts across segments — store as object array
                        temporal_feature_maps = np.empty(len(feature_maps_per_seg), dtype=object)
                        for i, m in enumerate(feature_maps_per_seg):
                            temporal_feature_maps[i] = m

            # 8. Build hierarchy levels (when hierarchical mode is enabled)
            segment_hierarchy: list[list[dict]] = []
            hierarchy_embeddings: list[np.ndarray | None] = []
            if hierarchy_result is not None and len(hierarchy_result["levels"]) > 1:
                for lvl_idx in range(1, len(hierarchy_result["levels"])):
                    lvl_scenes = hierarchy_result["levels"][lvl_idx]
                    lvl_segments: list[dict] = []
                    for h_start, h_end in lvl_scenes:
                        # Find child segments from level 0 that fall within this range
                        child_captions = [
                            seg.get("caption", "")
                            for seg in segment_infos
                            if seg["start_time"] >= h_start and seg["end_time"] <= h_end
                        ]
                        merged_caption = " ".join(c for c in child_captions if c)
                        lvl_segments.append(
                            {
                                "start_time": h_start,
                                "end_time": h_end,
                                "caption": merged_caption,
                            }
                        )
                    segment_hierarchy.append(lvl_segments)

                    # Embed the merged captions for this level
                    lvl_captions = [s["caption"] for s in lvl_segments]
                    if any(lvl_captions):
                        lvl_emb = self._embed_captions(lvl_segments)[0]  # summary only
                        if lvl_emb is not None:
                            lvl_emb = self._smooth_embeddings(lvl_emb, window=3)
                        hierarchy_embeddings.append(lvl_emb)
                    else:
                        hierarchy_embeddings.append(None)

            # Always add a fixed-duration coarse level for multi-scale search
            if embeddings is not None:
                coarse_segs, coarse_embs = self._build_coarse_level(
                    segment_infos, embeddings, target_duration=30.0
                )
                if coarse_segs:
                    segment_hierarchy.append(coarse_segs)
                    hierarchy_embeddings.append(coarse_embs)

            # Save embeddings to sidecar cache
            if sc is not None:
                _emb_arrays: dict[str, np.ndarray] = {}
                if embeddings is not None:
                    _emb_arrays["embeddings"] = embeddings
                if action_embeddings is not None:
                    _emb_arrays["action_embeddings"] = action_embeddings
                if frame_embeddings is not None:
                    _emb_arrays["frame_embeddings"] = frame_embeddings
                if temporal_embeddings is not None:
                    _emb_arrays["temporal_embeddings"] = temporal_embeddings
                if temporal_feature_maps is not None and temporal_feature_maps.dtype != object:
                    _emb_arrays["temporal_feature_maps"] = temporal_feature_maps
                for _lvl, _h_emb in enumerate(hierarchy_embeddings):
                    if _h_emb is not None:
                        _emb_arrays[f"hierarchy_emb_L{_lvl}"] = _h_emb
                if _emb_arrays:
                    sc.save_npz("embeddings", _emb_arrays)
                sc.save_json(
                    "embeddings_meta",
                    {
                        "quality": quality,
                        "segment_hierarchy": segment_hierarchy,
                    },
                )
                logger.info("[stage-cache] saved embeddings to sidecar")
        else:
            # Embeddings stage skipped — initialise with empty defaults
            embeddings = None
            action_embeddings = None
            frame_embeddings = None
            temporal_embeddings = None
            temporal_feature_maps = None
            quality = {}
            segment_hierarchy = []
            hierarchy_embeddings = []

        logger.info(
            "[pipeline] search index: %d segments, %d scenes",
            len(segment_infos),
            len(scene_boundaries),
        )

        index = VideoIndex(
            segments=segment_infos,
            embeddings=embeddings,
            action_embeddings=action_embeddings,
            frame_embeddings=frame_embeddings,
            temporal_embeddings=temporal_embeddings,
            temporal_feature_maps=temporal_feature_maps,
            transcript=transcript_data,
            scene_boundaries=scene_boundaries,
            embedding_quality=quality,
            embed_fn=self._encode_query,
            visual_embed_fn=self._encode_query_siglip,
            segment_hierarchy=segment_hierarchy,
            hierarchy_embeddings=hierarchy_embeddings,
        )

        # --- Attach predictor closures so search tools can use them ---
        if self._scene_predictor is not None:
            indexer_ref = self

            def _predict_fn(time_point: float) -> np.ndarray | None:
                """Predict future embedding from a time point using V-JEPA 2 predictor."""
                # Find the segment at/just before time_point
                seg_idx = None
                for i, seg in enumerate(index.segments):
                    if seg["start_time"] <= time_point <= seg["end_time"]:
                        seg_idx = i
                        break
                    if seg["end_time"] <= time_point:
                        seg_idx = i
                if seg_idx is None:
                    return None
                # Need feature maps to feed the predictor
                if index.temporal_feature_maps is None or seg_idx >= len(
                    index.temporal_feature_maps
                ):
                    return None
                feature_map = index.temporal_feature_maps[seg_idx]
                predicted = indexer_ref._predict_future_embedding(feature_map, 16)
                if predicted is None:
                    return None
                # Mean-pool to a single embedding vector
                return predicted.mean(axis=0)

            def _predict_future_fn(
                feature_map: np.ndarray, n_future_tokens: int = 16
            ) -> np.ndarray | None:
                return indexer_ref._predict_future_embedding(feature_map, n_future_tokens)

            index._predict_fn = _predict_fn
            index._predict_future_fn = _predict_future_fn

        # --- Cache save ---
        if cache_path is not None:
            try:
                index.save(cache_path)
                logger.info("Saved index cache to %s", cache_path)
            except Exception:
                logger.warning("Failed to save index cache to %s", cache_path, exc_info=True)

        if mem_key is not None:
            self._memory_cache[mem_key] = index
        return index

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _segments_from_loaded(self, loaded_video: LoadedVideo) -> list[dict]:
        """Convert :class:`VideoSegment` objects to plain dicts."""
        results: list[dict] = []
        for seg in loaded_video.segments:
            seg_frames = seg.frames
            if self._max_frames_per_segment and len(seg_frames) > self._max_frames_per_segment:
                step = len(seg_frames) / self._max_frames_per_segment
                seg_frames = [
                    seg_frames[int(i * step)] for i in range(self._max_frames_per_segment)
                ]
            results.append(
                {
                    "start_time": seg.start_time,
                    "end_time": seg.end_time,
                    "caption": "",
                    "_frames": seg_frames,
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
                f for f, t in zip(frames, timestamps, strict=False) if start <= t < end or t == end
            ]
            # Cap frames per segment for memory/cost efficiency
            if self._max_frames_per_segment and len(seg_frames) > self._max_frames_per_segment:
                step = len(seg_frames) / self._max_frames_per_segment
                seg_frames = [
                    seg_frames[int(i * step)] for i in range(self._max_frames_per_segment)
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

    def enhance_index(
        self,
        index: VideoIndex,
        loaded_video: LoadedVideo,
        *,
        caption_fn: Callable | None = None,
        frame_caption_fn: Callable | None = None,
        refine_fn: Callable | None = None,
        refine_rounds: int = 3,
    ) -> VideoIndex:
        """Run full captioning and Self-Refine on a fast-mode :class:`VideoIndex`.

        Takes an index produced by ``index_video(mode="fast")`` and runs the
        full Tree-of-Captions + Self-Refine pipeline on the segments, returning
        an updated :class:`VideoIndex` with richer annotations and embeddings.

        Args:
            index: Existing :class:`VideoIndex` (typically from fast-mode indexing).
            loaded_video: The original :class:`LoadedVideo` used to build the index.
            caption_fn: Segment-level captioning function (Tree-of-Captions node).
            frame_caption_fn: Keyframe captioning function (Tree-of-Captions leaf).
            refine_fn: Self-Refine function ``(draft, context, effort) -> str``.
            refine_rounds: Number of Self-Refine iterations.

        Returns:
            The same :class:`VideoIndex` instance with updated segments,
            embeddings, and ``embed_fn`` re-attached.
        """
        fps = loaded_video.metadata.extraction_fps
        frames = loaded_video.frames
        timestamps = [i / fps for i in range(len(frames))]
        transcript_data = index.transcript
        segment_infos = index.segments

        # Re-populate _frames for each segment from the loaded video
        for seg in segment_infos:
            seg_frames = [
                f
                for f, t in zip(frames, timestamps, strict=False)
                if seg["start_time"] <= t <= seg["end_time"]
            ]
            if self._max_frames_per_segment and len(seg_frames) > self._max_frames_per_segment:
                step = len(seg_frames) / self._max_frames_per_segment
                seg_frames = [
                    seg_frames[int(i * step)] for i in range(self._max_frames_per_segment)
                ]
            seg["_frames"] = seg_frames

        # Run full captioning pipeline (steps 5-6 of index_video)
        if caption_fn is not None or frame_caption_fn is not None:
            caption_tasks = []
            for seg in segment_infos:
                seg_frames = seg.pop("_frames")
                transcript_text = self._transcript_for_range(
                    transcript_data,
                    seg["start_time"],
                    seg["end_time"],
                )
                if transcript_text:
                    seg_frames = [f"[transcript] {transcript_text}"] + seg_frames
                caption_tasks.append((seg, seg_frames))

            if frame_caption_fn is not None:

                def _frame_caption_one(args):
                    seg, seg_frames = args
                    real_frames = [f for f in seg_frames if not isinstance(f, str)]
                    if real_frames:
                        mid_idx = len(real_frames) // 2
                        mid_frame = real_frames[mid_idx]
                        result = frame_caption_fn([mid_frame])
                        return seg, result if isinstance(result, str) else str(result)
                    return seg, ""

                with ThreadPoolExecutor(max_workers=8) as pool:
                    futures = [pool.submit(_frame_caption_one, task) for task in caption_tasks]
                    for future in as_completed(futures):
                        try:
                            seg, frame_cap = future.result()
                            seg["frame_caption"] = frame_cap
                        except Exception:
                            logger.warning(
                                "Frame caption future raised an exception", exc_info=True
                            )

            if caption_fn is not None:

                def _caption_segment(args):
                    seg, seg_frames = args
                    if self._model is not None:
                        real_frames = [f for f in seg_frames if not isinstance(f, str)]
                        if len(real_frames) >= 5:
                            filtered_real = self._filter_edge_frames(real_frames, threshold=0.5)
                            str_tokens = [f for f in seg_frames if isinstance(f, str)]
                            seg_frames = str_tokens + filtered_real
                    if self._caption_resize:
                        import cv2

                        resized = []
                        for f in seg_frames:
                            if isinstance(f, str):
                                resized.append(f)
                            else:
                                resized.append(cv2.resize(f, self._caption_resize))
                        seg_frames = resized
                    frame_cap = seg.get("frame_caption", "")
                    if frame_cap:
                        seg_frames = [f"[frame_caption] {frame_cap}"] + seg_frames
                    result = caption_fn(seg_frames)
                    if isinstance(result, str):
                        annotation = {
                            "summary": {"brief": result, "detailed": result},
                            "action": {"brief": "", "detailed": "", "actor": None},
                        }
                    else:
                        annotation = result
                    return seg, annotation

                with ThreadPoolExecutor(max_workers=8) as pool:
                    futures = [pool.submit(_caption_segment, task) for task in caption_tasks]
                    for future in as_completed(futures):
                        try:
                            seg, annotation = future.result()
                            seg["annotation"] = annotation
                            seg["annotation"]["frame_caption"] = seg.get("frame_caption", "")
                            seg["caption"] = annotation.get("summary", {}).get("brief", "")
                            action_brief = annotation.get("action", {}).get("brief", "").strip()
                            if not action_brief or action_brief.upper() == "N/A":
                                seg["is_non_action"] = True
                        except Exception:
                            logger.warning("Caption future raised an exception", exc_info=True)
        else:
            for seg in segment_infos:
                seg.pop("_frames", None)

        # Self-Refine
        self._refine_annotations(
            segment_infos,
            transcript_data,
            refine_fn,
            video_metadata=loaded_video.metadata,
            rounds=refine_rounds,
        )

        # Re-embed with updated captions
        embeddings, action_embeddings = self._embed_captions(segment_infos)
        if embeddings is not None:
            embeddings = self._smooth_embeddings(embeddings, window=3)
        if action_embeddings is not None:
            action_embeddings = self._smooth_embeddings(action_embeddings, window=3)

        index.embeddings = embeddings
        index.action_embeddings = action_embeddings
        index.embed_fn = self._encode_query
        return index
