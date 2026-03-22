"""Video indexing: scene detection, captioning, embedding, and ASR transcript.

This module is the orchestration layer. The actual work is delegated to:

- :mod:`kuavi.encoding` — frame/text encoding (LanguageBind, V-JEPA 2)
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

_embedder = None
def get_embedder():
    global _embedder
    if _embedder is None:
        from kuavi.embedders import create_embedder
        _embedder = create_embedder()
    return _embedder



@dataclass
class VideoIndex:
    """Pre-computed searchable index for a video."""

    video_path: str | None = None
    segments: list[dict] = field(default_factory=list)
    embeddings: np.ndarray | None = None
    action_embeddings: np.ndarray | None = None
    transcript: list[dict] = field(default_factory=list)
    scene_boundaries: list[float] = field(default_factory=list)
    embedding_quality: dict = field(default_factory=dict)
    frame_embeddings: np.ndarray | None = None
    temporal_embeddings: np.ndarray | None = None  # (N_segments, 1024) from V-JEPA 2
    temporal_feature_maps: np.ndarray | None = None  # (N_segments, num_patches, D) from V-JEPA 2
    segment_hierarchy: list[list[dict]] = field(default_factory=list)
    hierarchy_embeddings: list[np.ndarray | None] = field(default_factory=list)

    def save(self, path: str | Path) -> None:
        """Persist index to *path* (a directory).

        Embeddings are stored as a ``.npz`` file; metadata (segments,
        transcript, scene_boundaries) as ``metadata.json``.
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
            "video_path": self.video_path,
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
            video_path=metadata.get("video_path"),
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

    ALL_STAGES: list[str] = [
        "scenes",
        "transcript",
        "segments",
        "captions",
        "languagebind",
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
        device: str = "auto",
        temporal_window: int = 4,
        max_frames_per_segment: int = 32,
        cache_dir: str | Path | None = None,
        caption_resize: tuple[int, int] | None = None,
        embedding_stride: int | None = None,
        hierarchical: bool = False,
        scene_model: str | None = None,
        scene_clip_size: int = 16,
        scene_stride: int = 8,
        scene_model_preset: str | None = None,
    ):
        from kuavi.types import VJEPA2_PRESETS

        self._device = device
        self._temporal_window = temporal_window
        self._max_frames_per_segment = max_frames_per_segment
        self._cache_dir = Path(cache_dir) if cache_dir is not None else None
        self._caption_resize = caption_resize
        self._embedding_stride = embedding_stride
        self._hierarchical = hierarchical
        self._memory_cache: dict[str, VideoIndex] = {}
        self._scene_model = None
        self._scene_processor = None
        self._scene_predictor = None
        self._asr_model = None
        self._asr_model_name: str | None = None
        self._faster_whisper_model = None
        self._faster_whisper_model_size = None

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
        pass

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
        return seg_frames

    def _check_embedding_quality(self, embeddings: np.ndarray, label: str = "caption") -> dict:
        return emb_mod.check_embedding_quality(embeddings, label)

    @staticmethod
    def _smooth_embeddings(embs: np.ndarray, window: int = 3) -> np.ndarray:
        return emb_mod.smooth_embeddings(embs, window)

    def _embed_captions(
        self,
        segments: list[dict],
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        return None, None

    def _deduplicate_segments(self, segments: list[dict], threshold: float = 0.95) -> None:
        pass

    def _global_deduplicate(self, segments: list[dict], threshold: float = 0.90) -> None:
        pass

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
        pass

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
        pass

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
                pass
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
                raise ValueError("V-JEPA 2 scene model is required for scene detection")
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

        # 7. Embed segments (languagebind)
        if (
            not force_reindex
            and _should_run("languagebind", stages)
            and sc is not None
            and sc.has_json("languagebind_embeddings")
        ):
            logger.info("[stage-cache] loading languagebind_embeddings from sidecar")
        elif _should_run("languagebind", stages):
            embedder = get_embedder()
            video_path_str = loaded_video.metadata.path
            import tempfile, subprocess
            from pathlib import Path
            logger.info("[pipeline] Embed: processing %d segments", len(segment_infos))
            
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_audio:
                audio_path = tmp_audio.name
            try:
                subprocess.run([
                    "ffmpeg", "-y", "-i", video_path_str,
                    "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
                    audio_path
                ], check=True, capture_output=True)
            except Exception:
                audio_path = None
                
            emb_data: list[dict] = []
            for i, seg in enumerate(segment_infos):
                logger.info("[embed] segment %d/%d done", i + 1, len(segment_infos))
                
                try:
                    video_emb = embedder.embed_video_segment(video_path_str, seg["start_time"], seg["end_time"])
                except Exception:
                    logger.warning("Video embedding failed for segment %d", i, exc_info=True)
                    video_emb = None
                    
                audio_emb = None
                if audio_path and Path(audio_path).exists():
                    try:
                        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as seg_audio:
                            seg_audio_path = seg_audio.name
                        duration = min(seg["end_time"] - seg["start_time"], 120.0)
                        subprocess.run([
                            "ffmpeg", "-y", "-ss", str(seg["start_time"]),
                            "-i", audio_path, "-t", str(duration), "-c", "copy", seg_audio_path
                        ], check=True, capture_output=True)
                        audio_emb = embedder.embed_audio_segment(seg_audio_path)
                        Path(seg_audio_path).unlink(missing_ok=True)
                    except Exception:
                        logger.warning("Audio embedding failed for segment %d", i, exc_info=True)
                
                text_content = seg.get("transcript") or seg.get("caption") or ""
                try:
                    text_emb = embedder.embed_text(text_content) if text_content else None
                except Exception:
                    logger.warning("Text embedding failed for segment %d", i, exc_info=True)
                    text_emb = None
                    
                emb_data.append({
                    "segment_id": seg.get("id", i),
                    "start": seg["start_time"],
                    "end": seg["end_time"],
                    "video_emb": video_emb,
                    "audio_emb": audio_emb,
                    "text_emb": text_emb,
                })
                
            if audio_path and Path(audio_path).exists():
                Path(audio_path).unlink(missing_ok=True)
                
            if sc is not None:
                sc.save_json("languagebind_embeddings", emb_data)
                logger.info("[stage-cache] saved languagebind_embeddings to sidecar")

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
            video_path=loaded_video.metadata.path,
            segments=segment_infos,
            embeddings=embeddings,
            action_embeddings=action_embeddings,
            frame_embeddings=frame_embeddings,
            temporal_embeddings=temporal_embeddings,
            temporal_feature_maps=temporal_feature_maps,
            transcript=transcript_data,
            scene_boundaries=scene_boundaries,
            embedding_quality=quality,
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
            and embeddings re-attached.
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

        return index
