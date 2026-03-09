"""Video indexing: scene detection, captioning, embedding, and ASR transcript."""

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

from rlm.video import video_caption_pipeline as caption_pipeline
from rlm.video import video_dedup as dedup
from rlm.video import video_embedding as emb_mod
from rlm.video import video_encoding as encoding
from rlm.video import video_transcript as transcript
from rlm.video.scene_detection import detect_scenes, detect_scenes_hierarchical
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
        temporal_feature_maps = npz["temporal_feature_maps"] if "temporal_feature_maps" in npz else None

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
            AutoModel.from_pretrained(self._scene_model_name, dtype=torch.float16)
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
            device = (
                "cuda" if torch.cuda.is_available() else "cpu"
            )
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
    # Thin delegation wrappers
    # ------------------------------------------------------------------

    def _encode_frames(
        self, frames: list[np.ndarray], temporal_window: int = 1, stride: int | None = None
    ) -> np.ndarray:
        return encoding.encode_frames(
            self._model, self._image_processor, self._torch_device,
            frames, temporal_window=temporal_window, stride=stride,
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
            self._model, self._tokenizer, self._torch_device, texts,
        )

    def _encode_clips_vjepa(
        self, clips: list[list[np.ndarray]], return_full: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, list[np.ndarray]]:
        return encoding.encode_clips_vjepa(
            self._scene_model, clips, self._scene_processor,
            self._scene_torch_device,
            scene_embed_dim=self._scene_embed_dim,
            return_full=return_full,
        )

    def _group_frames_into_clips(
        self, frames: list[np.ndarray], timestamps: list[float], clip_size: int,
    ) -> tuple[list[list[np.ndarray]], list[float]]:
        return encoding.group_frames_into_clips(frames, timestamps, clip_size)

    def _encode_frames_overlapping_vjepa(
        self, frames, timestamps, clip_size=64, stride=8, store_feature_maps=False,
    ):
        return encoding.encode_frames_overlapping_vjepa(
            self._encode_clips_vjepa, frames, timestamps,
            clip_size=clip_size, stride=stride,
            scene_embed_dim=self._scene_embed_dim,
            store_feature_maps=store_feature_maps,
        )

    def _encode_query(self, text: str) -> np.ndarray:
        self._ensure_model()
        return self._encode_texts([text])[0]

    def _encode_query_siglip(self, text: str) -> np.ndarray:
        self._ensure_model()
        return self._encode_texts_siglip([text])[0]

    def _predict_future_embedding(
        self, context_features: np.ndarray, n_future_tokens: int = 16,
    ) -> np.ndarray | None:
        return emb_mod.predict_future_embedding(
            self._scene_predictor, self._scene_torch_device,
            context_features, n_future_tokens,
        )

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
            refine_rounds: Number of Self-Refine iterations.
            mode: Indexing mode — ``"full"`` or ``"fast"``.
            store_feature_maps: Whether to store V-JEPA 2 spatial feature maps.
            overlapping_vjepa: Use overlapping V-JEPA 2 windows.
            semantic_dedup: Enable semantic deduplication via k-means.

        Returns:
            A :class:`VideoIndex` ready for use with the search-tool factories.
        """
        # --- In-memory / disk cache lookup ---
        mem_key: str | None = None
        try:
            mem_key = _cache_key(loaded_video.metadata.path)
        except (FileNotFoundError, OSError):
            pass

        if mem_key is not None and mem_key in self._memory_cache:
            logger.info("Returning in-memory cached index for %s", loaded_video.metadata.path)
            return self._memory_cache[mem_key]

        cache_path: Path | None = None
        if mem_key is not None and self._cache_dir is not None:
            cache_path = self._cache_dir / mem_key
            if (cache_path / "metadata.json").exists():
                logger.info("Loading cached index from %s", cache_path)
                idx = VideoIndex.load(cache_path)
                idx.embed_fn = self._encode_query
                self._memory_cache[mem_key] = idx
                return idx

        fps = loaded_video.metadata.extraction_fps
        frames = loaded_video.frames

        # 1. Compute per-frame timestamps
        timestamps = [i / fps for i in range(len(frames))]

        # 2. Detect scene boundaries
        hierarchy_result: dict | None = None
        vjepa_clip_embeddings: np.ndarray | None = None
        vjepa_clip_timestamps: list[float] | None = None
        vjepa_clip_feature_maps: list[np.ndarray] | None = None
        logger.info("[pipeline] V-JEPA 2: detecting scenes in %d frames", len(frames))
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
                ovl_window_ts: list[float] = []
                for start in range(0, len(frames), self._scene_stride):
                    end = min(start + self._scene_clip_size, len(frames))
                    if end - start < 2:
                        continue
                    mid = min(start + (end - start) // 2, len(frames) - 1)
                    ovl_window_ts.append(timestamps[mid])
                vjepa_clip_feature_maps = ovl_feature_maps
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
        logger.info("[pipeline] V-JEPA 2: %d scenes detected", len(scenes))
        scene_boundaries = [start for start, _end in scenes]

        # 3. Build segment dicts — prefer existing segments, fall back to scenes
        if loaded_video.segments:
            segment_infos = self._segments_from_loaded(loaded_video)
        else:
            segment_infos = self._segments_from_scenes(scenes, frames, timestamps)

        # 4. Transcript (Qwen3-ASR or pre-existing file) — run before captioning
        self._ensure_asr_model(asr_model)
        trans_result, fw_model, fw_size = transcript.get_transcript(
            loaded_video.metadata.path,
            asr_model_name=asr_model,
            transcript_path=transcript_path,
            asr_model=self._asr_model,
            asr_batch_size=getattr(self, "_asr_batch_size", None),
            faster_whisper_model=getattr(self, "_faster_whisper_model", None),
            faster_whisper_model_size=getattr(self, "_faster_whisper_model_size", None),
        )
        # Cache faster-whisper model for repeated calls
        if fw_model is not None:
            self._faster_whisper_model = fw_model
            self._faster_whisper_model_size = fw_size
        logger.info("[pipeline] Qwen3-ASR: %d transcript segments", len(trans_result))

        # 4b. Pre-captioning dedup: identify visually similar segments
        self._ensure_model()
        dedup.pre_caption_dedup(segment_infos, self._encode_frames)

        # 4c. Selective decoding: 3-tier (dead / static-informative / dynamic)
        caption_pipeline.selective_decode(
            segment_infos,
            frames,
            timestamps,
            self._encode_frames,
            temporal_clip_embeddings=vjepa_clip_embeddings,
            temporal_clip_timestamps=vjepa_clip_timestamps,
        )

        if mode == "fast":
            # Fast mode: use midpoint frame captions only
            caption_pipeline.action_first_pass(segment_infos, frame_caption_fn)

            # Propagate captions from representatives to skipped duplicates
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
        else:
            # Full mode: Tree-of-Captions + Self-Refine

            # 5. Caption each segment (if a caption function was provided)
            if caption_fn is not None or frame_caption_fn is not None:
                # Prepare all segments first (skip near-duplicates)
                caption_tasks = []
                for seg in segment_infos:
                    seg_frames = seg.pop("_frames")
                    if seg.get("_skip_caption"):
                        continue
                    # ASR context injection
                    transcript_text = caption_pipeline.transcript_for_range(
                        trans_result,
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
                                filtered_real = caption_pipeline.filter_edge_frames(
                                    real_frames, self._encode_frames, threshold=0.5
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
                        futures = [pool.submit(_caption_segment, task) for task in caption_tasks]
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
                                logger.warning("Caption future raised an exception", exc_info=True)
            else:
                for seg in segment_infos:
                    seg.pop("_frames", None)

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

            # 6. Self-Refine annotations
            caption_pipeline.refine_annotations(
                segment_infos,
                trans_result,
                refine_fn,
                video_metadata=loaded_video.metadata,
                rounds=refine_rounds,
            )

            # 6b. Mark near-duplicate adjacent segments before embedding
            dedup.deduplicate_segments(segment_infos, self._encode_texts)

            # 6c. Global dedup: find duplicates anywhere (non-adjacent)
            dedup.global_deduplicate(segment_infos, self._encode_texts)

            # 6.5 Score annotations and re-caption low-quality ones
            caption_pipeline.score_annotations(
                segment_infos,
                loaded_video_frames=frames,
                timestamps=timestamps,
                encode_frames_fn=self._encode_frames,
                encode_texts_fn=self._encode_texts,
                text_embedding_model_name=self._text_embedding_model_name,
            )

        # 7. Embed captions
        logger.info("[pipeline] Gemma: embedding captions for %d segments", len(segment_infos))
        embeddings, action_embeddings = emb_mod.embed_captions(segment_infos, self._encode_texts)
        logger.info("[pipeline] Gemma: caption embeddings complete")

        # 7b. Smooth embeddings to reduce noise across adjacent segments
        if embeddings is not None:
            embeddings = emb_mod.smooth_embeddings(embeddings, window=3)
        if action_embeddings is not None:
            action_embeddings = emb_mod.smooth_embeddings(action_embeddings, window=3)

        quality = emb_mod.check_embedding_quality(embeddings, label="caption")

        # 7b2. Semantic deduplication via k-means clustering (optional)
        if semantic_dedup:
            dedup.semantic_deduplicate(
                segment_infos,
                embeddings,
                action_embeddings=action_embeddings,
            )

        # 7c. Embed representative frame per segment for visual search
        rep_frames = []
        for seg in segment_infos:
            seg_frames_list = [
                f for f, t in zip(frames, timestamps, strict=False)
                if seg["start_time"] <= t <= seg["end_time"]
            ]
            if seg_frames_list:
                rep_frames.append(seg_frames_list[len(seg_frames_list) // 2])
            else:
                rep_frames.append(frames[0])  # fallback

        logger.info("[pipeline] SigLIP2: building frame embeddings for %d segments", len(rep_frames))
        self._ensure_model()
        frame_embeddings = self._encode_frames(rep_frames)
        frame_embeddings = emb_mod.smooth_embeddings(frame_embeddings, window=3)
        emb_mod.check_embedding_quality(frame_embeddings, label="frame")
        logger.info("[pipeline] SigLIP2: %d frame embeddings built", len(rep_frames))

        # 7d. Aggregate V-JEPA 2 temporal embeddings per segment
        temporal_embeddings: np.ndarray | None = None
        temporal_feature_maps: np.ndarray | None = None
        # Feature maps may use different timestamps than clip embeddings
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
                        fmap_indices = [
                            i
                            for i, ct in enumerate(fmap_ts)
                            if seg["start_time"] <= ct <= seg["end_time"]
                        ]
                        if fmap_indices:
                            maps = [vjepa_clip_feature_maps[i] for i in fmap_indices]
                            shapes = {m.shape for m in maps}
                            if len(shapes) == 1:
                                feature_maps_per_seg.append(
                                    np.stack(maps).mean(axis=0)
                                )
                            else:
                                seg_mid = (seg["start_time"] + seg["end_time"]) / 2
                                best_idx = min(
                                    fmap_indices,
                                    key=lambda i: abs(fmap_ts[i] - seg_mid),
                                )
                                feature_maps_per_seg.append(
                                    vjepa_clip_feature_maps[best_idx]
                                )
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
            if vjepa_clip_feature_maps is not None and feature_maps_per_seg:
                shapes = {m.shape for m in feature_maps_per_seg}
                if len(shapes) == 1:
                    temporal_feature_maps = np.stack(feature_maps_per_seg)
                else:
                    temporal_feature_maps = np.empty(
                        len(feature_maps_per_seg), dtype=object
                    )
                    for i, m in enumerate(feature_maps_per_seg):
                        temporal_feature_maps[i] = m
            temporal_embeddings = emb_mod.smooth_embeddings(temporal_embeddings, window=3)
            emb_mod.check_embedding_quality(temporal_embeddings, label="temporal")

        # 8. Build hierarchy levels (when hierarchical mode is enabled)
        segment_hierarchy: list[list[dict]] = []
        hierarchy_embeddings: list[np.ndarray | None] = []
        if hierarchy_result is not None and len(hierarchy_result["levels"]) > 1:
            for lvl_idx in range(1, len(hierarchy_result["levels"])):
                lvl_scenes = hierarchy_result["levels"][lvl_idx]
                lvl_segments: list[dict] = []
                for h_start, h_end in lvl_scenes:
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

                lvl_captions = [s["caption"] for s in lvl_segments]
                if any(lvl_captions):
                    lvl_emb = emb_mod.embed_captions(lvl_segments, self._encode_texts)[0]
                    if lvl_emb is not None:
                        lvl_emb = emb_mod.smooth_embeddings(lvl_emb, window=3)
                    hierarchy_embeddings.append(lvl_emb)
                else:
                    hierarchy_embeddings.append(None)

        # Always add a fixed-duration coarse level for multi-scale search
        if embeddings is not None:
            coarse_segs, coarse_embs = emb_mod.build_coarse_level(
                segment_infos, embeddings, target_duration=30.0
            )
            if coarse_segs:
                segment_hierarchy.append(coarse_segs)
                hierarchy_embeddings.append(coarse_embs)

        index = VideoIndex(
            segments=segment_infos,
            embeddings=embeddings,
            action_embeddings=action_embeddings,
            frame_embeddings=frame_embeddings,
            temporal_embeddings=temporal_embeddings,
            temporal_feature_maps=temporal_feature_maps,
            transcript=trans_result,
            scene_boundaries=scene_boundaries,
            embedding_quality=quality,
            embed_fn=self._encode_query,
            visual_embed_fn=self._encode_query_siglip,
            segment_hierarchy=segment_hierarchy,
            hierarchy_embeddings=hierarchy_embeddings,
        )
        logger.info(
            "[pipeline] search index: %d segments, %d transcript entries",
            len(segment_infos), len(trans_result),
        )

        # --- Attach predictor closures so search tools can use them ---
        if self._scene_predictor is not None:
            indexer_ref = self

            def _predict_fn(time_point: float) -> np.ndarray | None:
                seg_idx = None
                for i, seg in enumerate(index.segments):
                    if seg["start_time"] <= time_point <= seg["end_time"]:
                        seg_idx = i
                        break
                    if seg["end_time"] <= time_point:
                        seg_idx = i
                if seg_idx is None:
                    return None
                if index.temporal_feature_maps is None or seg_idx >= len(
                    index.temporal_feature_maps
                ):
                    return None
                feature_map = index.temporal_feature_maps[seg_idx]
                predicted = indexer_ref._predict_future_embedding(feature_map, 16)
                if predicted is None:
                    return None
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
            seg_frames = [f for f, t in zip(frames, timestamps, strict=False) if start <= t < end or t == end]
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

    def _ensure_asr_model(self, model_name: str) -> None:
        """Lazily load and cache the Qwen3-ASR model."""
        if transcript.is_faster_whisper_model(model_name):
            return  # faster-whisper models are loaded in run_faster_whisper
        asr_model, name, batch_size = transcript.ensure_asr_model(
            self._asr_model, self._asr_model_name, model_name,
        )
        self._asr_model = asr_model
        self._asr_model_name = name
        self._asr_batch_size = batch_size

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
        full Tree-of-Captions + Self-Refine pipeline on the segments.
        """
        fps = loaded_video.metadata.extraction_fps
        frames = loaded_video.frames
        timestamps = [i / fps for i in range(len(frames))]
        trans = index.transcript
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
                transcript_text = caption_pipeline.transcript_for_range(
                    trans,
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
                            filtered_real = caption_pipeline.filter_edge_frames(
                                real_frames, self._encode_frames, threshold=0.5
                            )
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
                            action_brief = (
                                annotation.get("action", {}).get("brief", "").strip()
                            )
                            if not action_brief or action_brief.upper() == "N/A":
                                seg["is_non_action"] = True
                        except Exception:
                            logger.warning("Caption future raised an exception", exc_info=True)
        else:
            for seg in segment_infos:
                seg.pop("_frames", None)

        # Self-Refine
        caption_pipeline.refine_annotations(
            segment_infos,
            trans,
            refine_fn,
            video_metadata=loaded_video.metadata,
            rounds=refine_rounds,
        )

        # Re-embed with updated captions
        embeddings, action_embeddings = emb_mod.embed_captions(segment_infos, self._encode_texts)
        if embeddings is not None:
            embeddings = emb_mod.smooth_embeddings(embeddings, window=3)
        if action_embeddings is not None:
            action_embeddings = emb_mod.smooth_embeddings(action_embeddings, window=3)

        index.embeddings = embeddings
        index.action_embeddings = action_embeddings
        index.embed_fn = self._encode_query
        return index
