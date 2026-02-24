"""Video indexing: scene detection, captioning, embedding, and ASR transcript."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import tempfile
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

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

    def _predict_future_embedding(
        self,
        context_features: np.ndarray,
        n_future_tokens: int = 16,
    ) -> np.ndarray | None:
        """Predict future frame representation using V-JEPA 2 predictor.

        The predictor uses a context/target masking scheme: context_mask selects
        patches from the encoder output that serve as context, and target_mask
        selects positions to predict. Both are index tensors of shape
        ``[batch, num_selected]`` with int64 patch indices.

        We use the first ``num_patches - n_future_tokens`` patches as context
        and the last ``n_future_tokens`` patches as target (to predict).

        Args:
            context_features: (num_patches, D) spatial feature map from a segment.
            n_future_tokens: Number of future token positions to predict.

        Returns:
            (n_future_tokens, D) predicted feature map, or None if predictor unavailable.
        """
        if self._scene_predictor is None:
            return None

        import torch

        num_patches = context_features.shape[0]
        if n_future_tokens >= num_patches:
            n_future_tokens = max(1, num_patches // 4)

        n_context = num_patches - n_future_tokens

        # Convert to tensor and add batch dimension: (1, num_patches, D)
        encoder_hidden_states = torch.from_numpy(context_features).unsqueeze(0).to(
            self._scene_torch_device, dtype=torch.float16
        )

        # Index masks: context = first n_context patches, target = last n_future_tokens
        context_mask = [torch.arange(n_context, dtype=torch.int64).unsqueeze(0).to(
            self._scene_torch_device
        )]
        target_mask = [torch.arange(n_context, num_patches, dtype=torch.int64).unsqueeze(0).to(
            self._scene_torch_device
        )]

        with torch.no_grad():
            try:
                output = self._scene_predictor(
                    encoder_hidden_states, context_mask, target_mask
                )
                return output.last_hidden_state.squeeze(0).cpu().float().numpy()
            except Exception as e:
                logger.warning("Predictor forward pass failed: %s", e)
                return None

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
            refine_rounds: Number of Self-Refine iterations. Default 0 (single-pass
                captioning). Set to 3 for the original multi-round refinement.
            mode: Indexing mode — ``"full"`` (default) runs the captioning
                pipeline; ``"fast"`` skips segment captioning, using only
                midpoint frame captions to produce a quickly searchable index.

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
                for thresh, min_dur in zip((0.10, 0.20, 0.35), (0.5, 2.0, 4.0)):
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

        # 3. Build segment dicts — prefer existing segments, fall back to scenes
        if loaded_video.segments:
            segment_infos = self._segments_from_loaded(loaded_video)
        else:
            segment_infos = self._segments_from_scenes(scenes, frames, timestamps)

        # 4. Transcript (Qwen3-ASR or pre-existing file) — run before captioning
        #    so ASR context can be injected into caption prompts.
        #    Free V-JEPA 2 from GPU first — it stays loaded after scene detection
        #    and leaves too little VRAM for the ASR model on 11-12 GiB GPUs.
        self._free_scene_model()
        transcript = self._get_transcript(
            loaded_video.metadata.path,
            asr_model=asr_model,
            transcript_path=transcript_path,
        )

        # 4b. Pre-captioning dedup: identify visually similar segments
        #     and only caption representatives, propagating results afterward.
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
            # 5 (fast). Action-first pass: frame captions for non-skipped segments
            self._action_first_pass(segment_infos, frame_caption_fn)

            # 5c (fast). Propagate captions from representatives to skipped duplicates
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
            # Full mode: Tree-of-Captions + Self-Refine (original behavior)

            # 5. Caption each segment (if a caption function was provided)
            if caption_fn is not None or frame_caption_fn is not None:
                # Prepare all segments first (skip near-duplicates)
                caption_tasks = []
                for seg in segment_infos:
                    seg_frames = seg.pop("_frames")
                    if seg.get("_skip_caption"):
                        continue
                    # ASR context injection: prepend transcript text for this segment
                    transcript_text = self._transcript_for_range(
                        transcript,
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
                                filtered_real = self._filter_edge_frames(real_frames, threshold=0.5)
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
            self._refine_annotations(
                segment_infos,
                transcript,
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

        # 7. Embed captions
        embeddings, action_embeddings = self._embed_captions(segment_infos)

        # 7b. Smooth embeddings to reduce noise across adjacent segments
        if embeddings is not None:
            embeddings = self._smooth_embeddings(embeddings, window=3)
        if action_embeddings is not None:
            action_embeddings = self._smooth_embeddings(action_embeddings, window=3)

        quality = self._check_embedding_quality(embeddings, label="caption")

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
                f for f, t in zip(frames, timestamps, strict=False)
                if seg["start_time"] <= t <= seg["end_time"]
            ]
            if seg_frames_list:
                rep_frames.append(seg_frames_list[len(seg_frames_list) // 2])
            else:
                rep_frames.append(frames[0])  # fallback

        self._ensure_model()
        frame_embeddings = self._encode_frames(rep_frames)
        frame_embeddings = self._smooth_embeddings(frame_embeddings, window=3)
        self._check_embedding_quality(frame_embeddings, label="frame")

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
                                feature_maps_per_seg.append(
                                    np.stack(maps).mean(axis=0)
                                )
                            else:
                                # Variable shapes (different clip lengths) —
                                # pick the clip closest to segment midpoint
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
            temporal_embeddings = self._smooth_embeddings(temporal_embeddings, window=3)
            self._check_embedding_quality(temporal_embeddings, label="temporal")
            if vjepa_clip_feature_maps is not None and feature_maps_per_seg:
                shapes = {m.shape for m in feature_maps_per_seg}
                if len(shapes) == 1:
                    temporal_feature_maps = np.stack(feature_maps_per_seg)
                else:
                    # Variable patch counts across segments — store as object array
                    temporal_feature_maps = np.empty(
                        len(feature_maps_per_seg), dtype=object
                    )
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

        index = VideoIndex(
            segments=segment_infos,
            embeddings=embeddings,
            action_embeddings=action_embeddings,
            frame_embeddings=frame_embeddings,
            temporal_embeddings=temporal_embeddings,
            temporal_feature_maps=temporal_feature_maps,
            transcript=transcript,
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
            seg_frames = [f for f, t in zip(frames, timestamps, strict=False) if start <= t < end or t == end]
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

    def _pre_caption_dedup(
        self,
        segments: list[dict],
        threshold: float = 0.90,
    ) -> None:
        """Identify visually near-duplicate segments before captioning."""
        if len(segments) < 2:
            return

        self._ensure_model()

        seg_embeddings = []
        valid_indices = []
        for i, seg in enumerate(segments):
            frames = seg.get("_frames", [])
            real_frames = [f for f in frames if not isinstance(f, str)]
            if not real_frames:
                seg_embeddings.append(None)
                continue
            try:
                embs = self._encode_frames(real_frames)
                mean_emb = embs.mean(axis=0)
                norm = np.linalg.norm(mean_emb)
                if norm > 1e-10:
                    mean_emb = mean_emb / norm
                seg_embeddings.append(mean_emb)
                valid_indices.append(i)
            except Exception:
                logger.warning("Failed to encode frames for segment %d", i, exc_info=True)
                seg_embeddings.append(None)

        if len(valid_indices) < 2:
            return

        valid_embs = np.stack([seg_embeddings[i] for i in valid_indices])

        from sklearn.metrics.pairwise import cosine_similarity

        sim_matrix = cosine_similarity(valid_embs)

        representatives: dict[int, int] = {}
        for vi in range(len(valid_indices)):
            seg_idx = valid_indices[vi]
            found_rep = False
            for rep_vi, rep_seg_idx in representatives.items():
                if sim_matrix[vi, rep_vi] > threshold:
                    segments[seg_idx]["_skip_caption"] = True
                    segments[seg_idx]["_caption_source"] = rep_seg_idx
                    found_rep = True
                    break
            if not found_rep:
                representatives[vi] = seg_idx

        skipped = sum(1 for s in segments if s.get("_skip_caption"))
        if skipped:
            logger.info(
                "Pre-caption dedup: %d/%d segments skipped (threshold=%.2f)",
                skipped,
                len(segments),
                threshold,
            )

    def _semantic_deduplicate(
        self,
        segment_infos: list[dict],
        embeddings: np.ndarray | None,
        action_embeddings: np.ndarray | None = None,
        n_clusters: int | None = None,
        similarity_threshold: float = 0.92,
    ) -> np.ndarray | None:
        """Semantic deduplication via k-means clustering.

        Clusters segments by embedding similarity and marks near-duplicates
        within each cluster. Also stores cluster_id on each segment for
        downstream cluster-aware search diversity.

        Args:
            segment_infos: List of segment dicts (modified in place).
            embeddings: (N, D) caption embeddings. If None, skips dedup.
            action_embeddings: Optional (N, D) action embeddings for combined clustering.
            n_clusters: Number of clusters. If None, auto-computed as
                max(2, len(segments) // 5).
            similarity_threshold: Cosine similarity above which segments in the
                same cluster are considered duplicates (default 0.92).

        Returns:
            cluster_labels array of shape (N,) or None if skipped.
        """
        if embeddings is None or len(embeddings) < 3:
            return None

        from sklearn.cluster import KMeans
        from sklearn.metrics.pairwise import cosine_similarity

        n = len(embeddings)
        if n_clusters is None:
            n_clusters = max(2, n // 5)
        n_clusters = min(n_clusters, n)

        # Combine embeddings if action embeddings available
        if action_embeddings is not None and len(action_embeddings) == n:
            combined = np.concatenate([embeddings, action_embeddings], axis=1)
            norms = np.linalg.norm(combined, axis=1, keepdims=True)
            combined = combined / np.maximum(norms, 1e-10)
        else:
            combined = embeddings

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(combined)

        # Assign cluster_id to each segment
        for i, seg in enumerate(segment_infos):
            seg["cluster_id"] = int(labels[i])

        # Within each cluster, mark duplicates (keep highest-quality representative)
        clusters: dict[int, list[int]] = {}
        for i, label in enumerate(labels):
            clusters.setdefault(int(label), []).append(i)

        dedup_count = 0
        for cluster_indices in clusters.values():
            if len(cluster_indices) < 2:
                continue

            cluster_embs = embeddings[cluster_indices]
            sim_matrix = cosine_similarity(cluster_embs)

            # Sort by quality score (descending), keep best as representative
            scored = [
                (idx, segment_infos[idx].get("quality_score", 0.5)) for idx in cluster_indices
            ]
            scored.sort(key=lambda x: x[1], reverse=True)

            kept: set[int] = set()
            for idx, _score in scored:
                is_dup = False
                local_pos = cluster_indices.index(idx)
                for kept_idx in kept:
                    kept_local = cluster_indices.index(kept_idx)
                    if sim_matrix[local_pos, kept_local] > similarity_threshold:
                        is_dup = True
                        segment_infos[idx]["is_semantic_duplicate"] = True
                        segment_infos[idx]["_semantic_dup_of"] = kept_idx
                        dedup_count += 1
                        break
                if not is_dup:
                    kept.add(idx)

        if dedup_count > 0:
            logger.info(
                "Semantic dedup: %d/%d segments marked as duplicates"
                " (threshold=%.2f, clusters=%d)",
                dedup_count,
                n,
                similarity_threshold,
                n_clusters,
            )

        return labels

    def _filter_edge_frames(self, seg_frames: list, threshold: float = 0.5) -> list:
        """Filter visually dissimilar edge frames from a segment."""
        real_frames = [f for f in seg_frames if not isinstance(f, str)]
        if len(real_frames) < 5:
            return seg_frames

        str_tokens = [f for f in seg_frames if isinstance(f, str)]

        try:
            embs = self._encode_frames(real_frames)
        except AttributeError:
            return seg_frames
        n = len(real_frames)

        start_20 = max(1, int(n * 0.2))
        end_80 = min(n - 1, int(n * 0.8))

        central_embs = embs[start_20:end_80]
        central_mean = central_embs.mean(axis=0)
        norm = np.linalg.norm(central_mean)
        if norm > 1e-10:
            central_mean = central_mean / norm

        keep_indices = set(range(start_20, end_80))
        for i in list(range(0, start_20)) + list(range(end_80, n)):
            sim = float(np.dot(embs[i], central_mean))
            if sim >= threshold:
                keep_indices.add(i)

        filtered_real = [real_frames[i] for i in sorted(keep_indices)]
        return str_tokens + filtered_real

    def _check_embedding_quality(
        self, embeddings: np.ndarray, label: str = "caption"
    ) -> dict:
        """Compute embedding quality metrics (uniformity, pairwise similarity)."""
        if embeddings is None or embeddings.shape[0] < 2:
            return {}

        n = embeddings.shape[0]
        rng = np.random.default_rng(42)
        num_pairs = min(500, n * (n - 1) // 2)
        pairs_i = rng.integers(0, n, size=num_pairs)
        pairs_j = rng.integers(0, n - 1, size=num_pairs)
        pairs_j = np.where(pairs_j >= pairs_i, pairs_j + 1, pairs_j)

        ei = embeddings[pairs_i]
        ej = embeddings[pairs_j]

        sq_dists = np.sum((ei - ej) ** 2, axis=1)
        uniformity = float(np.log(np.mean(np.exp(-2.0 * sq_dists))))

        dot_products = np.sum(ei * ej, axis=1)
        mean_pairwise_similarity = float(np.mean(dot_products))

        is_degenerate = mean_pairwise_similarity > 0.99
        if is_degenerate:
            logger.warning(
                "Embedding quality check (%s): DEGENERATE — mean pairwise similarity %.4f > 0.99",
                label,
                mean_pairwise_similarity,
            )
        else:
            logger.info(
                "Embedding quality check (%s): OK — mean pairwise similarity %.4f",
                label,
                mean_pairwise_similarity,
            )

        return {
            "uniformity": uniformity,
            "mean_pairwise_similarity": mean_pairwise_similarity,
            "is_degenerate": is_degenerate,
        }

    def _smooth_embeddings(self, embs: np.ndarray, window: int = 3) -> np.ndarray:
        """Apply centered moving average smoothing to embedding rows."""
        if embs.shape[0] < window:
            return embs

        w = window // 2
        n = embs.shape[0]
        smoothed = np.empty_like(embs)
        for i in range(n):
            lo = max(0, i - w)
            hi = min(n, i + w + 1)
            smoothed[i] = embs[lo:hi].mean(axis=0)

        norms = np.linalg.norm(smoothed, axis=1, keepdims=True)
        smoothed = smoothed / np.maximum(norms, 1e-10)
        return smoothed

    def _embed_captions(
        self,
        segments: list[dict],
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Encode segment captions and action briefs into embedding matrices."""
        captions = [seg.get("caption", "") for seg in segments]
        actions = [
            ""
            if (b := seg.get("annotation", {}).get("action", {}).get("brief", "").strip())
            in ("", "N/A")
            else b
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

    def _deduplicate_segments(self, segments: list[dict], threshold: float = 0.95) -> None:
        """Mark near-duplicate adjacent segments."""
        if len(segments) < 2:
            return

        captions = [seg.get("caption", "") for seg in segments]
        if not any(captions):
            return

        self._ensure_model()
        try:
            embs = self._encode_texts(captions)
        except AttributeError:
            return

        for i in range(len(segments) - 1):
            if not captions[i] or not captions[i + 1]:
                continue
            sim = float(np.dot(embs[i], embs[i + 1]))
            if sim > threshold:
                dur_i = segments[i]["end_time"] - segments[i]["start_time"]
                dur_j = segments[i + 1]["end_time"] - segments[i + 1]["start_time"]
                shorter = i if dur_i <= dur_j else i + 1
                segments[shorter]["is_duplicate"] = True

    def _global_deduplicate(self, segments: list[dict], threshold: float = 0.90) -> None:
        """Mark globally duplicate segments (non-adjacent) by caption similarity."""
        if len(segments) < 3:
            return

        captions = [seg.get("caption", "") for seg in segments]
        non_empty = [i for i, c in enumerate(captions) if c]
        if len(non_empty) < 2:
            return

        self._ensure_model()
        try:
            all_embs = self._encode_texts(captions)
        except AttributeError:
            return

        from sklearn.metrics.pairwise import cosine_similarity

        valid_embs = all_embs[non_empty]
        sim_matrix = cosine_similarity(valid_embs)

        for vi in range(len(non_empty)):
            i = non_empty[vi]
            if segments[i].get("is_duplicate"):
                continue
            for vj in range(vi + 1, len(non_empty)):
                j = non_empty[vj]
                if abs(i - j) <= 1:
                    continue
                if segments[j].get("is_duplicate"):
                    continue
                if sim_matrix[vi, vj] > threshold:
                    dur_i = segments[i]["end_time"] - segments[i]["start_time"]
                    dur_j = segments[j]["end_time"] - segments[j]["start_time"]
                    shorter = i if dur_i <= dur_j else j
                    segments[shorter]["is_duplicate"] = True

        n_marked = sum(1 for s in segments if s.get("is_duplicate"))
        if n_marked:
            logger.info(
                "Global dedup: %d/%d segments marked as duplicate (threshold=%.2f)",
                n_marked,
                len(segments),
                threshold,
            )

    @staticmethod
    def _transcript_for_range(
        transcript: list[dict],
        start: float,
        end: float,
    ) -> str:
        """Return concatenated transcript text overlapping a time range."""
        return " ".join(
            e["text"] for e in transcript if e["end_time"] >= start and e["start_time"] <= end
        )

    def _refine_annotations(
        self,
        segments: list[dict],
        transcript: list[dict],
        refine_fn: Callable | None,
        video_metadata=None,
        rounds: int = 3,
    ) -> None:
        """Iteratively refine segment annotations using the Self-Refine pattern."""
        if refine_fn is None:
            return

        global_context = ""
        if len(segments) > 1:
            first_cap = segments[0].get("caption", "")
            last_cap = segments[-1].get("caption", "")
            global_context = f"Video starts with: {first_cap}\nVideo ends with: {last_cap}"

        metadata_text = ""
        if video_metadata:
            path = getattr(video_metadata, "path", "") or ""
            duration = float(getattr(video_metadata, "duration", 0) or 0)
            metadata_text = f"Video: {Path(path).name}, Duration: {duration:.1f}s"

        _JSON_SCHEMA = (
            "### Output Format (strict JSON)\n"
            "{\n"
            '  "summary": {"brief": "<single sentence, ~20 words>", "detailed": "<~95 words>"},\n'
            '  "action": {"brief": "<imperative verb phrase, 2-5 words>", '
            '"detailed": "<imperative sentence>", "actor": "<noun phrase or null>"}\n'
            "}"
        )

        def _build_tree_text(segs: list[dict]) -> str:
            lines = ["## Tree of Captions"]
            for j, s in enumerate(segs):
                fc = s.get("frame_caption", "")
                sc = s.get("caption", "")
                lines.append(f"### Seg {j} [{s['start_time']:.1f}s-{s['end_time']:.1f}s]")
                if fc:
                    lines.append(f"- **Frame**: {fc}")
                if sc:
                    lines.append(f"- **Segment**: {sc}")
            return "\n".join(lines)

        for _round in range(rounds):
            tree_text = _build_tree_text(segments)
            refine_tasks = []
            skipped = 0
            for i, seg in enumerate(segments):
                seg_duration = seg["end_time"] - seg["start_time"]
                if seg_duration < 4.0:
                    skipped += 1
                    continue
                neighbors = segments[max(0, i - 1) : i + 2]
                neighbor_text = " | ".join(n.get("caption", "") for n in neighbors if n is not seg)
                transcript_text = self._transcript_for_range(
                    transcript,
                    seg["start_time"],
                    seg["end_time"],
                )
                context = f"""# Video Metadata
{metadata_text}

# Global Video Context
{global_context}

{tree_text}

# Neighbor Segments
{neighbor_text}

# Transcript
{transcript_text}"""
                annotation_json = json.dumps(seg.get("annotation", {}))
                if _round > 0:
                    draft = (
                        "Carefully analyze, verify, and revise the previous draft. "
                        "Correct factual errors, resolve inconsistencies, and remove "
                        "unsupported statements.\n\n"
                        "### Verification Checklist\n"
                        "- Remove any claims not supported by at least 2 frame observations\n"
                        "- Remove names, speech content, or internal states unless directly visible\n"
                        "- Ensure chronological ordering without timestamps\n"
                        "- Verify action.brief is an imperative verb phrase (2-5 words)\n\n"
                        f"{_JSON_SCHEMA}\n\n"
                        f"Previous draft:\n{annotation_json}"
                    )
                else:
                    draft = (
                        "Analyze this video segment annotation and produce a refined version.\n\n"
                        "### Task 1: Summarization\n"
                        "Generate summary.brief (single sentence, ~20 words) and summary.detailed (~95 words).\n"
                        "Describe events in chronological order. Do not mention exact timestamps.\n\n"
                        "### Task 2: Action Identification\n"
                        "Identify the primary action (action.brief: imperative verb phrase, 2-5 words).\n"
                        "Describe the actor performing the action (action.actor).\n"
                        "Use 'N/A' for action.brief if no identifiable action exists.\n\n"
                        "### Anti-Hallucination Rules\n"
                        "- Be cautious and conservative. Rely on majority consensus across frame captions.\n"
                        "- Do not add visually unobservable information (speech content, names, internal states).\n"
                        "- Use global context and metadata only for disambiguation, not for adding new claims.\n"
                        "- If frame captions conflict, describe only what is consistently observed.\n\n"
                        f"{_JSON_SCHEMA}\n\n"
                        f"Current annotation:\n{annotation_json}"
                    )
                refine_tasks.append((i, seg, draft, context))
            if skipped:
                logger.debug("Self-Refine round %d: skipped %d short segments (< 4s)", _round, skipped)

            effort = "high" if _round == 0 else "low"

            def _refine_one(args, _effort=effort):
                i, seg, draft, context = args
                try:
                    refined = refine_fn(draft, context, _effort)
                except TypeError:
                    refined = refine_fn(draft, context)
                return i, refined

            with ThreadPoolExecutor(max_workers=8) as pool:
                futures = [pool.submit(_refine_one, task) for task in refine_tasks]
                results = {}
                for future in as_completed(futures):
                    try:
                        i, refined = future.result()
                        results[i] = refined
                    except Exception:
                        logger.warning("Refine future raised an exception", exc_info=True)

            for i, seg in enumerate(segments):
                refined = results.get(i)
                if refined is None:
                    continue
                try:
                    seg["annotation"] = json.loads(refined)
                    seg["caption"] = (
                        seg["annotation"].get("summary", {}).get("brief", seg.get("caption", ""))
                    )
                except (json.JSONDecodeError, TypeError):
                    pass

    @staticmethod
    def _score_format_compliance(seg: dict) -> float:
        """Score annotation format compliance (0.0-1.0, pure string checks)."""
        import re

        annotation = seg.get("annotation", {})
        summary = annotation.get("summary", {}) if isinstance(annotation, dict) else {}
        action = annotation.get("action", {}) if isinstance(annotation, dict) else {}

        score = 0.0

        # summary.brief exists and is non-empty (0.25)
        summary_brief = summary.get("brief", "") if isinstance(summary, dict) else ""
        if summary_brief and isinstance(summary_brief, str) and summary_brief.strip():
            score += 0.25

        # action.brief is 2-5 words starting with imperative verb (0.25)
        action_brief = action.get("brief", "") if isinstance(action, dict) else ""
        if action_brief and isinstance(action_brief, str) and action_brief.strip():
            words = action_brief.strip().split()
            if 2 <= len(words) <= 5 and words[0][0].isupper():
                score += 0.25

        # No timestamps in summary text (0.25)
        if summary_brief and not re.search(r"\bat\s+\d+(?:\.\d+)?s\b", summary_brief):
            score += 0.25

        # action.actor is present when action.brief is not "N/A" (0.25)
        if isinstance(action, dict):
            ab = action.get("brief", "")
            if ab and ab != "N/A":
                actor = action.get("actor")
                if actor is not None and str(actor).strip():
                    score += 0.25
            else:
                # action is N/A — actor field not required
                score += 0.25

        return round(score, 4)

    @staticmethod
    def _score_action_frequency(segments: list[dict]) -> None:
        """Score each segment's action.brief frequency across all segments (in-place)."""
        action_counts: dict[str, int] = {}
        total = len(segments)
        if total == 0:
            return

        for seg in segments:
            annotation = seg.get("annotation", {})
            action = annotation.get("action", {}) if isinstance(annotation, dict) else {}
            ab = action.get("brief", "") if isinstance(action, dict) else ""
            if ab and ab != "N/A":
                action_counts[ab] = action_counts.get(ab, 0) + 1

        for seg in segments:
            annotation = seg.get("annotation", {})
            action = annotation.get("action", {}) if isinstance(annotation, dict) else {}
            ab = action.get("brief", "") if isinstance(action, dict) else ""
            if not ab or ab == "N/A":
                seg["action_frequency_score"] = 1.0
                continue

            freq = action_counts.get(ab, 0) / total
            if freq <= 0.30:
                freq_score = 1.0
            elif freq >= 0.50:
                freq_score = 0.0
            else:
                freq_score = 1.0 - (freq - 0.30) / 0.20

            seg["action_frequency_score"] = round(freq_score, 4)

    def _score_annotations(
        self,
        segments: list[dict],
        loaded_video_frames: list[np.ndarray],
        timestamps: list[float],
        min_similarity: float = 0.3,
    ) -> None:
        """Score annotation quality using model-free signals."""
        self._ensure_model()

        # Signal 2: Format compliance — no model needed
        for seg in segments:
            seg["format_compliance_score"] = self._score_format_compliance(seg)

        # Signal 5: Action frequency — no model needed, needs all segments
        self._score_action_frequency(segments)

        # Collect caption embeddings for signals 3 and 4 (text model required)
        caption_embeddings: dict[int, np.ndarray] = {}

        for idx, seg in enumerate(segments):
            caption = seg.get("caption", "")
            if not caption:
                continue

            if self._text_embedding_model_name is not None:
                # Skip signal 1 and signal 3 when using separate text embedding model
                continue

            seg_frames = [
                f
                for f, t in zip(loaded_video_frames, timestamps, strict=False)
                if seg["start_time"] <= t <= seg["end_time"]
            ]
            if not seg_frames:
                continue

            try:
                caption_emb = self._encode_texts([caption])
                frame_embs = self._encode_frames(seg_frames)
            except AttributeError:
                continue
            mean_frame_emb = frame_embs.mean(axis=0, keepdims=True)
            norm = np.linalg.norm(mean_frame_emb, axis=1, keepdims=True)
            mean_frame_emb = mean_frame_emb / np.maximum(norm, 1e-10)

            similarity = float(np.dot(caption_emb[0], mean_frame_emb[0]))
            seg["caption_quality_score"] = round(similarity, 4)

            # Store caption embedding for signals 3 and 4
            caption_embeddings[idx] = caption_emb[0]

            # Signal 3: Summary-Action Coherence
            annotation = seg.get("annotation", {})
            action = annotation.get("action", {}) if isinstance(annotation, dict) else {}
            action_brief = action.get("brief", "") if isinstance(action, dict) else ""
            if action_brief and action_brief != "N/A":
                summary = annotation.get("summary", {}) if isinstance(annotation, dict) else {}
                summary_brief = summary.get("brief", "") if isinstance(summary, dict) else ""
                if summary_brief:
                    try:
                        action_emb = self._encode_texts([action_brief])
                        summary_emb = self._encode_texts([summary_brief])
                        coherence = float(np.dot(action_emb[0], summary_emb[0]))
                        seg["coherence_score"] = round(coherence, 4)
                    except AttributeError:
                        pass

        # Signal 4: Temporal consistency (needs all caption embeddings)
        for idx, seg in enumerate(segments):
            if idx not in caption_embeddings:
                continue
            emb = caption_embeddings[idx]
            sims = []
            if idx - 1 in caption_embeddings:
                sims.append(float(np.dot(emb, caption_embeddings[idx - 1])))
            if idx + 1 in caption_embeddings:
                sims.append(float(np.dot(emb, caption_embeddings[idx + 1])))
            if sims:
                max_sim = max(sims)
                seg["temporal_consistency_score"] = round(max(0.0, min(1.0, 1.0 - max_sim)), 4)

        # Aggregate quality_score: average of all available signals
        signal_keys = [
            "caption_quality_score",
            "format_compliance_score",
            "coherence_score",
            "temporal_consistency_score",
            "action_frequency_score",
        ]
        for seg in segments:
            values = [seg[k] for k in signal_keys if k in seg]
            if values:
                seg["quality_score"] = round(sum(values) / len(values), 4)

    def _fix_low_quality_annotations(
        self,
        segments: list[dict],
        loaded_video_frames: list[np.ndarray],
        timestamps: list[float],
        caption_fn: Callable | None = None,
        threshold: float = 0.3,
        num_retries: int = 3,
    ) -> None:
        """Re-caption segments where any quality signal is below *threshold*."""
        if caption_fn is None:
            return

        signal_keys = [
            "caption_quality_score",
            "format_compliance_score",
            "coherence_score",
            "temporal_consistency_score",
            "action_frequency_score",
        ]

        for seg in segments:
            low_quality = any(
                seg.get(k, 1.0) < threshold for k in signal_keys if k in seg
            )
            if not low_quality:
                continue

            seg_frames = [
                f
                for f, t in zip(loaded_video_frames, timestamps, strict=False)
                if seg["start_time"] <= t <= seg["end_time"]
            ]
            if not seg_frames:
                continue

            best_annotation = seg.get("annotation", {})
            best_caption = seg.get("caption", "")
            best_score = seg.get("quality_score", 0.0)

            for _ in range(num_retries):
                try:
                    result = caption_fn(seg_frames)
                    if isinstance(result, str):
                        new_annotation = {
                            "summary": {"brief": result, "detailed": result},
                            "action": {"brief": "", "detailed": "", "actor": None},
                        }
                        new_caption = result
                    else:
                        new_annotation = result
                        new_caption = result.get("summary", {}).get("brief", "")

                    if new_caption and new_caption != best_caption:
                        best_annotation = new_annotation
                        best_caption = new_caption
                        break
                except Exception:
                    logger.warning("_fix_low_quality_annotations re-caption failed", exc_info=True)

            if best_caption and best_caption != seg.get("caption", ""):
                seg["annotation"] = best_annotation
                seg["caption"] = best_caption
                logger.info(
                    "Fixed low-quality segment %.1f-%.1fs",
                    seg["start_time"],
                    seg["end_time"],
                )

    def _encode_frames(
        self, frames: list[np.ndarray], temporal_window: int = 1, stride: int | None = None
    ) -> np.ndarray:
        """Encode a batch of BGR frames into an (N, D) embedding matrix."""
        import torch
        from PIL import Image

        images = [Image.fromarray(f[:, :, ::-1]) for f in frames]

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

        all_embs_arr = np.concatenate(all_embs, axis=0)

        if stride is not None and stride < temporal_window and len(all_embs_arr) >= temporal_window:
            n = len(all_embs_arr)
            accum = np.zeros_like(all_embs_arr)
            counts = np.zeros(n, dtype=np.float32)
            for start in range(0, n - temporal_window + 1, stride):
                window_mean = all_embs_arr[start : start + temporal_window].mean(axis=0)
                for k in range(start, min(start + temporal_window, n)):
                    accum[k] += window_mean
                    counts[k] += 1
            counts = np.maximum(counts, 1)
            result = accum / counts[:, None]
            norms = np.linalg.norm(result, axis=1, keepdims=True)
            result = result / np.maximum(norms, 1e-10)
            return result

        if temporal_window > 1 and len(all_embs_arr) >= temporal_window:
            n = len(all_embs_arr)
            n_groups = n // temporal_window
            grouped = all_embs_arr[: n_groups * temporal_window].reshape(
                n_groups, temporal_window, -1
            )
            averaged = grouped.mean(axis=1)
            norms = np.linalg.norm(averaged, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-10)
            averaged = averaged / norms
            remainder = all_embs_arr[n_groups * temporal_window :]
            if len(remainder) > 0:
                rem_avg = remainder.mean(axis=0, keepdims=True)
                rem_avg = rem_avg / np.maximum(
                    np.linalg.norm(rem_avg, axis=1, keepdims=True), 1e-10
                )
                averaged = np.concatenate([averaged, rem_avg], axis=0)
            return averaged

        return all_embs_arr

    def _encode_texts(self, texts: list[str]) -> np.ndarray:
        """Encode a list of text strings into an (N, D) embedding matrix."""
        self._ensure_text_model()

        if self._text_model is not None and self._text_embedding_model_name is not None:
            if self._text_model_type == "sentence_transformers":
                emb = self._text_model.encode(texts, normalize_embeddings=True)
                return np.asarray(emb)
            else:
                import torch

                inputs = self._text_tokenizer(
                    texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                )
                with torch.no_grad():
                    out = self._text_model(**inputs)
                    emb = out.last_hidden_state[:, 0, :]
                    emb = emb / emb.norm(p=2, dim=-1, keepdim=True)
                return emb.cpu().numpy()

        import torch

        inputs = self._tokenizer(
            texts,
            padding="max_length",
            max_length=64,
            return_tensors="pt",
        ).to(self._torch_device)
        with torch.no_grad():
            out = self._model.get_text_features(**inputs)
            emb = out.pooler_output if hasattr(out, "pooler_output") else out
            emb = emb / emb.norm(p=2, dim=-1, keepdim=True)
        return emb.cpu().numpy()

    def _selective_decode(
        self,
        segments: list[dict],
        frames: list[np.ndarray],
        timestamps: list[float],
        similarity_threshold: float = 0.98,
        temporal_clip_embeddings: np.ndarray | None = None,
        temporal_clip_timestamps: list[float] | None = None,
    ) -> None:
        """3-tier selective decoding to optimize captioning cost.

        Tier 0 — DEAD: Skip captioning entirely (black/blank frames).
        Tier 1 — STATIC-INFORMATIVE: Caption with 1 keyframe only (slides, charts).
        Tier 2 — DYNAMIC: Full captioning pipeline (no change).

        V-JEPA temporal variance can promote Tier 1 → Tier 2 when subtle motion
        is detected that SigLIP2 misses.
        """
        import cv2

        self._ensure_model()

        tier_0_count = 0
        tier_1_count = 0
        tier_2_count = 0

        for seg in segments:
            if seg.get("_skip_caption"):
                continue
            seg_frames = [
                f for f, t in zip(frames, timestamps, strict=False)
                if seg["start_time"] <= t <= seg["end_time"]
            ]
            if not seg_frames:
                continue

            # --- Tier 0: DEAD frame detection ---
            # Sample 1-2 frames and check pixel variance + edge density
            sample_indices = [len(seg_frames) // 2]
            if len(seg_frames) >= 4:
                sample_indices.append(len(seg_frames) // 4)
            is_dead = True
            for si in sample_indices:
                sample = seg_frames[si]
                gray = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)
                pixel_std = float(gray.std())
                laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                edge_density = laplacian_var / max(gray.mean(), 1.0)
                if pixel_std >= 5.0 and edge_density >= 0.01:
                    is_dead = False
                    break

            if is_dead:
                seg["_skip_caption"] = True
                seg["_selective_tier"] = 0
                seg["caption"] = "Dead frame (black/blank)"
                seg["annotation"] = {
                    "summary": {
                        "brief": "Dead frame (black/blank)",
                        "detailed": "This segment contains dead frames with no visual content.",
                    },
                    "action": {"brief": "N/A", "detailed": "", "actor": None},
                }
                seg["is_non_action"] = True
                tier_0_count += 1
                continue

            # --- Compute SigLIP2 visual similarity for Tier 1/2 ---
            if len(seg_frames) < 3:
                seg["_selective_tier"] = 2
                tier_2_count += 1
                continue

            try:
                embs = self._encode_frames(seg_frames)
                sim_matrix = embs @ embs.T
                n = len(sim_matrix)
                if n < 2:
                    seg["_selective_tier"] = 2
                    tier_2_count += 1
                    continue
                mean_sim = float(
                    (sim_matrix.sum() - np.trace(sim_matrix)) / (n * (n - 1))
                )
                seg["_visual_variance"] = round(1.0 - mean_sim, 6)
            except Exception:
                seg["_selective_tier"] = 2
                tier_2_count += 1
                continue

            if mean_sim <= similarity_threshold:
                # Dynamic — full captioning
                seg["_selective_tier"] = 2
                tier_2_count += 1
                continue

            # --- Tier 1 candidate: check V-JEPA temporal variance for promotion ---
            if temporal_clip_embeddings is not None and temporal_clip_timestamps is not None:
                clip_indices = [
                    i
                    for i, ct in enumerate(temporal_clip_timestamps)
                    if seg["start_time"] <= ct <= seg["end_time"]
                ]
                if len(clip_indices) >= 2:
                    clip_embs = temporal_clip_embeddings[clip_indices]
                    temporal_var = float(np.var(clip_embs, axis=0).mean())
                    seg["_temporal_variance"] = round(temporal_var, 6)
                    if temporal_var > 0.05:
                        # Subtle motion detected — promote to dynamic
                        seg["_selective_tier"] = 2
                        tier_2_count += 1
                        continue

            # --- Tier 1: STATIC-INFORMATIVE — keep only middle keyframe ---
            seg["_selective_tier"] = 1
            seg["_static_informative"] = True
            real_frames = [f for f in seg.get("_frames", []) if not isinstance(f, str)]
            if real_frames:
                mid_frame = real_frames[len(real_frames) // 2]
                str_tokens = [f for f in seg.get("_frames", []) if isinstance(f, str)]
                seg["_frames"] = str_tokens + [mid_frame]
            tier_1_count += 1

        logger.info(
            "Selective decode: Tier 0 (dead): %d, Tier 1 (static-informative): %d, "
            "Tier 2 (dynamic): %d out of %d segments",
            tier_0_count,
            tier_1_count,
            tier_2_count,
            len(segments),
        )

    def _action_first_pass(
        self,
        segment_infos: list[dict],
        frame_caption_fn: Callable | None,
    ) -> None:
        """Set brief frame captions for fast-mode indexing (action-first pass).

        For each non-skipped segment, extracts the midpoint keyframe and calls
        ``frame_caption_fn`` to produce a brief caption.  Sets ``caption``,
        ``frame_caption``, and a minimal ``annotation`` structure so that
        ``_embed_captions`` can produce searchable embeddings immediately,
        without running the full Tree-of-Captions or Self-Refine pipeline.

        Skipped segments (Tier-0 dead frames or pre-caption dedup) have their
        ``_frames`` key removed; caption propagation from representatives is
        handled by the caller after this method returns.
        """
        caption_tasks = []
        for seg in segment_infos:
            seg_frames = seg.pop("_frames", [])
            if seg.get("_skip_caption"):
                # Already captioned (Tier 0) or dedup'd — propagation handled by caller
                continue
            real_frames = [f for f in seg_frames if not isinstance(f, str)]
            if real_frames:
                mid_frame = real_frames[len(real_frames) // 2]
                caption_tasks.append((seg, mid_frame))

        if frame_caption_fn is None or not caption_tasks:
            return

        def _caption_one(args):
            seg, mid_frame = args
            try:
                result = frame_caption_fn([mid_frame])
                caption = result if isinstance(result, str) else str(result)
            except Exception:
                logger.warning("Fast-mode frame caption failed", exc_info=True)
                caption = ""
            return seg, caption

        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(_caption_one, task) for task in caption_tasks]
            for future in as_completed(futures):
                try:
                    seg, caption = future.result()
                    seg["frame_caption"] = caption
                    seg["caption"] = caption
                    seg["annotation"] = {
                        "summary": {"brief": caption, "detailed": caption},
                        "action": {"brief": "", "detailed": "", "actor": None},
                    }
                except Exception:
                    logger.warning(
                        "Fast-mode caption future raised an exception", exc_info=True
                    )

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
        transcript = index.transcript
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
                    transcript,
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
        self._refine_annotations(
            segment_infos,
            transcript,
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

    def _build_coarse_level(
        self,
        segments: list[dict],
        embeddings: np.ndarray,
        target_duration: float = 30.0,
    ) -> tuple[list[dict], np.ndarray | None]:
        """Merge fine segments into ~30s coarse chunks."""
        if not segments or embeddings is None or len(embeddings) == 0:
            return [], None

        coarse_segs: list[dict] = []
        coarse_embs: list[np.ndarray] = []

        group_start = 0
        group_duration = 0.0

        for i, seg in enumerate(segments):
            seg_dur = seg["end_time"] - seg["start_time"]
            group_duration += seg_dur

            is_last = i == len(segments) - 1
            if group_duration >= target_duration or is_last:
                group_segs = segments[group_start : i + 1]
                merged_caption = " ".join(
                    s.get("caption", "") for s in group_segs if s.get("caption")
                )
                coarse_segs.append(
                    {
                        "start_time": group_segs[0]["start_time"],
                        "end_time": group_segs[-1]["end_time"],
                        "caption": merged_caption,
                    }
                )

                group_emb = embeddings[group_start : i + 1].mean(axis=0)
                norm = np.linalg.norm(group_emb)
                if norm > 1e-10:
                    group_emb = group_emb / norm
                coarse_embs.append(group_emb)

                group_start = i + 1
                group_duration = 0.0

        if not coarse_embs:
            return [], None

        return coarse_segs, np.stack(coarse_embs)

    def _encode_texts_siglip(self, texts: list[str]) -> np.ndarray:
        """Encode texts using SigLIP2's text encoder (ignoring text_embedding_model)."""
        import torch

        inputs = self._tokenizer(
            texts,
            padding="max_length",
            max_length=64,
            return_tensors="pt",
        ).to(self._torch_device)
        with torch.no_grad():
            out = self._model.get_text_features(**inputs)
            emb = out.pooler_output if hasattr(out, "pooler_output") else out
            emb = emb / emb.norm(p=2, dim=-1, keepdim=True)
        return emb.cpu().numpy()

    def _encode_query_siglip(self, text: str) -> np.ndarray:
        """Encode a single query using SigLIP2's text encoder."""
        self._ensure_model()
        return self._encode_texts_siglip([text])[0]

    def _encode_query(self, text: str) -> np.ndarray:
        """Encode a single query string — stored as ``embed_fn`` on the index."""
        self._ensure_model()
        return self._encode_texts([text])[0]

    def _encode_clips_vjepa(
        self,
        clips: list[list[np.ndarray]],
        return_full: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, list[np.ndarray]]:
        """Embed video clips using V-JEPA 2.

        Args:
            clips: List of clips, each a list of frames (H, W, C) BGR.
            return_full: When False (default), returns pooled embeddings array
                of shape (N_clips, D). When True, returns a tuple
                ``(pooled_embs, feature_maps)`` where ``feature_maps`` is a
                list of arrays of shape ``(num_patches, D)``, one per clip.
        """
        from itertools import groupby

        import torch

        indexed_clips = list(enumerate(clips))
        indexed_clips.sort(key=lambda x: len(x[1]))

        result_embs = [None] * len(clips)
        result_maps: list[np.ndarray | None] | None = [None] * len(clips) if return_full else None
        batch_size = 2 if getattr(self, "_scene_embed_dim", 1024) >= 1280 else 4

        for _frame_count, group in groupby(indexed_clips, key=lambda x: len(x[1])):
            group_items = list(group)
            for i in range(0, len(group_items), batch_size):
                batch_items = group_items[i : i + batch_size]
                rgb_clips = []
                for _idx, clip in batch_items:
                    rgb_frames = [f[:, :, ::-1] for f in clip]
                    rgb_clips.append(rgb_frames)

                inputs = self._scene_processor(rgb_clips, return_tensors="pt").to(
                    self._scene_torch_device
                )
                with torch.no_grad():
                    outputs = self._scene_model(**inputs)
                    patch_tokens = outputs.last_hidden_state
                    clip_embs = patch_tokens.mean(dim=1)
                    clip_embs = clip_embs / clip_embs.norm(p=2, dim=-1, keepdim=True)
                embs_np = clip_embs.cpu().float().numpy()
                if return_full:
                    maps_np = patch_tokens.cpu().float().numpy()
                for j, (orig_idx, _clip) in enumerate(batch_items):
                    result_embs[orig_idx] = embs_np[j]
                    if return_full:
                        result_maps[orig_idx] = maps_np[j]

        pooled = np.stack(result_embs)
        if return_full:
            return pooled, result_maps
        return pooled

    def _group_frames_into_clips(
        self,
        frames: list[np.ndarray],
        timestamps: list[float],
        clip_size: int,
    ) -> tuple[list[list[np.ndarray]], list[float]]:
        """Group frames into clips with midpoint timestamps."""
        clips: list[list[np.ndarray]] = []
        clip_timestamps: list[float] = []
        for i in range(0, len(frames), clip_size):
            clip = frames[i : i + clip_size]
            clips.append(clip)
            mid = min(i + len(clip) // 2, len(frames) - 1)
            clip_timestamps.append(timestamps[mid])
        return clips, clip_timestamps

    def _encode_frames_overlapping_vjepa(
        self,
        frames: list[np.ndarray],
        timestamps: list[float],
        clip_size: int = 64,
        stride: int = 8,
        store_feature_maps: bool = False,
    ) -> tuple[np.ndarray, list[float]] | tuple[np.ndarray, list[float], list[np.ndarray]]:
        """Encode frames using overlapping V-JEPA 2 windows with per-frame averaging.

        Each frame appears in up to clip_size/stride windows. The final per-frame
        embedding is the L2-normalized average of all window embeddings containing
        that frame.

        Args:
            frames: BGR numpy arrays.
            timestamps: Per-frame timestamps in seconds.
            clip_size: Number of frames per V-JEPA 2 window (default 64).
            stride: Window stride in frames (default 8).
            store_feature_maps: If True, also return per-window spatial feature maps.

        Returns:
            Tuple of (per_frame_embeddings, timestamps) — or
            (per_frame_embeddings, timestamps, per_window_feature_maps) when
            store_feature_maps is True.
        """
        n = len(frames)
        if n == 0:
            if store_feature_maps:
                return np.empty((0, self._scene_embed_dim), dtype=np.float32), [], []
            return np.empty((0, self._scene_embed_dim), dtype=np.float32), []

        # Build overlapping windows
        windows = []
        window_frame_ranges = []  # (start_idx, end_idx) for each window
        for start in range(0, n, stride):
            end = min(start + clip_size, n)
            if end - start < 2:  # skip tiny windows
                continue
            windows.append(frames[start:end])
            window_frame_ranges.append((start, end))

        if not windows:
            # Fallback: single window with all frames
            windows = [frames]
            window_frame_ranges = [(0, n)]

        # Encode all windows via _encode_clips_vjepa
        feature_maps: list[np.ndarray] | None = None
        if store_feature_maps:
            clip_embeddings, feature_maps = self._encode_clips_vjepa(
                windows, return_full=True
            )
        else:
            clip_embeddings = self._encode_clips_vjepa(windows)  # (N_windows, D)

        # Per-frame averaging: accumulate embeddings for each frame
        D = clip_embeddings.shape[1]
        frame_emb_sum = np.zeros((n, D), dtype=np.float64)
        frame_emb_count = np.zeros(n, dtype=np.float64)

        for w_idx, (start, end) in enumerate(window_frame_ranges):
            for f_idx in range(start, end):
                frame_emb_sum[f_idx] += clip_embeddings[w_idx]
                frame_emb_count[f_idx] += 1.0

        # Average and L2-normalize
        mask = frame_emb_count > 0
        per_frame = np.zeros((n, D), dtype=np.float32)
        per_frame[mask] = (frame_emb_sum[mask] / frame_emb_count[mask, np.newaxis]).astype(
            np.float32
        )

        norms = np.linalg.norm(per_frame, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        per_frame = per_frame / norms

        if store_feature_maps and feature_maps is not None:
            return per_frame, timestamps, feature_maps
        return per_frame, timestamps

    def _get_transcript(
        self,
        video_path: str,
        *,
        asr_model: str = "Qwen/Qwen3-ASR-0.6B",
        transcript_path: str | None = None,
    ) -> list[dict]:
        """Return ASR transcript as a list of ``{start_time, end_time, text}`` dicts."""
        if transcript_path is not None:
            return self._load_transcript_file(transcript_path)

        return self._run_asr(video_path, asr_model)

    @staticmethod
    def _load_transcript_file(path: str) -> list[dict]:
        """Load a transcript from a JSON file.

        Validates that each entry has the required keys (start_time, end_time,
        text).  Malformed entries are skipped with a warning.
        """
        try:
            data = json.loads(Path(path).read_text())
            if not isinstance(data, list):
                logger.warning("Transcript file %s is not a JSON list; ignoring.", path)
                return []
            required = ("start_time", "end_time", "text")
            valid = [
                e for e in data
                if isinstance(e, dict) and all(k in e for k in required)
            ]
            if len(valid) < len(data):
                logger.warning(
                    "Skipped %d invalid transcript entries in %s",
                    len(data) - len(valid), path,
                )
            return valid
        except Exception:
            logger.warning("Failed to load transcript from %s", path, exc_info=True)
        return []

    @staticmethod
    def _extract_audio(video_path: str, out_wav: str) -> bool:
        """Extract audio track to a WAV file using ffmpeg."""
        try:
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    video_path,
                    "-vn",
                    "-acodec",
                    "pcm_s16le",
                    "-ar",
                    "16000",
                    "-ac",
                    "1",
                    out_wav,
                ],
                check=True,
                capture_output=True,
            )
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.warning("Audio extraction failed for %s (ffmpeg may be missing).", video_path)
            return False

    # Batch size for ASR inference.  On CPU/MPS the autoregressive decoder is
    # memory-bound; small batches avoid excessive padding.  On CUDA, larger
    # batches saturate the GPU.
    _ASR_BATCH_CPU = 4
    _ASR_BATCH_CUDA = 16
    # Duration (seconds) of each audio chunk we feed to qwen_asr.  Shorter
    # chunks decode faster (less autoregressive steps) and produce less padding
    # waste when batched.  30s is a sweet spot — short enough to keep decoding
    # fast, long enough to avoid excessive chunk overhead.
    # IMPORTANT: Must stay < 180 to avoid double-offset issues with
    # qwen_asr's internal MAX_FORCE_ALIGN_INPUT_SECONDS limit.
    _ASR_CHUNK_SEC = 30
    _ASR_OVERLAP_SEC = 1  # overlap between chunks to avoid boundary word loss

    def _ensure_asr_model(self, model_name: str) -> None:
        """Lazily load and cache the Qwen3-ASR model."""
        if self._asr_model is not None and self._asr_model_name == model_name:
            return

        try:
            from qwen_asr import Qwen3ASRModel
        except ImportError:
            logger.info("qwen_asr not installed; skipping ASR.")
            return

        import torch

        device = (
            "mps"
            if torch.backends.mps.is_available()
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        batch_size = self._ASR_BATCH_CUDA if device == "cuda" else self._ASR_BATCH_CPU
        dtype = torch.bfloat16 if device == "cuda" else torch.float16
        logger.info("[pipeline] Qwen3-ASR: loading model on %s (batch=%d)", device, batch_size)

        try:
            self._asr_model = Qwen3ASRModel.from_pretrained(
                model_name,
                dtype=dtype,
                device_map=device,
                max_inference_batch_size=batch_size,
                forced_aligner="Qwen/Qwen3-ForcedAligner-0.6B",
                forced_aligner_kwargs=dict(
                    dtype=dtype,
                    device_map=device,
                ),
            )
        except (OSError, RuntimeError, ImportError, ValueError):
            logger.info(
                "Qwen3-ForcedAligner-0.6B unavailable; "
                "proceeding without word-level alignment."
            )
            self._asr_model = Qwen3ASRModel.from_pretrained(
                model_name,
                dtype=dtype,
                device_map=device,
                max_inference_batch_size=batch_size,
            )
        self._asr_model_name = model_name
        self._asr_batch_size = batch_size
        logger.info("[pipeline] Qwen3-ASR: model loaded")

    @staticmethod
    def _split_audio_chunks(
        wav_path: str, chunk_sec: int, tmp_dir: str, overlap_sec: float = 0.0,
    ) -> list[tuple[str, float]]:
        """Split a WAV file into fixed-duration chunks using ffmpeg.

        Returns list of (chunk_path, offset_seconds) tuples.  When
        *overlap_sec* > 0 each chunk (except the first) starts that many
        seconds before the nominal boundary so words at the cut point are
        captured by both the previous and the current chunk.
        """
        assert chunk_sec < 180, (
            f"_ASR_CHUNK_SEC={chunk_sec} must be < 180 "
            "(qwen_asr MAX_FORCE_ALIGN_INPUT_SECONDS limit)"
        )
        import wave

        with wave.open(wav_path, "rb") as wf:
            duration = wf.getnframes() / wf.getframerate()

        if duration <= chunk_sec:
            return [(wav_path, 0.0)]

        stride = chunk_sec - overlap_sec
        chunks: list[tuple[str, float]] = []
        offset = 0.0
        idx = 0
        while offset < duration:
            chunk_dur = chunk_sec if idx == 0 else chunk_sec + overlap_sec
            chunk_path = str(Path(tmp_dir) / f"chunk_{idx:04d}.wav")
            try:
                subprocess.run(
                    [
                        "ffmpeg", "-y",
                        "-ss", str(offset),
                        "-t", str(chunk_dur),
                        "-i", wav_path,
                        "-acodec", "pcm_s16le",
                        "-ar", "16000",
                        "-ac", "1",
                        chunk_path,
                    ],
                    check=True,
                    capture_output=True,
                )
                chunks.append((chunk_path, offset))
            except subprocess.CalledProcessError:
                logger.warning("Failed to split audio chunk at offset %.1f", offset)
                break
            offset += stride if idx == 0 else stride
            idx += 1

        return chunks if chunks else [(wav_path, 0.0)]

    def _run_asr(self, video_path: str, model_name: str) -> list[dict]:
        """Transcribe audio using Qwen3-ASR with word-level timestamps.

        Splits audio into short chunks (default 30s) before transcription.
        Shorter chunks decode faster and produce less padding waste when batched.
        """
        self._ensure_asr_model(model_name)
        if self._asr_model is None:
            return []

        with tempfile.TemporaryDirectory() as tmp:
            wav_path = str(Path(tmp) / "audio.wav")
            if not self._extract_audio(video_path, wav_path):
                return []

            # Split into short chunks for faster decoding (with overlap for dedup)
            overlap = self._ASR_OVERLAP_SEC
            chunks = self._split_audio_chunks(
                wav_path, self._ASR_CHUNK_SEC, tmp, overlap_sec=overlap,
            )
            chunk_paths = [c[0] for c in chunks]
            chunk_offsets = [c[1] for c in chunks]
            logger.info(
                "[pipeline] Qwen3-ASR: transcribing %d chunk(s) of %ds (overlap=%ds)",
                len(chunks), self._ASR_CHUNK_SEC, overlap,
            )

            try:
                # Transcribe in batches so we can log progress on long videos.
                bs = getattr(self, "_asr_batch_size", None) or len(chunk_paths)
                all_results: list = []
                for i in range(0, len(chunk_paths), bs):
                    batch = chunk_paths[i : i + bs]
                    r = self._asr_model.transcribe(
                        audio=batch, return_time_stamps=True,
                    )
                    all_results.extend(r)
                    logger.info(
                        "[pipeline] Qwen3-ASR: %d/%d chunks done",
                        min(i + bs, len(chunk_paths)), len(chunk_paths),
                    )

                if not all_results:
                    return []

                # Merge results from all chunks, offsetting timestamps.
                # For chunks after the first, skip words in the overlap
                # region that are better covered by the previous chunk.
                transcript: list[dict] = []
                for chunk_idx, (asr_result, offset) in enumerate(
                    zip(all_results, chunk_offsets)
                ):
                    skip_before = (
                        offset + overlap / 2 if chunk_idx > 0 and overlap > 0 else 0.0
                    )
                    self._collect_transcript_segments(
                        asr_result, offset, transcript, skip_before=skip_before,
                    )

                logger.info("[pipeline] Qwen3-ASR: %d segments transcribed", len(transcript))
                return transcript
            except Exception:
                logger.warning("Qwen3-ASR transcription failed.", exc_info=True)
                return []

    @staticmethod
    def _collect_transcript_segments(
        asr_result,
        offset: float,
        transcript: list[dict],
        *,
        skip_before: float = 0.0,
    ) -> None:
        """Extract sentence-level segments from one ASR chunk result.

        Appends to *transcript* in-place, adding *offset* to all timestamps.
        When *skip_before* > 0, words whose offset-corrected start_time falls
        before that threshold are dropped — they belong to the overlap region
        and are better covered by the previous chunk.
        """
        if asr_result.time_stamps is not None and asr_result.time_stamps.items:
            items = asr_result.time_stamps.items
            seg_words: list[dict] = []
            seg_start: float | None = None
            seg_end: float = 0.0

            def _flush_segment() -> None:
                nonlocal seg_words, seg_start, seg_end
                if not seg_words:
                    return
                text = " ".join(w["text"] for w in seg_words).strip()
                if text:
                    transcript.append({
                        "start_time": round(seg_start + offset, 3),
                        "end_time": round(seg_end + offset, 3),
                        "text": text,
                        "words": seg_words,
                    })
                seg_words = []
                seg_start = None

            for idx, item in enumerate(items):
                # Skip words in the overlap region covered by previous chunk
                if skip_before > 0 and (item.start_time + offset) < skip_before:
                    continue

                if seg_start is not None and (item.start_time - seg_end) > 1.0:
                    _flush_segment()

                if seg_start is None:
                    seg_start = item.start_time
                seg_words.append({
                    "text": item.text,
                    "start_time": round(item.start_time + offset, 3),
                    "end_time": round(item.end_time + offset, 3),
                })
                seg_end = item.end_time

                is_sentence_end = item.text.rstrip().endswith((".", "!", "?"))
                at_end = idx == len(items) - 1
                if is_sentence_end or at_end:
                    _flush_segment()

        elif asr_result.text.strip():
            transcript.append({
                "start_time": round(offset, 3),
                "end_time": round(offset, 3),
                "text": asr_result.text.strip(),
                "words": [],
            })
