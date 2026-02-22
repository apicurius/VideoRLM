"""Corpus-level multi-video indexing and search."""

from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


@dataclass
class CorpusIndex:
    """A searchable index spanning multiple videos.

    Aggregates individual VideoIndex objects and provides cross-video
    search, action vocabulary, and corpus-level statistics.
    """

    video_indices: dict[str, Any] = field(default_factory=dict)  # video_id -> VideoIndex
    video_metadata: dict[str, dict] = field(default_factory=dict)  # video_id -> {path, duration, ...}
    action_vocabulary: dict[str, list[dict]] = field(default_factory=dict)  # action_brief -> [...]
    corpus_embeddings: np.ndarray | None = None  # (total_segments, D) stacked from all videos
    corpus_segment_map: list[dict] = field(default_factory=list)  # maps global idx -> {video_id, segment_idx}

    @property
    def num_videos(self) -> int:
        return len(self.video_indices)

    @property
    def total_segments(self) -> int:
        return len(self.corpus_segment_map)

    @property
    def total_duration(self) -> float:
        return sum(m.get("duration", 0) for m in self.video_metadata.values())

    def save(self, path: str | Path) -> None:
        """Save corpus index to directory."""
        directory = Path(path)
        directory.mkdir(parents=True, exist_ok=True)

        # Save each video index
        indices_dir = directory / "indices"
        indices_dir.mkdir(exist_ok=True)
        for video_id, idx in self.video_indices.items():
            idx.save(indices_dir / video_id)

        # Save corpus-level embeddings
        if self.corpus_embeddings is not None:
            np.savez(directory / "corpus_embeddings.npz", embeddings=self.corpus_embeddings)

        metadata = {
            "video_metadata": self.video_metadata,
            "action_vocabulary": self.action_vocabulary,
            "corpus_segment_map": self.corpus_segment_map,
        }
        (directory / "corpus_metadata.json").write_text(json.dumps(metadata))

    @classmethod
    def load(cls, path: str | Path) -> CorpusIndex:
        """Load a corpus index from directory."""
        from kuavi.indexer import VideoIndex

        directory = Path(path)
        metadata = json.loads((directory / "corpus_metadata.json").read_text())

        # Load video indices
        indices_dir = directory / "indices"
        video_indices = {}
        if indices_dir.exists():
            for sub in sorted(indices_dir.iterdir()):
                if sub.is_dir() and (sub / "metadata.json").exists():
                    video_indices[sub.name] = VideoIndex.load(sub)

        # Load corpus embeddings
        corpus_embeddings = None
        emb_path = directory / "corpus_embeddings.npz"
        if emb_path.exists():
            npz = np.load(emb_path)
            corpus_embeddings = npz["embeddings"] if "embeddings" in npz else None

        return cls(
            video_indices=video_indices,
            video_metadata=metadata.get("video_metadata", {}),
            action_vocabulary=metadata.get("action_vocabulary", {}),
            corpus_embeddings=corpus_embeddings,
            corpus_segment_map=metadata.get("corpus_segment_map", []),
        )


class CorpusIndexer:
    """Index multiple videos into a unified searchable corpus.

    Uses ThreadPoolExecutor for parallel video indexing and builds
    cross-video action vocabulary and stacked embeddings for corpus search.
    """

    def __init__(
        self,
        max_workers: int = 4,
        **indexer_kwargs,
    ):
        """
        Args:
            max_workers: Maximum parallel video indexing threads.
            **indexer_kwargs: Passed to VideoIndexer for each video.
        """
        self.max_workers = max_workers
        self._indexer_kwargs = indexer_kwargs

    def index_corpus(
        self,
        video_paths: list[str | Path],
        *,
        mode: str = "fast",
        caption_fn=None,
        frame_caption_fn=None,
        refine_fn=None,
        progress_callback=None,  # called with (video_path, status, elapsed)
    ) -> CorpusIndex:
        """Index multiple videos into a CorpusIndex.

        Args:
            video_paths: List of video file paths.
            mode: Indexing mode ("fast" or "full"). Default "fast" for corpus.
            caption_fn: Optional caption function (shared across videos).
            frame_caption_fn: Optional frame caption function.
            refine_fn: Optional refinement function.
            progress_callback: Optional callback(video_path, status, elapsed).

        Returns:
            A CorpusIndex with all videos indexed.
        """
        from kuavi.indexer import VideoIndexer
        from kuavi.loader import VideoLoader

        corpus = CorpusIndex()
        loader = VideoLoader(fps=self._indexer_kwargs.get("fps", 1.0))

        def _index_one(video_path: str | Path) -> tuple[str, Any, dict]:
            """Index a single video. Returns (video_id, VideoIndex, metadata)."""
            video_path = Path(video_path)
            video_id = video_path.stem
            t0 = time.time()

            try:
                loaded = loader.load(str(video_path))
                indexer = VideoIndexer(**self._indexer_kwargs)
                idx = indexer.index_video(
                    loaded,
                    caption_fn=caption_fn,
                    frame_caption_fn=frame_caption_fn,
                    refine_fn=refine_fn,
                    mode=mode,
                )
                meta = {
                    "path": str(video_path),
                    "duration": loaded.metadata.duration,
                    "num_segments": len(idx.segments),
                    "num_frames": loaded.metadata.extracted_frame_count,
                    "indexed_at": time.time(),
                }
                elapsed = time.time() - t0
                if progress_callback:
                    progress_callback(str(video_path), "done", elapsed)
                logger.info(
                    "Indexed %s in %.1fs (%d segments)", video_id, elapsed, len(idx.segments)
                )
                return video_id, idx, meta
            except Exception as e:
                elapsed = time.time() - t0
                logger.error("Failed to index %s: %s", video_path, e)
                if progress_callback:
                    progress_callback(str(video_path), f"error: {e}", elapsed)
                return video_id, None, {"path": str(video_path), "error": str(e)}

        # Parallel indexing
        if self.max_workers > 1 and len(video_paths) > 1:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {executor.submit(_index_one, vp): vp for vp in video_paths}
                for future in as_completed(futures):
                    video_id, idx, meta = future.result()
                    corpus.video_metadata[video_id] = meta
                    if idx is not None:
                        corpus.video_indices[video_id] = idx
        else:
            for vp in video_paths:
                video_id, idx, meta = _index_one(vp)
                corpus.video_metadata[video_id] = meta
                if idx is not None:
                    corpus.video_indices[video_id] = idx

        # Build cross-video structures
        self._build_action_vocabulary(corpus)
        self._build_corpus_embeddings(corpus)

        logger.info(
            "Corpus indexed: %d videos, %d total segments, %.1fs total duration",
            corpus.num_videos,
            corpus.total_segments,
            corpus.total_duration,
        )
        return corpus

    def _build_action_vocabulary(self, corpus: CorpusIndex) -> None:
        """Build cross-video action vocabulary from segment annotations."""
        vocab: dict[str, list[dict]] = {}

        for video_id, idx in corpus.video_indices.items():
            for seg_idx, seg in enumerate(idx.segments):
                annotation = seg.get("annotation", {})
                action = annotation.get("action", {}) if isinstance(annotation, dict) else {}
                brief = action.get("brief", "") if isinstance(action, dict) else ""
                if not brief:
                    continue

                brief_lower = brief.lower().strip()
                if brief_lower not in vocab:
                    vocab[brief_lower] = []
                vocab[brief_lower].append({
                    "video_id": video_id,
                    "segment_idx": seg_idx,
                    "start_time": seg["start_time"],
                    "end_time": seg["end_time"],
                })

        corpus.action_vocabulary = vocab

    def _build_corpus_embeddings(self, corpus: CorpusIndex) -> None:
        """Stack all video embeddings into a single matrix for corpus search."""
        all_embs = []
        segment_map = []

        for video_id, idx in sorted(corpus.video_indices.items()):
            if idx.embeddings is not None and len(idx.embeddings) > 0:
                all_embs.append(idx.embeddings)
                for seg_idx in range(len(idx.segments)):
                    segment_map.append({
                        "video_id": video_id,
                        "segment_idx": seg_idx,
                    })

        if all_embs:
            corpus.corpus_embeddings = np.vstack(all_embs)
        corpus.corpus_segment_map = segment_map


def search_corpus(
    corpus: CorpusIndex,
    query: str,
    top_k: int = 10,
    video_filter: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Search across all videos in the corpus.

    Args:
        corpus: The corpus index to search.
        query: Search query string.
        top_k: Number of results.
        video_filter: Optional list of video_ids to restrict search to.

    Returns:
        List of result dicts with video_id, start_time, end_time, score, caption.
    """
    from sklearn.metrics.pairwise import cosine_similarity

    if corpus.corpus_embeddings is None or len(corpus.corpus_embeddings) == 0:
        return []

    # Find an embed_fn from any video index
    embed_fn = None
    for idx in corpus.video_indices.values():
        if idx.embed_fn is not None:
            embed_fn = idx.embed_fn
            break

    if embed_fn is None:
        return []

    query_emb = np.asarray(embed_fn(query)).reshape(1, -1)
    scores = cosine_similarity(query_emb, corpus.corpus_embeddings)[0]

    # Apply video filter
    if video_filter:
        filter_set = set(video_filter)
        for i, entry in enumerate(corpus.corpus_segment_map):
            if entry["video_id"] not in filter_set:
                scores[i] = -np.inf

    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for gi in top_indices:
        if scores[gi] <= -np.inf:
            continue
        entry = corpus.corpus_segment_map[gi]
        video_id = entry["video_id"]
        seg_idx = entry["segment_idx"]
        idx = corpus.video_indices[video_id]
        seg = idx.segments[seg_idx]
        results.append({
            "video_id": video_id,
            "video_path": corpus.video_metadata.get(video_id, {}).get("path", ""),
            "start_time": seg["start_time"],
            "end_time": seg["end_time"],
            "score": round(float(scores[gi]), 4),
            "caption": seg.get("caption", ""),
            "annotation": seg.get("annotation", {}),
        })

    return results


def corpus_stats(corpus: CorpusIndex) -> dict[str, Any]:
    """Get statistics about the corpus."""
    action_counts = {k: len(v) for k, v in corpus.action_vocabulary.items()}
    top_actions = sorted(action_counts.items(), key=lambda x: x[1], reverse=True)[:20]

    return {
        "num_videos": corpus.num_videos,
        "total_segments": corpus.total_segments,
        "total_duration_seconds": round(corpus.total_duration, 1),
        "action_vocabulary_size": len(corpus.action_vocabulary),
        "top_actions": [{"action": a, "count": c} for a, c in top_actions],
        "videos": {
            vid: {
                "path": meta.get("path", ""),
                "duration": meta.get("duration", 0),
                "num_segments": meta.get("num_segments", 0),
            }
            for vid, meta in corpus.video_metadata.items()
        },
    }


def discover_videos(directory: str | Path) -> list[Path]:
    """Discover video files in a directory (non-recursive)."""
    directory = Path(directory)
    found = []
    for p in sorted(directory.iterdir()):
        if p.is_file() and p.suffix.lower() in _VIDEO_EXTENSIONS:
            found.append(p)
    return found
