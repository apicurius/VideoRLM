"""Factory functions that create REPL-injectable search tool closures.

Each ``make_*`` function accepts a :class:`~rlm.video.video_indexer.VideoIndex`
and returns a dict ``{"tool": callable, "description": str}`` compatible with
the custom-tools interface used by :class:`~rlm.video.video_rlm.VideoRLM`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from rlm.video.video_indexer import VideoIndex


def make_search_video(index: VideoIndex) -> dict[str, Any]:
    """Semantic search over video segment embeddings.

    Embeds the query text with the same model used during indexing and returns
    the *top_k* most similar segments by cosine similarity.
    """
    from sklearn.metrics.pairwise import cosine_similarity

    def search_video(query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Search video segments by semantic similarity to *query*.

        Args:
            query: Natural-language description of what to find.
            top_k: Number of results to return.

        Returns:
            List of dicts with ``start_time``, ``end_time``, ``score``, and
            ``caption`` for each matching segment, sorted by descending score.
        """
        if index.embeddings is None or len(index.embeddings) == 0:
            return []

        # Encode query with the same model stored in the index
        query_emb = index.embed_fn(query)
        query_emb = np.asarray(query_emb).reshape(1, -1)

        scores = cosine_similarity(query_emb, index.embeddings)[0]
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            seg = index.segments[idx]
            results.append(
                {
                    "start_time": seg["start_time"],
                    "end_time": seg["end_time"],
                    "score": round(float(scores[idx]), 4),
                    "caption": seg.get("caption", ""),
                }
            )
        return results

    return {
        "tool": search_video,
        "description": (
            "search_video(query, top_k=5) -> list[dict]. "
            "Semantic search over pre-indexed video segments. Returns the top_k "
            "most relevant segments with start_time, end_time, score, and caption."
        ),
    }


def make_search_transcript(index: VideoIndex) -> dict[str, Any]:
    """Keyword search over ASR transcript entries."""

    def search_transcript(query: str) -> list[dict[str, Any]]:
        """Search the video transcript for lines containing *query*.

        Args:
            query: Keyword or phrase to search for (case-insensitive).

        Returns:
            List of dicts with ``start_time``, ``end_time``, ``text``, and
            surrounding ``context`` for each match.
        """
        if not index.transcript:
            return []

        query_lower = query.lower()
        results = []
        for i, entry in enumerate(index.transcript):
            if query_lower in entry["text"].lower():
                # Build context from surrounding entries
                context_entries = index.transcript[max(0, i - 1) : i + 2]
                context = " ".join(e["text"] for e in context_entries)
                results.append(
                    {
                        "start_time": entry["start_time"],
                        "end_time": entry["end_time"],
                        "text": entry["text"],
                        "context": context,
                    }
                )
        return results

    return {
        "tool": search_transcript,
        "description": (
            "search_transcript(query) -> list[dict]. "
            "Search spoken words in the video transcript (ASR). Returns matching "
            "entries with start_time, end_time, text, and surrounding context."
        ),
    }


def make_get_transcript(index: VideoIndex) -> dict[str, Any]:
    """Retrieve transcript text for a time range."""

    def get_transcript(start_time: float, end_time: float) -> str:
        """Return transcript text for a specific time range.

        Args:
            start_time: Start of the range in seconds.
            end_time: End of the range in seconds.

        Returns:
            Concatenated transcript text for the given time range.
        """
        if not index.transcript:
            return ""

        lines = []
        for entry in index.transcript:
            # Include entries that overlap with the requested range
            if entry["end_time"] >= start_time and entry["start_time"] <= end_time:
                lines.append(f"[{entry['start_time']:.1f}s] {entry['text']}")
        return "\n".join(lines)

    return {
        "tool": get_transcript,
        "description": (
            "get_transcript(start_time, end_time) -> str. "
            "Get the spoken transcript for a specific time range of the video."
        ),
    }


def make_get_scene_list(index: VideoIndex) -> dict[str, Any]:
    """Return scene boundaries with captions."""

    def get_scene_list() -> list[dict[str, Any]]:
        """List all detected scene boundaries with descriptions.

        Returns:
            List of dicts with ``scene_index``, ``start_time``, ``end_time``,
            and ``caption`` for each scene.
        """
        scenes = []
        for i, seg in enumerate(index.segments):
            scenes.append(
                {
                    "scene_index": i,
                    "start_time": seg["start_time"],
                    "end_time": seg["end_time"],
                    "caption": seg.get("caption", ""),
                }
            )
        return scenes

    return {
        "tool": get_scene_list,
        "description": (
            "get_scene_list() -> list[dict]. "
            "List all detected scene boundaries with scene_index, start_time, "
            "end_time, and caption. Use this to understand the video structure."
        ),
    }
