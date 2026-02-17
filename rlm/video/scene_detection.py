"""Semantic scene boundary detection using frame embeddings."""

from __future__ import annotations

from typing import Callable

import numpy as np


def detect_scenes(
    frames: list[np.ndarray],
    timestamps: list[float],
    embed_fn: Callable[[list[np.ndarray]], np.ndarray],
    threshold: float = 0.3,
) -> list[tuple[float, float]]:
    """Detect scene boundaries via embedding-based semantic similarity.

    Embeds each frame with *embed_fn*, then marks a scene boundary wherever
    the cosine distance between consecutive frame embeddings exceeds
    *threshold*.

    Args:
        frames: BGR numpy arrays (OpenCV format).
        timestamps: Per-frame timestamps in seconds (same length as *frames*).
        embed_fn: A callable that takes a list of frames (numpy arrays) and
            returns an ``(N, D)`` embedding matrix.  Typically backed by a
            multimodal model like CLIP / SigLIP / Jina-CLIP.
        threshold: Cosine distance (1 - similarity) above which a scene cut
            is declared.  Range 0-2; higher → fewer cuts.  Default 0.3 works
            well for CLIP-family models.

    Returns:
        List of ``(start_time, end_time)`` tuples for each scene.
    """
    if len(frames) != len(timestamps):
        raise ValueError(
            f"frames ({len(frames)}) and timestamps ({len(timestamps)}) must have the same length"
        )
    if not frames:
        return []
    if len(frames) == 1:
        return [(timestamps[0], timestamps[0])]

    # Embed all frames in a single batch
    embeddings = np.asarray(embed_fn(frames))  # (N, D)

    # L2-normalise for cosine distance
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    embeddings = embeddings / norms

    # Detect boundaries: cosine distance between consecutive frames
    boundaries: list[int] = [0]
    for i in range(1, len(embeddings)):
        cosine_sim = float(np.dot(embeddings[i - 1], embeddings[i]))
        cosine_dist = 1.0 - cosine_sim
        if cosine_dist > threshold:
            boundaries.append(i)

    # Convert boundary indices to (start, end) time ranges
    scenes: list[tuple[float, float]] = []
    for i, bnd in enumerate(boundaries):
        start = timestamps[bnd]
        if i + 1 < len(boundaries):
            end = timestamps[boundaries[i + 1]]
        else:
            end = timestamps[-1]
        scenes.append((start, end))

    return scenes
