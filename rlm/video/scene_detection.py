"""Scene boundary detection using frame embeddings or histogram fallback."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def detect_scenes(
    frames: list[np.ndarray],
    timestamps: list[float],
    embed_fn: Callable[[list[np.ndarray]], np.ndarray] | None = None,
    threshold: float = 0.3,
    min_duration: float = 4.0,
) -> list[tuple[float, float]]:
    """Detect scene boundaries via embedding clustering or histogram fallback.

    When *embed_fn* is provided, embeds each frame and clusters temporally-adjacent
    frames using agglomerative clustering with Ward linkage.  Otherwise falls back
    to color histogram chi-squared distance.

    Args:
        frames: BGR numpy arrays (OpenCV format).
        timestamps: Per-frame timestamps in seconds (same length as *frames*).
        embed_fn: Optional callable that takes a list of frames and returns an
            ``(N, D)`` embedding matrix.
        threshold: Distance threshold controlling the number of scenes.
            Higher → fewer scenes.
        min_duration: Minimum scene duration in seconds. Defaults to 4.0.

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

    if embed_fn is not None:
        return _detect_scenes_embedding(frames, timestamps, embed_fn, threshold, min_duration)
    return _detect_scenes_histogram(frames, timestamps, threshold, min_duration)


def _detect_scenes_embedding(
    frames: list[np.ndarray],
    timestamps: list[float],
    embed_fn: Callable[[list[np.ndarray]], np.ndarray],
    threshold: float,
    min_duration: float,
) -> list[tuple[float, float]]:
    """Embedding-based scene detection via agglomerative clustering."""
    embeddings = np.asarray(embed_fn(frames), dtype=np.float64)
    n = len(embeddings)

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    embeddings = embeddings / norms

    from scipy.sparse import eye as speye
    from sklearn.cluster import AgglomerativeClustering

    connectivity = (
        speye(n, format="csr") + speye(n, k=1, format="csr") + speye(n, k=-1, format="csr")
    )

    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=threshold,
        linkage="ward",
        connectivity=connectivity,
    )
    labels = clustering.fit_predict(embeddings)

    scenes: list[tuple[float, float]] = []
    scene_start = timestamps[0]
    current_label = labels[0]
    for i in range(1, n):
        if labels[i] != current_label:
            scenes.append((scene_start, timestamps[i - 1]))
            scene_start = timestamps[i]
            current_label = labels[i]
    scenes.append((scene_start, timestamps[-1]))

    if min_duration > 0:
        scenes = [(s, e) for s, e in scenes if (e - s) >= min_duration]
        if not scenes:
            scenes = [(timestamps[0], timestamps[-1])]

    return scenes


def detect_scenes_hierarchical(
    frames: list[np.ndarray],
    timestamps: list[float],
    embed_fn: Callable[[list[np.ndarray]], np.ndarray],
    thresholds: tuple[float, ...] = (0.15, 0.30, 0.50),
    min_durations: tuple[float, ...] = (0.5, 2.0, 4.0),
) -> dict:
    """Detect scene boundaries at multiple granularity levels.

    Runs embedding-based scene detection at each ``(threshold, min_duration)``
    pair.  Lower thresholds produce finer (more) scenes; higher thresholds
    produce coarser (fewer) scenes.  Coarse-level boundaries are snapped to
    the nearest fine-level boundary so that the hierarchy is consistent.

    Args:
        frames: BGR numpy arrays (OpenCV format).
        timestamps: Per-frame timestamps in seconds.
        embed_fn: Callable that embeds a list of frames into an ``(N, D)``
            matrix.
        thresholds: Distance thresholds from finest to coarsest.
        min_durations: Minimum scene durations corresponding to each threshold.

    Returns:
        ``{"levels": [finest_scenes, ..., coarsest_scenes]}`` where each
        entry is a list of ``(start_time, end_time)`` tuples.
    """
    if len(frames) != len(timestamps):
        raise ValueError(
            f"frames ({len(frames)}) and timestamps ({len(timestamps)}) must have the same length"
        )
    if not frames:
        return {"levels": [[] for _ in thresholds]}
    if len(frames) == 1:
        single = [(timestamps[0], timestamps[0])]
        return {"levels": [single for _ in thresholds]}

    levels: list[list[tuple[float, float]]] = []
    for thresh, min_dur in zip(thresholds, min_durations):
        scenes = _detect_scenes_embedding(frames, timestamps, embed_fn, thresh, min_dur)
        levels.append(scenes)

    # Align coarser levels to finest boundaries
    if levels:
        fine_boundaries = sorted({s for scene in levels[0] for s in scene})
        for lvl_idx in range(1, len(levels)):
            aligned: list[tuple[float, float]] = []
            for start, end in levels[lvl_idx]:
                snap_start = min(fine_boundaries, key=lambda b: abs(b - start))
                snap_end = min(fine_boundaries, key=lambda b: abs(b - end))
                if snap_start >= snap_end:
                    snap_end = end  # fallback to original if snapping collapses
                aligned.append((snap_start, snap_end))
            levels[lvl_idx] = aligned

    return {"levels": levels}


def _detect_scenes_histogram(
    frames: list[np.ndarray],
    timestamps: list[float],
    threshold: float,
    min_duration: float,
) -> list[tuple[float, float]]:
    """Histogram-based scene detection (no ML model needed)."""
    import cv2

    histograms = []
    for frame in frames:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        histograms.append(hist.flatten())

    distances = []
    for i in range(1, len(histograms)):
        d = cv2.compareHist(
            histograms[i - 1].reshape(-1, 1).astype(np.float32),
            histograms[i].reshape(-1, 1).astype(np.float32),
            cv2.HISTCMP_CHISQR,
        )
        distances.append(d)

    if not distances:
        return [(timestamps[0], timestamps[-1])]

    dist_arr = np.array(distances)
    cut_threshold = np.mean(dist_arr) + threshold * np.std(dist_arr)

    scenes: list[tuple[float, float]] = []
    scene_start = timestamps[0]
    for i, d in enumerate(distances):
        if d > cut_threshold:
            scenes.append((scene_start, timestamps[i]))
            scene_start = timestamps[i + 1]
    scenes.append((scene_start, timestamps[-1]))

    if min_duration > 0:
        filtered = [(s, e) for s, e in scenes if (e - s) >= min_duration]
        if not filtered:
            filtered = [(timestamps[0], timestamps[-1])]
        scenes = filtered

    return scenes
