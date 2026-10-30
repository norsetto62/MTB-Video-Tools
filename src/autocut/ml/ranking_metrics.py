"""Video-level event coverage metrics for temporal ranking."""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def segment_recall_at_k(
    y_true: np.ndarray,
    scores: np.ndarray,
    group_sizes: Sequence[int],
    timestamps: np.ndarray,
    *,
    k: int = 5,
) -> dict[str, Any]:
    """Mean per-video recall of distinct class-3 events found in each video's Top K.

    Consecutive or overlapping class-3 intervals form one event. Events are
    separated by a positive temporal gap. Videos without class-3 events are
    excluded from the mean because their recall is undefined.
    """
    labels = np.asarray(y_true)
    ranking = np.asarray(scores, dtype=np.float64)
    intervals = np.asarray(timestamps, dtype=np.float64)
    sizes = list(group_sizes)
    if labels.ndim != 1 or ranking.ndim != 1 or len(labels) != len(ranking):
        raise ValueError("y_true and scores must be aligned one-dimensional arrays.")
    if intervals.shape != (len(labels), 2):
        raise ValueError("timestamps must have shape (N, 2) and align with y_true.")
    if not np.isfinite(ranking).all() or not np.isfinite(intervals).all():
        raise ValueError("scores and timestamps must contain only finite values.")
    if np.any(intervals[:, 1] <= intervals[:, 0]):
        raise ValueError("Every window timestamp must have end > start.")
    if not np.isin(labels, [0, 1, 2, 3]).all():
        raise ValueError("y_true must contain relevance grades 0, 1, 2, or 3.")
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer.")
    if not sizes or any(isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n <= 0 for n in sizes):
        raise ValueError("group_sizes must contain positive integers.")
    if sum(int(n) for n in sizes) != len(labels):
        raise ValueError("group_sizes must sum to the number of windows.")

    recalls: list[float] = []
    total_events = hit_events = videos_with_events = 0
    offset = 0
    per_video: dict[str, dict[str, int | float]] = {}
    for video_index, size_raw in enumerate(sizes):
        size = int(size_raw)
        stop = offset + size
        video_labels = labels[offset:stop]
        video_scores = ranking[offset:stop]
        video_intervals = intervals[offset:stop]
        positive = np.flatnonzero(video_labels == 3)
        events: list[list[int]] = []
        if len(positive):
            ordered = sorted(positive.tolist(), key=lambda i: (video_intervals[i, 0], video_intervals[i, 1], i))
            current = [ordered[0]]
            current_end = float(video_intervals[ordered[0], 1])
            for index in ordered[1:]:
                start, end = map(float, video_intervals[index])
                if start <= current_end + 1e-9:
                    current.append(index)
                    current_end = max(current_end, end)
                else:
                    events.append(current)
                    current = [index]
                    current_end = end
            events.append(current)
        if events:
            videos_with_events += 1
            top = np.argsort(-video_scores, kind="stable")[:min(k, size)]
            top_set = set(int(i) for i in top)
            hits = sum(bool(top_set.intersection(event)) for event in events)
            recall = hits / len(events)
            recalls.append(float(recall))
            total_events += len(events)
            hit_events += hits
            per_video[str(video_index)] = {"events": len(events), "hits": hits, "recall": float(recall)}
        offset = stop

    mean_recall = float(np.mean(recalls)) if recalls else 0.0
    return {
        "metric": f"segment_recall_at_{k}",
        "k": k,
        "segment_recall_at_k": mean_recall,
        "segment_recall_at_5": mean_recall if k == 5 else None,
        "videos_with_events": videos_with_events,
        "events": total_events,
        "hit_events": hit_events,
        "evaluated_videos": len(recalls),
        "per_video": per_video,
    }


def make_segment_recall_at_5(timestamps: np.ndarray, group_sizes: Sequence[int]):
    """Build LightGBM feval callback bound to validation video intervals."""
    fixed_timestamps = np.asarray(timestamps, dtype=np.float64).copy()
    fixed_groups = tuple(int(n) for n in group_sizes)

    def evaluate(predictions: np.ndarray, dataset: object):
        result = segment_recall_at_k(
            np.asarray(dataset.get_label(), dtype=np.int32),
            predictions,
            fixed_groups,
            fixed_timestamps,
            k=5,
        )
        return "segment_recall_at_5", float(result["segment_recall_at_5"]), True

    return evaluate
