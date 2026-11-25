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
    suppression_iou_threshold: float = 0.3,
) -> dict[str, Any]:
    """Mean per-video recall of distinct class-3 events found in each video's Top K.

    Consecutive or overlapping class-3 intervals form one event. Greedy
    temporal NMS suppresses redundant overlapping predictions before Top K.
    The headline score is pooled event recall (total hits / total events), so
    videos contribute in proportion to their number of annotated events.
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
    if (
        isinstance(suppression_iou_threshold, bool)
        or not isinstance(suppression_iou_threshold, (int, float))
        or not np.isfinite(suppression_iou_threshold)
        or not 0.0 <= float(suppression_iou_threshold) <= 1.0
    ):
        raise ValueError("suppression_iou_threshold must be finite and in [0, 1].")
    suppression_iou_threshold = float(suppression_iou_threshold)
    if not sizes or any(isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n <= 0 for n in sizes):
        raise ValueError("group_sizes must contain positive integers.")
    if sum(int(n) for n in sizes) != len(labels):
        raise ValueError("group_sizes must sum to the number of windows.")

    recalls: list[float] = []
    total_events = hit_events = videos_with_events = 0
    total_suppressed = 0
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
            ranked = np.argsort(-video_scores, kind="stable")
            selected: list[int] = []
            selected_mask = np.zeros(size, dtype=bool)
            suppressed = np.zeros(size, dtype=bool)
            starts = video_intervals[:, 0]
            ends = video_intervals[:, 1]
            durations = ends - starts
            for candidate_raw in ranked:
                candidate = int(candidate_raw)
                if suppressed[candidate]:
                    continue
                selected.append(candidate)
                selected_mask[candidate] = True
                if len(selected) >= min(k, size):
                    break
                c_start, c_end = video_intervals[candidate]
                intersection = np.maximum(
                    0.0,
                    np.minimum(c_end, ends) - np.maximum(c_start, starts),
                )
                union = durations[candidate] + durations - intersection
                iou = np.divide(
                    intersection, union,
                    out=np.zeros_like(intersection), where=union > 0,
                )
                suppressed |= (
                    (iou > suppression_iou_threshold)
                    & ~selected_mask
                )
            top_set = set(selected)
            hits = sum(bool(top_set.intersection(event)) for event in events)
            suppressed_count = int(np.count_nonzero(suppressed))
            total_suppressed += suppressed_count
            recall = hits / len(events)
            recalls.append(float(recall))
            total_events += len(events)
            hit_events += hits
            per_video[str(video_index)] = {
                "events": len(events), "hits": hits, "recall": float(recall),
                "predictions_after_suppression": len(selected),
                "predictions_suppressed": suppressed_count,
            }
        offset = stop

    pooled_recall = float(hit_events / total_events) if total_events else None
    mean_recall = float(np.mean(recalls)) if recalls else None
    return {
        "metric": f"segment_recall_at_{k}",
        "k": k,
        "segment_recall_at_k": pooled_recall,
        "segment_recall_at_5": pooled_recall if k == 5 else None,
        "pooled_event_recall": pooled_recall,
        "mean_per_video_event_recall": mean_recall,
        "suppression_iou_threshold": suppression_iou_threshold,
        "predictions_suppressed": total_suppressed,
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
        value = result["segment_recall_at_5"]
        if value is None:
            raise ValueError(
                "Validation data contains no class-3 events; Segment Recall@5 "
                "and metric-based early stopping are undefined for this fold."
            )
        return "segment_recall_at_5", float(value), True

    return evaluate
