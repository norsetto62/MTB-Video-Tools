"""Candidate-level event coverage metrics for temporal ranking."""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from autocut.selection.candidates import build_candidate_clips


METRIC_ID = "candidate_event_recall_at_10_v2"
TOP_K = 10
MIN_EVENT_OVERLAP_SECONDS = 1.0
INTERESTING_LABEL_FRACTION = 0.5


def candidate_event_metrics(
    scores: np.ndarray,
    group_sizes: Sequence[int],
    timestamps: np.ndarray,
    video_durations: Sequence[float],
    annotations_by_video: Sequence[Sequence[tuple[float, float, int]]],
    *,
    max_gap_seconds: float = 2.0,
    min_clip_seconds: float = 3.0,
    padding_seconds: float = 2.0,
) -> dict[str, Any]:
    """Evaluate production-equivalent selected clips against annotation events.

    Class-3 annotations are merged into events when contiguous/overlapping.
    An event is recalled when a padded, boundary-clipped selected candidate
    overlaps it by at least one second. Candidate usefulness and class-2/3
    duration coverage are reported separately from the primary event recall.
    """
    ranking = np.asarray(scores, dtype=np.float64)
    spans = np.asarray(timestamps, dtype=np.float64)
    sizes = [int(size) for size in group_sizes]
    if ranking.ndim != 1 or spans.shape != (len(ranking), 2):
        raise ValueError("scores and timestamps must align as (N,) and (N, 2).")
    if not np.isfinite(ranking).all() or not np.isfinite(spans).all():
        raise ValueError("scores and timestamps must be finite.")
    if not sizes or any(size <= 0 for size in sizes) or sum(sizes) != len(ranking):
        raise ValueError("group_sizes must be positive and sum to the number of windows.")
    if len(video_durations) != len(sizes) or len(annotations_by_video) != len(sizes):
        raise ValueError("Video durations and annotations must align with group_sizes.")

    total_events = hit_events = total_candidates = useful_candidates = 0
    total_duration = 0.0
    interesting_candidates = 0
    per_video: dict[str, dict[str, Any]] = {}
    offset = 0
    for video_index, size in enumerate(sizes):
        video_spans = spans[offset:offset + size]
        video_scores = ranking[offset:offset + size]
        offset += size
        duration = float(video_durations[video_index])
        annotations = tuple(
            (float(start), float(end), int(label))
            for start, end, label in annotations_by_video[video_index]
        )
        if any(end <= start or start < 0 or end > duration + 1e-6 or label not in (0, 1, 2, 3)
               for start, end, label in annotations):
            raise ValueError(f"Invalid annotation interval for validation video {video_index}.")

        class3 = sorted(
            ((start, end) for start, end, label in annotations if label == 3),
            key=lambda item: (item[0], item[1]),
        )
        events: list[tuple[float, float]] = []
        for start, end in class3:
            if events and start <= events[-1][1] + 1e-9:
                events[-1] = (events[-1][0], max(events[-1][1], end))
            else:
                events.append((start, end))

        built = build_candidate_clips(
            video_spans, video_scores, video_duration=duration, top_k=TOP_K,
            max_gap_seconds=max_gap_seconds, min_clip_seconds=min_clip_seconds,
            padding_seconds=padding_seconds,
        )
        candidates = [candidate for _, candidate in built]
        event_covered = [False] * len(events)
        useful = 0
        interesting = 0
        for candidate in candidates:
            event_overlaps = [
                max(0.0, min(candidate.end, event_end) - max(candidate.start, event_start))
                for event_start, event_end in events
            ]
            useful_candidate = any(
                overlap + 1e-9 >= MIN_EVENT_OVERLAP_SECONDS
                for overlap in event_overlaps
            )
            if useful_candidate:
                useful += 1
            for event_index, overlap in enumerate(event_overlaps):
                if overlap + 1e-9 >= MIN_EVENT_OVERLAP_SECONDS:
                    event_covered[event_index] = True

            class23_overlap = 0.0
            for ann_start, ann_end, label in annotations:
                if label in (2, 3):
                    class23_overlap += max(
                        0.0, min(candidate.end, ann_end) - max(candidate.start, ann_start)
                    )
            class23_fraction = min(1.0, class23_overlap / candidate.duration)
            if class23_fraction + 1e-9 >= INTERESTING_LABEL_FRACTION:
                interesting += 1
        hits = sum(event_covered)

        event_count = len(events)
        recall = hits / event_count if event_count else None
        precision = useful / len(candidates) if candidates else None
        total_events += event_count
        hit_events += hits
        total_candidates += len(candidates)
        useful_candidates += useful
        interesting_candidates += interesting
        candidate_duration = sum(item.duration for item in candidates)
        total_duration += candidate_duration
        per_video[str(video_index)] = {
            "events": event_count,
            "hit_events": hits,
            "event_recall": recall,
            "candidate_count": len(candidates),
            "useful_candidate_count": useful,
            "candidate_precision": precision,
            "interesting_candidate_count": interesting,
            "interesting_candidate_rate": interesting / len(candidates) if candidates else None,
            "total_candidate_duration_seconds": candidate_duration,
            "candidates": [
                {"start": item.start, "end": item.end, "duration": item.duration}
                for item in candidates
            ],
        }

    return {
        "metric": METRIC_ID,
        "top_k": TOP_K,
        "minimum_event_overlap_seconds": MIN_EVENT_OVERLAP_SECONDS,
        "max_gap_seconds": float(max_gap_seconds),
        "min_clip_seconds": float(min_clip_seconds),
        "padding_seconds": float(padding_seconds),
        "events": total_events,
        "hit_events": hit_events,
        "pooled_event_recall": hit_events / total_events if total_events else None,
        "selected_segments": total_candidates,
        "useful_segments": useful_candidates,
        "segment_precision": useful_candidates / total_candidates if total_candidates else None,
        "average_segments_per_video": total_candidates / len(sizes),
        "total_selected_duration_seconds": total_duration,
        "interesting_segments": interesting_candidates,
        "interesting_segment_rate": interesting_candidates / total_candidates if total_candidates else None,
        "evaluated_videos": len(sizes),
        "per_video": per_video,
    }


def make_candidate_event_recall_evaluator(
    timestamps: np.ndarray,
    group_sizes: Sequence[int],
    video_durations: Sequence[float],
    annotations_by_video: Sequence[Sequence[tuple[float, float, int]]],
    *,
    max_gap_seconds: float,
    min_clip_seconds: float,
    padding_seconds: float,
):
    """Create LightGBM's evaluator for the production-equivalent candidate metric."""
    fixed_timestamps = np.asarray(timestamps, dtype=np.float64).copy()
    fixed_groups = tuple(int(size) for size in group_sizes)

    def evaluate(predictions: np.ndarray, dataset: object):
        result = candidate_event_metrics(
            np.asarray(predictions, dtype=np.float64), fixed_groups, fixed_timestamps,
            video_durations, annotations_by_video,
            max_gap_seconds=max_gap_seconds,
            min_clip_seconds=min_clip_seconds,
            padding_seconds=padding_seconds,
        )
        value = result["pooled_event_recall"]
        if value is None:
            raise ValueError("Validation data contains no class-3 annotation events.")
        return METRIC_ID, float(value), True

    return evaluate
