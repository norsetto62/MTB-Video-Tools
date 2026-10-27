"""Group-aware training and validation orchestration for LightGBM.

Input data is prepared per video by autocut.dataset.lightgbm. This module
combines those per-video datasets and performs validation only across whole
groups, preventing overlapping windows from the same group leaking between
training and validation.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
from pathlib import Path
from typing import Mapping, Sequence, Any

import numpy as np

from autocut.data_models import FeatureWindows
from autocut.dataset.windows import LIGHTGBM_TABULAR_FEATURE_NAMES
from autocut.ml.model import LightGBMConfig, LightGBMRanker
from autocut.ml.training import train_lightgbm


@dataclass(frozen=True)
class LightGBMVideoData:
    """Prepared tabular data for one video, with labels aligned to its rows."""

    windows: FeatureWindows
    targets: np.ndarray

    def __post_init__(self) -> None:
        X = np.asarray(self.windows.X)
        y = np.asarray(self.targets)
        if X.ndim != 2 or X.shape[1] != len(LIGHTGBM_TABULAR_FEATURE_NAMES):
            raise ValueError(
                f"windows.X must have shape (N, {len(LIGHTGBM_TABULAR_FEATURE_NAMES)})."
            )
        if y.ndim != 1 or len(y) != len(X):
            raise ValueError("targets must be a 1D vector aligned with windows.X.")
        if not np.isfinite(X).all():
            raise ValueError("windows.X must contain only finite values.")
        if not np.isfinite(y).all() or not np.isin(y, [0, 1, 2, 3]).all():
            raise ValueError("targets must contain only finite MTB classes 0, 1, 2, or 3.")


@dataclass(frozen=True)
class LightGBMGroupFoldResult:
    """Training outcome and held-out metrics for one group fold."""

    held_out_group: str
    train_video_ids: tuple[str, ...]
    validation_video_ids: tuple[str, ...]
    summary: dict[str, Any]


def _validate_inputs(
    video_data: Mapping[str, LightGBMVideoData],
    group_by_video: Mapping[str, str],
) -> tuple[str, ...]:
    if not video_data:
        raise ValueError("video_data must contain at least one video.")
    missing = sorted(set(video_data) - set(group_by_video))
    if missing:
        raise ValueError(f"Missing group IDs for video(s): {missing}.")
    extras = sorted(set(group_by_video) - set(video_data))
    if extras:
        raise ValueError(f"group_by_video contains unknown video ID(s): {extras}.")
    for video_id, item in video_data.items():
        if not isinstance(video_id, str) or not video_id.strip():
            raise ValueError("Video IDs must be non-empty strings.")
        if not isinstance(item, LightGBMVideoData):
            raise TypeError(f"video_data[{video_id!r}] must be LightGBMVideoData.")
        group_id = group_by_video[video_id]
        if not isinstance(group_id, str) or not group_id.strip():
            raise ValueError(f"Video {video_id!r} has an empty group ID.")
    group_values = list(group_by_video.values())
    if any(not isinstance(group_id, str) or not group_id.strip() for group_id in group_values):
        raise ValueError("Group IDs must be non-empty strings.")
    groups = tuple(sorted(set(group_values)))
    if len(groups) < 2:
        raise ValueError("At least two distinct groups are required for grouped validation.")
    return groups


def _combine(
    video_data: Mapping[str, LightGBMVideoData],
    video_ids: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    feature_parts: list[np.ndarray] = []
    target_parts: list[np.ndarray] = []
    for video_id in video_ids:
        item = video_data[video_id]
        if len(item.targets) == 0:
            continue
        feature_parts.append(np.asarray(item.windows.X, dtype=np.float32))
        target_parts.append(np.asarray(item.targets, dtype=np.int32))
    if not feature_parts:
        return (
            np.empty((0, len(LIGHTGBM_TABULAR_FEATURE_NAMES)), dtype=np.float32),
            np.empty((0,), dtype=np.int32),
        )
    return np.concatenate(feature_parts, axis=0), np.concatenate(target_parts, axis=0)


def _combine_with_group_sizes(
    video_data: Mapping[str, LightGBMVideoData],
    video_ids: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Combine videos in order and preserve each video as a ranking query."""
    X, y = _combine(video_data, video_ids)
    group_sizes = [
        int(len(video_data[video_id].targets))
        for video_id in video_ids
        if len(video_data[video_id].targets) > 0
    ]
    return X, y, group_sizes


def _group_offsets(group_sizes: Sequence[int]):
    offset = 0
    for size in group_sizes:
        yield offset, size
        offset += size


def _ranking_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    group_sizes: Sequence[int],
    *,
    cutoffs: Sequence[int],
) -> dict[str, Any]:
    """Calculate mean per-video NDCG, the ranking metric used for training."""
    def dcg(relevance: np.ndarray, k: int) -> float:
        rel = relevance[:k].astype(np.float64)
        if not len(rel):
            return 0.0
        gains = np.exp2(rel) - 1.0
        discounts = np.log2(np.arange(2, len(rel) + 2, dtype=np.float64))
        return float(np.sum(gains / discounts))

    results: dict[str, Any] = {"validation_rows": int(len(y_true)), "queries": int(len(group_sizes))}
    offset = 0
    by_cutoff: dict[str, float | None] = {}
    for cutoff in cutoffs:
        query_values: list[float] = []
        offset = 0
        for size in group_sizes:
            labels = y_true[offset:offset + size]
            pred = scores[offset:offset + size]
            offset += size
            if size < 1 or len(labels) != size:
                continue
            order = np.argsort(-pred, kind="stable")
            ideal = np.sort(labels)[::-1]
            ideal_dcg = dcg(ideal, cutoff)
            if ideal_dcg > 0:
                query_values.append(dcg(labels[order], cutoff) / ideal_dcg)
        by_cutoff[str(int(cutoff))] = (
            float(np.mean(query_values)) if query_values else None
        )
    results["ndcg_at"] = by_cutoff
    return results


def train_lightgbm_group_fold(
    video_data: Mapping[str, LightGBMVideoData],
    group_by_video: Mapping[str, str],
    *,
    held_out_group: str,
    config: LightGBMConfig | None = None,
    checkpoint_path: str | Path | None = None,
    data_config: dict[str, Any] | None = None,
) -> tuple[LightGBMRanker, LightGBMGroupFoldResult]:
    """Train one fold with every video in held_out_group reserved for validation.

    No window-level random split is performed. By default, training must
    contain examples of all four MTB classes; this prevents a fold from
    silently producing a nominal four-class model that never saw a class.
    """
    groups = _validate_inputs(video_data, group_by_video)
    if held_out_group not in groups:
        raise ValueError(f"Unknown held-out group {held_out_group!r}; available groups: {groups}.")
    train_ids = tuple(sorted(video_id for video_id in video_data if group_by_video[video_id] != held_out_group))
    val_ids = tuple(sorted(video_id for video_id in video_data if group_by_video[video_id] == held_out_group))
    train_X, train_y, train_group_sizes = _combine_with_group_sizes(video_data, train_ids)
    val_X, val_y, val_group_sizes = _combine_with_group_sizes(video_data, val_ids)
    if len(train_y) == 0 or len(val_y) == 0:
        raise ValueError("Both training and held-out group must contain labeled windows.")
    if not any(
        len(np.unique(train_y[offset:offset + size])) > 1
        for offset, size in _group_offsets(train_group_sizes)
    ):
        raise ValueError(
            f"Training fold excluding group {held_out_group!r} has no video with multiple relevance grades."
        )
    metadata = {
        "validation_strategy": "leave_one_group_out",
        "held_out_group": held_out_group,
        "train_video_ids": list(train_ids),
        "validation_video_ids": list(val_ids),
        "feature_names": list(LIGHTGBM_TABULAR_FEATURE_NAMES),
    }
    model, summary = train_lightgbm(
        train_X,
        train_y,
        train_group_sizes=train_group_sizes,
        val_X=val_X,
        val_y=val_y,
        val_group_sizes=val_group_sizes,
        config=config,
        checkpoint_path=checkpoint_path,
        data_config={
            **(data_config or {}),
            "feature_names": list(LIGHTGBM_TABULAR_FEATURE_NAMES),
        },
        metadata=metadata,
    )
    ranking_scores = model.predict(val_X)
    summary["validation_metrics"] = _ranking_metrics(
        val_y, ranking_scores, val_group_sizes, cutoffs=config.ndcg_eval_at if config else (10,)
    )
    summary["held_out_group"] = held_out_group
    summary["train_video_ids"] = list(train_ids)
    summary["validation_video_ids"] = list(val_ids)
    result = LightGBMGroupFoldResult(
        held_out_group=held_out_group,
        train_video_ids=train_ids,
        validation_video_ids=val_ids,
        summary=summary,
    )
    return model, result


def train_lightgbm_lovo(
    video_data: Mapping[str, LightGBMVideoData],
    group_by_video: Mapping[str, str],
    *,
    config: LightGBMConfig | None = None,
    checkpoint_dir: str | Path | None = None,
    data_config: dict[str, Any] | None = None,
) -> tuple[LightGBMGroupFoldResult, ...]:
    """Run deterministic leave-one-group-out validation across all groups."""
    groups = _validate_inputs(video_data, group_by_video)
    results: list[LightGBMGroupFoldResult] = []
    for group_id in groups:
        path = None
        if checkpoint_dir is not None:
            safe_group = re.sub(r"[^A-Za-z0-9_.-]+", "_", group_id).strip("._") or "group"
            path = Path(checkpoint_dir) / f"lightgbm_lovo_{safe_group}.json"
        _, result = train_lightgbm_group_fold(
            video_data,
            group_by_video,
            held_out_group=group_id,
            config=config,
            checkpoint_path=path,
            data_config=data_config,
        )
        results.append(result)
    return tuple(results)
