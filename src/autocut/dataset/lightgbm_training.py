"""Group-safe assembly and training for four-class LightGBM datasets."""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ..data_models import FeatureWindows
from .windows import LIGHTGBM_TABULAR_FEATURE_NAMES


def _validate_video_dataset(
    video_id: str,
    dataset: tuple[FeatureWindows, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not isinstance(video_id, str) or not video_id.strip():
        raise ValueError("Video dataset IDs must be non-empty strings.")
    if not isinstance(dataset, tuple) or len(dataset) != 2:
        raise ValueError(
            f"Dataset for video {video_id!r} must be a (FeatureWindows, labels) tuple."
        )
    windows, targets = dataset
    if not isinstance(windows, FeatureWindows):
        raise TypeError(f"Dataset for video {video_id!r} must contain FeatureWindows.")
    X = np.asarray(windows.X, dtype=np.float32)
    y_raw = np.asarray(targets)
    timestamps = np.asarray(windows.timestamps, dtype=np.float64)
    expected_features = len(LIGHTGBM_TABULAR_FEATURE_NAMES)
    if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] != expected_features:
        raise ValueError(
            f"Video {video_id!r} features must have shape (N, {expected_features}) "
            "with N > 0."
        )
    if y_raw.ndim != 1 or len(y_raw) != len(X):
        raise ValueError(f"Video {video_id!r} labels must be a 1D vector aligned with X.")
    if timestamps.shape != (len(X), 2):
        raise ValueError(f"Video {video_id!r} timestamps must have shape (N, 2).")
    if not np.isfinite(X).all() or not np.isfinite(timestamps).all():
        raise ValueError(f"Video {video_id!r} features and timestamps must be finite.")
    try:
        numeric_y = np.asarray(y_raw, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"Video {video_id!r} labels must be integer classes 0..3.") from exc
    if (
        not np.isfinite(numeric_y).all()
        or not np.isin(numeric_y, [0.0, 1.0, 2.0, 3.0]).all()
    ):
        raise ValueError(f"Video {video_id!r} labels must contain only classes 0, 1, 2, or 3.")
    return X, numeric_y.astype(np.int32), timestamps


def prepare_lightgbm_group_split(
    video_datasets: Mapping[str, tuple[FeatureWindows, np.ndarray]],
    video_groups: Mapping[str, str],
    *,
    validation_groups: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Combine per-video datasets without allowing a group into both splits.

    Each video dataset must already be prepared by
    `build_lightgbm_training_data`. The supplied validation group IDs are held
    out as a unit; all remaining groups form the training set. The returned
    metadata records exact video/group membership and per-class counts.
    """
    if not isinstance(video_datasets, Mapping) or not video_datasets:
        raise ValueError("video_datasets must be a non-empty mapping.")
    if not isinstance(video_groups, Mapping):
        raise ValueError("video_groups must be a mapping of video ID to group ID.")
    if set(video_datasets) != set(video_groups):
        missing = sorted(set(video_datasets) - set(video_groups))
        extra = sorted(set(video_groups) - set(video_datasets))
        raise ValueError(
            f"video_datasets and video_groups must have identical IDs; "
            f"missing groups={missing}, unknown groups={extra}."
        )
    if isinstance(validation_groups, (str, bytes)) or not isinstance(validation_groups, Sequence):
        raise ValueError("validation_groups must be a non-empty sequence of group IDs.")
    val_groups = tuple(dict.fromkeys(validation_groups))
    if not val_groups or any(not isinstance(group, str) or not group.strip() for group in val_groups):
        raise ValueError("validation_groups must contain non-empty group IDs.")
    all_groups = {group for group in video_groups.values() if isinstance(group, str) and group.strip()}
    if len(all_groups) == 0 or any(group not in all_groups for group in val_groups):
        raise ValueError("Every validation group must exist in video_groups.")
    if any(not isinstance(group, str) or not group.strip() for group in video_groups.values()):
        raise ValueError("Every video group ID must be a non-empty string.")
    val_set = set(val_groups)
    train_groups = tuple(sorted(all_groups - val_set))
    if not train_groups:
        raise ValueError("At least one group must remain for training.")
    train_ids = tuple(video_id for video_id in video_datasets if video_groups[video_id] in train_groups)
    val_ids = tuple(video_id for video_id in video_datasets if video_groups[video_id] in val_set)
    if not train_ids:
        raise ValueError("No video datasets belong to the training groups.")
    if not val_ids:
        raise ValueError("No video datasets belong to the validation groups.")

    validated = {
        video_id: _validate_video_dataset(video_id, video_datasets[video_id])
        for video_id in video_datasets
    }

    def combine(ids: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.concatenate([validated[video_id][0] for video_id in ids], axis=0),
            np.concatenate([validated[video_id][1] for video_id in ids], axis=0),
        )

    train_X, train_y = combine(train_ids)
    val_X, val_y = combine(val_ids)
    train_classes = set(np.unique(train_y).tolist())
    val_classes = set(np.unique(val_y).tolist())
    if len(train_classes) < 2:
        raise ValueError("Training split must contain at least two distinct MTB classes.")
    if len(val_classes) < 2:
        raise ValueError("Validation split must contain at least two distinct MTB classes.")
    metadata: dict[str, Any] = {
        "split_strategy": "held_out_groups",
        "training_groups": list(train_groups),
        "validation_groups": list(val_groups),
        "training_video_ids": list(train_ids),
        "validation_video_ids": list(val_ids),
        "training_rows_by_class": {
            str(label): int(count) for label, count in sorted(Counter(train_y.tolist()).items())
        },
        "validation_rows_by_class": {
            str(label): int(count) for label, count in sorted(Counter(val_y.tolist()).items())
        },
    }
    return train_X, train_y, val_X, val_y, metadata


def train_lightgbm_grouped(
    video_datasets: Mapping[str, tuple[FeatureWindows, np.ndarray]],
    video_groups: Mapping[str, str],
    *,
    validation_groups: Sequence[str],
    config: Any = None,
    checkpoint_path: str | Path | None = None,
    data_config: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
):
    """Train LightGBM using explicitly held-out video groups for validation.

    Per-video rows must come from `build_lightgbm_training_data`. This function
    creates a group-disjoint split before fitting, uses validation groups for
    early stopping, and records split membership in the checkpoint metadata.
    """
    train_X, train_y, val_X, val_y, split_metadata = prepare_lightgbm_group_split(
        video_datasets, video_groups, validation_groups=validation_groups
    )
    training_classes = set(np.unique(train_y).tolist())
    if training_classes != {0, 1, 2, 3}:
        missing = sorted({0, 1, 2, 3} - training_classes)
        raise ValueError(
            "Four-class LightGBM training requires all MTB classes 0, 1, 2, and 3 "
            f"in the training groups; missing classes: {missing}."
        )
    from ..ml.training import train_lightgbm

    return train_lightgbm(
        train_X,
        train_y,
        val_X=val_X,
        val_y=val_y,
        config=config,
        checkpoint_path=checkpoint_path,
        data_config=data_config,
        metadata={**(metadata or {}), **split_metadata},
    )
