"""Dataset-to-checkpoint orchestration for the fixed-split training command."""

from __future__ import annotations

import json
import logging
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Sequence

import numpy as np

from ..annotations import load_training_annotations
from ..config_loader import TrainingRunConfig
from ..data_models import Clip
from ..dataset.manifest import DatasetRecord, get_unique_groups, load_manifest
from ..dataset.sequence import build_feature_sequence
from ..dataset.targets import build_targets
from ..dataset.windows import build_windows
from ..ml.training import train_single_fold
from ..video.probe import get_video_info

logger = logging.getLogger(__name__)


def _validate_split(
    records: Sequence[DatasetRecord],
    *,
    train_groups: Sequence[str],
    validation_groups: Sequence[str],
) -> None:
    """Require an explicit, complete, non-overlapping group split."""
    actual_groups = set(get_unique_groups(records))
    train = set(train_groups)
    validation = set(validation_groups)

    overlap = train & validation
    if overlap:
        raise ValueError(
            "Training and validation groups must be disjoint; overlap: "
            + ", ".join(sorted(overlap))
        )

    unknown = (train | validation) - actual_groups
    if unknown:
        raise ValueError(
            "Configured training/validation group(s) are absent from the manifest: "
            + ", ".join(sorted(unknown))
        )

    unassigned = actual_groups - (train | validation)
    if unassigned:
        raise ValueError(
            "Every manifest group must be assigned to training or validation; "
            "unassigned group(s): " + ", ".join(sorted(unassigned))
        )

    if not train or not validation:
        raise ValueError("Training and validation splits must each contain at least one group.")


def _record_windows(
    record: DatasetRecord,
    settings: TrainingRunConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract annotated windows for one manifest record."""
    embedded_video, annotations = load_training_annotations(record.annotation_path)
    if embedded_video.resolve() != record.video_path.resolve():
        raise ValueError(
            f"Manifest video and annotation source disagree for {record.id!r}: "
            f"manifest points to {record.video_path}, annotation points to {embedded_video}."
        )
    if not annotations:
        raise ValueError(f"No training annotations found for manifest entry {record.id!r}.")

    video_info = get_video_info(record.video_path)
    clip = Clip(video_name=record.id, start=0.0, end=video_info.duration)
    sequence = build_feature_sequence(
        record.video_path,
        clip,
        sample_fps=settings.sample_fps,
        processing_width=settings.processing_width,
        cache_dir=settings.cache_dir,
    )
    windows = build_windows(
        sequence,
        window=settings.common.window_duration,
        stride=settings.common.window_stride,
    )
    X, y = build_targets(windows, annotations)

    if len(X) == 0:
        raise ValueError(
            f"No feature windows overlap training annotations for {record.id!r}. "
            "Check annotation timestamps and configured window duration."
        )
    if X.ndim != 3 or X.shape[1:] != (8, 54):
        raise ValueError(
            f"Training windows for {record.id!r} have shape {X.shape}; "
            "the current classifier requires (N, 8, 54). Check sampling_fps "
            "and window_duration (normally 2 FPS and 4 seconds)."
        )
    if len(X) != len(y):
        raise ValueError(f"Feature/target count mismatch for {record.id!r}.")
    return X, y


def _collect_split(
    records: Sequence[DatasetRecord],
    selected_groups: Sequence[str],
    settings: TrainingRunConfig,
    *,
    split_name: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    selected = set(selected_groups)
    feature_parts: list[np.ndarray] = []
    target_parts: list[np.ndarray] = []
    counts: dict[str, int] = {}

    for record in records:
        if record.group_id not in selected:
            continue
        X, y = _record_windows(record, settings)
        feature_parts.append(X)
        target_parts.append(y)
        counts[record.id] = int(len(y))
        logger.info(
            "%s: %s (group=%s): %d annotated windows",
            split_name,
            record.id,
            record.group_id,
            len(y),
        )

    if not feature_parts:
        raise ValueError(f"The {split_name} split produced no annotated feature windows.")

    return (
        np.concatenate(feature_parts, axis=0).astype(np.float32, copy=False),
        np.concatenate(target_parts, axis=0).astype(np.float32, copy=False),
        counts,
    )


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Write a JSON artifact atomically so interrupted writes don't corrupt it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as stream:
            temporary_path = stream.name
            json.dump(payload, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            try:
                os.unlink(temporary_path)
            except FileNotFoundError:
                pass


def run_training(
    settings: TrainingRunConfig,
    *,
    manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run the configured group-held-out training workflow.

    Motion features are loaded from the versioned cache when compatible;
    annotated windows are assembled by group; the scaler is fitted only on
    training data inside `train_single_fold`; and checkpoint/metrics artifacts
    are written to the configured experiment directories.
    """
    if not isinstance(settings, TrainingRunConfig):
        raise TypeError("settings must be a TrainingRunConfig.")
    manifest = Path(manifest_path or settings.manifest).expanduser()
    records = load_manifest(manifest, verify_files=True)
    _validate_split(
        records,
        train_groups=settings.train_groups,
        validation_groups=settings.validation_groups,
    )

    # Experiment names become filenames; reject path components rather than
    # allowing a YAML value to write outside the configured artifact folders.
    name = settings.experiment_name
    if name in {"", ".", ".."} or Path(name).name != name or "/" in name or "\\" in name:
        raise ValueError("experiment.name must be a single filename-safe path component.")

    train_X, train_y, train_counts = _collect_split(
        records, settings.train_groups, settings, split_name="train"
    )
    val_X, val_y, val_counts = _collect_split(
        records, settings.validation_groups, settings, split_name="validation"
    )

    checkpoint_path = settings.checkpoint_dir / f"{name}.pt"
    metrics_path = settings.results_dir / f"{name}.json"
    data_config = {
        "flow_fps": settings.sample_fps,
        "window": settings.common.window_duration,
        "stride": settings.common.window_stride,
    }

    _, _, training_summary = train_single_fold(
        train_X,
        train_y,
        val_X,
        val_y,
        settings.training,
        checkpoint_path=checkpoint_path,
        data_config=data_config,
    )

    result: dict[str, Any] = {
        "schema_version": 1,
        "experiment_name": name,
        "manifest": str(manifest.resolve()),
        "train_groups": list(settings.train_groups),
        "validation_groups": list(settings.validation_groups),
        "train_windows": int(len(train_y)),
        "validation_windows": int(len(val_y)),
        "train_windows_by_video": train_counts,
        "validation_windows_by_video": val_counts,
        "checkpoint_path": str(checkpoint_path),
        "metrics_path": str(metrics_path),
        "training": training_summary,
    }
    _write_json_atomic(metrics_path, result)
    logger.info("Saved checkpoint: %s", checkpoint_path)
    logger.info("Saved training metrics: %s", metrics_path)
    return result
