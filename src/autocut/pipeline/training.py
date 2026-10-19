"""Dataset orchestration for the configured fixed-group training split.

This module joins the existing manifest, annotation, motion-feature, window,
target, scaler, trainer, and checkpoint APIs. The low-level trainer remains
agnostic about how a fold is constructed.
"""
from __future__ import annotations

from dataclasses import asdict
import json
import logging
import re
import time
from pathlib import Path
from typing import Any

import numpy as np

from ..annotations import load_training_annotations
from ..config_compat import TrainingRunConfig
from ..dataset.manifest import DatasetRecord, load_manifest
from ..dataset.sequence import build_feature_sequence
from ..dataset.targets import build_targets, validate_targets
from ..dataset.windows import build_windows
from ..ml.training import train_single_fold

logger = logging.getLogger(__name__)


def _experiment_slug(name: str) -> str:
    """Convert a user-supplied experiment name into a safe filename stem."""
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", name.strip()).strip(".-_")
    if not slug:
        raise ValueError("experiment.name must contain at least one filename-safe character.")
    return slug[:100]


def _load_record_windows(
    record: DatasetRecord,
    settings: TrainingRunConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract annotated training windows for one manifest record."""
    record_started = time.perf_counter()
    logger.info("Record %s: loading annotations from %s", record.id, record.annotation_path)
    video_path, annotations = load_training_annotations(record.annotation_path)
    if video_path.resolve() != record.video_path.resolve():
        raise ValueError(
            f"Manifest entry {record.id!r} points to {record.video_path}, but "
            f"its annotation file declares {video_path}."
        )
    if not annotations:
        raise ValueError(f"Manifest entry {record.id!r} has no training annotations.")

    # The training annotation format covers the complete video timeline. A
    # clip from 0 to the last annotation end keeps timestamps aligned with the
    # absolute annotation times and lets VideoReader enforce the clip bounds.
    from ..video.probe import get_video_info
    logger.info("Record %s: probing video metadata", record.id)
    info = get_video_info(record.video_path)
    clip_end = annotations[-1].end
    duration_tolerance = 0.05

    if clip_end > info.duration + duration_tolerance:
        raise ValueError(
            f"Annotation end ({clip_end:.3f}s) exceeds video duration "
            f"({info.duration:.3f}s) for manifest entry {record.id!r}."
        )
    
    from ..data_models import Clip
    clip = Clip(video_name=record.id, start=0.0, end=min(clip_end, info.duration))
    logger.info(
        "Record %s: extracting motion features (%.1fs clip at %.2f fps; cache %s)",
        record.id, clip.duration, settings.sample_fps,
        "enabled" if settings.cache_dir is not None else "disabled",
    )
    sequence = build_feature_sequence(
        record.video_path,
        clip,
        sample_fps=settings.sample_fps,
        processing_width=settings.processing_width,
        cache_dir=settings.cache_dir,
    )
    logger.info("Record %s: extracted %d feature rows; building windows", record.id, len(sequence.features))
    windows = build_windows(
        sequence,
        window=settings.common.window_duration,
        stride=settings.common.window_stride,
    )
    X, y = build_targets(windows, annotations)
    # build_targets drops uncovered windows, so the original window timestamp
    # array no longer aligns row-for-row with X. The validator checks only
    # timestamp count here; the target builder already selects those rows.
    errors = validate_targets(X, np.zeros((len(y), 2), dtype=np.float64), y)
    if errors:
        raise ValueError(
            f"Invalid training targets for manifest entry {record.id!r}: "
            + "; ".join(errors)
        )
    if len(X) == 0:
        raise ValueError(
            f"No feature windows overlap annotations for manifest entry {record.id!r}."
        )
    logger.info(
        "Record %s: ready (%d labelled windows) in %.1fs",
        record.id, len(X), time.perf_counter() - record_started,
    )
    return X, y


def _combine_records(
    records: tuple[DatasetRecord, ...],
    settings: TrainingRunConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Build and concatenate all labeled windows for selected records."""
    features: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for index, record in enumerate(records, start=1):
        logger.info(
            "Preparing %s data: video %d/%d (%.0f%%): %s (group %s)",
            "split", index, len(records), 100.0 * index / len(records),
            record.id, record.group_id,
        )
        X, y = _load_record_windows(record, settings)
        features.append(X)
        targets.append(y)
    if not features:
        raise ValueError("The selected group split contains no manifest records.")
    return np.concatenate(features, axis=0), np.concatenate(targets, axis=0)


def run_training(
    manifest_path: str | Path,
    settings: TrainingRunConfig,
) -> dict[str, Any]:
    """Train from the configured disjoint group split and persist artifacts."""
    records = load_manifest(manifest_path, verify_files=True)
    train_groups = set(settings.train_groups)
    validation_groups = set(settings.validation_groups)
    if train_groups & validation_groups:
        raise ValueError("Training and validation groups must be disjoint.")
    known_groups = {record.group_id for record in records}
    unknown = (train_groups | validation_groups) - known_groups
    if unknown:
        raise ValueError(
            "Configured group ID(s) not found in the manifest: "
            + ", ".join(sorted(unknown))
        )
    train_records = tuple(record for record in records if record.group_id in train_groups)
    val_records = tuple(record for record in records if record.group_id in validation_groups)
    if not train_records:
        raise ValueError("No manifest records match data.train_groups.")
    if not val_records:
        raise ValueError("No manifest records match data.validation_groups.")

    logger.info(
        "Training experiment %r with %d training videos and %d validation videos.",
        settings.experiment_name, len(train_records), len(val_records),
    )
    logger.info("Preparing training split: 0/%d videos", len(train_records))
    train_X, train_y = _combine_records(train_records, settings)
    logger.info("Training data ready: %d labelled windows from %d videos", len(train_X), len(train_records))
    logger.info("Preparing validation split: 0/%d videos", len(val_records))
    val_X, val_y = _combine_records(val_records, settings)
    logger.info("Validation data ready: %d labelled windows from %d videos", len(val_X), len(val_records))

    slug = _experiment_slug(settings.experiment_name)
    checkpoint_path = settings.checkpoint_dir / f"{slug}_best.pt"
    results_path = settings.results_dir / f"{slug}_train.json"
    data_config = {
        "flow_fps": settings.sample_fps,
        "processing_width": settings.processing_width,
        "window": settings.common.window_duration,
        "stride": settings.common.window_stride,
    }
    _, _, summary = train_single_fold(
        train_X,
        train_y,
        val_X,
        val_y,
        settings.training,
        checkpoint_path=checkpoint_path,
        data_config=data_config,
    )
    result: dict[str, Any] = {
        "experiment_name": settings.experiment_name,
        "manifest": str(Path(manifest_path).expanduser().resolve()),
        "train_groups": sorted(train_groups),
        "validation_groups": sorted(validation_groups),
        "train_records": [record.id for record in train_records],
        "validation_records": [record.id for record in val_records],
        "train_windows": int(len(train_X)),
        "validation_windows": int(len(val_X)),
        "training_config": asdict(settings.training),
        "data_config": data_config,
        "checkpoint": str(checkpoint_path),
        "summary": summary,
    }
    results_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = results_path.with_name(f".{results_path.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(result, stream, indent=2, sort_keys=True, default=str, allow_nan=False)
            stream.write("\n")
        temporary.replace(results_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    result["results_path"] = str(results_path)
    logger.info("Saved model checkpoint: %s", checkpoint_path)
    logger.info("Saved training summary: %s", results_path)
    return result
