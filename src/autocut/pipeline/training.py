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
from collections.abc import Mapping
from typing import Any

import numpy as np

from ..annotations import load_training_annotations
from ..config_loader import Config
from ..dataset.manifest import DatasetRecord, load_manifest
from ..dataset.sequence import build_feature_sequence
from ..dataset.targets import build_targets, validate_targets
from ..dataset.windows import build_windows
from ..ml.training import TrainingConfig, train_single_fold

logger = logging.getLogger(__name__)


def _section(settings: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = settings.get(name, {})
    if not isinstance(value, Mapping):
        raise ValueError(f"Configuration section {name!r} must be a mapping.")
    return value


def _positive_float(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite positive number.")
    result = float(value)
    if not np.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be a finite positive number.")
    return result


def _positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def _path(value: Any, *, name: str) -> Path:
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ValueError(f"{name} must be a non-empty path string.")
    return Path(value).expanduser()


def _groups(value: Any, *, name: str) -> tuple[str, ...]:
    if not isinstance(value, (tuple, list)) or not value:
        raise ValueError(f"{name} must be a non-empty list of group IDs.")
    groups: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"{name} must contain non-empty string group IDs.")
        group = item.strip()
        if group in groups:
            raise ValueError(f"{name} contains duplicate group ID {group!r}.")
        groups.append(group)
    return tuple(groups)


def _training_config(settings: Mapping[str, Any]) -> TrainingConfig:
    """Build the domain-specific trainer config from generic immutable values."""
    training = _section(settings, "training")
    try:
        values = training.to_dict() if isinstance(training, Config) else dict(training)
        return TrainingConfig(**values)
    except TypeError as exc:
        raise ValueError(f"Invalid training configuration: {exc}") from exc


(settings: Mapping[str, Any]) -> tuple[Path, float, int, float, float]:
    data = _section(settings, "data")
    manifest = _path(
        data.get("manifest", "data/manifest/annotations_manifest.yaml"),
        name="data.manifest",
    )
    sample_fps = _positive_float(data.get("sampling_fps", 2.0), name="data.sampling_fps")
    processing_width = _positive_int(
        data.get("processing_width", 480), name="data.processing_width"
    )
    window_duration = _positive_float(
        data.get("window_duration", 4.0), name="data.window_duration"
    )
    window_stride = _positive_float(
        data.get("window_stride", 2.0), name="data.window_stride"
    )
    return manifest, sample_fps, processing_width, window_duration, window_stride


def _experiment_slug(name: str) -> str:
    """Convert a user-supplied experiment name into a safe filename stem."""
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", name.strip()).strip(".-_")
    if not slug:
        raise ValueError("experiment.name must contain at least one filename-safe character.")
    return slug[:100]


def _load_record_windows(
    record: DatasetRecord,
    settings: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Extract annotated training windows for one manifest record."""
    record_started = time.perf_counter()
    _manifest, sample_fps, processing_width, window_duration, window_stride = _data_values(settings)
    paths = _section(settings, "paths")
    cache_dir = _path(paths.get("cache_dir", "data/cache"), name="paths.cache_dir")
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
        record.id, clip.duration, sample_fps,
        "enabled" if cache_dir is not None else "disabled",
    )
    sequence = build_feature_sequence(
        record.video_path,
        clip,
        sample_fps=sample_fps,
        processing_width=processing_width,
        cache_dir=cache_dir,
    )
    logger.info("Record %s: extracted %d feature rows; building windows", record.id, len(sequence.features))
    windows = build_windows(
        sequence,
        window=window_duration,
        stride=window_stride,
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
            f"No pure, non-conflicting feature windows for manifest entry {record.id!r}."
        )
    logger.info(
        "Record %s: ready (%d labelled windows) in %.1fs",
        record.id, len(X), time.perf_counter() - record_started,
    )
    return X, y


def _combine_records(
    records: tuple[DatasetRecord, ...],
    settings: Mapping[str, Any],
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
    settings: Mapping[str, Any],
) -> dict[str, Any]:
    """Train from the configured disjoint group split and persist artifacts."""
    if not isinstance(settings, Mapping):
        raise ValueError("Training settings must be a configuration mapping.")
    data = _section(settings, "data")
    experiment = _section(settings, "experiment")
    train_groups = set(_groups(data.get("train_groups"), name="data.train_groups"))
    validation_groups = set(
        _groups(data.get("validation_groups"), name="data.validation_groups")
    )
    overlap = train_groups & validation_groups
    if overlap:
        raise ValueError(
            "Training and validation groups must be disjoint; overlap: "
            + ", ".join(sorted(overlap))
        )
    training_config = _training_config(settings)
    experiment_name = experiment.get("name", "baseline")
    if not isinstance(experiment_name, str) or not experiment_name.strip():
        raise ValueError("experiment.name must be a non-empty string.")
    checkpoint_dir = _path(
        experiment.get("checkpoint_dir", "models/experiments"),
        name="experiment.checkpoint_dir",
    )
    results_dir = _path(
        experiment.get("results_dir", "output/experiments"),
        name="experiment.results_dir",
    )
    _manifest, sample_fps, processing_width, window_duration, window_stride = _data_values(settings)
    records = load_manifest(manifest_path, verify_files=True)
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
        experiment_name, len(train_records), len(val_records),
    )
    logger.info("Preparing training split: 0/%d videos", len(train_records))
    train_X, train_y = _combine_records(train_records, settings)
    logger.info("Training data ready: %d labelled windows from %d videos", len(train_X), len(train_records))
    logger.info("Preparing validation split: 0/%d videos", len(val_records))
    val_X, val_y = _combine_records(val_records, settings)
    logger.info("Validation data ready: %d labelled windows from %d videos", len(val_X), len(val_records))

    slug = _experiment_slug(experiment_name)
    checkpoint_path = checkpoint_dir / f"{slug}_best.pt"
    results_path = results_dir / f"{slug}_train.json"
    data_config = {
        "flow_fps": sample_fps,
        "processing_width": processing_width,
        "window": window_duration,
        "stride": window_stride,
        "target_generation": {
            "positive_set": "pure score-3 windows",
            "unlabeled_set": "pure score-0/1/2 windows",
            "mixed_windows": "discard",
            "label_rule": "all overlapping annotations must agree",
        },
    }
    _, _, summary = train_single_fold(
        train_X,
        train_y,
        val_X,
        val_y,
        training_config,
        checkpoint_path=checkpoint_path,
        data_config=data_config,
    )
    result: dict[str, Any] = {
        "experiment_name": experiment_name,
        "manifest": str(Path(manifest_path).expanduser().resolve()),
        "train_groups": sorted(train_groups),
        "validation_groups": sorted(validation_groups),
        "train_records": [record.id for record in train_records],
        "validation_records": [record.id for record in val_records],
        "train_windows": int(len(train_X)),
        "validation_windows": int(len(val_X)),
        "training_config": asdict(training_config),
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
