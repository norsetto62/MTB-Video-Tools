"""Manifest-to-LightGBM LambdaRank training pipeline.

It reuses the canonical manifest, NPZ feature cache, video feature extractor,
annotation parser, and tabular-window APIs without changing the legacy P/U
pipeline. Ordinal labels are ranked within each video query.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from ..annotations import load_training_annotations
from ..config_loader import Config
from ..data_models import Clip
from ..dataset.cache_paths import feature_cache_dir
from ..dataset.lightgbm import build_lightgbm_training_data
from ..dataset.manifest import DatasetRecord, load_manifest
from ..dataset.sequence import (
    DEFAULT_PROCESSING_WIDTH,
    DEFAULT_ROI_BOTTOM,
    DEFAULT_ROI_LEFT,
    DEFAULT_ROI_RIGHT,
    DEFAULT_ROI_TOP,
    DEFAULT_SAMPLE_FPS,
    build_feature_sequence,
    validate_roi,
)
from ..ml.lightgbm_training import (
    LightGBMGroupFoldResult,
    LightGBMVideoData,
    _combine,
    _combine_with_group_sizes,
    train_lightgbm_lovo,
)
from ..ml.training import train_lightgbm
from ..ml.model import LightGBMConfig
from ..video.probe import get_video_info


def _section(settings: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    section = settings.get(name, {})
    if not isinstance(section, Mapping):
        raise ValueError(f"Configuration section {name!r} must be a mapping.")
    return section


def _positive_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite positive number.")
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive number.")
    return value


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def _lightgbm_config(settings: Mapping[str, Any]) -> LightGBMConfig:
    """Read LightGBM options from the dedicated lightgbm config section."""
    section = dict(_section(settings, "lightgbm"))
    # Migrate the previous multiclass YAML defaults to LambdaRank. These
    # settings are obsolete in ranking mode and must not reach LightGBM.
    if section.get("objective") == "multiclass":
        section["objective"] = "lambdarank"
    if section.get("num_class") == 4:
        section.pop("num_class")
    allowed = set(LightGBMConfig.__dataclass_fields__)
    unknown = sorted(set(section) - allowed)
    if unknown:
        raise ValueError(f"Unknown lightgbm configuration key(s): {unknown}.")
    try:
        return LightGBMConfig(**section)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid lightgbm configuration: {exc}") from exc


def _data_config(settings: Mapping[str, Any]) -> dict[str, Any]:
    data = _section(settings, "data")
    sample_fps = _positive_float(
        data.get("sampling_fps", DEFAULT_SAMPLE_FPS), "data.sampling_fps"
    )
    processing_width = _positive_int(
        data.get("processing_width", DEFAULT_PROCESSING_WIDTH), "data.processing_width"
    )
    window = _positive_float(data.get("window_duration", 4.0), "data.window_duration")
    stride = _positive_float(data.get("window_stride", 2.0), "data.window_stride")
    roi = validate_roi(
        data.get("roi_left", DEFAULT_ROI_LEFT),
        data.get("roi_right", DEFAULT_ROI_RIGHT),
        data.get("roi_bottom", DEFAULT_ROI_BOTTOM),
        data.get("roi_top", DEFAULT_ROI_TOP),
    )
    return {
        "sampling_fps": sample_fps,
        "processing_width": processing_width,
        "window": window,
        "stride": stride,
        "roi_left": roi[0],
        "roi_right": roi[1],
        "roi_bottom": roi[2],
        "roi_top": roi[3],
        "feature_normalization": "per_video_zscore_all_windows_ddof0",
    }


def prepare_lightgbm_manifest_data(
    manifest_path: str | Path,
    settings: Mapping[str, Any],
    *,
    records: tuple[DatasetRecord, ...] | None = None,
    require_multiple_groups: bool = True,
    progress_callback: Callable[[str], None] | None = None,
) -> tuple[dict[str, LightGBMVideoData], dict[str, str], dict[str, Any]]:
    """Load/reuse per-video cached features and build labeled 48-column rows.

    Each manifest record must have an annotation file because this is a
    supervised MTB 0..3 workflow. Existing NPZ feature caches are only read
    through build_feature_sequence; cache misses are extracted and written by
    the canonical cache implementation.
    """
    if not isinstance(settings, Mapping):
        raise ValueError("settings must be a configuration mapping.")
    manifest = Path(manifest_path).expanduser()
    if progress_callback is not None:
        progress_callback(f"Loading and verifying manifest: {manifest}")
    if records is None:
        records = load_manifest(manifest, verify_files=True)
    if progress_callback is not None:
        progress_callback(f"Manifest ready: {len(records)} records; preparing cached features and labels")
    data = _section(settings, "data")
    params = _data_config(settings)
    cache_dir = feature_cache_dir(settings)
    prepared: dict[str, LightGBMVideoData] = {}
    group_by_video: dict[str, str] = {}

    total_records = len(records)
    for record_index, record in enumerate(records, start=1):
        if progress_callback is not None:
            progress_callback(f"Preparing video {record_index}/{total_records}: {record.id}")
        if record.annotation_path is None:
            raise ValueError(
                f"Manifest entry {record.id!r} has no annotation file; "
                "requires MTB 0..3 annotations (ordinal relevance grades for LightGBM ranking)."
            )
        declared_video, annotations = load_training_annotations(record.annotation_path)
        if declared_video.resolve() != record.video_path.resolve():
            raise ValueError(
                f"Manifest entry {record.id!r} points to {record.video_path}, but "
                f"its annotation file declares {declared_video}."
            )
        if not annotations:
            raise ValueError(f"Manifest entry {record.id!r} has no MTB annotations.")

        video_info = get_video_info(record.video_path)
        timeline_end = float(annotations[-1].end)
        if timeline_end > float(video_info.duration) + 0.05:
            raise ValueError(
                f"Annotation end ({timeline_end:.3f}s) exceeds video duration "
                f"({video_info.duration:.3f}s) for {record.id!r}."
            )
        timeline_end = min(timeline_end, float(video_info.duration))
        clip = Clip(video_name=record.id, start=0.0, end=timeline_end)
        sequence = build_feature_sequence(
            record.video_path,
            clip,
            sample_fps=params["sampling_fps"],
            processing_width=params["processing_width"],
            cache_dir=cache_dir,
            roi_left=params["roi_left"],
            roi_right=params["roi_right"],
            roi_bottom=params["roi_bottom"],
            roi_top=params["roi_top"],
        )
        windows, targets = build_lightgbm_training_data(
            sequence,
            annotations,
            window=params["window"],
            stride=params["stride"],
        )
        if len(targets) == 0:
            raise ValueError(
                f"Manifest entry {record.id!r} produced no windows with annotated duration."
            )
        if not np.isfinite(windows.X).all():
            raise ValueError(f"Manifest entry {record.id!r} produced non-finite LightGBM features.")
        prepared[record.id] = LightGBMVideoData(windows=windows, targets=targets)
        group_by_video[record.id] = record.group_id
        if progress_callback is not None:
            progress_callback(
                f"Prepared {record_index}/{total_records}: {record.id} "
                f"({len(targets)} windows; group={record.group_id})"
            )

    if not prepared:
        raise ValueError("The manifest contains no annotated videos for LightGBM training.")
    if require_multiple_groups and len({group_by_video[video_id] for video_id in prepared}) < 2:
        raise ValueError("At least two annotated manifest groups are required for LOVO validation.")
    return prepared, group_by_video, params


def run_lightgbm_lovo_from_manifest(
    manifest_path: str | Path,
    settings: Mapping[str, Any],
    *,
    checkpoint_dir: str | Path | None = None,
    config: LightGBMConfig | None = None,
) -> tuple[LightGBMGroupFoldResult, ...]:
    """Prepare data from manifest/cache/video sources and run grouped LOVO."""
    prepared, group_by_video, data_config = prepare_lightgbm_manifest_data(
        manifest_path, settings
    )
    if config is None:
        config = _lightgbm_config(settings)
    if checkpoint_dir is None:
        experiment = _section(settings, "experiment")
        checkpoint_dir = experiment.get(
            "lightgbm_checkpoint_dir", "models/lightgbm_lovo"
        )
    return train_lightgbm_lovo(
        prepared,
        group_by_video,
        config=config,
        checkpoint_dir=checkpoint_dir,
        data_config={
            **data_config,
            "manifest": str(Path(manifest_path).expanduser().resolve()),
            "feature_cache_dir": str(feature_cache_dir(settings)),
            "target_policy": "ordinal relevance grades 0..3 from annotated-duration-majority; ties to higher class",
            "ranking_query_policy": "each video is one independent ranking query",
        },
    )



def _group_offsets(group_sizes):
    offset = 0
    for size in group_sizes:
        yield offset, size
        offset += size


def run_lightgbm_train_from_manifest(
    manifest_path: str | Path,
    settings: Mapping[str, Any],
    *,
    checkpoint_path: str | Path | None = None,
    config: LightGBMConfig | None = None,
) -> dict[str, Any]:
    """Validate on configured held-out groups, then refit one final model.

    data.train_groups and data.validation_groups define a disjoint group-level
    split. Validation NDCG selects the iteration count; the final checkpoint
    is then refit on all annotated manifest videos using that iteration count.
    """
    if not isinstance(settings, Mapping):
        raise ValueError("settings must be a configuration mapping.")
    data = _section(settings, "data")

    def configured_groups(key: str) -> tuple[str, ...]:
        value = data.get(key)
        if not isinstance(value, (tuple, list)) or not value:
            raise ValueError(f"data.{key} must be a non-empty list of group IDs.")
        groups: list[str] = []
        for item in value:
            if not isinstance(item, str) or not item.strip():
                raise ValueError(f"data.{key} must contain non-empty string group IDs.")
            group = item.strip()
            if group in groups:
                raise ValueError(f"data.{key} contains duplicate group ID {group!r}.")
            groups.append(group)
        return tuple(groups)

    train_groups = configured_groups("train_groups")
    validation_groups = configured_groups("validation_groups")
    overlap = set(train_groups) & set(validation_groups)
    if overlap:
        raise ValueError(
            "Training and validation groups must be disjoint; overlap: "
            + ", ".join(sorted(overlap))
        )

    manifest_records = load_manifest(manifest_path, verify_files=True)
    known_groups = {record.group_id for record in manifest_records}
    unknown = (set(train_groups) | set(validation_groups)) - known_groups
    if unknown:
        raise ValueError(
            "Configured group ID(s) not found in the manifest: "
            + ", ".join(sorted(unknown))
        )
    train_records = tuple(r for r in manifest_records if r.group_id in train_groups)
    validation_records = tuple(r for r in manifest_records if r.group_id in validation_groups)
    if not train_records:
        raise ValueError("No manifest records match data.train_groups.")
    if not validation_records:
        raise ValueError("No manifest records match data.validation_groups.")

    prepared, group_by_video, data_config = prepare_lightgbm_manifest_data(
        manifest_path, settings, records=manifest_records, require_multiple_groups=False
    )
    train_ids = sorted(
        video_id for video_id in prepared if group_by_video[video_id] in train_groups
    )
    validation_ids = sorted(
        video_id for video_id in prepared if group_by_video[video_id] in validation_groups
    )
    if not train_ids:
        raise ValueError("No annotated videos were prepared from data.train_groups.")
    if not validation_ids:
        raise ValueError("No annotated videos were prepared from data.validation_groups.")

    if config is None:
        config = _lightgbm_config(settings)
    train_X, train_y, train_group_sizes = _combine_with_group_sizes(prepared, train_ids)
    val_X, val_y, val_group_sizes = _combine_with_group_sizes(prepared, validation_ids)
    val_timestamps = np.concatenate(
        [prepared[video_id].windows.timestamps for video_id in validation_ids], axis=0
    )
    if not any(
        len(np.unique(train_y[offset:offset + size])) > 1
        for offset, size in _group_offsets(train_group_sizes)
    ):
        raise ValueError("LambdaRank training requires at least one video with multiple relevance grades.")
    if len(val_y) == 0:
        raise ValueError("The configured validation groups produced no labeled windows.")

    # This first fit is strictly group-held-out. Its Segment Recall@5 history
    # and best iteration determine the final all-data refit's iteration count.
    _, validation_summary = train_lightgbm(
        train_X,
        train_y,
        train_group_sizes=train_group_sizes,
        val_X=val_X,
        val_y=val_y,
        val_group_sizes=val_group_sizes,
        val_timestamps=val_timestamps,
        config=config,
        data_config={
            **data_config,
            "manifest": str(Path(manifest_path).expanduser().resolve()),
            "feature_cache_dir": str(feature_cache_dir(settings)),
            "target_policy": "ordinal relevance grades 0..3 from annotated-duration-majority; ties to higher class",
        },
        metadata={
            "training_strategy": "configured_group_validation",
            "train_groups": list(train_groups),
            "validation_groups": list(validation_groups),
            "training_video_ids": train_ids,
            "validation_video_ids": validation_ids,
        },
    )
    best_iteration = int(validation_summary["best_iteration"])
    if best_iteration <= 0:
        raise ValueError(f"Validation training returned invalid best_iteration={best_iteration}.")

    # Refit the production checkpoint on all annotated videos, including the
    # validation group, for the selected number of boosting iterations.
    all_ids = sorted(prepared)
    all_X, all_y, all_group_sizes = _combine_with_group_sizes(prepared, all_ids)
    if not any(
        len(np.unique(all_y[offset:offset + size])) > 1
        for offset, size in _group_offsets(all_group_sizes)
    ):
        raise ValueError("LambdaRank final refit requires at least one video with multiple relevance grades.")
    if checkpoint_path is None:
        experiment = _section(settings, "experiment")
        checkpoint_path = experiment.get(
            "lightgbm_model_path", "models/lightgbm/ranker.json"
        )
    destination = Path(checkpoint_path).expanduser()
    if destination.exists() and destination.is_dir():
        raise ValueError(f"LightGBM checkpoint path is a directory: {destination}")

    from dataclasses import replace
    final_config = replace(config, num_iterations=best_iteration)
    training_config = {
        **data_config,
        "manifest": str(Path(manifest_path).expanduser().resolve()),
        "feature_cache_dir": str(feature_cache_dir(settings)),
        "target_policy": "ordinal relevance grades 0..3 from annotated-duration-majority; ties to higher class",
        "ranking_query_policy": "each video is one independent ranking query",
        "group_by_video": dict(sorted(group_by_video.items())),
        "training_group_sizes": all_group_sizes,
        "training_video_ids": all_ids,
        "validation_selection": {
            "train_groups": list(train_groups),
            "validation_groups": list(validation_groups),
            "best_iteration": best_iteration,
            "evaluation_history": validation_summary.get("evaluation_history", {}),
        },
    }
    _, summary = train_lightgbm(
        all_X,
        all_y,
        train_group_sizes=all_group_sizes,
        config=final_config,
        checkpoint_path=destination,
        data_config=training_config,
        metadata={
            "training_strategy": "configured_group_validation_then_all_data_refit",
            "training_video_ids": all_ids,
            "training_group_ids": sorted(set(group_by_video.values())),
            "validation_groups_for_iteration_selection": list(validation_groups),
            "validation_training_rows": int(len(train_X)),
            "validation_rows": int(len(val_X)),
            "best_iteration_from_validation": best_iteration,
        },
    )
    summary["validation_selection"] = {
        "train_groups": list(train_groups),
        "validation_groups": list(validation_groups),
        "training_video_ids": train_ids,
        "validation_video_ids": validation_ids,
        "training_rows": int(len(train_X)),
        "validation_rows": int(len(val_X)),
        "best_iteration": best_iteration,
        "evaluation_history": validation_summary.get("evaluation_history", {}),
        "early_stopping_rounds": config.early_stopping_rounds,
    }
    summary["training_video_ids"] = all_ids
    summary["training_group_ids"] = sorted(set(group_by_video.values()))
    return summary
