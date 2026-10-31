"""Export LightGBM LambdaRank scores for every window in every manifest video.

This is a diagnostic-only command: it does not select clips or render a montage.
"""
from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ..annotations import load_training_annotations
from ..config_loader import Config
from ..data_models import Clip
from ..dataset.cache_paths import feature_cache_dir
from ..dataset.manifest import load_manifest
from ..dataset.normalization import normalize_per_video_features
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
from ..dataset.windows import build_lightgbm_windows_from_features
from ..motion.engineered import engineer_lightgbm_sequence
from ..ml.checkpoint import load_lightgbm_checkpoint
from ..ml.model import LightGBMRanker
from ..video.probe import get_video_info

logger = logging.getLogger(__name__)


def _effective_data_config(
    checkpoint_payload: Mapping[str, Any],
    settings: Mapping[str, Any],
) -> dict[str, Any]:
    """Use the checkpoint's training-time preprocessing parameters."""
    saved = checkpoint_payload.get("data_config", {})
    configured = settings.get("data", {})
    if not isinstance(saved, Mapping) or not isinstance(configured, Mapping):
        raise ValueError("Checkpoint data_config and settings.data must be mappings.")

    def choose(key: str, default: Any) -> Any:
        # A checkpoint records the exact preprocessing used to train the model.
        value = saved.get(key)
        if value is None:
            value = configured.get(key, default)
        return value

    fps = float(choose("sampling_fps", DEFAULT_SAMPLE_FPS))
    width = choose("processing_width", DEFAULT_PROCESSING_WIDTH)
    window = float(choose("window", choose("window_duration", 4.0)))
    stride = float(choose("stride", choose("window_stride", 2.0)))
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("Effective sampling_fps must be finite and positive.")
    if isinstance(width, bool) or not isinstance(width, int) or width <= 0:
        raise ValueError("Effective processing_width must be a positive integer.")
    if not np.isfinite(window) or window <= 0 or not np.isfinite(stride) or stride <= 0:
        raise ValueError("Effective window and stride must be finite and positive.")
    roi = validate_roi(
        choose("roi_left", DEFAULT_ROI_LEFT),
        choose("roi_right", DEFAULT_ROI_RIGHT),
        choose("roi_bottom", DEFAULT_ROI_BOTTOM),
        choose("roi_top", DEFAULT_ROI_TOP),
    )
    return {
        "sampling_fps": fps,
        "processing_width": width,
        "window": window,
        "stride": stride,
        "roi_left": roi[0],
        "roi_right": roi[1],
        "roi_bottom": roi[2],
        "roi_top": roi[3],
    }


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_lightgbm_score_report(
    manifest_path: str | Path,
    settings: Config | Mapping[str, Any],
    *,
    checkpoint_path: str | Path | None = None,
    output_path: str | Path = "output/lightgbm_window_scores.csv",
) -> dict[str, Any]:
    """Score all windows video-by-video and write global/per-video CSV reports.

    The primary CSV is sorted by descending raw score across the entire archive.
    A companion CSV summarizes each video's score min/max/range. Each window
    also carries its within-video rank and its video's score range.
    """
    manifest = Path(manifest_path).expanduser()
    if not manifest.is_file():
        raise FileNotFoundError(f"Dataset manifest not found: {manifest}")

    settings_mapping = settings.to_dict() if isinstance(settings, Config) else settings
    if not isinstance(settings_mapping, Mapping):
        raise ValueError("settings must be a configuration mapping.")
    experiment = settings_mapping.get("experiment", {})
    if not isinstance(experiment, Mapping):
        raise ValueError("Configuration section 'experiment' must be a mapping.")
    if checkpoint_path is None:
        checkpoint_path = experiment.get("lightgbm_model_path", "models/lightgbm/ranker.json")
    checkpoint_path = Path(checkpoint_path).expanduser()
    payload = load_lightgbm_checkpoint(checkpoint_path)
    if payload.get("model_class") != "LightGBMRanker":
        raise ValueError(
            f"Expected a LightGBMRanker checkpoint, got {payload.get('model_class')!r}."
        )

    from ..dataset.windows import LIGHTGBM_TABULAR_FEATURE_NAMES

    expected_names = list(LIGHTGBM_TABULAR_FEATURE_NAMES)
    if payload["feature_names"] != expected_names:
        raise ValueError(
            "Checkpoint feature schema does not match the current LightGBM window "
            "preparation. Refusing to score windows with a different input schema."
        )
    params = _effective_data_config(payload, settings_mapping)
    cache_dir = feature_cache_dir(settings_mapping)
    records = load_manifest(manifest, verify_files=True)
    ranker = LightGBMRanker(
        payload["_booster"], feature_names=tuple(payload["feature_names"])
    )

    all_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for record_number, record in enumerate(records, start=1):
        if record.annotation_path is None:
            raise ValueError(
                f"Manifest entry {record.id!r} has no annotation file; "
                "the report uses the same annotated video extent as training."
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
        end = min(float(annotations[-1].end), float(video_info.duration))
        if end <= 0:
            raise ValueError(f"Video {record.id!r} has no positive annotated duration.")
        clip = Clip(video_name=record.id, start=0.0, end=end)
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
        signals = engineer_lightgbm_sequence(sequence.features)
        windows = build_lightgbm_windows_from_features(
            signals, sequence.timestamps, window=params["window"], stride=params["stride"]
        )
        # Match training: normalize the complete video's windows before labels
        # are filtered. Here no windows are filtered because this is inference.
        normalized = normalize_per_video_features(windows.X)
        scores = ranker.predict(normalized)
        if len(scores) != len(windows.timestamps):
            raise ValueError(f"Score/window count mismatch for video {record.id!r}.")

        video_min = float(np.min(scores))
        video_max = float(np.max(scores))
        video_range = video_max - video_min
        order = np.argsort(-scores, kind="stable")
        ranks = np.empty(len(scores), dtype=np.int64)
        ranks[order] = np.arange(1, len(scores) + 1)
        video_rows: list[dict[str, Any]] = []
        for index, ((start, finish), score) in enumerate(zip(windows.timestamps, scores)):
            video_rows.append({
                "video_id": record.id,
                "group_id": record.group_id,
                "window_index": index,
                "start_seconds": round(float(start), 3),
                "end_seconds": round(float(finish), 3),
                "duration_seconds": round(float(finish - start), 3),
                "score": float(score),
                "rank_in_video": int(ranks[index]),
                "rank_global": 0,
                "video_score_min": video_min,
                "video_score_max": video_max,
                "video_score_range": video_range,
                "video_window_count": len(scores),
            })
        video_rows.sort(key=lambda row: (row["rank_in_video"], row["window_index"]))
        all_rows.extend(video_rows)
        summaries.append({
            "video_id": record.id,
            "group_id": record.group_id,
            "window_count": len(scores),
            "score_min": video_min,
            "score_max": video_max,
            "score_range": video_range,
            "score_mean": float(np.mean(scores)),
            "score_median": float(np.median(scores)),
            "score_std": float(np.std(scores, ddof=0)),
            "best_window_index": int(order[0]),
            "best_window_start_seconds": round(float(windows.timestamps[order[0], 0]), 3),
            "best_window_end_seconds": round(float(windows.timestamps[order[0], 1]), 3),
            "best_window_score": float(scores[order[0]]),
        })
        logger.info(
            "Scored %s (%d/%d): %d windows; score range [%.6f, %.6f]",
            record.id, record_number, len(records), len(scores), video_min, video_max,
        )

    if not all_rows:
        raise ValueError("No windows were scored from the manifest.")
    global_order = sorted(
        range(len(all_rows)),
        key=lambda index: (-float(all_rows[index]["score"]), all_rows[index]["video_id"],
                           int(all_rows[index]["window_index"])),
    )
    for rank, row_index in enumerate(global_order, start=1):
        all_rows[row_index]["rank_global"] = rank
    all_rows.sort(key=lambda row: (row["rank_global"], row["video_id"], row["window_index"]))

    output = Path(output_path).expanduser()
    if output.suffix.lower() != ".csv":
        output = output / "lightgbm_window_scores.csv"
    video_output = output.with_name(f"{output.stem}_video_ranges.csv")
    per_video_output = output.with_name(f"{output.stem}_by_video.csv")
    fields = [
        "rank_global", "video_id", "group_id", "window_index", "start_seconds",
        "end_seconds", "duration_seconds", "score", "rank_in_video",
        "video_score_min", "video_score_max", "video_score_range", "video_window_count",
    ]
    summary_fields = [
        "video_id", "group_id", "window_count", "score_min", "score_max",
        "score_range", "score_mean", "score_median", "score_std",
        "best_window_index", "best_window_start_seconds", "best_window_end_seconds",
        "best_window_score",
    ]
    _write_csv(output, fields, all_rows)
    per_video_rows = sorted(
        all_rows,
        key=lambda row: (row["video_id"], row["rank_in_video"], row["window_index"]),
    )
    _write_csv(per_video_output, fields, per_video_rows)
    summaries.sort(key=lambda row: (-row["score_max"], row["video_id"]))
    _write_csv(video_output, summary_fields, summaries)
    return {
        "checkpoint": str(checkpoint_path.resolve()),
        "manifest": str(manifest.resolve()),
        "videos_scored": len(summaries),
        "windows_scored": len(all_rows),
        "global_scores_csv": str(output.resolve()),
        "per_video_scores_csv": str(per_video_output.resolve()),
        "video_ranges_csv": str(video_output.resolve()),
        "score_semantics": "raw LambdaRank score; not a probability or calibrated cross-video quality",
        "video_summary": summaries,
    }
