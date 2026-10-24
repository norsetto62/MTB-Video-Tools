"""Leakage-safe leave-one-manifest-group-out cross-validation."""

from __future__ import annotations

from dataclasses import asdict, replace
import csv
import json
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .training import _combine_records, _experiment_slug, _data_values, _path, _section, _training_config
from ..dataset.manifest import get_unique_groups, load_manifest
from ..dataset.sequence import build_feature_sequence
from ..dataset.windows import build_windows
from ..motion.engineered import engineer_motion_features
from ..annotations import load_training_annotations
from ..data_models import Clip
from ..selection.candidates_generation import CandidateGenerationConfig, generate_candidates
from ..ml.inference import InferenceResult
from ..ml.model import MTBClassifier
from ..ml.scaling import FeatureScaler
from ..ml.training import targets_to_classes, train_fixed_epochs, train_single_fold

logger = logging.getLogger(__name__)


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    """Report PU selection statistics without treating unlabeled windows as negative."""
    positive = actual == 1
    unlabeled = actual == 0
    predicted_positive = predicted == 1
    return {
        "windows": int(len(actual)),
        "positive_windows": int(np.sum(positive)),
        "unlabeled_windows": int(np.sum(unlabeled)),
        "positive_recall_on_P": float(np.mean(predicted_positive[positive])) if np.any(positive) else None,
        "selected_fraction_of_U": float(np.mean(predicted_positive[unlabeled])) if np.any(unlabeled) else None,
        "predicted_positive_fraction": float(np.mean(predicted_positive)) if len(actual) else 0.0,
    }

def _threshold_sweep(
    actual: np.ndarray,
    probabilities: np.ndarray,
) -> list[dict[str, Any]]:
    """Compute diagnostic PU metrics over fixed probability thresholds."""
    rows: list[dict[str, Any]] = []
    for threshold_step in range(1, 20):
        threshold = threshold_step * 0.05
        predicted = (probabilities >= threshold).astype(np.int64)
        metrics = _metrics(actual, predicted)
        recall = metrics["positive_recall_on_P"]
        u_selection = metrics["selected_fraction_of_U"]
        rows.append({
            "threshold": round(threshold, 2),
            **metrics,
            "score": (
                float(recall * (1.0 - u_selection))
                if recall is not None and u_selection is not None
                else None
            ),
        })
    return rows


def _fit_fixed_epochs(
    windows: np.ndarray,
    targets: np.ndarray,
    config: Any,
    *,
    epochs: int,
    checkpoint_path: Path,
    data_config: dict[str, Any],
    metadata: dict[str, Any],
) -> tuple[MTBClassifier, FeatureScaler, dict[str, Any]]:
    """Delegate the leakage-safe final refit to the shared training primitive."""
    return train_fixed_epochs(
        windows,
        targets,
        config,
        epochs=epochs,
        checkpoint_path=checkpoint_path,
        data_config=data_config,
        metadata=metadata,
    )


def _combine_prepared_records(
    records: tuple[Any, ...],
    prepared: dict[str, tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate already-prepared record windows without reloading videos."""
    selected = [prepared[record.id] for record in records]
    if not selected:
        raise ValueError("The selected group split contains no manifest records.")
    return (
        np.concatenate([item[0] for item in selected], axis=0),
        np.concatenate([item[1] for item in selected], axis=0),
    )


def _predict(
    model: MTBClassifier,
    scaler: FeatureScaler,
    windows: np.ndarray,
    positive_threshold: float = 0.50,
) -> tuple[np.ndarray, np.ndarray]:
    """Return P probabilities and binary P/U predictions."""
    if isinstance(positive_threshold, bool) or not isinstance(positive_threshold, (int, float)) or not np.isfinite(positive_threshold) or not 0.0 <= positive_threshold <= 1.0:
        raise ValueError("positive_threshold must be between 0 and 1.")
    transformed = scaler.transform(windows)
    tensor = torch.from_numpy(np.ascontiguousarray(transformed, dtype=np.float32))
    batches: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(tensor), 256):
            logits = model(tensor[start:start + 256])
            if logits.ndim != 2 or logits.shape[1] != 1:
                raise RuntimeError("P/U model must return one logit per window.")
            batches.append(torch.sigmoid(logits[:, 0]).cpu().numpy())
    probabilities = np.concatenate(batches, axis=0)
    predictions = (probabilities >= float(positive_threshold)).astype(np.int64)
    return predictions, probabilities

def _write_selection_diagnostics(
    path: Path,
    *,
    record: Any,
    settings: Mapping[str, Any],
    model: MTBClassifier,
    scaler: FeatureScaler,
    positive_threshold: float,
    fold_metadata: Mapping[str, Any],
) -> None:
    """Export every source-timeline window and its candidate-selection outcome."""
    from ..video.probe import get_video_info

    _manifest, sample_fps, processing_width, window_duration, window_stride, roi_left, roi_right, roi_bottom, roi_top = _data_values(settings)
    paths = _section(settings, "paths")
    cache_dir = _path(paths.get("cache_dir", "data/cache"), name="paths.cache_dir")
    declared_video, annotations = load_training_annotations(record.annotation_path)
    if declared_video.resolve() != record.video_path.resolve():
        raise ValueError(
            f"Manifest entry {record.id!r} points to {record.video_path}, but "
            f"its annotation file declares {declared_video}."
        )
    if not annotations:
        raise ValueError(f"Manifest entry {record.id!r} has no annotations for diagnostics.")

    video_info = get_video_info(record.video_path)
    timeline_end = float(annotations[-1].end)
    if timeline_end > video_info.duration + 0.05:
        raise ValueError(
            f"Annotation end ({timeline_end:.3f}s) exceeds video duration "
            f"({video_info.duration:.3f}s) for {record.id!r}."
        )
    timeline_end = min(timeline_end, float(video_info.duration))
    sequence = build_feature_sequence(
        record.video_path,
        Clip(video_name=record.id, start=0.0, end=timeline_end),
        sample_fps=sample_fps,
        processing_width=processing_width,
        cache_dir=cache_dir,
        roi_left=roi_left, roi_right=roi_right,
        roi_bottom=roi_bottom, roi_top=roi_top,
    )
    windows = build_windows(sequence, window=window_duration, stride=window_stride)
    compact_windows = engineer_motion_features(windows.X)
    _predicted, probabilities = _predict(
        model, scaler, compact_windows, positive_threshold=positive_threshold
    )

    selection = _section(settings, "selection")
    candidate_config = CandidateGenerationConfig(
        positive_threshold=positive_threshold,
        min_candidate_duration=float(selection.get("min_clip", selection.get("min_candidate_duration", 3.0))),
        max_gap_to_merge=float(selection.get("max_gap_to_merge", 3.0)),
        padding_seconds=float(selection.get("padding_seconds", 2.0)),
    )
    candidates = generate_candidates(
        InferenceResult(timestamps=windows.timestamps, probabilities=probabilities),
        timeline_end,
        config=candidate_config,
    )
    candidate_for_window: dict[int, tuple[int, Any, bool]] = {}
    # Map both threshold-selected evidence windows and lower-scoring windows
    # covered only because a candidate was padded around its evidence.
    timestamp_to_index = {
        (round(float(start), 6), round(float(end), 6)): index
        for index, (start, end) in enumerate(windows.timestamps)
    }
    for candidate_index, candidate in enumerate(candidates, start=1):
        evidence_indices = {
            timestamp_to_index[(round(item.start, 6), round(item.end, 6))]
            for item in candidate.window_evidence
            if (round(item.start, 6), round(item.end, 6)) in timestamp_to_index
        }
        for window_index, (start, end) in enumerate(windows.timestamps):
            if float(start) < candidate.end and float(end) > candidate.start:
                candidate_for_window[window_index] = (
                    candidate_index, candidate, window_index in evidence_indices
                )

    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "trial_number", "record_id", "group_id", "source_video",
            "window_index", "window_start_s", "window_end_s", "p_probability",
            "threshold", "selected_by_threshold", "annotation_status",
            "overlapping_annotation_scores", "overlapping_annotation_remarks",
            "selection_outcome", "candidate_id", "candidate_start_s",
            "candidate_end_s", "candidate_duration_s", "candidate_mean_probability",
            "candidate_max_probability",
        ])
        if write_header:
            writer.writeheader()
        trial_number = fold_metadata.get("diagnostic_trial_number", "")
        for window_index, (start, end) in enumerate(windows.timestamps):
            overlapping = [
                annotation for annotation in annotations
                if min(float(end), annotation.end) - max(float(start), annotation.start) > 0.0
            ]
            scores = sorted({int(annotation.score) for annotation in overlapping})
            if not overlapping:
                annotation_status = "UNCOVERED"
            elif len(scores) > 1:
                annotation_status = "MIXED"
            elif scores[0] == 3:
                annotation_status = "P"
            else:
                annotation_status = "U"
            selected = bool(probabilities[window_index] >= positive_threshold)
            mapped = candidate_for_window.get(window_index)
            if mapped:
                candidate_index, candidate, is_evidence = mapped
                outcome = "IN_FINAL_CANDIDATE" if is_evidence else "IN_CANDIDATE_PADDING"
                candidate_id = f"{record.id}-C{candidate_index:03d}"
                candidate_values = {
                    "candidate_start_s": f"{candidate.start:.3f}",
                    "candidate_end_s": f"{candidate.end:.3f}",
                    "candidate_duration_s": f"{candidate.duration:.3f}",
                    "candidate_mean_probability": f"{candidate.mean_probability:.6f}",
                    "candidate_max_probability": f"{candidate.max_probability:.6f}",
                }
            else:
                candidate_id = ""
                candidate_values = {
                    "candidate_start_s": "", "candidate_end_s": "",
                    "candidate_duration_s": "", "candidate_mean_probability": "",
                    "candidate_max_probability": "",
                }
                outcome = (
                    "BELOW_THRESHOLD" if not selected
                    else "THRESHOLD_SELECTED_BUT_NO_FINAL_CANDIDATE"
                )
            writer.writerow({
                "trial_number": trial_number,
                "record_id": record.id,
                "group_id": record.group_id,
                "source_video": str(record.video_path),
                "window_index": window_index,
                "window_start_s": f"{float(start):.3f}",
                "window_end_s": f"{float(end):.3f}",
                "p_probability": f"{float(probabilities[window_index]):.6f}",
                "threshold": f"{positive_threshold:.6f}",
                "selected_by_threshold": selected,
                "annotation_status": annotation_status,
                "overlapping_annotation_scores": "|".join(map(str, scores)),
                "overlapping_annotation_remarks": " | ".join(
                    annotation.remarks for annotation in overlapping if annotation.remarks
                ),
                "selection_outcome": outcome,
                "candidate_id": candidate_id,
                **candidate_values,
            })



def _resolve_group_splits(
    all_groups: tuple[str, ...],
    *,
    test_groups: tuple[str, ...] | None = None,
    train_groups: tuple[str, ...] | None = None,
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Resolve evaluation folds and training pools, optionally isolating proxy groups."""
    if test_groups is None:
        if train_groups is not None:
            raise ValueError("train_groups can only be supplied when test_groups are specified.")
        if len(all_groups) < 3:
            raise ValueError(
                f"LOVO requires at least 3 distinct manifest groups (found {len(all_groups)}): "
                "one outer test group, one inner validation group, and at least one inner training group."
            )
        return tuple(
            (test_group, tuple(group for group in all_groups if group != test_group))
            for test_group in all_groups
        )

    if not test_groups:
        raise ValueError("test_groups must contain at least one proxy test group.")
    if len(set(test_groups)) != len(test_groups):
        raise ValueError("test_groups must not contain duplicates.")
    unknown_test = set(test_groups) - set(all_groups)
    if unknown_test:
        raise ValueError("Unknown test group(s): " + ", ".join(sorted(unknown_test)))

    if train_groups is None:
        resolved_train = tuple(group for group in all_groups if group not in test_groups)
    else:
        if not train_groups:
            raise ValueError("train_groups must contain at least one training group.")
        if len(set(train_groups)) != len(train_groups):
            raise ValueError("train_groups must not contain duplicates.")
        unknown_train = set(train_groups) - set(all_groups)
        if unknown_train:
            raise ValueError("Unknown training group(s): " + ", ".join(sorted(unknown_train)))
        overlap = set(train_groups) & set(test_groups)
        if overlap:
            raise ValueError(
                "Proxy test groups must be excluded from training; overlap: "
                + ", ".join(sorted(overlap))
            )
        resolved_train = tuple(group for group in all_groups if group in set(train_groups))

    if len(resolved_train) < 2:
        raise ValueError(
            "Proxy HPO evaluation requires at least two training groups so inner "
            "epoch selection has both training and validation groups."
        )
    return tuple((test_group, resolved_train) for test_group in test_groups)


def run_lovo(
    manifest_path: str | Path,
    settings: Mapping[str, Any],
    *,
    test_groups: tuple[str, ...] | None = None,
    train_groups: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Run nested group CV, optionally evaluating proxy groups only.

    When test_groups are supplied, train_groups is the exclusive training pool:
    proxy test groups are excluded from all fitting and inner epoch selection.
    """
    if not isinstance(settings, Mapping):
        raise ValueError("LOVO settings must be a configuration mapping.")
    experiment = _section(settings, "experiment")
    experiment_name = experiment.get("name", "baseline")
    if not isinstance(experiment_name, str) or not experiment_name.strip():
        raise ValueError("experiment.name must be a non-empty string.")
    results_dir = _path(experiment.get("results_dir", "output/experiments"), name="experiment.results_dir")
    checkpoint_dir = _path(experiment.get("checkpoint_dir", "models/experiments"), name="experiment.checkpoint_dir")
    _manifest, sample_fps, processing_width, window_duration, window_stride, roi_left, roi_right, roi_bottom, roi_top = _data_values(settings)
    training_config = _training_config(settings)
    debug_settings = _section(settings, "debug")
    threshold_sweep_enabled = debug_settings.get("threshold_sweep", False)
    if not isinstance(threshold_sweep_enabled, bool):
        raise ValueError("debug.threshold_sweep must be a boolean.")
    records = load_manifest(manifest_path, verify_files=True)
    all_groups = get_unique_groups(records)
    group_splits = _resolve_group_splits(
        all_groups, test_groups=test_groups, train_groups=train_groups
    )
    groups = tuple(test_group for test_group, _ in group_splits)
    selected_groups = set(groups)
    for _, training_pool in group_splits:
        selected_groups.update(training_pool)
    # Do not even prepare records outside the selected proxy/test and training
    # groups; this keeps proxy trials cheaper as well as preventing leakage.
    records = tuple(record for record in records if record.group_id in selected_groups)

    slug = _experiment_slug(experiment_name)
    fold_root = results_dir / f"{slug}_lovo"
    checkpoint_root = checkpoint_dir / f"{slug}_lovo"
    data_config = {
        "flow_fps": sample_fps,
        "processing_width": processing_width,
        "roi": {"left": roi_left, "right": roi_right, "bottom": roi_bottom, "top": roi_top},
        "window": window_duration,
        "stride": window_stride,
        "pu_training": {
            "objective": "non-negative PU risk",
            "positive_prior": training_config.positive_prior,
            "positive_score": 3,
            "unlabeled_scores": [0, 1, 2],
            "model_outputs": 1,
        },
    }
    
    # Each record's windows/targets are independent of the fold. Prepare them
    # once so repeated folds do not re-hash the same source videos.
    logger.info("Preparing LOVO data once for %d manifest records", len(records))
    prepared_records: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for record_index, record in enumerate(records, start=1):
        logger.info("Preparing LOVO record %d/%d: %s (group %s)", record_index, len(records), record.id, record.group_id)
        prepared_records[record.id] = _combine_records((record,), settings)

    folds: list[dict[str, Any]] = []
    pooled_actual: list[np.ndarray] = []
    pooled_predicted: list[np.ndarray] = []
    fold_threshold_sweeps: list[list[dict[str, Any]]] = []
    
    for fold_index, (test_group, outer_train_groups) in enumerate(group_splits):
        outer_train_records = tuple(r for r in records if r.group_id in outer_train_groups)
        test_records = tuple(r for r in records if r.group_id == test_group)

        logger.info("LOVO fold %d/%d: test=%s", fold_index + 1, len(groups), test_group)

        # Robust Inner Validation: Evaluate across all possible inner validation groups
        # to find a consensus best epoch, avoiding single-group early-stopping bias.
        candidate_best_epochs = []
        for inner_val_group in outer_train_groups:
            inner_train_g = tuple(g for g in outer_train_groups if g != inner_val_group)
            if not inner_train_g:
                continue
            
            inner_train_recs = tuple(r for r in records if r.group_id in inner_train_g)
            inner_val_recs = tuple(r for r in records if r.group_id == inner_val_group)
            
            inner_X, inner_y = _combine_prepared_records(inner_train_recs, prepared_records)
            inner_val_X, inner_val_y = _combine_prepared_records(inner_val_recs, prepared_records)
            
            selection_config = replace(training_config, seed=training_config.seed + fold_index)
            _, _, selection = train_single_fold(
                inner_X,
                inner_y,
                inner_val_X,
                inner_val_y,
                selection_config,
                fold_video_id=f"inner-validation:{inner_val_group}",
                data_config=data_config,
            )
            candidate_best_epochs.append(int(selection["best_epoch"]))

        # Take the median epoch count across all inner validation splits as the consensus
        selected_epochs = int(np.median(candidate_best_epochs)) if candidate_best_epochs else 10
        if selected_epochs < 1:
            selected_epochs = 1

        logger.info(
            "LOVO fold %d consensus selected_epochs: %d (from candidates: %s)", 
            fold_index + 1, selected_epochs, candidate_best_epochs,
        )

        inner_validation_group = "consensus_multi_group"
        inner_train_groups = outer_train_groups  # Full outer train pool used for refit

        outer_X, outer_y = _combine_prepared_records(outer_train_records, prepared_records)
        # Keep record boundaries for interpretable out-of-fold predictions.
        test_parts = [
            (record, *_combine_prepared_records((record,), prepared_records))
            for record in test_records
        ]
        test_X = np.concatenate([part[1] for part in test_parts], axis=0)
        test_y = np.concatenate([part[2] for part in test_parts], axis=0)
        test_window_records = [
            (record.id, local_index)
            for record, part_X, _part_y in test_parts
            for local_index in range(len(part_X))
        ]
        fold_checkpoint = checkpoint_root / f"fold_{fold_index + 1:02d}_{_experiment_slug(test_group)}.pt"
        fold_metadata = {
            "experiment_name": experiment_name,
            "evaluation": "proxy_group_holdout" if test_groups is not None else "nested_leave_one_group_out",
            "outer_test_group": test_group,
            "outer_training_groups": list(outer_train_groups),
            "inner_validation_group": inner_validation_group,
            "inner_training_groups": list(inner_train_groups),
            "manifest": str(Path(manifest_path).expanduser().resolve()),
        }
        final_config = replace(training_config, seed=training_config.seed + fold_index)
        model, scaler, refit_summary = _fit_fixed_epochs(
            outer_X,
            outer_y,
            final_config,
            epochs=selected_epochs,
            checkpoint_path=fold_checkpoint,
            data_config=data_config,
            metadata=fold_metadata,
        )

        actual = targets_to_classes(test_y)
        training_settings = _section(settings, "training")
        positive_threshold = float(training_settings.get(
            "positive_threshold", training_settings.get("c2_threshold", 0.50)
        ))
        predicted, probabilities = _predict(
            model, scaler, test_X, positive_threshold=positive_threshold
        )
        diagnostics_path_value = debug_settings.get("selection_diagnostics_path")
        if diagnostics_path_value:
            diagnostics_path = _path(
                diagnostics_path_value, name="debug.selection_diagnostics_path"
            )
            for diagnostic_record in test_records:
                _write_selection_diagnostics(
                    diagnostics_path,
                    record=diagnostic_record,
                    settings=settings,
                    model=model,
                    scaler=scaler,
                    positive_threshold=positive_threshold,
                    fold_metadata={
                        **fold_metadata,
                        "diagnostic_trial_number": debug_settings.get("selection_diagnostics_trial", ""),
                    },
                )
        metrics = _metrics(actual, predicted)
        threshold_sweep = _threshold_sweep(actual, probabilities) if threshold_sweep_enabled else None
        if threshold_sweep is not None:
            fold_threshold_sweeps.append(threshold_sweep)
        pooled_actual.append(actual)
        pooled_predicted.append(predicted)
        predictions_path = fold_root / f"fold_{fold_index + 1:02d}_{_experiment_slug(test_group)}_predictions.json"
        fold_result: dict[str, Any] = {
            "fold": fold_index + 1,
            "test_group": test_group,
            "outer_training_groups": list(outer_train_groups),
            "inner_training_groups": list(inner_train_groups),
            "inner_validation_group": inner_validation_group,
            "selected_epochs": selected_epochs,
            "checkpoint": str(fold_checkpoint),
            "test_records": [r.id for r in test_records],
            "train_windows": int(len(outer_X)),
            "test_windows": int(len(test_X)),
            "metrics": metrics,
            "refit": refit_summary,
        }
        if threshold_sweep is not None:
            fold_result["threshold_sweep"] = threshold_sweep
        predictions_path.parent.mkdir(parents=True, exist_ok=True)
        predictions_payload = {
            **fold_metadata,
            "fold": fold_index + 1,
            "selected_epochs": selected_epochs,
            "checkpoint": str(fold_checkpoint),
            "metrics": metrics,
            "predictions": [
                {
                    "record_id": test_window_records[index][0],
                    "window_index": test_window_records[index][1],
                    "actual_pu_label": int(actual[index]),
                    "predicted_pu_label": int(predicted[index]),
                    "p_probability": float(probabilities[index]),
                }
                for index in range(len(actual))
            ],
        }
        _write_json(predictions_path, predictions_payload)
        fold_result["predictions_path"] = str(predictions_path)
        folds.append(fold_result)

        log_msg = (
            f"LOVO fold {fold_index + 1} complete: group={test_group} "
            f"windows={metrics['windows']} "
            f"P_recall={metrics['positive_recall_on_P']} "
            f"U_selected={metrics['selected_fraction_of_U']}"
        )

        # Let the logger do its job if INFO is enabled
        logger.info(log_msg)

        # If the logger is silenced at WARNING or above, print directly to the console
        if logger.getEffectiveLevel() >= logging.WARNING:
            print(log_msg)
    all_actual = np.concatenate(pooled_actual)
    all_predicted = np.concatenate(pooled_predicted)
    pooled_metrics = _metrics(all_actual, all_predicted)
    positive_recall_values = [
        f["metrics"]["positive_recall_on_P"] for f in folds
        if f["metrics"]["positive_recall_on_P"] is not None
    ]
    u_selection_values = [
        f["metrics"]["selected_fraction_of_U"] for f in folds
        if f["metrics"]["selected_fraction_of_U"] is not None
    ]
    aggregation: dict[str, Any] = {
        "micro": pooled_metrics,
        "macro_group_mean": {
            "positive_recall_on_P": float(np.mean(positive_recall_values)) if positive_recall_values else None,
            "selected_fraction_of_U": float(np.mean(u_selection_values)) if u_selection_values else None,
        },
        "macro_group_mean_definition": "Unweighted arithmetic mean of per-held-out-group PU selection statistics; U is not treated as negative.",
    }
    if threshold_sweep_enabled:
        macro_sweep: list[dict[str, Any]] = []
        for index, threshold_row in enumerate(fold_threshold_sweeps[0] if fold_threshold_sweeps else []):
            recalls_at_threshold = [sweep[index]["positive_recall_on_P"] for sweep in fold_threshold_sweeps if sweep[index]["positive_recall_on_P"] is not None]
            u_rates_at_threshold = [sweep[index]["selected_fraction_of_U"] for sweep in fold_threshold_sweeps if sweep[index]["selected_fraction_of_U"] is not None]
            mean_recall = float(np.mean(recalls_at_threshold)) if recalls_at_threshold else None
            mean_u_rate = float(np.mean(u_rates_at_threshold)) if u_rates_at_threshold else None
            macro_sweep.append({
                "threshold": threshold_row["threshold"],
                "positive_recall_on_P": mean_recall,
                "selected_fraction_of_U": mean_u_rate,
                "score": float(mean_recall * (1.0 - mean_u_rate)) if mean_recall is not None and mean_u_rate is not None else None,
            })
        aggregation["threshold_sweep"] = macro_sweep
        aggregation["threshold_sweep_definition"] = "Diagnostic only: unweighted mean of per-fold recall and U-selection metrics at each threshold; score is mean recall * (1 - mean U selection)."

    result: dict[str, Any] = {
        "experiment_name": experiment_name,
        "evaluation": "proxy_group_holdout" if test_groups is not None else "nested_leave_one_group_out",
        "manifest": str(Path(manifest_path).expanduser().resolve()),
        "groups": list(groups),
        "training_groups": list(group_splits[0][1]) if test_groups is not None else None,
        "fold_count": len(folds),
        "training_config": asdict(training_config),
        "data_config": data_config,
        "aggregation": aggregation,
        "folds": folds,
    }
    result_path = fold_root / f"{slug}_lovo_summary.json"
    _write_json(result_path, result)
    result["results_path"] = str(result_path)
    return result


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()