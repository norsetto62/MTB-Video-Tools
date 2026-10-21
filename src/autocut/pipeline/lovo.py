"""Leakage-safe leave-one-manifest-group-out cross-validation."""

from __future__ import annotations

from dataclasses import asdict, replace
import json
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .training import _combine_records, _experiment_slug, _data_values, _path, _section, _training_config
from ..dataset.manifest import get_unique_groups, load_manifest
from ..ml.model import MTBClassifier
from ..ml.scaling import FeatureScaler
from ..ml.training import targets_to_classes, train_fixed_epochs, train_single_fold

logger = logging.getLogger(__name__)


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    """Report PU selection statistics without treating unlabeled windows as negative."""
    positive = actual == 2
    unlabeled = actual != 2
    predicted_positive = predicted == 2
    return {
        "windows": int(len(actual)),
        "positive_windows": int(np.sum(positive)),
        "unlabeled_windows": int(np.sum(unlabeled)),
        "positive_recall_on_P": float(np.mean(predicted_positive[positive])) if np.any(positive) else None,
        "selected_fraction_of_U": float(np.mean(predicted_positive[unlabeled])) if np.any(unlabeled) else None,
        "predicted_positive_fraction": float(np.mean(predicted_positive)) if len(actual) else 0.0,
    }

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
    good_threshold: float = 0.50,
) -> tuple[np.ndarray, np.ndarray]:
    """Return binary predictions and unmodified (bad, good) probabilities."""
    if isinstance(good_threshold, bool) or not isinstance(good_threshold, (int, float)) or not np.isfinite(good_threshold) or not 0.0 <= good_threshold <= 1.0:
        raise ValueError("good_threshold must be between 0 and 1.")
    transformed = scaler.transform(windows)
    tensor = torch.from_numpy(np.ascontiguousarray(transformed, dtype=np.float32))
    batches: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(tensor), 256):
            batches.append(torch.softmax(model(tensor[start:start + 256]), dim=1).cpu().numpy())
    probabilities = np.concatenate(batches, axis=0)
    predictions = (probabilities[:, 1] >= float(good_threshold)).astype(np.int64)
    return predictions, probabilities


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
    _manifest, sample_fps, processing_width, window_duration, window_stride = _data_values(settings)
    training_config = _training_config(settings)
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
        "window": window_duration,
        "stride": window_stride,
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
        good_threshold = training_config.c2_threshold
        predicted, probabilities = _predict(model, scaler, test_X, good_threshold=good_threshold)
        metrics = _metrics(actual, predicted)
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
                    "actual_class": int(actual[index]),
                    "predicted_class": int(predicted[index]),
                    "probabilities": {str(i): float(probabilities[index, i]) for i in range(3)},
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
    result: dict[str, Any] = {
        "experiment_name": experiment_name,
        "evaluation": "proxy_group_holdout" if test_groups is not None else "nested_leave_one_group_out",
        "manifest": str(Path(manifest_path).expanduser().resolve()),
        "groups": list(groups),
        "training_groups": list(group_splits[0][1]) if test_groups is not None else None,
        "fold_count": len(folds),
        "training_config": asdict(training_config),
        "data_config": data_config,
        "aggregation": {
            "micro": pooled_metrics,
            "macro_group_mean": {
                "positive_recall_on_P": float(np.mean([f["metrics"]["positive_recall_on_P"] for f in folds if f["metrics"]["positive_recall_on_P"] is not None])),
                "selected_fraction_of_U": float(np.mean([f["metrics"]["selected_fraction_of_U"] for f in folds if f["metrics"]["selected_fraction_of_U"] is not None])),
            },
            "macro_group_mean_definition": "Unweighted arithmetic mean of per-held-out-group PU selection statistics; U is not treated as negative.",
        },
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