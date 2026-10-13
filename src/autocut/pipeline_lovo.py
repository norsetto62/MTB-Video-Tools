"""Leakage-safe leave-one-manifest-group-out cross-validation."""

from __future__ import annotations

from dataclasses import asdict, replace
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .config_loader import TrainingRunConfig
from .dataset.manifest import get_unique_groups, load_manifest
from .ml.model import MTBClassifier
from .ml.scaling import FeatureScaler
from .ml.training import targets_to_classes, train_fixed_epochs, train_single_fold
from .pipeline_training import _combine_records, _experiment_slug

logger = logging.getLogger(__name__)


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    """Return fixed-three-class metrics, including absent classes as zero F1."""
    confusion = [
        [int(np.sum((actual == truth) & (predicted == guess))) for guess in range(3)]
        for truth in range(3)
    ]
    class_f1 = []
    for class_id in range(3):
        tp = confusion[class_id][class_id]
        fp = sum(confusion[row][class_id] for row in range(3) if row != class_id)
        fn = sum(confusion[class_id][col] for col in range(3) if col != class_id)
        denom = 2 * tp + fp + fn
        class_f1.append(float(2 * tp / denom) if denom else 0.0)
    return {
        "windows": int(len(actual)),
        "accuracy": float(np.mean(actual == predicted)),
        # For single-label multiclass classification, micro-F1 equals accuracy.
        "micro_f1": float(np.mean(actual == predicted)),
        "macro_f1": float(np.mean(class_f1)),
        "class_f1": {str(i): class_f1[i] for i in range(3)},
        "confusion_matrix": confusion,
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


def _predict(
    model: MTBClassifier,
    scaler: FeatureScaler,
    windows: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    transformed = scaler.transform(windows)
    tensor = torch.from_numpy(np.ascontiguousarray(transformed, dtype=np.float32))
    probability_batches: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(tensor), 256):
            batch = tensor[start : start + 256]
            probability_batches.append(
                torch.softmax(model(batch), dim=1).cpu().numpy()
            )
    probabilities = np.concatenate(probability_batches, axis=0)
    return probabilities.argmax(axis=1).astype(np.int64), probabilities


def run_lovo(
    manifest_path: str | Path,
    settings: TrainingRunConfig,
) -> dict[str, Any]:
    """Run deterministic nested leave-one-group-out CV and persist fold artifacts.

    The outer held-out group is excluded from inner epoch selection, scaler
    fitting, and final refit. Epoch count is selected on one deterministic
    inner validation group from the outer-training groups, then the model and
    scaler are refit on every outer-training group for exactly that many epochs.
    """
    records = load_manifest(manifest_path, verify_files=True)
    groups = get_unique_groups(records)
    if len(groups) < 3:
        raise ValueError(
            f"LOVO requires at least 3 distinct manifest groups (found {len(groups)}): "
            "one outer test group, one inner validation group, and at least one inner training group."
        )

    slug = _experiment_slug(settings.experiment_name)
    fold_root = settings.results_dir / f"{slug}_lovo"
    checkpoint_root = settings.checkpoint_dir / f"{slug}_lovo"
    data_config = {
        "flow_fps": settings.sample_fps,
        "processing_width": settings.processing_width,
        "window": settings.common.window_duration,
        "stride": settings.common.window_stride,
    }
    folds: list[dict[str, Any]] = []
    pooled_actual: list[np.ndarray] = []
    pooled_predicted: list[np.ndarray] = []

    for fold_index, test_group in enumerate(groups):
        outer_train_groups = tuple(group for group in groups if group != test_group)
        # Rotate through the remaining sorted groups for deterministic inner validation.
        inner_validation_group = outer_train_groups[fold_index % len(outer_train_groups)]
        inner_train_groups = tuple(g for g in outer_train_groups if g != inner_validation_group)
        if not inner_train_groups:
            raise ValueError(f"LOVO fold {test_group!r} has no inner-training groups.")

        outer_train_records = tuple(r for r in records if r.group_id in outer_train_groups)
        inner_train_records = tuple(r for r in records if r.group_id in inner_train_groups)
        inner_val_records = tuple(r for r in records if r.group_id == inner_validation_group)
        test_records = tuple(r for r in records if r.group_id == test_group)

        logger.info(
            "LOVO fold %d/%d: test=%s, inner validation=%s",
            fold_index + 1, len(groups), test_group, inner_validation_group,
        )
        inner_X, inner_y = _combine_records(inner_train_records, settings)
        inner_val_X, inner_val_y = _combine_records(inner_val_records, settings)
        selection_config = replace(settings.training, seed=settings.training.seed + fold_index)
        _, _, selection = train_single_fold(
            inner_X,
            inner_y,
            inner_val_X,
            inner_val_y,
            selection_config,
            fold_video_id=f"inner-validation:{inner_validation_group}",
            data_config=data_config,
        )
        selected_epochs = int(selection["best_epoch"])
        if selected_epochs < 1:
            raise RuntimeError(f"Inner epoch selection returned invalid epoch {selected_epochs}.")

        outer_X, outer_y = _combine_records(outer_train_records, settings)
        # Keep record boundaries for interpretable out-of-fold predictions.
        test_parts = [
            (record, *_combine_records((record,), settings))
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
            "experiment_name": settings.experiment_name,
            "evaluation": "nested_leave_one_group_out",
            "outer_test_group": test_group,
            "outer_training_groups": list(outer_train_groups),
            "inner_validation_group": inner_validation_group,
            "inner_training_groups": list(inner_train_groups),
            "manifest": str(Path(manifest_path).expanduser().resolve()),
        }
        final_config = replace(settings.training, seed=settings.training.seed + fold_index)
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
        predicted, probabilities = _predict(model, scaler, test_X)
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
        logger.info(
            "LOVO fold %d complete: group=%s windows=%d accuracy=%.4f macro_f1=%.4f",
            fold_index + 1, test_group, metrics["windows"], metrics["accuracy"], metrics["macro_f1"],
        )

    all_actual = np.concatenate(pooled_actual)
    all_predicted = np.concatenate(pooled_predicted)
    pooled_metrics = _metrics(all_actual, all_predicted)
    result: dict[str, Any] = {
        "experiment_name": settings.experiment_name,
        "evaluation": "nested_leave_one_group_out",
        "manifest": str(Path(manifest_path).expanduser().resolve()),
        "groups": list(groups),
        "fold_count": len(folds),
        "training_config": asdict(settings.training),
        "data_config": data_config,
        "aggregation": {
            "micro": pooled_metrics,
            "macro_group_mean": {
                "accuracy": float(np.mean([f["metrics"]["accuracy"] for f in folds])),
                "micro_f1": float(np.mean([f["metrics"]["micro_f1"] for f in folds])),
                "macro_f1": float(np.mean([f["metrics"]["macro_f1"] for f in folds])),
            },
            "macro_group_mean_definition": "Unweighted arithmetic mean of per-held-out-group scores.",
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
