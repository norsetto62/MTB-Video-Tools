"""Low-level LightGBM LambdaRank fitting and validation."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from autocut.dataset.windows import LIGHTGBM_TABULAR_FEATURE_NAMES
from autocut.ml.lightgbm.model import LightGBMConfig, LightGBMRanker
from autocut.ml.lightgbm.checkpoint import save_lightgbm_checkpoint

# LightGBM ranking training for the tabular representation.
# ---------------------------------------------------------------------------



def _validate_lightgbm_data(
    X: np.ndarray,
    y: np.ndarray,
    *,
    name: str,
) -> tuple[np.ndarray, np.ndarray]:
    try:
        features = np.asarray(X, dtype=np.float32)
        labels_raw = np.asarray(y)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} features and relevance labels must be numeric arrays.") from exc
    if features.ndim != 2 or features.shape[0] == 0 or features.shape[1] != len(LIGHTGBM_TABULAR_FEATURE_NAMES):
        raise ValueError(f"{name} X must have shape (N, {len(LIGHTGBM_TABULAR_FEATURE_NAMES)}).")
    if labels_raw.ndim != 1 or labels_raw.size != len(features):
        raise ValueError(f"{name} y must be a 1D vector matching X rows.")
    if not np.isfinite(features).all():
        raise ValueError(f"{name} X must contain only finite values.")
    try:
        numeric_labels = np.asarray(labels_raw, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} relevance labels must be integers 0..3.") from exc
    if not np.isfinite(numeric_labels).all() or not np.isin(numeric_labels, [0, 1, 2, 3]).all():
        raise ValueError(f"{name} relevance labels must contain only integer grades 0, 1, 2, or 3.")
    return features, numeric_labels.astype(np.int32)


def _validate_rank_groups(
    group_sizes: Sequence[int] | None,
    row_count: int,
    *,
    name: str,
) -> list[int]:
    if group_sizes is None:
        raise ValueError(f"{name} group sizes are required for LambdaRank.")
    values = list(group_sizes)
    if not values or any(
        isinstance(size, bool) or not isinstance(size, (int, np.integer)) or int(size) <= 0
        for size in values
    ):
        raise ValueError(f"{name} group sizes must be a non-empty list of positive integers.")
    result = [int(size) for size in values]
    if sum(result) != row_count:
        raise ValueError(
            f"{name} group sizes sum to {sum(result)} rows but the dataset contains {row_count}."
        )
    return result


def train_lightgbm(
    train_X: np.ndarray,
    train_y: np.ndarray,
    *,
    train_group_sizes: Sequence[int] | None = None,
    val_X: np.ndarray | None = None,
    val_y: np.ndarray | None = None,
    val_group_sizes: Sequence[int] | None = None,
    val_timestamps: np.ndarray | None = None,
    val_video_durations: Sequence[float] | None = None,
    val_annotations_by_video: Sequence[Sequence[tuple[float, float, int]]] | None = None,
    metric_config: dict[str, float] | None = None,
    config: LightGBMConfig | None = None,
    checkpoint_path: str | Path | None = None,
    data_config: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    early_stopping: bool = True,
    collect_evaluation_history: bool = True,
) -> tuple[LightGBMRanker, dict[str, Any]]:
    """Train LambdaRank on ordinal relevance labels grouped by query/video.

    Each group-size entry describes the consecutive rows for one video. Ranking
    is learned within videos, not across unrelated videos with different
    annotation coverage or riding contexts.
    """
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise ImportError(
            'LightGBM is required for this training path. Install it with pip install -e ".[ml]".'
        ) from exc
    config = config or LightGBMConfig()
    if not isinstance(config, LightGBMConfig):
        raise TypeError("config must be a LightGBMConfig.")
    if config.task != "train":
        raise ValueError("LightGBMConfig.task must be 'train' for train_lightgbm().")
    train_features, train_labels = _validate_lightgbm_data(train_X, train_y, name="training")
    train_groups = _validate_rank_groups(train_group_sizes, len(train_features), name="training")
    if (val_X is None) != (val_y is None):
        raise ValueError("val_X and val_y must be supplied together.")
    if (val_X is None) != (val_group_sizes is None):
        raise ValueError("val_group_sizes must be supplied exactly when validation data is supplied.")
    evals_result: dict[str, Any] = {}
    valid_sets = None
    valid_names = None
    callbacks = []
    candidate_event_eval = None
    has_validation_events = False
    if val_X is not None:
        if val_timestamps is None:
            raise ValueError("val_timestamps are required for candidate-level event validation.")
        from autocut.ml.ranking_metrics import make_candidate_event_recall_evaluator
        valid_features, valid_labels = _validate_lightgbm_data(val_X, val_y, name="validation")
        valid_groups = _validate_rank_groups(val_group_sizes, len(valid_features), name="validation")
        validation_timestamps = np.asarray(val_timestamps, dtype=np.float64)
        if validation_timestamps.shape != (len(valid_features), 2):
            raise ValueError("val_timestamps must have shape (N, 2) aligned with val_X.")
        if val_video_durations is None or val_annotations_by_video is None:
            raise ValueError("Validation video durations and annotation intervals are required for candidate event metrics.")
        if len(val_video_durations) != len(valid_groups) or len(val_annotations_by_video) != len(valid_groups):
            raise ValueError("Validation video metadata must align with validation groups.")
        has_validation_events = any(
            int(label) == 3
            for video_annotations in val_annotations_by_video
            for _, _, label in video_annotations
        )
        metric_values = {
            "max_gap_seconds": 2.0,
            "min_clip_seconds": 3.0,
            "padding_seconds": 2.0,
            **(metric_config or {}),
        }
        candidate_event_eval = (
            make_candidate_event_recall_evaluator(
                validation_timestamps, valid_groups, val_video_durations,
                val_annotations_by_video, **metric_values,
            )
            if has_validation_events and collect_evaluation_history else None
        )
        if valid_features.shape[1] != train_features.shape[1]:
            raise ValueError("Training and validation feature dimensions must match.")
        train_dataset = lgb.Dataset(
            train_features, label=train_labels, group=train_groups,
            feature_name=list(LIGHTGBM_TABULAR_FEATURE_NAMES), free_raw_data=False,
        )
        valid_dataset = lgb.Dataset(
            valid_features, label=valid_labels, group=valid_groups,
            feature_name=list(LIGHTGBM_TABULAR_FEATURE_NAMES),
            reference=train_dataset, free_raw_data=False,
        )
        # The custom evaluator is bound to validation video metadata; evaluating
        # the training Dataset with that callback would misalign its rows.
        valid_sets, valid_names = ([valid_dataset], ["validation"]) if has_validation_events and collect_evaluation_history else (None, None)
        if has_validation_events and collect_evaluation_history:
            callbacks.extend([
                lgb.record_evaluation(evals_result),
                lgb.log_evaluation(period=10),
            ])
            if early_stopping:
                callbacks.insert(0, lgb.early_stopping(
                    config.early_stopping_rounds, first_metric_only=True, verbose=False
                ))
    else:
        train_dataset = lgb.Dataset(
            train_features, label=train_labels, group=train_groups,
            feature_name=list(LIGHTGBM_TABULAR_FEATURE_NAMES), free_raw_data=False,
        )
    params = config.to_params()
    if val_X is not None:
        # Disable LightGBM's native NDCG: candidate event recall is the sole
        # validation metric and therefore the early-stopping criterion.
        params["metric"] = "None"
    booster = lgb.train(
        params, train_dataset, num_boost_round=config.num_iterations,
        valid_sets=valid_sets, valid_names=valid_names, callbacks=callbacks,
        feval=candidate_event_eval,
    )
    model = LightGBMRanker(booster, feature_names=LIGHTGBM_TABULAR_FEATURE_NAMES)
    best_iteration = int(booster.best_iteration or config.num_iterations)
    summary: dict[str, Any] = {
        "objective": config.objective,
        "ranking_metric": "candidate_event_recall_at_10_v2" if val_X is not None and has_validation_events else None,
        "selection_metric": "candidate_event_recall_at_10_v2" if val_X is not None and has_validation_events else None,
        "validation_has_class3_events": bool(has_validation_events) if val_X is not None else None,
        "best_iteration": best_iteration,
        "configured_max_iterations": config.num_iterations,
        "early_stopping": bool(early_stopping) if val_X is not None and has_validation_events else False,
        "early_stopping_rounds": config.early_stopping_rounds if val_X is not None and has_validation_events and early_stopping else None,
        "training_relevance_grades": sorted(int(x) for x in np.unique(train_labels)),
        "validation_relevance_grades": (
            sorted(int(x) for x in np.unique(valid_labels)) if val_X is not None else None
        ),
        "training_group_sizes": train_groups,
        "validation_group_sizes": valid_groups if val_X is not None else None,
        "evaluation_history": evals_result,
        "evaluation_history_collected": bool(collect_evaluation_history and has_validation_events),
        "feature_count": len(LIGHTGBM_TABULAR_FEATURE_NAMES),
        "training_rows": int(len(train_features)),
        "validation_rows": int(len(val_X)) if val_X is not None else 0,
    }
    if checkpoint_path is not None:
        payload = save_lightgbm_checkpoint(
            checkpoint_path, model,
            feature_names=LIGHTGBM_TABULAR_FEATURE_NAMES,
            training_config=config.__dict__,
            data_config=data_config,
            metadata={**(metadata or {}), **summary},
        )
        summary["checkpoint_path"] = str(checkpoint_path)
        summary["checkpoint_version"] = payload["checkpoint_version"]
    return model, summary
