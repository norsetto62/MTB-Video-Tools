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
    config: LightGBMConfig | None = None,
    checkpoint_path: str | Path | None = None,
    data_config: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
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
    segment_recall_eval = None
    has_validation_events = False
    if val_X is not None:
        if val_timestamps is None:
            raise ValueError("val_timestamps are required for Segment Recall@5 validation.")
        from autocut.ml.ranking_metrics import make_segment_recall_at_5
        valid_features, valid_labels = _validate_lightgbm_data(val_X, val_y, name="validation")
        valid_groups = _validate_rank_groups(val_group_sizes, len(valid_features), name="validation")
        validation_timestamps = np.asarray(val_timestamps, dtype=np.float64)
        if validation_timestamps.shape != (len(valid_features), 2):
            raise ValueError("val_timestamps must have shape (N, 2) aligned with val_X.")
        has_validation_events = bool(np.any(valid_labels == 3))
        segment_recall_eval = (
            make_segment_recall_at_5(validation_timestamps, valid_groups)
            if has_validation_events else None
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
        # The custom evaluator is bound to validation timestamps; evaluating
        # the training Dataset with that callback would misalign its rows.
        valid_sets, valid_names = ([valid_dataset], ["validation"]) if has_validation_events else (None, None)
        if has_validation_events:
            callbacks.extend([
                lgb.early_stopping(config.early_stopping_rounds, first_metric_only=True, verbose=False),
                lgb.record_evaluation(evals_result),
                lgb.log_evaluation(period=10),
            ])
    else:
        train_dataset = lgb.Dataset(
            train_features, label=train_labels, group=train_groups,
            feature_name=list(LIGHTGBM_TABULAR_FEATURE_NAMES), free_raw_data=False,
        )
    params = config.to_params()
    if val_X is not None:
        # Disable LightGBM's native NDCG: Segment Recall@5 is the sole
        # validation metric and therefore the early-stopping criterion.
        params["metric"] = "None"
    booster = lgb.train(
        params, train_dataset, num_boost_round=config.num_iterations,
        valid_sets=valid_sets, valid_names=valid_names, callbacks=callbacks,
        feval=segment_recall_eval,
    )
    model = LightGBMRanker(booster, feature_names=LIGHTGBM_TABULAR_FEATURE_NAMES)
    best_iteration = int(booster.best_iteration or config.num_iterations)
    summary: dict[str, Any] = {
        "objective": config.objective,
        "ranking_metric": "segment_recall_at_5" if val_X is not None and has_validation_events else None,
        "selection_metric": "segment_recall_at_5" if val_X is not None and has_validation_events else None,
        "validation_has_class3_events": bool(has_validation_events) if val_X is not None else None,
        "best_iteration": best_iteration,
        "configured_max_iterations": config.num_iterations,
        "early_stopping_rounds": config.early_stopping_rounds if val_X is not None else None,
        "training_relevance_grades": sorted(int(x) for x in np.unique(train_labels)),
        "validation_relevance_grades": (
            sorted(int(x) for x in np.unique(valid_labels)) if val_X is not None else None
        ),
        "training_group_sizes": train_groups,
        "validation_group_sizes": valid_groups if val_X is not None else None,
        "evaluation_history": evals_result,
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
