"""Single-logit non-negative PU training for the AutoCut MTB interest model.

Pure score-3 windows are P. Pure score-0/1/2 windows are U. Conflicting and
uncovered windows are expected to have been removed by target construction.
U is never treated as a confirmed negative set.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import logging
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from autocut.ml.checkpoint import save_checkpoint
from autocut.ml.model import MTBClassifier
from autocut.ml.scaling import FeatureScaler
from autocut.motion.engineered import MODEL_FEATURE_NAMES

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainingConfig:
    """Validated hyperparameters for one P/U training fold."""

    epochs: int = 15
    lr: float = 1e-3
    batch_size: int = 32
    hidden_dim: int = 128
    classifier_mid_dim: int = 64
    dropout: float = 0.4
    use_scaling: bool = True
    scaler_strategy: str = "robust"
    patience: int = 10
    min_delta: float = 1e-4
    weight_decay: float = 1e-2
    seed: int = 42
    positive_prior: float = 0.1

    def __post_init__(self) -> None:
        for name in ("epochs", "batch_size", "hidden_dim", "classifier_mid_dim"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        if self.batch_size < 2:
            raise ValueError("batch_size must be at least 2 for PU batches.")
        if isinstance(self.patience, bool) or not isinstance(self.patience, int) or self.patience < 0:
            raise ValueError("patience must be a non-negative integer.")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed must be an integer.")
        for name, minimum, inclusive in (
            ("lr", 0.0, False), ("weight_decay", 0.0, True), ("min_delta", 0.0, True),
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be a finite number.")
            if not math.isfinite(float(value)) or (float(value) < minimum if inclusive else float(value) <= minimum):
                raise ValueError(f"{name} must be finite and {'non-negative' if inclusive else 'positive'}.")
        if isinstance(self.dropout, bool) or not isinstance(self.dropout, (int, float)) or not math.isfinite(float(self.dropout)) or not 0 <= float(self.dropout) < 1:
            raise ValueError("dropout must be finite and in the range [0, 1).")
        if not isinstance(self.use_scaling, bool):
            raise ValueError("use_scaling must be a bool.")
        if self.scaler_strategy not in {"standard", "robust"}:
            raise ValueError("scaler_strategy must be 'standard' or 'robust'.")
        if isinstance(self.positive_prior, bool) or not isinstance(self.positive_prior, (int, float)) or not math.isfinite(float(self.positive_prior)) or not 0 < float(self.positive_prior) < 1:
            raise ValueError("positive_prior must be finite and strictly between 0 and 1.")
        for name in ("lr", "weight_decay", "min_delta", "dropout", "positive_prior"):
            object.__setattr__(self, name, float(getattr(self, name)))


def targets_to_classes(targets: np.ndarray) -> np.ndarray:
    """Map pure annotation scores to binary labels: P=1 for score 3, U=0 for 0/1/2."""
    try:
        values = np.asarray(targets, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("targets must be a numeric array.") from exc
    if values.ndim != 1 or values.size == 0:
        raise ValueError("targets must be a non-empty one-dimensional array.")
    if not np.isfinite(values).all() or not np.isin(values, [0.0, 1.0, 2.0, 3.0]).all():
        raise ValueError("targets must contain only finite pure scores 0, 1, 2, or 3.")
    return (values == 3.0).astype(np.int64)


def _validate_windows(windows: np.ndarray, *, name: str, feature_dim: int = 54) -> np.ndarray:
    try:
        values = np.asarray(windows, dtype=np.float32)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be convertible to a float32 array.") from exc
    if (
        values.ndim != 3
        or values.shape[0] == 0
        or values.shape[1] < 2
        or values.shape[2] != feature_dim
    ):
        raise ValueError(
            f"{name} must have non-empty shape (N, T, {feature_dim}) "
            "with at least two time steps."
        )
    if not np.isfinite(values).all():
        raise ValueError(f"{name} must contain only finite values.")
    return values


def _pu_risk(logits: torch.Tensor, labels: torch.Tensor, positive_prior: float) -> torch.Tensor:
    """Compute the requested nnPU risk from one logit per window.

    labels are 1 for P and 0 for U; logits are raw, un-sigmoided model outputs.
    """
    scores = logits.reshape(-1)
    targets = labels.reshape(-1)
    if len(scores) != len(targets):
        raise ValueError("logits and labels must contain the same number of windows.")
    positive = targets == 1
    unlabeled = targets == 0
    if not bool(positive.any()) or not bool(unlabeled.any()):
        raise ValueError("PU risk requires both pure score-3 positives (P) and score-0/1/2 unlabeled windows (U).")
    positive_logits = scores[positive]
    unlabeled_logits = scores[unlabeled]
    risk_positive = positive_prior * torch.nn.functional.softplus(-positive_logits).mean()
    risk_negative = torch.nn.functional.softplus(unlabeled_logits).mean() - positive_prior * torch.nn.functional.softplus(positive_logits).mean()
    return risk_positive + torch.clamp(risk_negative, min=0.0)


def _validate_targets_presence(labels: np.ndarray, *, name: str) -> None:
    if not np.any(labels == 1):
        raise ValueError(f"{name} must contain at least one pure score-3 positive (P) window.")
    if not np.any(labels == 0):
        raise ValueError(f"{name} must contain at least one unlabeled (U) window with score 0, 1, or 2.")


def _training_loss(logits: torch.Tensor, labels: torch.Tensor, config: TrainingConfig) -> torch.Tensor:
    """The sole training objective: non-negative PU risk, with no auxiliary loss."""
    return _pu_risk(logits, labels, config.positive_prior)


def _batch_indices(
    positive_indices: torch.Tensor,
    unlabeled_indices: torch.Tensor,
    *,
    batch_size: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample equal-sized P and U subsets, uniformly within each set."""
    p_size = max(1, batch_size // 2)
    u_size = max(1, batch_size - p_size)
    p = positive_indices[torch.randint(len(positive_indices), (p_size,), generator=generator)]
    u = unlabeled_indices[torch.randint(len(unlabeled_indices), (u_size,), generator=generator)]
    return p, u


def _evaluate(model: MTBClassifier, X: torch.Tensor, y: torch.Tensor, config: TrainingConfig) -> float:
    """Evaluate nnPU risk over all validation P and U examples."""
    model.eval()
    with torch.no_grad():
        logits = torch.cat([
            model(X[start : start + config.batch_size])
            for start in range(0, len(X), config.batch_size)
        ], dim=0)
    return float(_pu_risk(logits, y, config.positive_prior).item())


def _fit(
    train_windows: np.ndarray,
    train_targets: np.ndarray,
    config: TrainingConfig,
    *,
    epochs: int,
    val_windows: np.ndarray | None = None,
    val_targets: np.ndarray | None = None,
    checkpoint_path: str | Path | None = None,
    fold_video_id: str | None = None,
    data_config: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    early_stopping: bool = True,
) -> tuple[MTBClassifier, FeatureScaler, dict[str, Any]]:
    train_raw = np.asarray(train_windows)
    feature_dim = train_raw.shape[2] if train_raw.ndim == 3 and train_raw.shape[2] in (4, 7, 54) else 54
    train_X = _validate_windows(train_windows, name="train_windows", feature_dim=feature_dim)
    train_y = targets_to_classes(train_targets)
    if len(train_X) != len(train_y):
        raise ValueError("train_windows and train_targets must contain the same number of samples.")
    _validate_targets_presence(train_y, name="Training targets")
    if (val_windows is None) != (val_targets is None):
        raise ValueError("val_windows and val_targets must be supplied together.")
    if val_windows is not None:
        val_X = _validate_windows(val_windows, name="val_windows", feature_dim=feature_dim)
        val_y = targets_to_classes(val_targets)
        if len(val_X) != len(val_y):
            raise ValueError("val_windows and val_targets must contain the same number of samples.")
        if train_X.shape[1:] != val_X.shape[1:]:
            raise ValueError("Training and validation windows must have matching time and feature dimensions.")
        _validate_targets_presence(val_y, name="Validation targets")
    else:
        val_X = val_y = None
    if data_config is not None and not isinstance(data_config, dict):
        raise ValueError("data_config must be a dictionary or None.")
    if metadata is not None and not isinstance(metadata, dict):
        raise ValueError("metadata must be a dictionary or None.")
    if data_config is not None and "flow_fps" in data_config and "window" in data_config:
        flow_fps = data_config["flow_fps"]
        window_duration = data_config["window"]
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0
            for value in (flow_fps, window_duration)
        ):
            raise ValueError("data_config flow_fps and window must be finite positive numbers.")
        expected_steps_float = float(flow_fps) * float(window_duration)
        expected_steps = int(round(expected_steps_float))
        if not math.isclose(expected_steps_float, expected_steps, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError("window duration multiplied by flow_fps must be an integer number of feature steps.")
        if train_X.shape[1] != expected_steps:
            raise ValueError(
                f"train_windows must contain {expected_steps} time steps from "
                f"window={window_duration}s and flow_fps={flow_fps}; got {train_X.shape[1]}."
            )
        if val_X is not None and val_X.shape[1] != expected_steps:
            raise ValueError(
                f"val_windows must contain {expected_steps} time steps from "
                f"window={window_duration}s and flow_fps={flow_fps}; got {val_X.shape[1]}."
            )
    if fold_video_id is not None and (not isinstance(fold_video_id, str) or not fold_video_id.strip()):
        raise ValueError("fold_video_id must be a non-empty string or None.")

    scaler = FeatureScaler(enabled=config.use_scaling, strategy=config.scaler_strategy)
    train_scaled = scaler.fit_transform(train_X)
    train_tensor = torch.from_numpy(np.ascontiguousarray(train_scaled, dtype=np.float32))
    train_label_tensor = torch.from_numpy(train_y)
    if val_X is not None:
        val_tensor = torch.from_numpy(np.ascontiguousarray(scaler.transform(val_X), dtype=np.float32))
        val_label_tensor = torch.from_numpy(val_y)
    else:
        val_tensor = val_label_tensor = None

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config.seed)
        model = MTBClassifier(
            feature_dim=train_X.shape[2], hidden_dim=config.hidden_dim,
            classifier_mid_dim=config.classifier_mid_dim, dropout=config.dropout,
        ).to(torch.device("cpu"))
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(config.seed)
        positive_indices = torch.from_numpy(np.flatnonzero(train_y == 1))
        unlabeled_indices = torch.from_numpy(np.flatnonzero(train_y == 0))
        steps = max(1, math.ceil(max(len(positive_indices), len(unlabeled_indices)) / max(1, config.batch_size // 2)))
        best_state: dict[str, torch.Tensor] | None = None
        best_loss = math.inf
        best_epoch = 0
        epochs_without_improvement = 0
        history: list[dict[str, float | int]] = []

        for epoch in range(1, epochs + 1):
            model.train()
            loss_total = 0.0
            for _ in range(steps):
                p_idx, u_idx = _batch_indices(
                    positive_indices, unlabeled_indices, batch_size=config.batch_size,
                    generator=generator,
                )
                batch_idx = torch.cat((p_idx, u_idx))
                optimizer.zero_grad(set_to_none=True)
                logits = model(train_tensor[batch_idx])
                loss = _training_loss(logits, train_label_tensor[batch_idx], config)
                if not torch.isfinite(loss).item():
                    raise RuntimeError(f"Non-finite nnPU risk at epoch {epoch}.")
                loss.backward()
                optimizer.step()
                loss_total += float(loss.item())
            train_loss = loss_total / steps
            if val_tensor is not None:
                val_loss = _evaluate(model, val_tensor, val_label_tensor, config)
                history.append({"epoch": epoch, "train_pu_risk": train_loss, "val_pu_risk": val_loss})
                if not math.isfinite(val_loss):
                    raise RuntimeError(f"Non-finite validation PU risk at epoch {epoch}.")
                if best_state is None or val_loss < best_loss - config.min_delta:
                    best_loss, best_epoch = val_loss, epoch
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                    if early_stopping and epochs_without_improvement >= max(1, config.patience):
                        break
            else:
                history.append({"epoch": epoch, "train_pu_risk": train_loss})
        if val_tensor is not None:
            if best_state is None:
                raise RuntimeError("Training completed without a valid best model state.")
            model.load_state_dict(best_state, strict=True)
        else:
            best_epoch = epochs
        model.eval()

    p_count = int(np.sum(train_y == 1))
    u_count = int(np.sum(train_y == 0))
    summary: dict[str, Any] = {
        "history": history,
        "best_epoch": best_epoch,
        "best_val_pu_risk": float(best_loss) if val_tensor is not None else None,
        "epochs_completed": len(history),
        "stopped_early": early_stopping and len(history) < epochs,
        "positive_windows": p_count,
        "unlabeled_windows": u_count,
        "positive_prior": config.positive_prior,
        "fold_video_id": fold_video_id,
    }
    if checkpoint_path is not None:
        save_checkpoint(
            checkpoint_path, model, scaler=scaler,
            feature_names=(
                list(data_config["feature_names"])
                if data_config is not None and "feature_names" in data_config
                else (
                    ["vertical_flow_gradient", "lateral_flow_asymmetry", "expansion_index", "roll_index"]
                    if model.feature_dim == 4
                    else (list(MODEL_FEATURE_NAMES) if model.feature_dim == 7 else None)
                )
            ),
            training_config=asdict(config),
            data_config=data_config, epoch=best_epoch,
            best_metric=float(best_loss) if val_tensor is not None else None,
            metadata={
                **(metadata or {}),
                "best_val_pu_risk": float(best_loss) if val_tensor is not None else None,
                "positive_windows": p_count,
                "unlabeled_windows": u_count,
                "positive_prior": config.positive_prior,
                "fold_video_id": fold_video_id,
            },
        )
        summary["checkpoint_path"] = str(checkpoint_path)
    return model, scaler, summary


def train_single_fold(
    train_windows: np.ndarray, train_targets: np.ndarray,
    val_windows: np.ndarray, val_targets: np.ndarray,
    config: TrainingConfig | None = None, *,
    checkpoint_path: str | Path | None = None,
    fold_video_id: str | None = None,
    data_config: dict[str, Any] | None = None,
) -> tuple[MTBClassifier, FeatureScaler, dict[str, Any]]:
    """Train one fold, selecting the checkpoint by non-negative PU validation risk."""
    config = config or TrainingConfig()
    if not isinstance(config, TrainingConfig):
        raise TypeError("config must be a TrainingConfig.")
    return _fit(
        train_windows, train_targets, config, epochs=config.epochs,
        val_windows=val_windows, val_targets=val_targets,
        checkpoint_path=checkpoint_path, fold_video_id=fold_video_id,
        data_config=data_config, early_stopping=True,
    )


def train_fixed_epochs(
    train_windows: np.ndarray, train_targets: np.ndarray,
    config: TrainingConfig | None = None, *, epochs: int,
    checkpoint_path: str | Path | None = None,
    data_config: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> tuple[MTBClassifier, FeatureScaler, dict[str, Any]]:
    """Refit on all supplied P/U data for an explicit number of epochs."""
    config = config or TrainingConfig()
    if not isinstance(config, TrainingConfig):
        raise TypeError("config must be a TrainingConfig.")
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs <= 0:
        raise ValueError("epochs must be a positive integer.")
    fit_config = replace(config, epochs=epochs)
    return _fit(
        train_windows, train_targets, fit_config, epochs=epochs,
        checkpoint_path=checkpoint_path, data_config=data_config,
        metadata=metadata, early_stopping=False,
    )


# ---------------------------------------------------------------------------
# Four-class LightGBM training for the 48-column tabular representation.
# ---------------------------------------------------------------------------

from autocut.ml.model import LightGBMConfig, LightGBMRanker
from autocut.dataset.windows import LIGHTGBM_TABULAR_FEATURE_NAMES
from autocut.ml.checkpoint import save_lightgbm_checkpoint


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
    if val_X is not None:
        valid_features, valid_labels = _validate_lightgbm_data(val_X, val_y, name="validation")
        valid_groups = _validate_rank_groups(val_group_sizes, len(valid_features), name="validation")
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
        valid_sets, valid_names = [train_dataset, valid_dataset], ["train", "validation"]
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
    booster = lgb.train(
        config.to_params(), train_dataset, num_boost_round=config.num_iterations,
        valid_sets=valid_sets, valid_names=valid_names, callbacks=callbacks,
    )
    model = LightGBMRanker(booster, feature_names=LIGHTGBM_TABULAR_FEATURE_NAMES)
    best_iteration = int(booster.best_iteration or config.num_iterations)
    summary: dict[str, Any] = {
        "objective": config.objective,
        "ranking_metric": "ndcg",
        "ndcg_eval_at": list(config.ndcg_eval_at),
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
