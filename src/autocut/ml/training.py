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
from typing import Any

import numpy as np
import torch

from autocut.ml.checkpoint import save_checkpoint
from autocut.ml.model import MTBClassifier
from autocut.ml.scaling import FeatureScaler

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
    train_X = _validate_windows(train_windows, name="train_windows")
    train_y = targets_to_classes(train_targets)
    if len(train_X) != len(train_y):
        raise ValueError("train_windows and train_targets must contain the same number of samples.")
    _validate_targets_presence(train_y, name="Training targets")
    if (val_windows is None) != (val_targets is None):
        raise ValueError("val_windows and val_targets must be supplied together.")
    if val_windows is not None:
        val_X = _validate_windows(val_windows, name="val_windows")
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

    scaler = FeatureScaler(enabled=config.use_scaling)
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
            checkpoint_path, model, scaler=scaler, training_config=asdict(config),
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
