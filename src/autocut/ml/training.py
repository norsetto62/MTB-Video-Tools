"""Three-output PU training for the AutoCut MTB interest model.

Pure annotation score 3 is the positive set P. Pure scores 0, 1, and 2 form
the unlabeled set U. The three-output classifier from main is retained:
scores 0 and 1 map to output 0 and 1, score 2 maps to output 1, and score 3
maps to output 2. PU risk treats output 2 as positive and outputs 0/1 jointly
as the unlabeled side; it never treats U as confirmed negatives.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from autocut.ml.checkpoint import save_checkpoint
from autocut.ml.model import MTBClassifier
from autocut.ml.scaling import FeatureScaler

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainingConfig:
    """Validated hyperparameters for one PU training fold."""

    epochs: int = 15
    lr: float = 1e-3
    batch_size: int = 32
    hidden_dim: int = 128
    classifier_mid_dim: int = 64
    dropout: float = 0.4
    num_classes: int = 3
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
        if isinstance(self.num_classes, bool) or not isinstance(self.num_classes, int) or self.num_classes != 3:
            raise ValueError("num_classes must be 3 for the established classifier contract.")
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
        object.__setattr__(self, "lr", float(self.lr))
        object.__setattr__(self, "weight_decay", float(self.weight_decay))
        object.__setattr__(self, "min_delta", float(self.min_delta))
        object.__setattr__(self, "dropout", float(self.dropout))
        object.__setattr__(self, "positive_prior", float(self.positive_prior))


def targets_to_classes(targets: np.ndarray) -> np.ndarray:
    """Map pure annotation scores 0..3 to the existing three output classes.

    Scores 0 and 1 map to output 0; score 2 maps to output 1; score 3 maps
    to output 2 (the confirmed positive set P).
    """
    try:
        values = np.asarray(targets, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("targets must be a numeric array.") from exc
    if values.ndim != 1 or values.size == 0:
        raise ValueError("targets must be a non-empty one-dimensional array.")
    if not np.isfinite(values).all() or not np.isin(values, [0.0, 1.0, 2.0, 3.0]).all():
        raise ValueError("targets must contain only finite pure scores 0, 1, 2, or 3.")
    classes = np.zeros(values.shape, dtype=np.int64)
    classes[values == 2.0] = 1
    classes[values == 3.0] = 2
    return classes


def _validate_windows(windows: np.ndarray, *, name: str, feature_dim: int = 54) -> np.ndarray:
    try:
        values = np.asarray(windows, dtype=np.float32)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be convertible to a float32 array.") from exc
    if values.ndim != 3 or values.shape[0] == 0 or values.shape[1] != 8 or values.shape[2] != feature_dim:
        raise ValueError(f"{name} must have non-empty shape (N, 8, {feature_dim}).")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} must contain only finite values.")
    return values


def _pu_risk(logits: torch.Tensor, classes: torch.Tensor, positive_prior: float) -> torch.Tensor:
    """Non-negative PU risk, with output 2 positive and outputs 0/1 unlabeled."""
    positive = classes == 2
    unlabeled = classes != 2
    if not positive.any() or not unlabeled.any():
        raise ValueError("PU risk requires both pure score-3 positives and unlabeled score-0/1/2 windows.")
    # logit of P(class 3) versus P(not class 3), while retaining 3 output logits.
    binary_logit = logits[:, 2] - torch.logsumexp(logits[:, :2], dim=1)
    positive_logits = binary_logit[positive]
    unlabeled_logits = binary_logit[unlabeled]
    positive_risk = positive_prior * torch.nn.functional.softplus(-positive_logits).mean()
    negative_risk = torch.nn.functional.softplus(unlabeled_logits).mean() - positive_prior * torch.nn.functional.softplus(positive_logits).mean()
    return positive_risk + torch.clamp(negative_risk, min=0.0)


def _validate_targets_presence(classes: np.ndarray, *, name: str) -> None:
    if not np.any(classes == 2):
        raise ValueError(f"{name} must contain at least one pure score-3 positive (P) window.")
    if not np.any(classes != 2):
        raise ValueError(f"{name} must contain at least one unlabeled (U) window with score 0, 1, or 2.")


def _batch_indices(
    positive_indices: torch.Tensor,
    unlabeled_indices: torch.Tensor,
    *,
    batch_size: int,
    steps: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Draw class-conditional batches uniformly within P and U."""
    p_size = max(1, batch_size // 2)
    u_size = max(1, batch_size - p_size)
    p = positive_indices[torch.randint(len(positive_indices), (p_size,), generator=generator)]
    u = unlabeled_indices[torch.randint(len(unlabeled_indices), (u_size,), generator=generator)]
    return p, u


def _evaluate(model: MTBClassifier, X: torch.Tensor, y: torch.Tensor, config: TrainingConfig) -> float:
    model.eval()
    with torch.no_grad():
        logits = model(X)
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
    if fold_video_id is not None and (not isinstance(fold_video_id, str) or not fold_video_id.strip()):
        raise ValueError("fold_video_id must be a non-empty string or None.")

    scaler = FeatureScaler(enabled=config.use_scaling)
    train_scaled = scaler.fit_transform(train_X)
    train_tensor = torch.from_numpy(np.ascontiguousarray(train_scaled, dtype=np.float32))
    train_label_tensor = torch.from_numpy(train_y)
    if val_X is not None:
        val_scaled = scaler.transform(val_X)
        val_tensor = torch.from_numpy(np.ascontiguousarray(val_scaled, dtype=np.float32))
        val_label_tensor = torch.from_numpy(val_y)
    else:
        val_tensor = val_label_tensor = None

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config.seed)
        model = MTBClassifier(
            feature_dim=train_X.shape[2], hidden_dim=config.hidden_dim,
            classifier_mid_dim=config.classifier_mid_dim, num_classes=3, dropout=config.dropout,
        ).to(torch.device("cpu"))
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(config.seed)
        positive_indices = torch.from_numpy(np.flatnonzero(train_y == 2))
        unlabeled_indices = torch.from_numpy(np.flatnonzero(train_y != 2))
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
                    steps=steps, generator=generator,
                )
                batch_idx = torch.cat((p_idx, u_idx))
                optimizer.zero_grad(set_to_none=True)
                loss = _pu_risk(model(train_tensor[batch_idx]), train_label_tensor[batch_idx], config.positive_prior)
                if not torch.isfinite(loss).item():
                    raise RuntimeError(f"Non-finite PU training risk at epoch {epoch}.")
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
                    best_loss = val_loss
                    best_epoch = epoch
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

    class_counts = {str(i): int(np.sum(train_y == i)) for i in range(3)}
    summary: dict[str, Any] = {
        "history": history,
        "best_epoch": best_epoch,
        "best_val_pu_risk": float(best_loss) if val_tensor is not None else None,
        "epochs_completed": len(history),
        "stopped_early": early_stopping and len(history) < epochs,
        "class_counts": class_counts,
        "positive_prior": config.positive_prior,
        "positive_windows": class_counts["2"],
        "unlabeled_windows": class_counts["0"] + class_counts["1"],
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
                "class_counts": class_counts,
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
    return _fit(
        train_windows, train_targets, config, epochs=epochs,
        checkpoint_path=checkpoint_path, data_config=data_config,
        metadata=metadata, early_stopping=False,
    )
