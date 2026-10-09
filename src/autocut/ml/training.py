"""Training utilities for the three-class AutoCut MTB interest model.

Dataset discovery and LOVO fold construction deliberately live outside this
module. This module accepts one already-separated train/validation fold.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from autocut.ml.checkpoint import save_checkpoint
from autocut.ml.model import MTBClassifier
from autocut.ml.scaling import FeatureScaler


@dataclass(frozen=True)
class TrainingConfig:
    """Validated hyperparameters for one training fold."""

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
    class_weights: tuple[float, float, float] = (0.2, 1.0, 6.5)

    def __post_init__(self) -> None:
        for name in ("epochs", "batch_size", "hidden_dim", "classifier_mid_dim"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        if isinstance(self.patience, bool) or not isinstance(self.patience, int) or self.patience < 0:
            raise ValueError("patience must be a non-negative integer.")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed must be an integer.")
        if isinstance(self.num_classes, bool) or not isinstance(self.num_classes, int) or self.num_classes != 3:
            raise ValueError("num_classes must be the integer 3 for the established classifier contract.")
        for name, minimum, inclusive in (
            ("lr", 0.0, False),
            ("weight_decay", 0.0, True),
            ("min_delta", 0.0, True),
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be a finite number.")
            if not math.isfinite(float(value)) or (
                float(value) < minimum if inclusive else float(value) <= minimum
            ):
                qualifier = "non-negative" if inclusive else "positive"
                raise ValueError(f"{name} must be finite and {qualifier}.")
        if isinstance(self.dropout, bool) or not isinstance(self.dropout, (int, float)):
            raise ValueError("dropout must be a number in the range [0, 1).")
        if not math.isfinite(float(self.dropout)) or not 0.0 <= float(self.dropout) < 1.0:
            raise ValueError("dropout must be a number in the range [0, 1).")
        if not isinstance(self.use_scaling, bool):
            raise ValueError("use_scaling must be a bool.")
        if not isinstance(self.class_weights, (tuple, list)) or len(self.class_weights) != 3:
            raise ValueError("class_weights must contain exactly three positive values.")
        weights = tuple(self.class_weights)
        for weight in weights:
            if (
                isinstance(weight, bool)
                or not isinstance(weight, (int, float))
                or not math.isfinite(float(weight))
                or float(weight) <= 0.0
            ):
                raise ValueError("class_weights must contain exactly three positive finite values.")
        object.__setattr__(self, "class_weights", tuple(float(w) for w in weights))
        object.__setattr__(self, "lr", float(self.lr))
        object.__setattr__(self, "weight_decay", float(self.weight_decay))
        object.__setattr__(self, "min_delta", float(self.min_delta))
        object.__setattr__(self, "dropout", float(self.dropout))


def targets_to_classes(targets: np.ndarray) -> np.ndarray:
    """Map continuous AutoCut targets to classes 0, 1, and 2.

    Values below 0.5 map to class 0; [0.5, 2.5) maps to class 1;
    values greater than or equal to 2.5 map to class 2.
    """
    try:
        values = np.asarray(targets, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("targets must be a numeric array.") from exc
    if values.ndim != 1 or values.size == 0:
        raise ValueError("targets must be a non-empty one-dimensional array.")
    if not np.isfinite(values).all():
        raise ValueError("targets must contain only finite values.")
    if np.any((values < 0.0) | (values > 3.0)):
        raise ValueError("targets must lie within the continuous range [0, 3].")
    classes = np.ones(values.shape, dtype=np.int64)
    classes[values < 0.5] = 0
    classes[values >= 2.5] = 2
    return classes


def _validate_windows(windows: np.ndarray, *, name: str, feature_dim: int = 54) -> np.ndarray:
    try:
        values = np.asarray(windows, dtype=np.float32)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be convertible to a float32 array.") from exc
    if values.ndim != 3:
        raise ValueError(f"{name} must have shape (N, 8, {feature_dim}).")
    if values.shape[0] == 0 or values.shape[1] == 0 or values.shape[2] != feature_dim:
        raise ValueError(f"{name} must have non-empty shape (N, 8, {feature_dim}).")
    if values.shape[1] != 8:
        raise ValueError(f"{name} must contain exactly 8 time steps per window.")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} must contain only finite values.")
    return values


def _macro_f1(predicted: np.ndarray, actual: np.ndarray, *, num_classes: int = 3) -> float:
    scores = []
    for class_id in range(num_classes):
        tp = int(np.sum((predicted == class_id) & (actual == class_id)))
        fp = int(np.sum((predicted == class_id) & (actual != class_id)))
        fn = int(np.sum((predicted != class_id) & (actual == class_id)))
        denominator = 2 * tp + fp + fn
        scores.append((2.0 * tp / denominator) if denominator else 0.0)
    return float(np.mean(scores))


def _evaluate(
    model: MTBClassifier,
    X: torch.Tensor,
    y: torch.Tensor,
    criterion: nn.Module,
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluate without enabling autograd or changing model parameters."""
    model.eval()
    total_loss = 0.0
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            batch_X = X[start : start + batch_size].to(device)
            batch_y = y[start : start + batch_size].to(device)
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            total_loss += float(loss.item()) * len(batch_X)
            predictions.append(logits.argmax(dim=1).cpu().numpy())
            targets.append(batch_y.cpu().numpy())
    pred = np.concatenate(predictions)
    actual = np.concatenate(targets)
    return total_loss / len(X), _macro_f1(pred, actual)


def train_single_fold(
    train_windows: np.ndarray,
    train_targets: np.ndarray,
    val_windows: np.ndarray,
    val_targets: np.ndarray,
    config: TrainingConfig | None = None,
    *,
    checkpoint_path: str | Path | None = None,
    fold_video_id: str | None = None,
    data_config: dict[str, Any] | None = None,
) -> tuple[MTBClassifier, FeatureScaler, dict[str, Any]]:
    """Train one preconstructed fold and restore its best validation weights.

    The scaler is fitted exclusively on training windows. Validation loss
    selects the best epoch; macro F1 is recorded as a secondary metric.
    If checkpoint_path is supplied, the best model and fitted scaler are saved
    through the existing versioned checkpoint API.
    """
    config = config or TrainingConfig()
    if not isinstance(config, TrainingConfig):
        raise TypeError("config must be a TrainingConfig.")
    train_X = _validate_windows(train_windows, name="train_windows")
    val_X = _validate_windows(val_windows, name="val_windows")
    train_y = targets_to_classes(train_targets)
    val_y = targets_to_classes(val_targets)
    if len(train_X) != len(train_y):
        raise ValueError("train_windows and train_targets must contain the same number of samples.")
    if len(val_X) != len(val_y):
        raise ValueError("val_windows and val_targets must contain the same number of samples.")
    if train_X.shape[1:] != val_X.shape[1:]:
        raise ValueError("Training and validation windows must have matching time and feature dimensions.")
    if len(np.unique(train_y)) < 2:
        raise ValueError("Training targets must contain at least two distinct classes.")
    if fold_video_id is not None and (not isinstance(fold_video_id, str) or not fold_video_id.strip()):
        raise ValueError("fold_video_id must be a non-empty string or None.")
    if data_config is not None and not isinstance(data_config, dict):
        raise ValueError("data_config must be a dictionary or None.")

    scaler = FeatureScaler(enabled=config.use_scaling)
    transformed_train = scaler.fit_transform(train_X)
    transformed_val = scaler.transform(val_X)
    train_tensor = torch.from_numpy(np.ascontiguousarray(transformed_train, dtype=np.float32))
    train_label_tensor = torch.from_numpy(train_y.astype(np.int64, copy=False))
    val_tensor = torch.from_numpy(np.ascontiguousarray(transformed_val, dtype=np.float32))
    val_label_tensor = torch.from_numpy(val_y.astype(np.int64, copy=False))

    # fork_rng keeps the caller's CPU RNG state unchanged while making model
    # initialization, dropout, and batch shuffling reproducible for this fold.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config.seed)
        model = MTBClassifier(
            feature_dim=train_X.shape[2],
            hidden_dim=config.hidden_dim,
            classifier_mid_dim=config.classifier_mid_dim,
            num_classes=config.num_classes,
            dropout=config.dropout,
        )
        device = torch.device("cpu")
        model.to(device)
        class_weights = torch.tensor(config.class_weights, dtype=torch.float32, device=device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )
        generator = torch.Generator(device="cpu")
        generator.manual_seed(config.seed)
        loader = DataLoader(
            TensorDataset(train_tensor, train_label_tensor),
            batch_size=config.batch_size,
            shuffle=True,
            generator=generator,
        )

        best_state: dict[str, torch.Tensor] | None = None
        best_loss = math.inf
        best_f1 = 0.0
        best_epoch = 0
        epochs_without_improvement = 0
        history: list[dict[str, float | int]] = []

        for epoch in range(1, config.epochs + 1):
            model.train()
            train_loss_total = 0.0
            for batch_X, batch_y in loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(batch_X)
                loss = criterion(logits, batch_y)
                if not torch.isfinite(loss).item():
                    raise RuntimeError(f"Non-finite training loss at epoch {epoch}.")
                loss.backward()
                optimizer.step()
                train_loss_total += float(loss.item()) * len(batch_X)

            val_loss, val_f1 = _evaluate(
                model,
                val_tensor,
                val_label_tensor,
                criterion,
                batch_size=config.batch_size,
                device=device,
            )
            train_loss = train_loss_total / len(train_tensor)
            history.append({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_macro_f1": val_f1,
            })

            if not math.isfinite(val_loss) or not math.isfinite(val_f1):
                raise RuntimeError(f"Non-finite validation metric at epoch {epoch}.")
            if best_state is None or val_loss < best_loss - config.min_delta:
                best_loss = val_loss
                best_f1 = val_f1
                best_epoch = epoch
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= max(1, config.patience):
                    break

        if best_state is None:
            raise RuntimeError("Training completed without a valid best model state.")
        model.load_state_dict(best_state, strict=True)
        model.eval()

        summary: dict[str, Any] = {
            "history": history,
            "best_epoch": best_epoch,
            "best_val_loss": float(best_loss),
            "best_val_f1": float(best_f1),
            "epochs_completed": len(history),
            "stopped_early": len(history) < config.epochs,
            "class_counts": {
                str(class_id): int(np.sum(train_y == class_id))
                for class_id in range(config.num_classes)
            },
            "class_weights": list(config.class_weights),
            "fold_video_id": fold_video_id,
        }

        if checkpoint_path is not None:
            save_checkpoint(
                checkpoint_path,
                model,
                scaler=scaler,
                training_config=asdict(config),
                data_config=data_config,
                epoch=best_epoch,
                best_metric=best_loss,
                metadata={
                    "best_val_loss": float(best_loss),
                    "best_val_f1": float(best_f1),
                    "class_weights": list(config.class_weights),
                    "class_counts": summary["class_counts"],
                    "fold_video_id": fold_video_id,
                },
            )
            summary["checkpoint_path"] = str(checkpoint_path)

    return model, scaler, summary
