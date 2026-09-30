#!/usr/bin/env python3
"""
Train a small temporal 1D CNN for the five-class MTB feature detector.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


LABELS = [
    "drop",
    "rock_garden",
    "switchback",
    "stairs",
    "technical_climb",
]


class MTBTemporalCNN(nn.Module):
    """Temporal 1D CNN with expanded dilated receptive field."""

    def __init__(self, n_features: int, n_labels: int = 5) -> None:
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(n_features, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            
            nn.Conv1d(32, 64, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            
            nn.Conv1d(64, 64, kernel_size=3, padding=4, dilation=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )

        self.pool = nn.AdaptiveMaxPool1d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(32, n_labels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input: [batch, time, features] -> Conv1d: [batch, features, time]
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = self.pool(x)
        return self.classifier(x)


def load_dataset(paths: list[Path]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load and concatenate NPZ datasets, preserving each file as a group."""
    if not paths:
        raise ValueError("at least one --dataset is required")

    all_x: list[np.ndarray] = []
    all_y: list[np.ndarray] = []
    feature_names: list[str] | None = None

    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            if not {"X", "y", "feature_names", "labels"} <= set(data.files):
                raise ValueError(f"{path}: expected X, y, feature_names and labels")

            x = np.asarray(data["X"], dtype=np.float32)
            y = np.asarray(data["y"], dtype=np.float32)
            current_features = [str(v) for v in data["feature_names"]]
            current_labels = [str(v) for v in data["labels"]]

        if x.ndim != 3:
            raise ValueError(f"{path}: X must have shape [N,T,F], got {x.shape}")
        if y.ndim != 2 or y.shape[1] != len(LABELS):
            raise ValueError(f"{path}: y must have shape [N,5], got {y.shape}")
        if len(x) != len(y):
            raise ValueError(f"{path}: X/y row counts differ")
        if current_labels != LABELS:
            raise ValueError(f"{path}: labels differ from expected order: {current_labels}")

        if feature_names is None:
            feature_names = current_features
        elif current_features != feature_names:
            raise ValueError(f"{path}: feature_names differ from other datasets")

        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError(f"{path}: X/y contains NaN or Inf")

        all_x.append(x)
        all_y.append(y)

        print(f"Loaded {path}: X={x.shape}, y={y.shape}")

    X = np.concatenate(all_x, axis=0)
    y = np.concatenate(all_y, axis=0)

    assert feature_names is not None
    return X, y, feature_names


def split_by_dataset(
    paths: list[Path],
    validation_index: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Load datasets and hold out one complete dataset for validation."""
    if len(paths) < 2:
        raise ValueError("at least two --dataset files are required for video-level validation holdout")

    if not 0 <= validation_index < len(paths):
        raise ValueError("validation index is out of range")

    train_paths = [path for i, path in enumerate(paths) if i != validation_index]
    validation_path = paths[validation_index]

    train_x, train_y, feature_names = load_dataset(train_paths)
    val_x, val_y, val_features = load_dataset([validation_path])

    if feature_names != val_features:
        raise ValueError("training and validation feature names differ")

    return train_x, train_y, val_x, val_y, feature_names


def fit_normalizer(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = X.reshape(-1, X.shape[-1]).mean(axis=0)
    std = X.reshape(-1, X.shape[-1]).std(axis=0)
    std[std < 1e-6] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def normalize(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((X - mean[None, None, :]) / std[None, None, :]).astype(np.float32)


def make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def positive_recall_precision(
    logits: torch.Tensor, targets: torch.Tensor, threshold: float
) -> tuple[np.ndarray, np.ndarray]:
    predictions = torch.sigmoid(logits) >= threshold
    precision, recall = [], []

    for i in range(targets.shape[1]):
        pred = predictions[:, i]
        truth = targets[:, i].bool()

        tp = (pred & truth).sum().item()
        fp = (pred & ~truth).sum().item()
        fn = (~pred & truth).sum().item()

        precision.append(tp / (tp + fp) if tp + fp else 0.0)
        recall.append(tp / (tp + fn) if tp + fn else 0.0)

    return np.asarray(precision), np.asarray(recall)


@torch.no_grad()
def evaluate(
    model: nn.Module, loader: DataLoader, criterion: nn.Module, threshold: float, device: torch.device
) -> tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()

    total_loss, total_count = 0.0, 0
    all_logits, all_targets = [], []

    for X, y in loader:
        X, y = X.to(device), y.to(device)
        logits = model(X)
        loss = criterion(logits, y)

        count = len(X)
        total_loss += float(loss.item()) * count
        total_count += count
        all_logits.append(logits)
        all_targets.append(y)

    logits = torch.cat(all_logits)
    targets = torch.cat(all_targets)

    loss = total_loss / total_count
    
    preds = (torch.sigmoid(logits) >= threshold).bool()
    accuracy = float((preds == targets.bool()).float().mean().item())
    
    precision, recall = positive_recall_precision(logits, targets, threshold)

    return loss, accuracy, precision, recall


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    learning_rate: float,
    threshold: float,
    pos_weight: torch.Tensor,
    device: torch.device,
) -> list[dict]:
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    history: list[dict] = []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss, count = 0.0, 0

        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(X)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            batch_count = len(X)
            running_loss += float(loss.item()) * batch_count
            count += batch_count

        train_loss = running_loss / count
        val_loss, val_accuracy, precision, recall = evaluate(
            model, val_loader, criterion, threshold, device
        )

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "precision_mean": float(precision.mean()),
            "recall_mean": float(recall.mean()),
        }
        history.append(row)

        per_label = " ".join(
            f"{label}:P{precision[i]:.2f}/R{recall[i]:.2f}"
            for i, label in enumerate(LABELS)
        )

        print(
            f"epoch {epoch:3d}/{epochs} "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} "
            f"val_acc={val_accuracy:.3f} "
            f"{per_label}"
        )

    return history


def save_checkpoint(
    path: Path,
    model: nn.Module,
    mean: np.ndarray,
    std: np.ndarray,
    feature_names: list[str],
    history: list[dict],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "n_features": len(feature_names),
            "n_labels": len(LABELS),
            "labels": LABELS,
            "feature_names": feature_names,
            "normalization_mean": mean,
            "normalization_std": std,
            "history": history,
        },
        path,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Train temporal 1D CNN for MTB features.")
    parser.add_argument("--dataset", type=Path, action="append", required=True)
    parser.add_argument("--validation-index", type=int, default=-1)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("output/mtb_temporal_cnn.pt"))

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using compute device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    validation_index = len(args.dataset) - 1 if args.validation_index == -1 else args.validation_index

    train_x, train_y, val_x, val_y, feature_names = split_by_dataset(args.dataset, validation_index)

    # Calculate class imbalance weights for BCEWithLogitsLoss
    pos_counts = train_y.sum(axis=0)
    neg_counts = len(train_y) - pos_counts
    pos_weight = torch.from_numpy(neg_counts / np.maximum(pos_counts, 1.0)).float().to(device)

    mean, std = fit_normalizer(train_x)
    train_x = normalize(train_x, mean, std)
    val_x = normalize(val_x, mean, std)

    train_loader = make_loader(train_x, train_y, args.batch_size, shuffle=True)
    val_loader = make_loader(val_x, val_y, args.batch_size, shuffle=False)

    model = MTBTemporalCNN(n_features=train_x.shape[-1], n_labels=len(LABELS)).to(device)

    history = train(
        model, train_loader, val_loader, args.epochs, args.learning_rate, args.threshold, pos_weight, device
    )

    save_checkpoint(args.output, model, mean, std, feature_names, history)
    print(f"\nSaved checkpoint to {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())