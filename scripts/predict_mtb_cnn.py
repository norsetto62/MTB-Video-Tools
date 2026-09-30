#!/usr/bin/env python3
"""
Run the trained MTB temporal CNN on selected dataset examples.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from torch import nn


LABELS = [
    "drop",
    "rock_garden",
    "switchback",
    "stairs",
    "technical_climb",
]


class MTBTemporalCNN(nn.Module):
    """Same architecture used by train_mtb_cnn.py."""

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
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = self.pool(x)
        return self.classifier(x)


def load_manifest(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the trained MTB CNN on an NPZ dataset."
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("output/mtb_temporal_cnn.pt"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Dataset manifest CSV. Defaults to <dataset>_manifest.csv",
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)

    parser.add_argument(
        "--detected",
        choices=LABELS,
        help="Show only examples whose highest predicted label matches this label.",
    )
    parser.add_argument(
        "--expected",
        choices=LABELS,
        help="Show only examples whose actual annotation contains this label.",
    )

    args = parser.parse_args()

    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
    )

    feature_names = [str(v) for v in checkpoint["feature_names"]]
    mean = np.asarray(
        checkpoint["normalization_mean"],
        dtype=np.float32,
    )
    std = np.asarray(
        checkpoint["normalization_std"],
        dtype=np.float32,
    )

    if len(feature_names) != checkpoint["n_features"]:
        raise ValueError("checkpoint feature_names/n_features mismatch")

    with np.load(args.dataset, allow_pickle=False) as data:
        X = np.asarray(data["X"], dtype=np.float32)
        y = np.asarray(data["y"], dtype=np.float32)
        dataset_features = [str(v) for v in data["feature_names"]]
        dataset_labels = [str(v) for v in data["labels"]]

    if dataset_labels != LABELS:
        raise ValueError(f"Unexpected labels: {dataset_labels}")

    indices = []
    for name in feature_names:
        if name not in dataset_features:
            raise ValueError(
                f"Dataset is missing checkpoint feature: {name}"
            )
        indices.append(dataset_features.index(name))

    X = X[:, :, indices]

    if X.shape[-1] != len(feature_names):
        raise ValueError("Feature selection produced unexpected shape")

    X = (X - mean[None, None, :]) / std[None, None, :]
    X = X.astype(np.float32)

    if args.manifest is None:
        args.manifest = args.dataset.with_name(
            args.dataset.stem.replace(
                "_dataset",
                "_dataset_manifest",
            ) + ".csv"
        )

    manifest = load_manifest(args.manifest)

    if len(manifest) != len(X):
        raise ValueError(
            f"Manifest examples ({len(manifest)}) do not match "
            f"dataset examples ({len(X)})"
        )

    start = max(0, args.start)
    end = len(X) if args.end is None else min(args.end, len(X))

    if start >= end:
        raise ValueError("Invalid --start/--end range")

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = MTBTemporalCNN(
        n_features=checkpoint["n_features"],
        n_labels=checkpoint["n_labels"],
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    with torch.no_grad():
        tensor = torch.from_numpy(X[start:end]).to(device)
        probabilities = torch.sigmoid(model(tensor)).cpu().numpy()

    print(f"Dataset:   {args.dataset}")
    print(f"Manifest:  {args.manifest}")
    print(f"Examples:  {start}-{end - 1}")
    print()

    header = (
        "example  start_time  end_time    expected"
        "                 "
        "drop  rock_garden  switchback  stairs  technical_climb"
    )
    print(header)
    print("-" * len(header))

    for offset, probs in enumerate(probabilities):
        i = start + offset
        row = manifest[i]

        actual_labels = [
            label
            for label, value in zip(LABELS, y[i])
            if value > 0.5
        ]
        actual = "+".join(actual_labels) if actual_labels else "negative"

        predicted_index = int(np.argmax(probs))
        predicted_label = LABELS[predicted_index]

        if args.expected is not None and args.expected not in actual_labels:
            continue

        if args.detected is not None and predicted_label != args.detected:
            continue

        print(
            f"{i:7d}  "
            f"{float(row['start']):10.2f}  "
            f"{float(row['end']):8.2f}  "
            f"{actual:20s}  "
            f"{probs[0]:.3f}      "
            f"{probs[1]:.3f}       "
            f"{probs[2]:.3f}       "
            f"{probs[3]:.3f}     "
            f"{probs[4]:.3f}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())