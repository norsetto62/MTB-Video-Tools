#!/usr/bin/env python3
"""
Run the trained MTB temporal CNN on selected dataset examples
or directly on a flow CSV.
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


def load_flow_csv(
    path: Path,
    feature_names: list[str],
    mean: np.ndarray,
    std: np.ndarray,
) -> tuple[np.ndarray, list[dict]]:
    """Build 4-second / 2-second-stride inference windows from a flow CSV."""

    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise ValueError(f"Flow CSV is empty: {path}")

    for name in ["time", *feature_names]:
        if name not in rows[0]:
            raise ValueError(f"Flow CSV is missing feature: {name}")

    timestamps = np.asarray(
        [float(row["time"]) for row in rows],
        dtype=np.float64,
    )

    dt = float(np.median(np.diff(timestamps)))
    n_window_samples = int(round(4.0 / dt))
    stride_samples = int(round(2.0 / dt))

    if n_window_samples < 2 or stride_samples < 1:
        raise ValueError(
            f"Invalid flow sampling interval: dt={dt:.6f}s"
        )

    values = np.asarray(
        [
            [float(row[name]) for name in feature_names]
            for row in rows
        ],
        dtype=np.float32,
    )

    examples = []
    manifest = []

    for start_index in range(
        0,
        len(values) - n_window_samples + 1,
        stride_samples,
    ):
        end_index = start_index + n_window_samples

        examples.append(values[start_index:end_index])

        manifest.append(
            {
                "start": float(timestamps[start_index]),
                "end": float(timestamps[start_index] + 4.0),
            }
        )

    if not examples:
        raise ValueError("no inference windows were generated")

    X = np.stack(examples).astype(np.float32)

    X = (X - mean[None, None, :]) / std[None, None, :]

    return X.astype(np.float32), manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the trained MTB CNN on an NPZ dataset or flow CSV."
    )

    parser.add_argument(
        "--dataset",
        type=Path,
    )

    parser.add_argument(
        "--flow-csv",
        type=Path,
        help="Run inference directly on a flow CSV using 4-second windows.",
    )

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

    if (args.dataset is None) == (args.flow_csv is None):
        parser.error("specify exactly one of --dataset or --flow-csv")

    if args.flow_csv is not None and args.expected is not None:
        parser.error("--expected requires --dataset")

    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
    )

    feature_names = [
        str(v) for v in checkpoint["feature_names"]
    ]

    mean = np.asarray(
        checkpoint["normalization_mean"],
        dtype=np.float32,
    )

    std = np.asarray(
        checkpoint["normalization_std"],
        dtype=np.float32,
    )

    if len(feature_names) != checkpoint["n_features"]:
        raise ValueError(
            "checkpoint feature_names/n_features mismatch"
        )

    if args.flow_csv is not None:
        X, manifest = load_flow_csv(
            args.flow_csv,
            feature_names,
            mean,
            std,
        )

        y = None

    else:
        with np.load(args.dataset, allow_pickle=False) as data:
            X = np.asarray(
                data["X"],
                dtype=np.float32,
            )

            y = np.asarray(
                data["y"],
                dtype=np.float32,
            )

            dataset_features = [
                str(v) for v in data["feature_names"]
            ]

            dataset_labels = [
                str(v) for v in data["labels"]
            ]

        if dataset_labels != LABELS:
            raise ValueError(
                f"Unexpected labels: {dataset_labels}"
            )

        indices = []

        for name in feature_names:
            if name not in dataset_features:
                raise ValueError(
                    f"Dataset is missing checkpoint feature: {name}"
                )

            indices.append(
                dataset_features.index(name)
            )

        X = X[:, :, indices]

        if X.shape[-1] != len(feature_names):
            raise ValueError(
                "Feature selection produced unexpected shape"
            )

        X = (
            X
            - mean[None, None, :]
        ) / std[None, None, :]

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
                f"Manifest examples ({len(manifest)}) "
                f"do not match dataset examples ({len(X)})"
            )

    start = max(0, args.start)
    end = (
        len(X)
        if args.end is None
        else min(args.end, len(X))
    )

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

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(device)
    model.eval()

    with torch.no_grad():
        tensor = torch.from_numpy(
            X[start:end]
        ).to(device)

        probabilities = (
            torch.sigmoid(model(tensor))
            .cpu()
            .numpy()
        )

    if args.flow_csv is not None:
        print(f"Flow CSV:  {args.flow_csv}")
    else:
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

        if y is not None:
            actual_labels = [
                label
                for label, value in zip(
                    LABELS,
                    y[i],
                )
                if value > 0.5
            ]

            actual = (
                "+".join(actual_labels)
                if actual_labels
                else "negative"
            )

        else:
            actual_labels = []
            actual = "unknown"

        predicted_index = int(
            np.argmax(probs)
        )

        predicted_label = LABELS[
            predicted_index
        ]

        if (
            args.expected is not None
            and args.expected not in actual_labels
        ):
            continue

        if (
            args.detected is not None
            and predicted_label != args.detected
        ):
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