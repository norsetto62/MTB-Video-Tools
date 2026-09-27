#!/usr/bin/env python3
"""
Build temporal training windows for the five-class MTB feature model.

Inputs:
    - flow CSV produced by analyze_bike_flow.py
    - manual annotation TXT (e.g. data/annotations/Mentorella.txt)

Outputs:
    - manifest CSV: one row per temporal training window
    - NPZ: X [N, T, F], y [N, 5]
    - human-readable summary

The generator deliberately does NOT treat unlabeled video as negative.
Only explicit annotated intervals are used:
    * target-feature annotations -> positive labels
    * explicit low-interest annotations (MTB <= 2) -> all-zero negatives

Windows are kept only when the complete window lies inside one annotated
interval. This avoids contaminating labels at annotation boundaries.

The five labels are independent:
    drop, rock_garden, switchback, stairs, technical_climb

Category names are inferred from annotation remarks using explicit keyword
rules. Unknown remarks are reported and are not silently assigned a class.
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


LABELS = [
    "drop",
    "rock_garden",
    "switchback",
    "stairs",
    "technical_climb",
]

DEFAULT_WINDOW = 4.0
DEFAULT_STRIDE = 2.0
DEFAULT_DT_TOLERANCE = 0.15

# Explicit mapping from the current annotation vocabulary to model classes.
CATEGORY_PATTERNS = {
    "drop": re.compile(r"\bdrops?\b", re.IGNORECASE),
    "rock_garden": re.compile(
        r"\brock(?:y)?\s+garden(?:s)?\b", re.IGNORECASE
    ),
    "switchback": re.compile(r"\bswitchback\b", re.IGNORECASE),
    "stairs": re.compile(r"\bstairs?\b", re.IGNORECASE),
    # Deliberately require "technical climb"; "technical passages" is NOT
    # considered a positive technical_climb example.
    "technical_climb": re.compile(r"\btechnical\s+climb\b", re.IGNORECASE),
}


@dataclass(frozen=True)
class Annotation:
    start: float
    end: float
    mtb: int
    video: int
    remarks: str
    category: str | None


def parse_time(value: str) -> float:
    value = value.strip()
    parts = value.split(":")
    if len(parts) == 1:
        return float(parts[0])
    if len(parts) == 2:
        return float(parts[0]) * 60.0 + float(parts[1])
    if len(parts) == 3:
        return (
            float(parts[0]) * 3600.0
            + float(parts[1]) * 60.0
            + float(parts[2])
        )
    raise ValueError(f"invalid time: {value!r}")


def classify_remarks(remarks: str) -> str | None:
    matches = [
        label
        for label, pattern in CATEGORY_PATTERNS.items()
        if pattern.search(remarks)
    ]

    if len(matches) > 1:
        raise ValueError(
            f"annotation matches multiple target classes: "
            f"{matches!r}: {remarks!r}"
        )

    return matches[0] if matches else None


def load_annotations(path: Path) -> tuple[Path, list[Annotation]]:
    lines = path.read_text(encoding="utf-8-sig").splitlines()

    video_path: Path | None = None
    annotations: list[Annotation] = []

    for line_no, raw in enumerate(lines, start=1):
        line = raw.strip()

        if not line or line.startswith("#"):
            continue

        lower = line.lower()

        if all(token in lower for token in ("start", "end", "mtb", "video")):
            continue

        # The annotation file contains the source video path on its own line.
        if (
            video_path is None
            and len(line) >= 3
            and line[1] == ":"
            and line[2] in ("\\", "/")
        ):
            video_path = Path(line)
            continue

        parts = line.split(maxsplit=4)

        if len(parts) < 4:
            raise ValueError(
                f"{path}:{line_no}: expected Start End MTB Video [Remarks]"
            )

        try:
            start = parse_time(parts[0])
            end = parse_time(parts[1])
            mtb = int(parts[2])
            video = int(parts[3])
        except ValueError as exc:
            raise ValueError(
                f"{path}:{line_no}: {exc}"
            ) from exc

        remarks = parts[4].strip() if len(parts) == 5 else ""

        if end <= start:
            raise ValueError(
                f"{path}:{line_no}: End must be greater than Start"
            )
        if not 1 <= mtb <= 5:
            raise ValueError(
                f"{path}:{line_no}: MTB must be 1..5"
            )
        if not 1 <= video <= 5:
            raise ValueError(
                f"{path}:{line_no}: Video must be 1..5"
            )

        category = classify_remarks(remarks)

        annotations.append(
            Annotation(
                start=start,
                end=end,
                mtb=mtb,
                video=video,
                remarks=remarks,
                category=category,
            )
        )

    if video_path is None:
        raise ValueError(f"{path}: source video path not found")

    annotations.sort(key=lambda a: a.start)
    return video_path, annotations


def load_flow_csv(path: Path) -> tuple[list[float], list[str], np.ndarray]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)

        if reader.fieldnames is None:
            raise ValueError(f"{path}: missing CSV header")

        fields = [name.strip() for name in reader.fieldnames]

        if "timestamp" not in fields:
            raise ValueError(f"{path}: missing timestamp column")

        feature_names = [name for name in fields if name != "timestamp"]

        if not feature_names:
            raise ValueError(f"{path}: no feature columns")

        timestamps: list[float] = []
        rows: list[list[float]] = []

        for line_no, raw in enumerate(reader, start=2):
            try:
                timestamp = float(raw["timestamp"])
                values = [float(raw[name]) for name in feature_names]
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{path}:{line_no}: non-numeric value: {exc}"
                ) from exc

            if not np.isfinite(timestamp) or not np.all(np.isfinite(values)):
                raise ValueError(
                    f"{path}:{line_no}: NaN/Inf values are not allowed"
                )

            timestamps.append(timestamp)
            rows.append(values)

    if len(timestamps) < 2:
        raise ValueError(f"{path}: need at least two rows")

    x = np.asarray(rows, dtype=np.float32)

    return timestamps, feature_names, x


def check_sampling(timestamps: list[float], tolerance: float) -> float:
    dt = np.diff(np.asarray(timestamps, dtype=np.float64))
    if np.any(dt <= 0):
        raise ValueError("flow CSV timestamps must be strictly increasing")

    median_dt = float(np.median(dt))
    max_relative_error = float(
        np.max(np.abs(dt - median_dt) / median_dt)
    )

    if max_relative_error > tolerance:
        print(
            f"WARNING: sampling is not uniform: median dt={median_dt:.6f}s, "
            f"max relative deviation={max_relative_error:.3f}"
        )

    return median_dt


def make_windows(
    timestamps: list[float],
    x: np.ndarray,
    annotations: list[Annotation],
    window: float,
    stride: float,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Create only windows fully contained in an explicit annotation."""

    if window <= 0 or stride <= 0:
        raise ValueError("window and stride must be > 0")

    timestamp_array = np.asarray(timestamps)

    examples: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    manifest: list[dict] = []

    # A window is represented by rows whose timestamps are inside
    # [start, end). We require its first and last sample to fit within
    # the annotation, and require a stable number of samples.
    for ann_index, ann in enumerate(annotations, start=1):
        category = ann.category

        # MTB <= 2 annotations are explicit low-interest negatives.
        is_negative = ann.mtb <= 2

        # MTB > 2 with no recognized target category is not a training
        # example yet; report it rather than inventing a class.
        if not is_negative and category is None:
            continue

        label = np.zeros(len(LABELS), dtype=np.float32)

        if category is not None:
            label[LABELS.index(category)] = 1.0

        start = ann.start

        while start + window <= ann.end + 1e-9:
            end = start + window

            mask = (
                (timestamp_array >= start - 1e-9)
                & (timestamp_array < end - 1e-9)
            )
            indices = np.flatnonzero(mask)

            if len(indices) >= 2:
                first = float(timestamp_array[indices[0]])
                last = float(timestamp_array[indices[-1]])

                # The complete temporal sample must lie inside the manual
                # annotation. We don't allow a window to cross its boundary.
                if (
                    first >= ann.start - 1e-9
                    and last <= ann.end + 1e-9
                ):
                    examples.append(x[indices])
                    labels.append(label.copy())

                    manifest.append(
                        {
                            "example_id": len(manifest),
                            "start": first,
                            "end": last,
                            "requested_start": start,
                            "requested_end": end,
                            "duration": last - first,
                            "annotation_id": ann_index,
                            "mtb": ann.mtb,
                            "video": ann.video,
                            "category": category or "negative",
                            "remarks": ann.remarks,
                            "n_samples": len(indices),
                        }
                    )

            start += stride

    if not examples:
        raise ValueError("no training windows were generated")

    lengths = {len(a) for a in examples}

    if len(lengths) != 1:
        raise ValueError(
            f"generated windows have different sample counts: {sorted(lengths)}"
        )

    return (
        np.stack(examples).astype(np.float32),
        np.stack(labels).astype(np.float32),
        manifest,
    )


def write_manifest(path: Path, manifest: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "example_id",
        "start",
        "end",
        "requested_start",
        "requested_end",
        "duration",
        "annotation_id",
        "mtb",
        "video",
        "category",
        "remarks",
        "n_samples",
    ]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for row in manifest:
            writer.writerow(row)


def print_annotation_audit(annotations: list[Annotation]) -> None:
    print()
    print("=" * 90)
    print("ANNOTATION AUDIT")
    print("=" * 90)

    for i, ann in enumerate(annotations, start=1):
        role = (
            ann.category
            if ann.category is not None
            else ("negative" if ann.mtb <= 2 else "ignored")
        )

        print(
            f"{i:2d}  {ann.start:7.2f}-{ann.end:7.2f}  "
            f"MTB={ann.mtb}  Video={ann.video}  "
            f"role={role:14s}  {ann.remarks}"
        )


def print_dataset_preview(
    manifest: list[dict],
    y: np.ndarray,
    max_rows: int,
) -> None:
    print()
    print("=" * 90)
    print("GENERATED TRAINING EXAMPLES")
    print("=" * 90)

    header = (
        "id    start     end    "
        "drop rock_garden switchback stairs technical_climb   category"
    )
    print(header)
    print("-" * len(header))

    for row, labels in zip(manifest[:max_rows], y[:max_rows]):
        values = " ".join(f"{int(v):4d}" for v in labels)
        print(
            f"{row['example_id']:3d}  "
            f"{row['start']:7.2f}  "
            f"{row['end']:7.2f}  "
            f"{values}    "
            f"{row['category']}"
        )

    if len(manifest) > max_rows:
        print(f"... {len(manifest) - max_rows} more examples")


def print_summary(manifest: list[dict], y: np.ndarray) -> None:
    print()
    print("=" * 90)
    print("DATASET SUMMARY")
    print("=" * 90)

    print(f"Examples:       {len(manifest)}")
    print(f"Window samples: {y.shape[0]} x {y.shape[1]} labels")

    for i, label in enumerate(LABELS):
        positive = int(np.sum(y[:, i] == 1))
        print(f"  {label:16s}: {positive:4d} positive")

    negatives = int(np.sum(np.all(y == 0, axis=1)))
    print(f"  {'all-zero negative':16s}: {negatives:4d}")

    print()
    print("By annotation:")
    counts: dict[int, int] = {}
    for row in manifest:
        counts[row["annotation_id"]] = counts.get(row["annotation_id"], 0) + 1

    for annotation_id, count in counts.items():
        print(f"  annotation {annotation_id:2d}: {count:4d} windows")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the five-class MTB temporal NN dataset."
    )
    parser.add_argument(
        "--flow-csv",
        type=Path,
        required=True,
        help="Optical-flow CSV produced by analyze_bike_flow.py",
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=Path("data/annotations/Mentorella.txt"),
        help="Manual annotation file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/datasets"),
        help="Directory for generated dataset files",
    )
    parser.add_argument(
        "--window",
        type=float,
        default=DEFAULT_WINDOW,
        help="Temporal window in seconds (default: 4)",
    )
    parser.add_argument(
        "--stride",
        type=float,
        default=DEFAULT_STRIDE,
        help="Window stride in seconds (default: 2)",
    )
    parser.add_argument(
        "--dt-tolerance",
        type=float,
        default=DEFAULT_DT_TOLERANCE,
        help="Maximum relative sampling deviation before warning",
    )
    parser.add_argument(
        "--preview",
        type=int,
        default=40,
        help="Number of generated examples to print (default: 40)",
    )

    args = parser.parse_args()

    video_path, annotations = load_annotations(args.annotations)
    timestamps, feature_names, x = load_flow_csv(args.flow_csv)

    median_dt = check_sampling(timestamps, args.dt_tolerance)

    X, y, manifest = make_windows(
        timestamps,
        x,
        annotations,
        args.window,
        args.stride,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    stem = args.annotations.stem.lower()

    manifest_path = args.output_dir / f"{stem}_dataset_manifest.csv"
    npz_path = args.output_dir / f"{stem}_dataset.npz"

    write_manifest(manifest_path, manifest)

    np.savez_compressed(
        npz_path,
        X=X,
        y=y,
        timestamps=np.asarray(timestamps, dtype=np.float64),
        feature_names=np.asarray(feature_names),
        labels=np.asarray(LABELS),
    )

    print(f"Annotation video: {video_path}")
    print(f"Flow CSV:         {args.flow_csv}")
    print(f"Flow rows:        {len(timestamps)}")
    print(f"Features/row:     {len(feature_names)}")
    print(f"Median dt:        {median_dt:.6f}s")
    print(f"Window:           {args.window:.2f}s")
    print(f"Stride:           {args.stride:.2f}s")
    print(f"Output X shape:   {X.shape}")
    print(f"Output y shape:   {y.shape}")

    print_annotation_audit(annotations)
    print_dataset_preview(manifest, y, args.preview)
    print_summary(manifest, y)

    print()
    print(f"Manifest: {manifest_path}")
    print(f"Dataset:  {npz_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
