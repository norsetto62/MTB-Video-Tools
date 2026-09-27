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

A window is accepted when at least 50% of its duration overlaps an explicit
annotation. Target-feature annotations contribute independent labels, so an
annotation such as "Rock garden + Drop" produces both positive labels.

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
    categories: tuple[str, ...]


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


def classify_remarks(remarks: str) -> list[str]:
    """Return all target classes explicitly mentioned in the remarks."""
    return [
        label
        for label, pattern in CATEGORY_PATTERNS.items()
        if pattern.search(remarks)
    ]


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

        categories = classify_remarks(remarks)

        annotations.append(
            Annotation(
                start=start,
                end=end,
                mtb=mtb,
                video=video,
                remarks=remarks,
                categories=categories,
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

        # Flow CSVs produced by analyze_bike_flow.py use "time".
        # Accept "timestamp" only as a compatibility alias for older/external CSVs.
        if "time" in fields:
            time_field = "time"
        elif "timestamp" in fields:
            time_field = "timestamp"
        else:
            raise ValueError(f"{path}: missing time column")

        feature_names = [name for name in fields if name != time_field]

        if not feature_names:
            raise ValueError(f"{path}: no feature columns")

        timestamps: list[float] = []
        rows: list[list[float]] = []

        for line_no, raw in enumerate(reader, start=2):
            try:
                timestamp = float(raw[time_field])
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
    min_overlap: float = 0.50,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Create temporal windows using a minimum annotation-overlap rule.

    Each window is evaluated against all annotations. Target annotations
    contribute independent positive labels, so multi-feature annotations
    such as "Rock garden + Drop" become multi-hot targets.

    Explicit low-interest annotations (MTB <= 2) produce all-zero negatives
    when they cover at least min_overlap of the window and no target-feature
    annotation covers the window.
    """
    if window <= 0 or stride <= 0:
        raise ValueError("window and stride must be > 0")
    if not 0 < min_overlap <= 1:
        raise ValueError("min_overlap must be in (0, 1]")

    timestamp_array = np.asarray(timestamps, dtype=np.float64)
    video_start = float(timestamp_array[0])
    video_end = float(timestamp_array[-1])

    examples: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    manifest: list[dict] = []

    # Generate the temporal grid once. This prevents duplicate windows when
    # annotations overlap and allows several annotations to contribute labels.
    start = video_start
    while start + window <= video_end + 1e-9:
        end = start + window

        mask = (
            (timestamp_array >= start - 1e-9)
            & (timestamp_array < end - 1e-9)
        )
        indices = np.flatnonzero(mask)

        if len(indices) >= 2:
            labels_vector = np.zeros(len(LABELS), dtype=np.float32)
            target_matches: list[tuple[int, Annotation, float]] = []
            negative_matches: list[tuple[int, Annotation, float]] = []

            for ann_index, ann in enumerate(annotations, start=1):
                overlap = max(
                    0.0,
                    min(end, ann.end) - max(start, ann.start),
                )
                overlap_fraction = overlap / window

                if overlap_fraction + 1e-12 < min_overlap:
                    continue

                if ann.categories:
                    target_matches.append(
                        (ann_index, ann, overlap_fraction)
                    )
                    for category in ann.categories:
                        labels_vector[LABELS.index(category)] = 1.0
                elif ann.mtb <= 2:
                    negative_matches.append(
                        (ann_index, ann, overlap_fraction)
                    )

            # Explicit negatives are used only when no target annotation
            # covers the window. Target labels take precedence.
            if target_matches:
                selected = target_matches
                category = "+".join(
                    label
                    for label in LABELS
                    if labels_vector[LABELS.index(label)] == 1
                )
            elif negative_matches:
                selected = negative_matches
                category = "negative"
            else:
                selected = []

            if selected:
                annotation_ids = [item[0] for item in selected]
                source_annotations = [item[1] for item in selected]
                overlap_fractions = [item[2] for item in selected]

                examples.append(x[indices])
                labels.append(labels_vector)

                manifest.append(
                    {
                        "example_id": len(manifest),
                        "start": float(timestamp_array[indices[0]]),
                        "end": float(timestamp_array[indices[-1]]),
                        "requested_start": start,
                        "requested_end": end,
                        "duration": float(
                            timestamp_array[indices[-1]]
                            - timestamp_array[indices[0]]
                        ),
                        "annotation_id": ";".join(
                            map(str, annotation_ids)
                        ),
                        "mtb": ";".join(
                            str(a.mtb) for a in source_annotations
                        ),
                        "video": ";".join(
                            str(a.video) for a in source_annotations
                        ),
                        "category": category,
                        "remarks": " | ".join(
                            a.remarks for a in source_annotations
                        ),
                        "overlap_fraction": ";".join(
                            f"{v:.3f}" for v in overlap_fractions
                        ),
                        "n_samples": len(indices),
                    }
                )

        start += stride

    if not examples:
        raise ValueError("no training windows were generated")

    lengths = {len(example) for example in examples}
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
        "example_id", "start", "end", "requested_start", "requested_end",
        "duration", "annotation_id", "mtb", "video", "category",
        "remarks", "overlap_fraction", "n_samples",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(manifest)
def print_annotation_audit(annotations: list[Annotation]) -> None:
    print()
    print("=" * 90)
    print("ANNOTATION AUDIT")
    print("=" * 90)

    for i, ann in enumerate(annotations, start=1):
        role = (
            "+".join(ann.categories)
            if ann.categories
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
        "--min-overlap",
        type=float,
        default=0.50,
        help="Minimum fraction of each window overlapping an annotation (default: 0.50)",
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
        args.min_overlap,
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
