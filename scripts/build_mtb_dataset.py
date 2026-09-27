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


def write_manifest