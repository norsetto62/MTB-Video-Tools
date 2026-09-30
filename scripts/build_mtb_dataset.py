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
    * valid target-feature annotations -> positive labels
    * explicit low-interest non-target annotations (MTB <= 2) -> all-zero negatives
    * annotations marked "Slow" or "Dismounted" -> excluded from training

A window is accepted when at least 50% of its duration overlaps an explicit
annotation. Target-feature annotations contribute independent labels, so an
annotation such as "Rock garden + Drop" produces both positive labels.

The five labels are independent:
    drop, rock_garden, switchback, stairs, technical_climb

"Steep" is recognized for audit purposes only. It is deliberately NOT an NN
label yet, so steep-only annotations retain the existing training behavior.

Category names are inferred from annotation remarks using explicit keyword
rules. Unknown remarks are reported and are not silently assigned a class.
Annotations marked "Slow" or "Dismounted" are excluded entirely: they are
not used as positives or negatives because their motion characteristics may
differ from the real riding events the model is intended to detect.
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
EXCLUDE_PATTERNS = {
    "slow": re.compile(r"\bslow\b", re.IGNORECASE),
    "dismounted": re.compile(r"\bdismounted\b", re.IGNORECASE),
}

AUDIT_CATEGORY_PATTERNS = {
    "steep": re.compile(r"\bsteep\b", re.IGNORECASE),
}

CATEGORY_PATTERNS = {
    "drop": re.compile(r"\bdrops?\b", re.IGNORECASE),
    "rock_garden": re.compile(
        r"\brock(?:y)?\s+garden(?:s)?\b", re.IGNORECASE
    ),
    "switchback": re.compile(r"\bswitchbacks?\b", re.IGNORECASE),
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
    excluded: bool
    audit_categories: tuple[str, ...] = ()


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


def annotation_is_excluded(remarks: str) -> bool:
    """Return True for events unsuitable for feature-model training."""
    return any(pattern.search(remarks) for pattern in EXCLUDE_PATTERNS.values())


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
        audit_categories = tuple(
            label
            for label, pattern in AUDIT_CATEGORY_PATTERNS.items()
            if pattern.search(remarks)
        )
        excluded = annotation_is_excluded(remarks)

        annotations.append(
            Annotation(
                start=start,
                end=end,
                mtb=mtb,
                video=video,
                remarks=remarks,
                categories=categories,
                excluded=excluded,
                audit_categories=audit_categories,
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
    flow_start: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Create temporal windows using a minimum annotation-overlap rule.

    Each window is evaluated against all annotations. Target annotations
    contribute independent positive labels, so multi-feature annotations
    such as "Rock garden + Drop" become multi-hot targets.

    Explicit low-interest non-target annotations (MTB <= 2) produce all-zero
    negatives when they cover at least min_overlap of the window and no
    target-feature annotation covers the window. Annotations marked Slow or
    Dismounted are excluded entirely and never produce training examples.
    """
    if window <= 0 or stride <= 0:
        raise ValueError("window and stride must be > 0")
    if not 0 < min_overlap <= 1:
        raise ValueError("min_overlap must be in (0, 1]")

    timestamp_array = np.asarray(timestamps, dtype=np.float64)
    # Work on the flow CSV timeline internally; add flow_start only when
    # comparing against source-video annotation times or writing manifest times.
    video_start = float(timestamp_array[0])
    video_end = float(timestamp_array[-1])

    # Keep every temporal window at a fixed number of samples.
    dt = float(np.median(np.diff(timestamp_array)))
    n_window_samples = int(round(window / dt))

    if n_window_samples < 2:
        raise ValueError(
            f"window={window:.3f}s is too short for flow sampling dt={dt:.6f}s"
        )

    examples: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    manifest: list[dict] = []

    # Generate the temporal grid once. This prevents duplicate windows when
    # annotations overlap and allows several annotations to contribute labels.
    start_index = 0
    while start_index + n_window_samples <= len(timestamp_array):
        indices = np.arange(
            start_index,
            start_index + n_window_samples,
            dtype=np.int64,
        )

        start = float(timestamp_array[start_index])
        end = start + window

        labels_vector = np.zeros(len(LABELS), dtype=np.float32)
        target_matches: list[tuple[int, Annotation, float]] = []
        negative_matches: list[tuple[int, Annotation, float]] = []

        annotation_start = start + flow_start
        annotation_end = end + flow_start

        for ann_index, ann in enumerate(annotations, start=1):
            # Slow/Dismounted occurrences are deliberately excluded from
            # both positive and negative training data.
            if ann.excluded:
                continue

            overlap = max(
                0.0,
                min(annotation_end, ann.end) - max(annotation_start, ann.start),
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
                    "start": float(timestamp_array[indices[0]] + flow_start),
                    "end": float(timestamp_array[indices[-1]] + flow_start),
                    "requested_start": annotation_start,
                    "requested_end": annotation_end,
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

        start_index += int(round(stride / dt))

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
def parse_audit_pair(value: str) -> tuple[Path, Path]:
    """Parse an audit pair as FLOW_CSV=ANNOTATIONS_TXT."""
    if "=" not in value:
        raise ValueError(
            f"invalid --audit-pair {value!r}: expected FLOW_CSV=ANNOTATIONS_TXT"
        )
    flow_csv, annotations = value.split("=", 1)
    if not flow_csv.strip() or not annotations.strip():
        raise ValueError(
            f"invalid --audit-pair {value!r}: both paths are required"
        )
    return Path(flow_csv.strip()), Path(annotations.strip())


def audit_multiple_datasets(
    pairs: list[tuple[Path, Path]],
    output_dir: Path,
    window: float,
    stride: float,
    dt_tolerance: float,
    min_overlap: float,
    flow_start: float,
) -> None:
    """Audit several flow/annotation pairs and print one combined summary."""
    combined_counts = np.zeros(len(LABELS), dtype=np.int64)
    combined_negative = 0
    combined_examples = 0
    combined_annotations = 0
    combined_excluded = 0
    combined_steep = 0

    print()
    print("#" * 90)
    print("MULTI-VIDEO DATASET AUDIT")
    print("#" * 90)
    print(f"Videos: {len(pairs)}")

    for flow_csv, annotation_path in pairs:
        video_path, annotations = load_annotations(annotation_path)
        timestamps, feature_names, x = load_flow_csv(flow_csv)
        median_dt = check_sampling(timestamps, dt_tolerance)
        try:
            X, y, manifest = make_windows(
                timestamps, x, annotations, window, stride, min_overlap, flow_start
            )
        except ValueError as exc:
            if str(exc) != "no training windows were generated":
                raise
            X = np.empty((0, 0, len(feature_names)), dtype=np.float32)
            y = np.empty((0, len(LABELS)), dtype=np.float32)
            manifest = []
            no_training_windows = True
        else:
            no_training_windows = False

        print()
        print("-" * 90)
        print(f"VIDEO: {annotation_path.stem}")
        print(f"  Source video: {video_path}")
        print(f"  Flow CSV:     {flow_csv}")
        print(f"  Flow rows:    {len(timestamps)}")
        print(f"  Features:     {len(feature_names)}")
        print(f"  Median dt:    {median_dt:.6f}s")
        print(f"  Annotations:  {len(annotations)}")
        steep_count = sum(bool(a.audit_categories) for a in annotations)
        print(f"  Steep annotations (audit-only): {steep_count}")
        if no_training_windows:
            print("  Training examples: 0 (no target or explicit negative annotations)")

        for i, label in enumerate(LABELS):
            count = int(np.sum(y[:, i] == 1))
            combined_counts[i] += count
            print(f"  {label:16s}: {count:4d}")

        negative_count = int(np.sum(np.all(y == 0, axis=1)))
        combined_negative += negative_count
        combined_examples += len(manifest)
        combined_annotations += len(annotations)
        combined_excluded += sum(a.excluded for a in annotations)
        combined_steep += steep_count

        print_annotation_audit(annotations)
        if no_training_windows:
            print()
            print("DATASET AUDIT")
            print("=" * 90)
            print("  No training windows were generated for this video.")
            print("  This is valid for audit-only videos containing no target")
            print("  labels and no explicit MTB<=2 non-target negatives.")
            print("  Annotation parsing and audit classification are still reported above.")
        else:
            audit_dataset(annotations, manifest, y, window, stride, min_overlap)

        audit_csv_path = output_dir / f"{annotation_path.stem.lower()}_dataset_audit.csv"
        write_example_audit(
            audit_csv_path, manifest, y, annotations, stride, min_overlap
        )
        print(f"  Audit CSV:    {audit_csv_path}")

    print()
    print("#" * 90)
    print("COMBINED AUDIT SUMMARY")
    print("#" * 90)
    print(f"  Videos:                    {len(pairs)}")
    print(f"  Annotations:               {combined_annotations}")
    print(f"  Excluded annotations:      {combined_excluded}")
    print(f"  Examples:                  {combined_examples}")
    for label, count in zip(LABELS, combined_counts):
        print(f"  {label:24s}: {int(count):4d}")
    print(f"  {'all-zero negative':24s}: {combined_negative:4d}")
    print(f"  {'steep annotations':24s}: {combined_steep:4d} (audit-only)")
    print()
    print("NOTE: steep is intentionally not included in the five NN labels yet.")
    print("      Unlabeled video is still NOT treated as negative.")
    print("      Normal single-video dataset generation is unchanged.")


def print_annotation_audit(annotations: list[Annotation]) -> None:
    print()
    print("=" * 90)
    print("ANNOTATION AUDIT")
    print("=" * 90)

    for i, ann in enumerate(annotations, start=1):
        role = (
            "excluded"
            if ann.excluded
            else "+".join(ann.categories)
            if ann.categories
            else "steep (audit-only)"
            if ann.audit_categories
            else ("negative" if ann.mtb <= 2 else "ignored")
        )

        print(
            f"{i:2d}  {ann.start:7.2f}-{ann.end:7.2f}  "
            f"MTB={ann.mtb}  Video={ann.video}  "
            f"role={role:14s}  {ann.remarks}"
        )



def audit_dataset(
    annotations: list[Annotation],
    manifest: list[dict],
    y: np.ndarray,
    window: float,
    stride: float,
    min_overlap: float,
) -> None:
    """Audit generated examples before they are used for NN training.

    This is deliberately human-readable rather than a full row dump. It checks
    coverage, label consistency, annotation boundaries, negative provenance,
    MTB-score distributions, temporal redundancy, and suspicious edge cases.
    """
    print()
    print("=" * 90)
    print("DATASET AUDIT")
    print("=" * 90)

    annotation_lookup = {
        str(i): ann for i, ann in enumerate(annotations, start=1)
    }

    # ------------------------------------------------------------------
    # 1. Coverage and class counts
    # ------------------------------------------------------------------
    print()
    print("1. COVERAGE")
    print(f"  Total examples: {len(manifest)}")
    print(f"  Window: {window:.2f}s   Stride: {stride:.2f}s")
    print(f"  Minimum annotation overlap: {min_overlap:.0%}")
    print(f"  Samples/example: {int(manifest[0]['n_samples'])}")

    for i, label in enumerate(LABELS):
        count = int(np.sum(y[:, i] == 1))
        print(f"  {label:16s}: {count:4d} positive")

    negative_count = int(np.sum(np.all(y == 0, axis=1)))
    if manifest:
        print(f"  Samples/example: {int(manifest[0]['n_samples'])}")
    else:
        print("  Samples/example: n/a (no generated windows)")

    annotation_counts: dict[str, int] = {}
    for row in manifest:
        for annotation_id in row["annotation_id"].split(";"):
            annotation_counts[annotation_id] = (
                annotation_counts.get(annotation_id, 0) + 1
            )

    print()
    print("  Windows per source annotation:")
    for annotation_id in sorted(
        annotation_counts, key=lambda value: int(value)
    ):
        ann = annotation_lookup[annotation_id]
        print(
            f"    {int(annotation_id):2d}: {annotation_counts[annotation_id]:3d} "
            f"windows  ({ann.start:.2f}-{ann.end:.2f}s, "
            f"MTB={ann.mtb}, {ann.remarks})"
        )

    # ------------------------------------------------------------------
    # 2. Label correctness
    # ------------------------------------------------------------------
    excluded_ids = [
        i for i, ann in enumerate(annotations, start=1) if ann.excluded
    ]
    print()
    print(f"  Excluded annotations: {len(excluded_ids)}")
    for annotation_id in excluded_ids:
        ann = annotation_lookup[str(annotation_id)]
        print(
            f"    annotation {annotation_id:2d}: {ann.start:.2f}-{ann.end:.2f}s  "
            f"{ann.remarks}"
        )

    # ------------------------------------------------------------------
    # 2. Label correctness
    # ------------------------------------------------------------------
    print()
    print("2. LABEL CORRECTNESS")

    multi_label_ids = [
        i for i, row in enumerate(manifest)
        if int(np.sum(y[i])) > 1
    ]
    multi_annotation_ids = [
        i for i, row in enumerate(manifest)
        if ";" in row["annotation_id"]
    ]

    print(f"  Multi-label examples: {len(multi_label_ids)}")
    if multi_label_ids:
        for i in multi_label_ids:
            row = manifest[i]
            labels = [
                label for j, label in enumerate(LABELS) if y[i, j] == 1
            ]
            print(
                f"    id {row['example_id']:3d} "
                f"{float(row['start']):7.2f}-{float(row['end']):7.2f}s: "
                f"{'+'.join(labels)}  "
                f"annotations={row['annotation_id']}"
            )

    print(f"  Multiple source annotations: {len(multi_annotation_ids)}")
    if multi_annotation_ids:
        for i in multi_annotation_ids:
            row = manifest[i]
            print(
                f"    id {row['example_id']:3d} "
                f"{float(row['start']):7.2f}-{float(row['end']):7.2f}s: "
                f"annotations={row['annotation_id']} "
                f"category={row['category']}"
            )

    consistency_errors: list[str] = []
    for i, (row, labels) in enumerate(zip(manifest, y)):
        expected = [
            label for j, label in enumerate(LABELS) if labels[j] == 1
        ]
        if all(v == 0 for v in labels):
            if row["category"] != "negative":
                consistency_errors.append(
                    f"id {row['example_id']}: all-zero target but category={row['category']}"
                )
        elif row["category"] != "+".join(expected):
            consistency_errors.append(
                f"id {row['example_id']}: y={'+'.join(expected)} "
                f"but category={row['category']}"
            )

    print(f"  Internal label/category contradictions: {len(consistency_errors)}")
    for error in consistency_errors[:20]:
        print(f"    {error}")
    if len(consistency_errors) > 20:
        print(f"    ... {len(consistency_errors) - 20} more")

    # ------------------------------------------------------------------
    # 3. Boundary behavior
    # ------------------------------------------------------------------
    print()
    print("3. BOUNDARY BEHAVIOR")

    boundary_rows: list[tuple[int, str]] = []
    overlap_values: list[float] = []

    for row in manifest:
        fractions = [
            float(value) for value in row["overlap_fraction"].split(";")
        ]
        overlap_values.extend(fractions)

        for annotation_id, fraction in zip(
            row["annotation_id"].split(";"), fractions
        ):
            ann = annotation_lookup[annotation_id]
            start = float(row["requested_start"])
            end = float(row["requested_end"])

            distance_to_start = abs(start - ann.start)
            distance_to_end = abs(end - ann.end)

            if distance_to_start <= stride + 1e-9:
                boundary_rows.append(
                    (int(row["example_id"]),
                     f"near start of annotation {annotation_id} "
                     f"(window {start:.2f}-{end:.2f}, overlap={fraction:.3f})")
                )
            if distance_to_end <= stride + 1e-9:
                boundary_rows.append(
                    (int(row["example_id"]),
                     f"near end of annotation {annotation_id} "
                     f"(window {start:.2f}-{end:.2f}, overlap={fraction:.3f})")
                )

    below_threshold = [
        (row["example_id"], row["overlap_fraction"])
        for row in manifest
        if any(
            float(value) + 1e-12 < min_overlap
            for value in row["overlap_fraction"].split(";")
        )
    ]

    near_threshold = [
        (row["example_id"], row["overlap_fraction"])
        for row in manifest
        if any(
            abs(float(value) - min_overlap) <= 0.03
            for value in row["overlap_fraction"].split(";")
        )
    ]

    print(f"  Recorded annotation overlaps: {len(overlap_values)}")
    if overlap_values:
        print(
            f"  Overlap range: {min(overlap_values):.3f} .. "
            f"{max(overlap_values):.3f}"
        )
    print(f"  Windows below {min_overlap:.0%}: {len(below_threshold)}")
    print(f"  Windows near threshold (+/- 3 percentage points): {len(near_threshold)}")
    print(f"  Boundary-near examples: {len(set(i for i, _ in boundary_rows))}")

    if below_threshold:
        for example_id, fractions in below_threshold[:20]:
            print(f"    ERROR: id {example_id}: overlap={fractions}")

    print("  Boundary examples:")
    for example_id, description in boundary_rows[:30]:
        print(f"    id {example_id:3d}: {description}")
    if len(boundary_rows) > 30:
        print(f"    ... {len(boundary_rows) - 30} more")

    # ------------------------------------------------------------------
    # 4. Negative provenance
    # ------------------------------------------------------------------
    print()
    print("4. NEGATIVE PROVENANCE")

    negative_rows = [
        (row, i) for i, row in enumerate(manifest)
        if np.all(y[i] == 0)
    ]
    invalid_negative_rows: list[str] = []

    for row, _ in negative_rows:
        annotation_ids = row["annotation_id"].split(";")
        if row["category"] != "negative":
            invalid_negative_rows.append(
                f"id {row['example_id']}: category={row['category']}"
            )
        for annotation_id in annotation_ids:
            ann = annotation_lookup[annotation_id]
            if ann.excluded or ann.categories or ann.mtb > 2:
                invalid_negative_rows.append(
                    f"id {row['example_id']}: annotation {annotation_id} "
                    f"is not an explicit MTB<=2 non-target annotation"
                )

    print(f"  All-zero examples: {len(negative_rows)}")
    print(
        f"  Valid explicit-negative sources: "
        f"{len(set(row['annotation_id'] for row, _ in negative_rows))}"
    )
    print(f"  Invalid negative provenance: {len(invalid_negative_rows)}")

    print("  Negative source annotations:")
    negative_source_ids = sorted(
        {
            annotation_id
            for row, _ in negative_rows
            for annotation_id in row["annotation_id"].split(";")
        },
        key=lambda value: int(value),
    )
    for annotation_id in negative_source_ids:
        ann = annotation_lookup[annotation_id]
        count = sum(
            1 for row, _ in negative_rows
            if annotation_id in row["annotation_id"].split(";")
        )
        print(
            f"    annotation {int(annotation_id):2d}: {count:3d} windows  "
            f"MTB={ann.mtb}  {ann.start:.2f}-{ann.end:.2f}s  {ann.remarks}"
        )

    # ------------------------------------------------------------------
    # 5. MTB-score distribution by feature
    # ------------------------------------------------------------------
    print()
    print("5. MTB SCORE DISTRIBUTION BY FEATURE")

    for label_index, label in enumerate(LABELS):
        scores: list[int] = []
        for i, row in enumerate(manifest):
            if y[i, label_index] != 1:
                continue
            for annotation_id in row["annotation_id"].split(";"):
                ann = annotation_lookup[annotation_id]
                if label in ann.categories:
                    scores.append(ann.mtb)

        if scores:
            values, counts = np.unique(scores, return_counts=True)
            distribution = ", ".join(
                f"{int(value)}:{int(count)}" for value, count in zip(values, counts)
            )
            print(
                f"  {label:16s}: n={len(scores):3d}  "
                f"mean={np.mean(scores):.2f}  "
                f"range={min(scores)}..{max(scores)}  "
                f"[{distribution}]"
            )
        else:
            print(f"  {label:16s}: no positive examples")

    # ------------------------------------------------------------------
    # 6. Temporal redundancy / leakage
    # ------------------------------------------------------------------
    print()
    print("6. TEMPORAL REDUNDANCY / LEAKAGE")

    starts = np.asarray(
        [float(row["requested_start"]) for row in manifest],
        dtype=np.float64,
    )
    ends = starts + window

    adjacent_overlap = max(0.0, window - stride) / window
    print(
        f"  Adjacent grid windows overlap by {adjacent_overlap:.0%} "
        f"when both are present."
    )

    if len(starts) > 1:
        order = np.argsort(starts)
        sorted_starts = starts[order]
        sorted_ends = ends[order]
        overlaps = np.maximum(
            0.0,
            np.minimum(sorted_ends[:-1], sorted_ends[1:])
            - np.maximum(sorted_starts[:-1], sorted_starts[1:]),
        )
        print(
            f"  Adjacent generated-example overlap: "
            f"{np.mean(overlaps > 0):.1%} of pairs have temporal overlap; "
            f"mean overlap={np.mean(overlaps):.2f}s"
        )

        # Same source annotation with overlapping windows is expected. The
        # important warning is that random train/validation splitting would
        # put near-identical temporal samples in both sets.
        adjacent_same_label = 0
        for left, right in zip(order[:-1], order[1:]):
            if np.any((y[left] == 1) & (y[right] == 1)):
                adjacent_same_label += 1
        print(
            f"  Adjacent pairs sharing at least one label: "
            f"{adjacent_same_label}/{len(order) - 1}"
        )

    print(
        "  IMPORTANT: use grouped/temporal train-validation splitting; "
        "do not randomly split these overlapping windows."
    )

    # ------------------------------------------------------------------
    # 7. Suspicious cases
    # ------------------------------------------------------------------
    print()
    print("7. SUSPICIOUS CASES")

    suspicious: list[str] = []

    for example_id, fractions in near_threshold:
        suspicious.append(
            f"id {example_id}: overlap near minimum threshold ({fractions})"
        )

    for example_id, description in boundary_rows:
        suspicious.append(f"id {example_id}: {description}")

    for i in multi_annotation_ids:
        row = manifest[i]
        suspicious.append(
            f"id {row['example_id']}: multiple source annotations "
            f"({row['annotation_id']})"
        )

    print(f"  Suspicious examples/conditions reported: {len(suspicious)}")
    for item in suspicious[:60]:
        print(f"    {item}")
    if len(suspicious) > 60:
        print(f"    ... {len(suspicious) - 60} more")

    print()
    print("AUDIT COMPLETE")

def write_example_audit(
    path: Path,
    manifest: list[dict],
    y: np.ndarray,
    annotations: list[Annotation],
    stride: float,
    min_overlap: float,
) -> None:
    """Write one inspectable row per generated example, including audit flags."""
    path.parent.mkdir(parents=True, exist_ok=True)
    annotation_lookup = {
        str(i): ann for i, ann in enumerate(annotations, start=1)
    }

    fields = [
        "example_id", "start", "end", "requested_start", "requested_end",
        "duration", "annotation_id", "mtb", "video", "category", "remarks",
        "overlap_fraction", "n_samples", "labels", "is_negative",
        "near_threshold", "near_annotation_boundary", "suspicious",
    ]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for i, row in enumerate(manifest):
            annotation_ids = row["annotation_id"].split(";")
            fractions = [float(v) for v in row["overlap_fraction"].split(";")]
            requested_start = float(row["requested_start"])
            requested_end = float(row["requested_end"])

            near_threshold = any(
                abs(fraction - min_overlap) <= 0.03
                for fraction in fractions
            )
            near_boundary = False
            for annotation_id in annotation_ids:
                ann = annotation_lookup[annotation_id]
                if (
                    abs(requested_start - ann.start) <= stride + 1e-9
                    or abs(requested_end - ann.end) <= stride + 1e-9
                ):
                    near_boundary = True
                    break

            suspicious = near_threshold or near_boundary or ";" in row["annotation_id"]
            labels = "+".join(
                label for j, label in enumerate(LABELS) if y[i, j] == 1
            )

            out = dict(row)
            out.update({
                "labels": labels,
                "is_negative": int(np.all(y[i] == 0)),
                "near_threshold": int(near_threshold),
                "near_annotation_boundary": int(near_boundary),
                "suspicious": int(suspicious),
            })
            writer.writerow(out)


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
    counts: dict[str, int] = {}
    for row in manifest:
        annotation_id = str(row["annotation_id"])
        counts[annotation_id] = counts.get(annotation_id, 0) + 1

    for annotation_id, count in counts.items():
        print(f"  annotation {annotation_id:>4}: {count:4d} windows")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the five-class MTB temporal NN dataset."
    )
    parser.add_argument(
        "--flow-csv",
        type=Path,
        required=False,
        help="Optical-flow CSV produced by analyze_bike_flow.py (required unless --audit-only)",
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
        "--flow-start",
        type=float,
        default=0.0,
        help="Start time of the analyzed flow segment in the source video (default: 0)",
    )
    parser.add_argument(
        "--preview",
        type=int,
        default=40,
        help="Number of generated examples to print (default: 40)",
    )
    parser.add_argument(
        "--audit-pair",
        action="append",
        metavar="FLOW_CSV=ANNOTATIONS_TXT",
        help=(
            "Add a flow CSV / annotation TXT pair for multi-video auditing. "
            "Repeat for each video. Does not alter normal dataset generation."
        ),
    )
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Run the multi-video audit only; requires --audit-pair.",
    )

    args = parser.parse_args()

    if not args.audit_only and args.flow_csv is None:
        parser.error("--flow-csv is required unless --audit-only is used")

    if args.audit_only:
        if not args.audit_pair:
            parser.error("--audit-only requires at least one --audit-pair")
        try:
            pairs = [parse_audit_pair(value) for value in args.audit_pair]
        except ValueError as exc:
            parser.error(str(exc))
        audit_multiple_datasets(
            pairs,
            args.output_dir,
            args.window,
            args.stride,
            args.dt_tolerance,
            args.min_overlap,
            args.flow_start,
        )
        return 0

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
        args.flow_start,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    stem = args.annotations.stem.lower()

    manifest_path = args.output_dir / f"{stem}_dataset_manifest.csv"
    audit_csv_path = args.output_dir / f"{stem}_dataset_audit.csv"
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
    print(f"Flow start:       {args.flow_start:.3f}s")
    print(f"Flow rows:        {len(timestamps)}")
    print(f"Features/row:     {len(feature_names)}")
    print(f"Median dt:        {median_dt:.6f}s")
    print(f"Window:           {args.window:.2f}s")
    print(f"Stride:           {args.stride:.2f}s")
    print(f"Output X shape:   {X.shape}")
    print(f"Output y shape:   {y.shape}")

    print_annotation_audit(annotations)
    audit_dataset(
        annotations,
        manifest,
        y,
        args.window,
        args.stride,
        args.min_overlap,
    )
    write_example_audit(
        audit_csv_path,
        manifest,
        y,
        annotations,
        args.stride,
        args.min_overlap,
    )
    print_dataset_preview(manifest, y, args.preview)
    print_summary(manifest, y)

    print()
    print(f"Manifest: {manifest_path}")
    print(f"Audit CSV: {audit_csv_path}")
    print(f"Dataset:  {npz_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
