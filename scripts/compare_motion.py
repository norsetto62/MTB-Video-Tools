#!/usr/bin/env python3
"""
compare_motion.py

Compare automatic motion scores with manual video annotations.

Motion CSV format:
    timestamp,motion_score
    0.000,0.000000
    0.500,27.761929
    ...

Annotation format:
    start,end,mtb,filmmaking,description

Example:
    start,end,mtb,filmmaking,description
    273,300,5,5,incontro
    860,870,5,3,drop + incontro
    1528,1562,5,4,rock slab

All times are seconds from the beginning of the video.

Usage:
    python compare_motion.py MOTION_CSV ANNOTATIONS_TXT

Example:
python .\scripts\compare_motion.py .\data\motion_scores\DJI_20250710074533_0001_D.csv .\data\annotations\DJI_20250710074533_0001_D.txt
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from statistics import mean, median
from typing import NamedTuple


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class Annotation(NamedTuple):
    start: float
    end: float
    mtb: int
    filmmaking: int
    description: str


class MotionSample(NamedTuple):
    timestamp: float
    score: float


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def format_time(seconds: float) -> str:
    """Format seconds as HH:MM:SS or MM:SS."""
    seconds = max(0.0, seconds)

    total_seconds = int(seconds)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60

    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    return f"{minutes:02d}:{secs:02d}"


def format_number(value: float) -> str:
    """Format a floating-point number consistently."""
    return f"{value:.2f}"


def percentile(values: list[float], p: float) -> float:
    """
    Calculate a percentile using linear interpolation.

    p is in the range 0..100.
    """
    if not values:
        return float("nan")

    if len(values) == 1:
        return values[0]

    values = sorted(values)

    position = (len(values) - 1) * (p / 100.0)
    lower = math.floor(position)
    upper = math.ceil(position)

    if lower == upper:
        return values[lower]

    fraction = position - lower
    return values[lower] + (values[upper] - values[lower]) * fraction


# ---------------------------------------------------------------------------
# Motion CSV
# ---------------------------------------------------------------------------

def load_motion_scores(path: Path) -> list[MotionSample]:
    """Load timestamp/motion_score samples from a CSV file."""

    if not path.exists():
        raise FileNotFoundError(f"Motion CSV not found: {path}")

    samples: list[MotionSample] = []

    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)

        if reader.fieldnames is None:
            raise ValueError(f"Motion CSV has no header: {path}")

        fieldnames = {name.strip() for name in reader.fieldnames}

        if "timestamp" not in fieldnames:
            raise ValueError(
                f"Motion CSV is missing 'timestamp' column: {path}"
            )

        if "motion_score" not in fieldnames:
            raise ValueError(
                f"Motion CSV is missing 'motion_score' column: {path}"
            )

        for row_number, row in enumerate(reader, start=2):
            try:
                timestamp = float(row["timestamp"])
                score = float(row["motion_score"])
            except (TypeError, ValueError) as exc:
                print(
                    f"Warning: skipping invalid row {row_number}: {exc}"
                )
                continue

            if not math.isfinite(timestamp) or not math.isfinite(score):
                continue

            samples.append(MotionSample(timestamp, score))

    if not samples:
        raise ValueError(f"No valid motion samples found in: {path}")

    return samples


# ---------------------------------------------------------------------------
# Annotations
# ---------------------------------------------------------------------------

def load_annotations(path: Path) -> list[Annotation]:
    """
    Load manual annotations.

    Expected format:

        start,end,mtb,filmmaking,description
        273,300,5,5,incontro
        860,870,5,3,drop + incontro

    The parser also accepts:
      - blank lines
      - lines beginning with #
      - whitespace around fields
      - files without the header
    """

    if not path.exists():
        raise FileNotFoundError(f"Annotation file not found: {path}")

    annotations: list[Annotation] = []

    with path.open("r", encoding="utf-8-sig") as f:
        lines = f.readlines()

    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()

        if not line or line.startswith("#"):
            continue

        # Skip the expected header.
        lower = line.lower()
        if lower.startswith("start,end,"):
            continue

        parts = line.split(",", 4)

        if len(parts) < 4:
            print(
                f"Warning: skipping annotation line {line_number}: "
                f"expected at least 4 fields"
            )
            continue

        try:
            start = float(parts[0].strip())
            end = float(parts[1].strip())
            mtb = int(parts[2].strip())
            filmmaking = int(parts[3].strip())
        except ValueError as exc:
            print(
                f"Warning: skipping annotation line {line_number}: {exc}"
            )
            continue

        description = parts[4].strip() if len(parts) >= 5 else ""

        if end <= start:
            print(
                f"Warning: skipping annotation line {line_number}: "
                f"end must be greater than start"
            )
            continue

        if not 1 <= mtb <= 5:
            print(
                f"Warning: skipping annotation line {line_number}: "
                f"MTB rating must be 1-5"
            )
            continue

        if not 1 <= filmmaking <= 5:
            print(
                f"Warning: skipping annotation line {line_number}: "
                f"filmmaking rating must be 1-5"
            )
            continue

        annotations.append(
            Annotation(
                start=start,
                end=end,
                mtb=mtb,
                filmmaking=filmmaking,
                description=description,
            )
        )

    annotations.sort(key=lambda a: a.start)

    return annotations


# ---------------------------------------------------------------------------
# Annotation matching
# ---------------------------------------------------------------------------

def sample_is_annotated(
    timestamp: float,
    annotations: list[Annotation],
) -> bool:
    """Return True if timestamp falls inside any annotation interval."""

    for annotation in annotations:
        if annotation.start <= timestamp <= annotation.end:
            return True

        # Since annotations are sorted, we can stop once we've passed it.
        if annotation.start > timestamp:
            break

    return False


def find_annotation(
    timestamp: float,
    annotations: list[Annotation],
) -> Annotation | None:
    """Return the annotation containing timestamp, if any."""

    for annotation in annotations:
        if annotation.start <= timestamp <= annotation.end:
            return annotation

        if annotation.start > timestamp:
            break

    return None


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def print_statistics(title: str, values: list[float]) -> None:
    """Print basic statistics for a list of motion scores."""

    if not values:
        print(f"{title}")
        print("  No samples")
        return

    print(title)
    print(f"  Samples: {len(values)}")
    print(f"  Mean:    {format_number(mean(values))}")
    print(f"  Median:  {format_number(median(values))}")
    print(f"  Min:     {format_number(min(values))}")
    print(f"  Max:     {format_number(max(values))}")
    print(f"  P90:     {format_number(percentile(values, 90))}")
    print(f"  P95:     {format_number(percentile(values, 95))}")
    print(f"  P99:     {format_number(percentile(values, 99))}")


def print_rating_statistics(
    samples: list[MotionSample],
    annotations: list[Annotation],
    attribute: str,
) -> None:
    """Print statistics grouped by MTB or filmmaking rating."""

    print()
    print("=" * 70)

    if attribute == "mtb":
        title = "MTB rating"
    else:
        title = "Filmmaking rating"

    print(title)
    print("=" * 70)

    for rating in range(1, 6):
        values: list[float] = []

        for sample in samples:
            annotation = find_annotation(sample.timestamp, annotations)

            if annotation is None:
                continue

            if getattr(annotation, attribute) == rating:
                values.append(sample.score)

        if values:
            print(
                f"  {rating}: "
                f"n={len(values):5d}  "
                f"mean={mean(values):7.2f}  "
                f"median={median(values):7.2f}  "
                f"p90={percentile(values, 90):7.2f}"
            )
        else:
            print(f"  {rating}: no samples")


# ---------------------------------------------------------------------------
# High-motion unannotated samples
# ---------------------------------------------------------------------------

def find_high_motion_outside(
    samples: list[MotionSample],
    annotations: list[Annotation],
    count: int,
    percentile_threshold: float,
) -> list[MotionSample]:
    """
    Find the highest motion samples outside manually annotated intervals.

    Only samples at or above percentile_threshold of the complete video
    are considered.
    """

    all_scores = [sample.score for sample in samples]
    threshold = percentile(all_scores, percentile_threshold)

    candidates = [
        sample
        for sample in samples
        if sample.score >= threshold
        and not sample_is_annotated(sample.timestamp, annotations)
    ]

    candidates.sort(key=lambda s: s.score, reverse=True)

    return candidates[:count]


# ---------------------------------------------------------------------------
# Annotation summary
# ---------------------------------------------------------------------------

def print_annotation_summary(
    samples: list[MotionSample],
    annotations: list[Annotation],
) -> None:
    """Print each annotation and its motion statistics."""

    print()
    print("=" * 70)
    print("Manual annotations")
    print("=" * 70)

    if not annotations:
        print("  No annotations.")
        return

    for index, annotation in enumerate(annotations, start=1):
        values = [
            sample.score
            for sample in samples
            if annotation.start <= sample.timestamp <= annotation.end
        ]

        print(
            f"  {index:2d}. "
            f"{format_time(annotation.start)} - "
            f"{format_time(annotation.end)}  "
            f"MTB={annotation.mtb}  "
            f"FILM={annotation.filmmaking}"
        )

        if annotation.description:
            print(f"      {annotation.description}")

        if values:
            print(
                f"      samples={len(values)}  "
                f"mean={mean(values):.2f}  "
                f"median={median(values):.2f}  "
                f"max={max(values):.2f}"
            )
        else:
            print("      no motion samples in interval")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compare automatic motion scores with manual annotations."
        )
    )

    parser.add_argument(
        "motion_csv",
        type=Path,
        help="Motion score CSV produced by analyze_motion.py",
    )

    parser.add_argument(
        "annotations",
        type=Path,
        help="Manual annotation TXT/CSV file",
    )

    parser.add_argument(
        "--top",
        type=int,
        default=20,
        help="Number of high-motion unannotated samples to show "
             "(default: 20)",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=90.0,
        help="Percentile threshold for high-motion samples "
             "(default: 90)",
    )

    args = parser.parse_args()

    if not 0 <= args.threshold <= 100:
        parser.error("--threshold must be between 0 and 100")

    if args.top < 1:
        parser.error("--top must be at least 1")

    try:
        samples = load_motion_scores(args.motion_csv)
        annotations = load_annotations(args.annotations)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}")
        return 1

    # ------------------------------------------------------------------
    # Header
    # ------------------------------------------------------------------

    print()
    print("=" * 70)
    print("Motion vs Manual Annotations")
    print("=" * 70)

    print(f"Motion CSV:   {args.motion_csv}")
    print(f"Annotations:  {args.annotations}")

    first_timestamp = samples[0].timestamp
    last_timestamp = samples[-1].timestamp

    print(
        f"Time range:   {format_time(first_timestamp)} - "
        f"{format_time(last_timestamp)}"
    )

    print(f"Motion samples: {len(samples)}")
    print(f"Annotations:    {len(annotations)}")

    # ------------------------------------------------------------------
    # Overall statistics
    # ------------------------------------------------------------------

    all_values = [sample.score for sample in samples]

    print()
    print("=" * 70)
    print("Overall")
    print("=" * 70)

    print(f"  Samples: {len(all_values)}")
    print(f"  Mean:    {mean(all_values):.2f}")
    print(f"  Median:  {median(all_values):.2f}")
    print(f"  Min:     {min(all_values):.2f}")
    print(f"  Max:     {max(all_values):.2f}")
    print(f"  P90:     {percentile(all_values, 90):.2f}")
    print(f"  P95:     {percentile(all_values, 95):.2f}")
    print(f"  P99:     {percentile(all_values, 99):.2f}")

    # ------------------------------------------------------------------
    # Inside vs outside annotations
    # ------------------------------------------------------------------

    annotated_values: list[float] = []
    outside_values: list[float] = []

    for sample in samples:
        if sample_is_annotated(sample.timestamp, annotations):
            annotated_values.append(sample.score)
        else:
            outside_values.append(sample.score)

    print()
    print("=" * 70)
    print("Annotated vs Outside")
    print("=" * 70)

    print_statistics("Annotated", annotated_values)
    print()
    print_statistics("Outside annotations", outside_values)

    if annotated_values and outside_values:
        annotated_mean = mean(annotated_values)
        outside_mean = mean(outside_values)

        difference = annotated_mean - outside_mean

        if outside_mean != 0:
            percentage = difference / outside_mean * 100
        else:
            percentage = float("nan")

        print()
        print(
            f"Mean difference: "
            f"{difference:+.2f} "
            f"({percentage:+.1f}%)"
        )

    # ------------------------------------------------------------------
    # Rating breakdown
    # ------------------------------------------------------------------

    print_rating_statistics(
        samples,
        annotations,
        "mtb",
    )

    print_rating_statistics(
        samples,
        annotations,
        "filmmaking",
    )

    # ------------------------------------------------------------------
    # Individual annotations
    # ------------------------------------------------------------------

    print_annotation_summary(samples, annotations)

    # ------------------------------------------------------------------
    # High-motion unannotated samples
    # ------------------------------------------------------------------

    outside_peaks = find_high_motion_outside(
        samples,
        annotations,
        args.top,
        args.threshold,
    )

    print()
    print("=" * 70)
    print(
        f"Highest Motion Outside Annotations "
        f"(top {args.top}, P{args.threshold:g}+)"
    )
    print("=" * 70)

    if outside_peaks:
        for index, sample in enumerate(outside_peaks, start=1):
            print(
                f"  {index:2d}. "
                f"{format_time(sample.timestamp)}  "
                f"motion={sample.score:.2f}"
            )
    else:
        print("  No qualifying samples.")

    print()
    print("Done.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())