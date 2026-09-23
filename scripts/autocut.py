#!/usr/bin/env python3
"""
autocut.py

Generate candidate highlight intervals from a motion-analysis CSV.

This is intentionally a first-stage candidate generator, not a final
"good clip" selector.

Pipeline:

    source video
        ↓
    analyze_motion.py
        ↓
    motion_scores/<video>.csv
        ↓
    autocut.py
        ↓
    autocut/<video>/candidates.csv

The script:
- reads an existing motion CSV
- calculates a configurable percentile threshold
- selects samples at or above that threshold
- groups nearby samples into events
- adds configurable padding
- filters very short events
- calculates additional diagnostic statistics
- ranks events by peak motion
- writes candidates.csv

No video rendering is performed in this version.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median


# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROJECT_DATA_DIR = PROJECT_ROOT / "data"
MOTION_DIR = PROJECT_DATA_DIR / "motion_scores"
AUTOCUT_DIR = PROJECT_DATA_DIR / "autocut"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class MotionSample:
    timestamp: float
    score: float


@dataclass
class Candidate:
    start: float
    end: float

    peak_score: float
    peak_time: float

    mean_score: float
    median_score: float

    baseline_mean: float
    baseline_median: float

    motion_excess: float
    peak_ratio: float

    high_motion_count: int
    sample_count: int

    high_motion_fraction: float

    threshold: float

    @property
    def duration(self) -> float:
        return self.end - self.start


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def format_time(seconds: float) -> str:
    """Format seconds as HH:MM:SS or MM:SS."""

    seconds = max(0.0, seconds)

    total_seconds = int(seconds)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60

    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    return f"{minutes:02d}:{secs:02d}"


def percentile(values: list[float], p: float) -> float:
    """
    Calculate percentile using linear interpolation.

    p is expressed as 0-100.
    """

    if not values:
        raise ValueError("Cannot calculate percentile of an empty list.")

    if not 0 <= p <= 100:
        raise ValueError("Percentile must be between 0 and 100.")

    values = sorted(values)

    if len(values) == 1:
        return values[0]

    position = (len(values) - 1) * (p / 100.0)

    lower = math.floor(position)
    upper = math.ceil(position)

    if lower == upper:
        return values[lower]

    fraction = position - lower

    return values[lower] + (
        values[upper] - values[lower]
    ) * fraction


def parse_float(
    value: str,
    field_name: str,
    row_number: int,
) -> float:
    """Parse a CSV numeric field with a useful error message."""

    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(
            f"Invalid {field_name} on CSV row "
            f"{row_number}: {value!r}"
        ) from exc


# ---------------------------------------------------------------------------
# Motion CSV handling
# ---------------------------------------------------------------------------

def load_motion_csv(csv_path: Path) -> list[MotionSample]:
    """Load timestamp/motion_score samples from a motion CSV."""

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Motion CSV not found: {csv_path}"
        )

    samples: list[MotionSample] = []

    with csv_path.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as f:
        reader = csv.DictReader(f)

        if reader.fieldnames is None:
            raise ValueError(
                f"CSV has no header: {csv_path}"
            )

        required = {
            "timestamp",
            "motion_score",
        }

        missing = required - set(reader.fieldnames)

        if missing:
            raise ValueError(
                f"CSV is missing required column(s): "
                f"{', '.join(sorted(missing))}"
            )

        for row_number, row in enumerate(reader, start=2):
            timestamp = parse_float(
                row["timestamp"],
                "timestamp",
                row_number,
            )

            score = parse_float(
                row["motion_score"],
                "motion_score",
                row_number,
            )

            samples.append(
                MotionSample(
                    timestamp=timestamp,
                    score=score,
                )
            )

    if not samples:
        raise ValueError(
            f"Motion CSV contains no samples: {csv_path}"
        )

    samples.sort(
        key=lambda sample: sample.timestamp
    )

    return samples


def find_motion_csv(video_path: Path) -> Path:
    """
    Find the motion CSV corresponding to a video.

    The normal location is:

        data/motion_scores/<video-stem>.csv
    """

    csv_path = (
        MOTION_DIR
        / f"{video_path.stem}.csv"
    )

    if csv_path.exists():
        return csv_path

    raise FileNotFoundError(
        "Could not find motion CSV for video.\n"
        f"Video:       {video_path}\n"
        f"Expected at: {csv_path}\n\n"
        "Run analyze_motion.py for this video first, or use "
        "--motion-csv to specify the CSV explicitly."
    )


# ---------------------------------------------------------------------------
# Candidate generation
# ---------------------------------------------------------------------------

def select_high_motion_samples(
    samples: list[MotionSample],
    threshold: float,
) -> list[MotionSample]:
    """Return samples whose score is at or above the threshold."""

    return [
        sample
        for sample in samples
        if sample.score >= threshold
    ]


def group_samples(
    samples: list[MotionSample],
    gap: float,
) -> list[list[MotionSample]]:
    """
    Group selected motion samples into events.

    Two samples belong to the same event when the time between them
    is no greater than 'gap'.
    """

    if not samples:
        return []

    samples = sorted(
        samples,
        key=lambda sample: sample.timestamp,
    )

    groups: list[list[MotionSample]] = []

    current_group = [samples[0]]

    for sample in samples[1:]:
        previous = current_group[-1]

        if (
            sample.timestamp
            - previous.timestamp
            <= gap
        ):
            current_group.append(sample)
        else:
            groups.append(current_group)
            current_group = [sample]

    groups.append(current_group)

    return groups


def samples_in_interval(
    samples: list[MotionSample],
    start: float,
    end: float,
) -> list[MotionSample]:
    """Return all motion samples inside an interval."""

    return [
        sample
        for sample in samples
        if start <= sample.timestamp <= end
    ]


def build_candidate(
    group: list[MotionSample],
    all_samples: list[MotionSample],
    threshold: float,
    padding: float,
) -> Candidate:
    """
    Convert one group of high-motion samples into a candidate.

    Two types of statistics are calculated:

    High-motion statistics:
        calculated only from samples >= threshold.

    Baseline statistics:
        calculated from every motion sample inside the padded
        candidate interval.

    This distinction is useful because a candidate containing one
    isolated motion spike should look different from one containing
    sustained motion.
    """

    peak_sample = max(
        group,
        key=lambda sample: sample.score,
    )

    raw_start = group[0].timestamp
    raw_end = group[-1].timestamp

    start = max(
        0.0,
        raw_start - padding,
    )

    end = raw_end + padding

    interval_samples = samples_in_interval(
        all_samples,
        start,
        end,
    )

    if not interval_samples:
        # This should not normally happen because the high-motion
        # group itself lies inside the interval.
        interval_samples = group

    high_motion_scores = [
        sample.score
        for sample in group
    ]

    baseline_scores = [
        sample.score
        for sample in interval_samples
    ]

    high_motion_mean = mean(
        high_motion_scores
    )

    high_motion_median = median(
        high_motion_scores
    )

    baseline_mean = mean(
        baseline_scores
    )

    baseline_median = median(
        baseline_scores
    )

    motion_excess = (
        high_motion_mean
        - baseline_mean
    )

    if baseline_mean > 0:
        peak_ratio = (
            peak_sample.score
            / baseline_mean
        )
    else:
        peak_ratio = float("inf")

    high_motion_count = len(group)
    sample_count = len(interval_samples)

    high_motion_fraction = (
        high_motion_count / sample_count
        if sample_count > 0
        else 0.0
    )

    return Candidate(
        start=start,
        end=end,
        peak_score=peak_sample.score,
        peak_time=peak_sample.timestamp,
        mean_score=high_motion_mean,
        median_score=high_motion_median,
        baseline_mean=baseline_mean,
        baseline_median=baseline_median,
        motion_excess=motion_excess,
        peak_ratio=peak_ratio,
        high_motion_count=high_motion_count,
        sample_count=sample_count,
        high_motion_fraction=high_motion_fraction,
        threshold=threshold,
    )


def generate_candidates(
    samples: list[MotionSample],
    threshold: float,
    gap: float,
    padding: float,
    min_duration: float,
) -> list[Candidate]:
    """Generate and filter candidate intervals."""

    selected = select_high_motion_samples(
        samples,
        threshold,
    )

    if not selected:
        return []

    groups = group_samples(
        selected,
        gap,
    )

    candidates: list[Candidate] = []

    for group in groups:
        candidate = build_candidate(
            group=group,
            all_samples=samples,
            threshold=threshold,
            padding=padding,
        )

        if candidate.duration < min_duration:
            continue

        candidates.append(candidate)

    return candidates


def rank_candidates(
    candidates: list[Candidate],
) -> list[Candidate]:
    """
    Rank candidates.

    The ranking is intentionally still based primarily on peak
    motion. We are collecting the additional statistics first,
    before deciding whether the ranking formula should change.
    """

    return sorted(
        candidates,
        key=lambda candidate: (
            candidate.peak_score,
            candidate.mean_score,
            candidate.duration,
        ),
        reverse=True,
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_statistics(
    samples: list[MotionSample],
    threshold: float,
) -> None:
    """Print basic statistics for the motion data."""

    scores = [
        sample.score
        for sample in samples
    ]

    duration = (
        samples[-1].timestamp
        if samples
        else 0.0
    )

    print()
    print("Motion statistics:")

    print(
        f"  Samples:       {len(samples)}"
    )

    print(
        f"  Duration:      {format_time(duration)}"
    )

    print(
        f"  Mean:          {mean(scores):.2f}"
    )

    print(
        f"  Median:        {median(scores):.2f}"
    )

    print(
        f"  Min:           {min(scores):.2f}"
    )

    print(
        f"  Max:           {max(scores):.2f}"
    )

    print(
        f"  P90:           {percentile(scores, 90):.2f}"
    )

    print(
        f"  P95:           {percentile(scores, 95):.2f}"
    )

    print(
        f"  P99:           {percentile(scores, 99):.2f}"
    )

    print(
        f"  Threshold:     {threshold:.2f}"
    )


def print_candidates(
    candidates: list[Candidate],
    top: int,
) -> None:
    """Print candidate intervals and diagnostics."""

    print()
    print("Candidates:")

    if not candidates:
        print("  No candidates found.")
        return

    displayed = candidates[:top]

    for index, candidate in enumerate(
        displayed,
        start=1,
    ):
        print(
            f"  {index:02d}. "
            f"{format_time(candidate.start)}-"
            f"{format_time(candidate.end)} "
            f"dur={candidate.duration:.1f}s "
            f"peak={candidate.peak_score:.2f} "
            f"@{format_time(candidate.peak_time)} "
            f"high_mean={candidate.mean_score:.2f} "
            f"base_mean={candidate.baseline_mean:.2f} "
            f"excess={candidate.motion_excess:.2f} "
            f"high={candidate.high_motion_count}/"
            f"{candidate.sample_count} "
            f"({candidate.high_motion_fraction:.0%})"
        )

    if len(candidates) > top:
        print(
            f"  ... {len(candidates) - top} more "
            f"candidate(s) not displayed."
        )


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

def write_candidates_csv(
    output_path: Path,
    candidates: list[Candidate],
) -> None:
    """Write ranked candidates to candidates.csv."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "rank",
        "start",
        "end",
        "duration",
        "start_time",
        "end_time",
        "peak_time",
        "peak_score",
        "mean_score",
        "median_score",
        "baseline_mean",
        "baseline_median",
        "motion_excess",
        "peak_ratio",
        "high_motion_count",
        "sample_count",
        "high_motion_fraction",
        "threshold",
    ]

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for rank, candidate in enumerate(
            candidates,
            start=1,
        ):
            writer.writerow(
                {
                    "rank": rank,
                    "start": (
                        f"{candidate.start:.3f}"
                    ),
                    "end": (
                        f"{candidate.end:.3f}"
                    ),
                    "duration": (
                        f"{candidate.duration:.3f}"
                    ),
                    "start_time": format_time(
                        candidate.start
                    ),
                    "end_time": format_time(
                        candidate.end
                    ),
                    "peak_time": (
                        f"{candidate.peak_time:.3f}"
                    ),
                    "peak_score": (
                        f"{candidate.peak_score:.6f}"
                    ),
                    "mean_score": (
                        f"{candidate.mean_score:.6f}"
                    ),
                    "median_score": (
                        f"{candidate.median_score:.6f}"
                    ),
                    "baseline_mean": (
                        f"{candidate.baseline_mean:.6f}"
                    ),
                    "baseline_median": (
                        f"{candidate.baseline_median:.6f}"
                    ),
                    "motion_excess": (
                        f"{candidate.motion_excess:.6f}"
                    ),
                    "peak_ratio": (
                        f"{candidate.peak_ratio:.6f}"
                    ),
                    "high_motion_count": (
                        candidate.high_motion_count
                    ),
                    "sample_count": (
                        candidate.sample_count
                    ),
                    "high_motion_fraction": (
                        f"{candidate.high_motion_fraction:.6f}"
                    ),
                    "threshold": (
                        f"{candidate.threshold:.6f}"
                    ),
                }
            )


# ---------------------------------------------------------------------------
# Argument handling
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate candidate highlight intervals "
            "from a motion CSV."
        )
    )

    parser.add_argument(
        "video",
        type=Path,
        help=(
            "Source video used to locate the corresponding "
            "motion CSV."
        ),
    )

    parser.add_argument(
        "--motion-csv",
        type=Path,
        default=None,
        help=(
            "Explicit motion CSV path. If omitted, the script "
            "looks in data\\motion_scores using the video's "
            "filename."
        ),
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=90.0,
        help=(
            "Motion percentile threshold (default: 90). "
            "For example, 90 means P90."
        ),
    )

    parser.add_argument(
        "--gap",
        type=float,
        default=3.0,
        help=(
            "Maximum gap in seconds between high-motion samples "
            "before starting a new candidate (default: 3)."
        ),
    )

    parser.add_argument(
        "--padding",
        type=float,
        default=5.0,
        help=(
            "Seconds added before and after each detected event "
            "(default: 5)."
        ),
    )

    parser.add_argument(
        "--min-duration",
        type=float,
        default=4.0,
        help=(
            "Minimum candidate duration in seconds "
            "(default: 4)."
        ),
    )

    parser.add_argument(
        "--top",
        type=int,
        default=20,
        help=(
            "Number of candidates to display in the terminal "
            "(default: 20)."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Optional output CSV path. If omitted, writes to "
            "data\\autocut\\<video-stem>\\candidates.csv."
        ),
    )

    return parser


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    video_path = args.video

    if args.threshold < 0 or args.threshold > 100:
        parser.error(
            "--threshold must be between 0 and 100."
        )

    if args.gap < 0:
        parser.error(
            "--gap cannot be negative."
        )

    if args.padding < 0:
        parser.error(
            "--padding cannot be negative."
        )

    if args.min_duration < 0:
        parser.error(
            "--min-duration cannot be negative."
        )

    if args.top < 1:
        parser.error(
            "--top must be at least 1."
        )

    print(
        "MTB AutoCut - Candidate Generator"
    )
    print(
        "----------------------------------"
    )

    print(
        f"Video:       {video_path}"
    )

    # Locate motion CSV.
    if args.motion_csv is not None:
        motion_csv = args.motion_csv
    else:
        motion_csv = find_motion_csv(
            video_path
        )

    print(
        f"Motion CSV:  {motion_csv}"
    )

    # Load motion data.
    samples = load_motion_csv(
        motion_csv
    )

    scores = [
        sample.score
        for sample in samples
    ]

    threshold_value = percentile(
        scores,
        args.threshold,
    )

    print(
        f"Threshold:   P{args.threshold:g} "
        f"= {threshold_value:.2f}"
    )

    print(
        f"Gap:         {args.gap:.1f}s"
    )

    print(
        f"Padding:     {args.padding:.1f}s"
    )

    print(
        f"Min duration:{args.min_duration:.1f}s"
    )

    print_statistics(
        samples=samples,
        threshold=threshold_value,
    )

    # Generate candidates.
    candidates = generate_candidates(
        samples=samples,
        threshold=threshold_value,
        gap=args.gap,
        padding=args.padding,
        min_duration=args.min_duration,
    )

    candidates = rank_candidates(
        candidates
    )

    print_candidates(
        candidates=candidates,
        top=args.top,
    )

    # Output.
    if args.output is not None:
        output_path = args.output
    else:
        output_path = (
            AUTOCUT_DIR
            / video_path.stem
            / "candidates.csv"
        )

    write_candidates_csv(
        output_path=output_path,
        candidates=candidates,
    )

    print()
    print(
        f"Candidates found: {len(candidates)}"
    )
    print(
        f"Output:            {output_path}"
    )


if __name__ == "__main__":
    main()