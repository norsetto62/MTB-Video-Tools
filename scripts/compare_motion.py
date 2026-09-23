#!/usr/bin/env python3
"""
compare_motion.py

Compare automatic motion scores with manual annotations.

The purpose of this script is to evaluate how useful the motion detector
is for finding interesting MTB/video moments.

Annotation file format
----------------------

    Start    End   MTB   Video   Remarks
    D:\\Prenestini\\Mentorella\\DJI_20250710075648_0002_D.MP4

    00:00   00:34   3      5     Landscape and path start
    00:45   00:53   5      3     Technical climb
    04:55   05:15   3      3     Pushing the bike uphill

Usage
-----

    python .\\scripts\\compare_motion.py .\\data\\annotations\\Mentorella.txt

Optional:

    --motion-csv PATH
        Explicitly specify the motion-score CSV.

    --threshold PERCENTILE
        Percentile used to define high-motion samples.
        Default: 90

    --gap SECONDS
        Maximum gap between high-motion samples for them to belong
        to the same event.
        Default: 3

    --padding SECONDS
        Extend detected events by this many seconds when checking
        overlap with manual annotations.
        Default: 2

    --min-duration SECONDS
        Ignore detected events shorter than this duration.
        Default: 2

    --top N
        Maximum number of unannotated high-motion events to display.
        Default: 20
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
    video: int
    remarks: str


class MotionSample(NamedTuple):
    timestamp: float
    score: float


class MotionEvent(NamedTuple):
    start: float
    end: float
    peak_time: float
    peak_score: float
    mean_score: float
    samples: int


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def format_time(seconds: float) -> str:
    """Format seconds as MM:SS or HH:MM:SS."""

    seconds = max(0.0, seconds)

    total_seconds = int(seconds)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60

    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    return f"{minutes:02d}:{secs:02d}"


def parse_time(value: str) -> float:
    """Parse MM:SS, HH:MM:SS, or plain seconds."""

    value = value.strip()

    if not value:
        raise ValueError("empty time")

    parts = value.split(":")

    try:
        if len(parts) == 1:
            return float(parts[0])

        if len(parts) == 2:
            minutes = int(parts[0])
            seconds = float(parts[1])

            if not 0 <= seconds < 60:
                raise ValueError(
                    "seconds must be between 0 and 59.999"
                )

            return minutes * 60 + seconds

        if len(parts) == 3:
            hours = int(parts[0])
            minutes = int(parts[1])
            seconds = float(parts[2])

            if not 0 <= minutes < 60:
                raise ValueError(
                    "minutes must be between 0 and 59"
                )

            if not 0 <= seconds < 60:
                raise ValueError(
                    "seconds must be between 0 and 59.999"
                )

            return hours * 3600 + minutes * 60 + seconds

    except ValueError as exc:
        raise ValueError(
            f"invalid time '{value}': {exc}"
        ) from exc

    raise ValueError(f"invalid time '{value}'")


def percentile(values: list[float], p: float) -> float:
    """Calculate percentile using linear interpolation."""

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

    return (
        values[lower]
        + (values[upper] - values[lower]) * fraction
    )


# ---------------------------------------------------------------------------
# Annotation file
# ---------------------------------------------------------------------------

def load_annotation_file(
    path: Path,
) -> tuple[Path, list[Annotation]]:

    if not path.exists():
        raise FileNotFoundError(
            f"Annotation file not found: {path}"
        )

    lines = path.read_text(
        encoding="utf-8-sig"
    ).splitlines()

    video_path: Path | None = None
    annotations: list[Annotation] = []

    for line_number, raw_line in enumerate(
        lines,
        start=1,
    ):

        line = raw_line.strip()

        if not line or line.startswith("#"):
            continue

        lower = line.lower()

        if (
            "start" in lower
            and "end" in lower
            and "mtb" in lower
            and "video" in lower
        ):
            continue

        # Windows absolute path.
        if video_path is None and (
            len(line) >= 3
            and line[1] == ":"
            and line[2] in ("\\", "/")
        ):
            video_path = Path(line)
            continue

        parts = line.split(maxsplit=4)

        if len(parts) < 4:
            print(
                f"Warning: skipping line {line_number}: "
                "not enough fields"
            )
            continue

        try:
            start = parse_time(parts[0])
            end = parse_time(parts[1])
            mtb = int(parts[2])
            video_rating = int(parts[3])

        except ValueError as exc:
            print(
                f"Warning: skipping line {line_number}: {exc}"
            )
            continue

        remarks = (
            parts[4].strip()
            if len(parts) >= 5
            else ""
        )

        if end <= start:
            print(
                f"Warning: skipping line {line_number}: "
                "End must be greater than Start"
            )
            continue

        if not 1 <= mtb <= 5:
            print(
                f"Warning: skipping line {line_number}: "
                "MTB rating must be 1-5"
            )
            continue

        if not 1 <= video_rating <= 5:
            print(
                f"Warning: skipping line {line_number}: "
                "Video rating must be 1-5"
            )
            continue

        annotations.append(
            Annotation(
                start=start,
                end=end,
                mtb=mtb,
                video=video_rating,
                remarks=remarks,
            )
        )

    if video_path is None:
        raise ValueError(
            f"No video path found in annotation file: {path}"
        )

    annotations.sort(key=lambda a: a.start)

    return video_path, annotations


# ---------------------------------------------------------------------------
# Motion CSV
# ---------------------------------------------------------------------------

def load_motion_scores(
    path: Path,
) -> list[MotionSample]:

    if not path.exists():
        raise FileNotFoundError(
            f"Motion CSV not found: {path}"
        )

    samples: list[MotionSample] = []

    with path.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as f:

        reader = csv.DictReader(f)

        if reader.fieldnames is None:
            raise ValueError(
                f"Motion CSV has no header: {path}"
            )

        fieldnames = {
            name.strip()
            for name in reader.fieldnames
        }

        if "timestamp" not in fieldnames:
            raise ValueError(
                "Motion CSV is missing 'timestamp' column"
            )

        if "motion_score" not in fieldnames:
            raise ValueError(
                "Motion CSV is missing 'motion_score' column"
            )

        for row_number, row in enumerate(
            reader,
            start=2,
        ):

            try:
                timestamp = float(row["timestamp"])
                score = float(row["motion_score"])

            except (TypeError, ValueError) as exc:
                print(
                    f"Warning: skipping CSV row "
                    f"{row_number}: {exc}"
                )
                continue

            if not math.isfinite(timestamp):
                continue

            if not math.isfinite(score):
                continue

            samples.append(
                MotionSample(
                    timestamp=timestamp,
                    score=score,
                )
            )

    if not samples:
        raise ValueError(
            f"No valid motion samples found in: {path}"
        )

    samples.sort(key=lambda s: s.timestamp)

    return samples


# ---------------------------------------------------------------------------
# Annotation matching
# ---------------------------------------------------------------------------

def find_annotation(
    timestamp: float,
    annotations: list[Annotation],
) -> Annotation | None:

    for annotation in annotations:

        if annotation.start <= timestamp <= annotation.end:
            return annotation

        if annotation.start > timestamp:
            break

    return None


def sample_is_annotated(
    timestamp: float,
    annotations: list[Annotation],
) -> bool:

    return find_annotation(
        timestamp,
        annotations,
    ) is not None


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def print_statistics(
    title: str,
    values: list[float],
) -> None:

    print(title)

    if not values:
        print("  No samples")
        return

    print(f"  Samples: {len(values)}")
    print(f"  Mean:    {mean(values):.2f}")
    print(f"  Median:  {median(values):.2f}")
    print(f"  Min:     {min(values):.2f}")
    print(f"  Max:     {max(values):.2f}")
    print(f"  P90:     {percentile(values, 90):.2f}")
    print(f"  P95:     {percentile(values, 95):.2f}")
    print(f"  P99:     {percentile(values, 99):.2f}")


def print_rating_statistics(
    samples: list[MotionSample],
    annotations: list[Annotation],
    attribute: str,
) -> None:

    if attribute == "mtb":
        title = "MTB rating"
        label = "MTB"
    else:
        title = "Video rating"
        label = "Video"

    print()
    print("=" * 70)
    print(title)
    print("=" * 70)

    for rating in range(1, 6):

        values: list[float] = []

        for sample in samples:

            annotation = find_annotation(
                sample.timestamp,
                annotations,
            )

            if annotation is None:
                continue

            if getattr(annotation, attribute) == rating:
                values.append(sample.score)

        if values:
            print(
                f"  {label} {rating}: "
                f"n={len(values):5d}  "
                f"mean={mean(values):7.2f}  "
                f"median={median(values):7.2f}  "
                f"p90={percentile(values, 90):7.2f}"
            )
        else:
            print(
                f"  {label} {rating}: no samples"
            )


# ---------------------------------------------------------------------------
# Automatic high-motion event detection
# ---------------------------------------------------------------------------

def detect_motion_events(
    samples: list[MotionSample],
    threshold: float,
    gap: float,
    min_duration: float,
) -> list[MotionEvent]:
    """
    Group consecutive high-motion samples into events.

    A new event begins when:
      - a sample is below threshold, or
      - the gap from the previous high-motion sample exceeds `gap`.

    Events shorter than min_duration are discarded.
    """

    high_samples = [
        sample
        for sample in samples
        if sample.score >= threshold
    ]

    if not high_samples:
        return []

    events: list[MotionEvent] = []

    current: list[MotionSample] = [
        high_samples[0]
    ]

    for sample in high_samples[1:]:

        previous = current[-1]

        if sample.timestamp - previous.timestamp <= gap:
            current.append(sample)
            continue

        event = make_motion_event(current)

        if event.end - event.start >= min_duration:
            events.append(event)

        current = [sample]

    event = make_motion_event(current)

    if event.end - event.start >= min_duration:
        events.append(event)

    return events


def make_motion_event(
    samples: list[MotionSample],
) -> MotionEvent:

    peak = max(
        samples,
        key=lambda sample: sample.score,
    )

    return MotionEvent(
        start=samples[0].timestamp,
        end=samples[-1].timestamp,
        peak_time=peak.timestamp,
        peak_score=peak.score,
        mean_score=mean(
            sample.score
            for sample in samples
        ),
        samples=len(samples),
    )


# ---------------------------------------------------------------------------
# Event / annotation overlap
# ---------------------------------------------------------------------------

def overlap_seconds(
    start1: float,
    end1: float,
    start2: float,
    end2: float,
) -> float:

    return max(
        0.0,
        min(end1, end2)
        - max(start1, start2),
    )


def event_annotations(
    event: MotionEvent,
    annotations: list[Annotation],
    padding: float = 0.0,
) -> list[Annotation]:

    result: list[Annotation] = []

    event_start = event.start - padding
    event_end = event.end + padding

    for annotation in annotations:

        if overlap_seconds(
            event_start,
            event_end,
            annotation.start,
            annotation.end,
        ) > 0:
            result.append(annotation)

    return result


# ---------------------------------------------------------------------------
# Detected event report
# ---------------------------------------------------------------------------

def print_detected_events(
    events: list[MotionEvent],
    annotations: list[Annotation],
    title: str,
    limit: int,
    padding: float,
    only_unannotated: bool = False,
) -> None:

    print()
    print("=" * 70)
    print(title)
    print("=" * 70)

    if not events:
        print("  No events found.")
        return

    shown = 0

    for event in events:

        matches = event_annotations(
            event,
            annotations,
            padding,
        )

        if only_unannotated and matches:
            continue

        shown += 1

        duration = event.end - event.start

        label = ""

        if matches:
            ratings = ", ".join(
                f"MTB={a.mtb}/Video={a.video}"
                for a in matches
            )

            remarks = "; ".join(
                a.remarks
                for a in matches
                if a.remarks
            )

            label = f"  [{ratings}]"

            if remarks:
                label += f"  {remarks}"

        print(
            f"  {shown:2d}. "
            f"{format_time(event.start)} - "
            f"{format_time(event.end)}  "
            f"dur={duration:5.1f}s  "
            f"peak={event.peak_score:6.2f} "
            f"@ {format_time(event.peak_time)}  "
            f"mean={event.mean_score:6.2f}  "
            f"n={event.samples:3d}"
            f"{label}"
        )

        if shown >= limit:
            break

    if shown == 0:
        print("  No matching events found.")


# ---------------------------------------------------------------------------
# Manual annotation detection report
# ---------------------------------------------------------------------------

def print_annotation_detection(
    annotations: list[Annotation],
    events: list[MotionEvent],
    padding: float,
) -> None:

    print()
    print("=" * 70)
    print("Manual annotations vs detected motion events")
    print("=" * 70)

    for index, annotation in enumerate(
        annotations,
        start=1,
    ):

        matches: list[MotionEvent] = []

        for event in events:

            if overlap_seconds(
                annotation.start,
                annotation.end,
                event.start - padding,
                event.end + padding,
            ) > 0:
                matches.append(event)

        if matches:

            best = max(
                matches,
                key=lambda event: event.peak_score,
            )

            print(
                f"  {index:2d}. "
                f"{format_time(annotation.start)} - "
                f"{format_time(annotation.end)}  "
                f"MTB={annotation.mtb}  "
                f"Video={annotation.video}"
            )

            print(
                f"      DETECTED: "
                f"peak={best.peak_score:.2f} "
                f"@ {format_time(best.peak_time)}"
            )

        else:

            print(
                f"  {index:2d}. "
                f"{format_time(annotation.start)} - "
                f"{format_time(annotation.end)}  "
                f"MTB={annotation.mtb}  "
                f"Video={annotation.video}"
            )

            print(
                "      NOT DETECTED by current "
                "motion threshold"
            )

        if annotation.remarks:
            print(
                f"      {annotation.remarks}"
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Compare automatic motion scores "
            "with manual annotations."
        )
    )

    parser.add_argument(
        "annotation_file",
        type=Path,
        help="Manual annotation file.",
    )

    parser.add_argument(
        "--motion-csv",
        type=Path,
        default=None,
        help="Explicit motion-score CSV.",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=90.0,
        help="High-motion percentile (default: 90).",
    )

    parser.add_argument(
        "--gap",
        type=float,
        default=3.0,
        help="Maximum gap between high-motion samples (default: 3s).",
    )

    parser.add_argument(
        "--padding",
        type=float,
        default=2.0,
        help="Detection overlap padding (default: 2s).",
    )

    parser.add_argument(
        "--min-duration",
        type=float,
        default=2.0,
        help="Minimum detected event duration (default: 2s).",
    )

    parser.add_argument(
        "--top",
        type=int,
        default=20,
        help="Maximum events to display (default: 20).",
    )

    args = parser.parse_args()

    if not 0 <= args.threshold <= 100:
        raise ValueError(
            "--threshold must be between 0 and 100."
        )

    if args.gap < 0:
        raise ValueError(
            "--gap must be >= 0."
        )

    if args.padding < 0:
        raise ValueError(
            "--padding must be >= 0."
        )

    if args.min_duration < 0:
        raise ValueError(
            "--min-duration must be >= 0."
        )

    if args.top < 1:
        raise ValueError(
            "--top must be >= 1."
        )

    annotation_file = (
        args.annotation_file.resolve()
    )

    # -----------------------------------------------------------------------
    # Load
    # -----------------------------------------------------------------------

    video_path, annotations = (
        load_annotation_file(annotation_file)
    )

    motion_csv = find_motion_csv(
        annotation_file,
        video_path,
        args.motion_csv,
    )

    samples = load_motion_scores(
        motion_csv
    )

    all_scores = [
        sample.score
        for sample in samples
    ]

    threshold = percentile(
        all_scores,
        args.threshold,
    )

    # -----------------------------------------------------------------------
    # Header
    # -----------------------------------------------------------------------

    print()
    print("=" * 70)
    print("Motion / Manual Annotation Comparison")
    print("=" * 70)

    print(
        f"Annotation file: {annotation_file}"
    )

    print(
        f"Video:           {video_path}"
    )

    print(
        f"Motion CSV:      {motion_csv}"
    )

    print(
        f"Annotations:     {len(annotations)}"
    )

    print(
        f"Motion samples:  {len(samples)}"
    )

    print(
        f"Threshold:       P{args.threshold:.0f} "
        f"= {threshold:.2f}"
    )

    # -----------------------------------------------------------------------
    # Basic statistics
    # -----------------------------------------------------------------------

    annotated_values: list[float] = []
    outside_values: list[float] = []

    for sample in samples:

        if sample_is_annotated(
            sample.timestamp,
            annotations,
        ):
            annotated_values.append(
                sample.score
            )
        else:
            outside_values.append(
                sample.score
            )

    print()
    print("=" * 70)
    print("Overall motion statistics")
    print("=" * 70)

    print_statistics(
        "All samples",
        all_scores,
    )

    print()
    print_statistics(
        "Annotated samples",
        annotated_values,
    )

    print()
    print_statistics(
        "Outside annotations",
        outside_values,
    )

    # -----------------------------------------------------------------------
    # Ratings
    # -----------------------------------------------------------------------

    print_rating_statistics(
        samples,
        annotations,
        "mtb",
    )

    print_rating_statistics(
        samples,
        annotations,
        "video",
    )

    # -----------------------------------------------------------------------
    # Manual annotations
    # -----------------------------------------------------------------------

    print()
    print("=" * 70)
    print("Manual annotations")
    print("=" * 70)

    for index, annotation in enumerate(
        annotations,
        start=1,
    ):

        values = [
            sample.score
            for sample in samples
            if (
                annotation.start
                <= sample.timestamp
                <= annotation.end
            )
        ]

        print(
            f"  {index:2d}. "
            f"{format_time(annotation.start)} - "
            f"{format_time(annotation.end)}  "
            f"MTB={annotation.mtb}  "
            f"Video={annotation.video}"
        )

        if annotation.remarks:
            print(
                f"      {annotation.remarks}"
            )

        if values:
            print(
                f"      samples={len(values)}  "
                f"mean={mean(values):.2f}  "
                f"median={median(values):.2f}  "
                f"max={max(values):.2f}"
            )
        else:
            print(
                "      no motion samples"
            )

    # -----------------------------------------------------------------------
    # Detect events
    # -----------------------------------------------------------------------

    events = detect_motion_events(
        samples,
        threshold,
        args.gap,
        args.min_duration,
    )

    # -----------------------------------------------------------------------
    # Event report
    # -----------------------------------------------------------------------

    print_detected_events(
        events,
        annotations,
        (
            f"Detected high-motion events "
            f"(P{args.threshold:.0f})"
        ),
        args.top,
        args.padding,
    )

    # -----------------------------------------------------------------------
    # Detection of manual annotations
    # -----------------------------------------------------------------------

    print_annotation_detection(
        annotations,
        events,
        args.padding,
    )

    # -----------------------------------------------------------------------
    # Unannotated candidates
    # -----------------------------------------------------------------------

    print_detected_events(
        events,
        annotations,
        (
            f"Potential missed events "
            f"(high motion, currently unannotated)"
        ),
        args.top,
        args.padding,
        only_unannotated=True,
    )


# ---------------------------------------------------------------------------
# Motion CSV discovery
# ---------------------------------------------------------------------------

def find_motion_csv(
    annotation_file: Path,
    video_path: Path,
    explicit_motion_csv: Path | None,
) -> Path:

    if explicit_motion_csv is not None:

        path = explicit_motion_csv

        if not path.exists():
            raise FileNotFoundError(
                f"Specified motion CSV not found: {path}"
            )

        return path

    csv_name = video_path.stem + ".csv"

    project_data_dir = (
        annotation_file.parent.parent
    )

    candidate = (
        project_data_dir
        / "motion_scores"
        / csv_name
    )

    if candidate.exists():
        return candidate

    candidate = (
        annotation_file.parent
        / csv_name
    )

    if candidate.exists():
        return candidate

    raise FileNotFoundError(
        "Could not find motion-score CSV.\n"
        f"Expected:\n  "
        f"{project_data_dir / 'motion_scores' / csv_name}\n"
        f"for video:\n  {video_path}\n\n"
        "Run analyze_motion.py first, or specify "
        "--motion-csv explicitly."
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()