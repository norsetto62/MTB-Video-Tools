#!/usr/bin/env python3

"""
Audit MTB annotation durations across one or more annotation files.

Expected annotation format:

    Start   End     MTB     Remarks
    D:/Ascoli/DJI_20250923122005_0005_D.MP4
    00:00   01:07   0       Dismounted/Push
    01:07   01:23   3       Drop and Rock Garden
    01:23   01:28   0       Stopped/Brambles

Reports, for each MTB score:
- number of annotation intervals
- total annotated duration
- min / max duration
- mean / median duration
- P25 / P75
- number and percentage of intervals below each threshold
- total time and percentage below each threshold

Also prints every short MTB2/MTB3 annotation below --short-threshold.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Annotation:
    source_file: Path
    video: str
    start: float
    end: float
    score: int
    remarks: str

    @property
    def duration(self) -> float:
        return self.end - self.start


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit MTB annotation duration distributions."
    )

    parser.add_argument(
        "--annotations",
        nargs="+",
        required=True,
        help="One or more MTB annotation .txt files.",
    )

    parser.add_argument(
        "--short-threshold",
        type=float,
        default=4.0,
        help=(
            "Threshold in seconds for the detailed short MTB2/MTB3 "
            "list (default: 4.0)."
        ),
    )

    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[2.0, 4.0, 6.0, 8.0],
        help=(
            "Duration thresholds in seconds used in the summary "
            "(default: 2 4 6 8)."
        ),
    )

    parser.add_argument(
        "--sort-short",
        choices=("duration", "video", "start"),
        default="duration",
        help="Sort order for the short MTB2/MTB3 event list.",
    )

    return parser.parse_args()


def parse_timestamp(value: str) -> float:
    """
    Convert a timestamp such as:

        01:23
        12:34
        01:02:34

    to seconds.
    """

    parts = value.strip().split(":")

    if len(parts) == 2:
        minutes = int(parts[0])
        seconds = int(parts[1])

        if not 0 <= seconds < 60:
            raise ValueError(
                f"Invalid timestamp: {value}"
            )

        return minutes * 60 + seconds

    if len(parts) == 3:
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = int(parts[2])

        if not 0 <= minutes < 60 or not 0 <= seconds < 60:
            raise ValueError(
                f"Invalid timestamp: {value}"
            )

        return hours * 3600 + minutes * 60 + seconds

    raise ValueError(
        f"Unsupported timestamp format: {value}"
    )


def parse_annotation_file(path: Path) -> list[Annotation]:
    annotations: list[Annotation] = []
    video = ""

    with path.open("r", encoding="utf-8-sig") as f:
        lines = [
            line.rstrip("\n")
            for line in f
        ]

    for line_number, raw_line in enumerate(lines, start=1):
        stripped = raw_line.strip()

        if not stripped:
            continue

        if stripped.startswith("#"):
            continue

        parts = stripped.split()

        # The annotation header.
        if (
            len(parts) >= 3
            and parts[0].lower() == "start"
            and parts[1].lower() == "end"
            and parts[2].lower() == "mtb"
        ):
            continue

        # The source video path is the line that does not begin
        # with a timestamp.
        if not video:
            try:
                parse_timestamp(parts[0])
            except ValueError:
                video = stripped
                continue

        if len(parts) < 3:
            continue

        try:
            start = parse_timestamp(parts[0])
            end = parse_timestamp(parts[1])
            score = int(parts[2])
        except ValueError:
            continue

        if not math.isfinite(start) or not math.isfinite(end):
            raise ValueError(
                f"{path}:{line_number}: non-finite timestamp"
            )

        if end <= start:
            raise ValueError(
                f"{path}:{line_number}: invalid interval "
                f"[{parts[0]}, {parts[1]})"
            )

        if score not in (0, 1, 2, 3):
            raise ValueError(
                f"{path}:{line_number}: MTB score must be 0-3, "
                f"got {score}"
            )

        remarks = " ".join(parts[3:])

        annotations.append(
            Annotation(
                source_file=path,
                video=video,
                start=start,
                end=end,
                score=score,
                remarks=remarks,
            )
        )

    if not video:
        raise ValueError(
            f"{path}: no source video path found"
        )

    return annotations


def percentile(values: np.ndarray, q: float) -> float:
    if len(values) == 0:
        return float("nan")

    return float(np.percentile(values, q))


def print_summary(
    annotations: list[Annotation],
    thresholds: list[float],
) -> None:
    print()
    print("=" * 110)
    print("MTB ANNOTATION DURATION AUDIT")
    print("=" * 110)

    files = sorted(
        {a.source_file for a in annotations},
        key=lambda p: p.name.lower(),
    )

    print(f"Annotation files: {len(files)}")
    print(f"Annotations:      {len(annotations)}")
    print(
        "Thresholds:       "
        + ", ".join(f"{x:g}s" for x in thresholds)
    )

    print()
    print(
        f"{'MTB':>3} "
        f"{'#':>6} "
        f"{'Total(s)':>11} "
        f"{'Min':>8} "
        f"{'P25':>8} "
        f"{'Median':>8} "
        f"{'Mean':>8} "
        f"{'P75':>8} "
        f"{'Max':>8}"
    )
    print("-" * 110)

    for score in range(4):
        durations = np.array(
            [
                a.duration
                for a in annotations
                if a.score == score
            ],
            dtype=float,
        )

        if len(durations) == 0:
            print(
                f"{score:>3} "
                f"{0:>6} "
                f"{0.0:>11.2f}"
            )
            continue

        print(
            f"{score:>3} "
            f"{len(durations):>6} "
            f"{durations.sum():>11.2f} "
            f"{durations.min():>8.2f} "
            f"{percentile(durations, 25):>8.2f} "
            f"{np.median(durations):>8.2f} "
            f"{durations.mean():>8.2f} "
            f"{percentile(durations, 75):>8.2f} "
            f"{durations.max():>8.2f}"
        )

    print()
    print("SHORT-DURATION DISTRIBUTION")
    print("-" * 110)

    for score in range(4):
        score_annotations = [
            a
            for a in annotations
            if a.score == score
        ]

        if not score_annotations:
            continue

        durations = np.array(
            [a.duration for a in score_annotations],
            dtype=float,
        )

        total_time = durations.sum()

        print(f"\nMTB{score}")

        for threshold in thresholds:
            short_mask = durations < threshold

            short_count = int(short_mask.sum())
            short_time = float(
                durations[short_mask].sum()
            )

            count_pct = (
                100.0 * short_count / len(durations)
            )

            time_pct = (
                100.0 * short_time / total_time
                if total_time
                else 0.0
            )

            print(
                f"  < {threshold:g}s: "
                f"{short_count:4d}/{len(durations):4d} intervals "
                f"({count_pct:5.1f}%), "
                f"{short_time:8.1f}s / {total_time:8.1f}s "
                f"({time_pct:5.1f}%)"
            )


def print_per_video_summary(
    annotations: list[Annotation],
) -> None:
    print()
    print("=" * 110)
    print("PER-VIDEO MTB2 / MTB3 DURATION SUMMARY")
    print("=" * 110)

    files = sorted(
        {a.source_file for a in annotations},
        key=lambda p: p.name.lower(),
    )

    print(
        f"{'File':<20} "
        f"{'MTB2 #':>7} "
        f"{'MTB2 time':>11} "
        f"{'MTB3 #':>7} "
        f"{'MTB3 time':>11}"
    )
    print("-" * 110)

    for path in files:
        file_annotations = [
            a
            for a in annotations
            if a.source_file == path
        ]

        mtb2 = [
            a
            for a in file_annotations
            if a.score == 2
        ]

        mtb3 = [
            a
            for a in file_annotations
            if a.score == 3
        ]

        print(
            f"{path.stem:<20} "
            f"{len(mtb2):>7} "
            f"{sum(a.duration for a in mtb2):>11.1f} "
            f"{len(mtb3):>7} "
            f"{sum(a.duration for a in mtb3):>11.1f}"
        )


def sort_short_annotations(
    annotations: list[Annotation],
    mode: str,
) -> list[Annotation]:
    if mode == "video":
        return sorted(
            annotations,
            key=lambda a: (
                a.source_file.name.lower(),
                a.start,
            ),
        )

    if mode == "start":
        return sorted(
            annotations,
            key=lambda a: (
                a.start,
                a.source_file.name.lower(),
            ),
        )

    return sorted(
        annotations,
        key=lambda a: (
            a.duration,
            a.source_file.name.lower(),
            a.start,
        ),
    )


def print_short_events(
    annotations: list[Annotation],
    threshold: float,
    sort_mode: str,
) -> None:
    short_events = [
        a
        for a in annotations
        if a.score in (2, 3)
        and a.duration < threshold
    ]

    short_events = sort_short_annotations(
        short_events,
        sort_mode,
    )

    print()
    print("=" * 130)
    print(
        f"SHORT MTB2/MTB3 EVENTS (< {threshold:g}s)"
    )
    print("=" * 130)

    if not short_events:
        print("None.")
        return

    print(
        f"{'File':<18} "
        f"{'MTB':>3} "
        f"{'Start':>9} "
        f"{'End':>9} "
        f"{'Duration':>9} "
        f"  Remarks"
    )
    print("-" * 130)

    for a in short_events:
        print(
            f"{a.source_file.stem:<18} "
            f"{a.score:>3} "
            f"{a.start:>9.2f} "
            f"{a.end:>9.2f} "
            f"{a.duration:>9.2f} "
            f"  {a.remarks}"
        )

    print()
    print(
        f"Short MTB2/MTB3 events: {len(short_events)}"
    )

    for score in (2, 3):
        score_events = [
            a
            for a in short_events
            if a.score == score
        ]

        total_short_time = sum(
            a.duration
            for a in score_events
        )

        print(
            f"  MTB{score}: "
            f"{len(score_events)} events, "
            f"{total_short_time:.1f}s total"
        )


def main() -> None:
    args = parse_args()

    if args.short_threshold <= 0:
        raise ValueError(
            "--short-threshold must be > 0"
        )

    if any(t <= 0 for t in args.thresholds):
        raise ValueError(
            "All --thresholds values must be > 0"
        )

    annotation_files = [
        Path(path)
        for path in args.annotations
    ]

    all_annotations: list[Annotation] = []

    for path in annotation_files:
        if not path.is_file():
            raise FileNotFoundError(
                f"Annotation file not found: {path}"
            )

        parsed = parse_annotation_file(path)

        print(
            f"Loaded {path}: "
            f"{len(parsed)} annotations"
        )

        all_annotations.extend(parsed)

    if not all_annotations:
        raise ValueError(
            "No annotations found."
        )

    print_summary(
        all_annotations,
        sorted(set(args.thresholds)),
    )

    print_per_video_summary(
        all_annotations
    )

    print_short_events(
        all_annotations,
        args.short_threshold,
        args.sort_short,
    )

    print()
    print("AUDIT COMPLETE")


if __name__ == "__main__":
    main()