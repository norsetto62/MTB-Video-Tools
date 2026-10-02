#!/usr/bin/env python3

"""
Audit generated 4-second window targets against MTB2/MTB3 annotations.

For each annotation/audit CSV pair, the script:

1. Reads the original annotation intervals.
2. Reads the validated generated-example audit CSV.
3. Recomputes which annotations overlap each generated window.
4. Identifies windows containing MTB2 and/or MTB3.
5. Reports target distributions.
6. Reports how often windows containing MTB3 have target < 3.
7. Reports how often windows containing MTB3 have target >= configurable
   thresholds.
8. Examines short MTB2/MTB3 annotations below --short-threshold and shows
   the generated windows that overlap them.

The generated-example audit CSV is expected to contain at least:

    start
    end
    target
    valid

The original annotation files use:

    Start   End     MTB     Remarks
    video_path
    00:00   01:07   0       Dismounted/Push
    01:07   01:23   3       Drop and Rock Garden

Window intervals and annotation intervals are treated as [start, end).
"""


from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path


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


@dataclass
class GeneratedWindow:
    source_file: Path
    example_id: str
    start: float
    end: float
    target: float
    valid: bool

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class WindowAnalysis:
    window: GeneratedWindow
    overlapping: list[Annotation]
    mtb2_overlap: float
    mtb3_overlap: float

    @property
    def has_mtb2(self) -> bool:
        return self.mtb2_overlap > 0.0

    @property
    def has_mtb3(self) -> bool:
        return self.mtb3_overlap > 0.0

    @property
    def has_mtb23(self) -> bool:
        return self.has_mtb2 or self.has_mtb3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit generated MTB window targets against "
            "MTB2/MTB3 annotations."
        )
    )

    parser.add_argument(
        "--pair",
        nargs=2,
        action="append",
        metavar=("ANNOTATIONS", "AUDIT_CSV"),
        required=True,
        help=(
            "Annotation file and corresponding validated "
            "dataset audit CSV. Repeat for any number of datasets."
        ),
    )

    parser.add_argument(
        "--short-threshold",
        type=float,
        default=4.0,
        help=(
            "Show generated windows overlapping MTB2/MTB3 annotations "
            "shorter than this duration (default: 4.0s)."
        ),
    )

    parser.add_argument(
        "--target-thresholds",
        type=float,
        nargs="+",
        default=[1.0, 1.5, 2.0, 2.5, 3.0],
        help=(
            "Target thresholds used in the distribution report "
            "(default: 1 1.5 2 2.5 3)."
        ),
    )

    parser.add_argument(
        "--mtb3-coverage-threshold",
        type=float,
        default=0.75,
        help=(
            "Minimum fraction of a generated window covered by MTB3 "
            "for the detailed MTB3 edge-window audit "
            "(default: 0.75)."
        ),
    )

    parser.add_argument(
        "--context",
        type=float,
        default=0.0,
        help=(
            "Extra seconds before/after a short MTB2/MTB3 annotation "
            "when displaying overlapping generated windows "
            "(default: 0)."
        ),
    )

    return parser.parse_args()


def parse_timestamp(value: str) -> float:
    """
    Convert MM:SS or HH:MM:SS into seconds.
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

        if not 0 <= minutes < 60:
            raise ValueError(
                f"Invalid timestamp: {value}"
            )

        if not 0 <= seconds < 60:
            raise ValueError(
                f"Invalid timestamp: {value}"
            )

        return (
            hours * 3600
            + minutes * 60
            + seconds
        )

    raise ValueError(
        f"Unsupported timestamp format: {value}"
    )


def parse_annotation_file(
    path: Path,
) -> list[Annotation]:
    annotations: list[Annotation] = []
    video = ""

    with path.open(
        "r",
        encoding="utf-8-sig",
    ) as f:
        lines = [
            line.rstrip("\n")
            for line in f
        ]

    for line_number, raw_line in enumerate(
        lines,
        start=1,
    ):
        stripped = raw_line.strip()

        if not stripped:
            continue

        if stripped.startswith("#"):
            continue

        parts = stripped.split()

        if (
            len(parts) >= 3
            and parts[0].lower() == "start"
            and parts[1].lower() == "end"
            and parts[2].lower() == "mtb"
        ):
            continue

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

        if not math.isfinite(start):
            raise ValueError(
                f"{path}:{line_number}: invalid start time"
            )

        if not math.isfinite(end):
            raise ValueError(
                f"{path}:{line_number}: invalid end time"
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


def parse_float(
    row: dict[str, str],
    field: str,
    path: Path,
    row_number: int,
) -> float:
    value = row.get(field)

    if value is None:
        raise ValueError(
            f"{path}: missing required column '{field}'"
        )

    try:
        result = float(value)
    except ValueError as exc:
        raise ValueError(
            f"{path}: row {row_number}: invalid "
            f"{field}={value!r}"
        ) from exc

    if not math.isfinite(result):
        raise ValueError(
            f"{path}: row {row_number}: "
            f"non-finite {field}"
        )

    return result


def parse_bool(
    row: dict[str, str],
    field: str,
    path: Path,
    row_number: int,
) -> bool:
    value = row.get(field)

    if value is None:
        raise ValueError(
            f"{path}: missing required column '{field}'"
        )

    normalized = value.strip().lower()

    if normalized in {
        "1",
        "true",
        "yes",
        "y",
    }:
        return True

    if normalized in {
        "0",
        "false",
        "no",
        "n",
    }:
        return False

    raise ValueError(
        f"{path}: row {row_number}: invalid "
        f"{field}={value!r}"
    )


def load_audit_csv(
    path: Path,
) -> list[GeneratedWindow]:
    windows: list[GeneratedWindow] = []

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        reader = csv.DictReader(f)

        if reader.fieldnames is None:
            raise ValueError(
                f"{path}: CSV has no header"
            )

        required = {
            "start",
            "end",
            "target",
            "valid",
        }

        missing = required - set(reader.fieldnames)

        if missing:
            raise ValueError(
                f"{path}: missing required columns: "
                + ", ".join(sorted(missing))
            )

        for row_number, row in enumerate(
            reader,
            start=2,
        ):
            start = parse_float(
                row,
                "start",
                path,
                row_number,
            )

            end = parse_float(
                row,
                "end",
                path,
                row_number,
            )

            target = parse_float(
                row,
                "target",
                path,
                row_number,
            )

            valid = parse_bool(
                row,
                "valid",
                path,
                row_number,
            )

            if end <= start:
                raise ValueError(
                    f"{path}: row {row_number}: "
                    f"end <= start"
                )

            example_id = row.get(
                "example_id",
                str(row_number - 1),
            )

            windows.append(
                GeneratedWindow(
                    source_file=path,
                    example_id=example_id,
                    start=start,
                    end=end,
                    target=target,
                    valid=valid,
                )
            )

    return windows


def overlap_duration(
    start_a: float,
    end_a: float,
    start_b: float,
    end_b: float,
) -> float:
    start = max(start_a, start_b)
    end = min(end_a, end_b)

    return max(0.0, end - start)


def analyze_window(
    window: GeneratedWindow,
    annotations: list[Annotation],
) -> WindowAnalysis:
    overlapping: list[Annotation] = []
    mtb2_overlap = 0.0
    mtb3_overlap = 0.0

    for annotation in annotations:
        overlap = overlap_duration(
            window.start,
            window.end,
            annotation.start,
            annotation.end,
        )

        if overlap <= 0.0:
            continue

        overlapping.append(annotation)

        if annotation.score == 2:
            mtb2_overlap += overlap

        elif annotation.score == 3:
            mtb3_overlap += overlap

    return WindowAnalysis(
        window=window,
        overlapping=overlapping,
        mtb2_overlap=mtb2_overlap,
        mtb3_overlap=mtb3_overlap,
    )


def analyze_dataset(
    annotations: list[Annotation],
    windows: list[GeneratedWindow],
) -> list[WindowAnalysis]:
    return [
        analyze_window(
            window,
            annotations,
        )
        for window in windows
        if window.valid
    ]


def format_pct(
    numerator: int,
    denominator: int,
) -> str:
    if denominator == 0:
        return "  0.0%"

    return f"{100.0 * numerator / denominator:5.1f}%"


def print_target_distribution(
    analyses: list[WindowAnalysis],
    label: str,
    predicate,
    thresholds: list[float],
) -> None:
    selected = [
        a
        for a in analyses
        if predicate(a)
    ]

    print()
    print(label)
    print("-" * 110)

    if not selected:
        print("No matching windows.")
        return

    targets = [
        a.window.target
        for a in selected
    ]

    print(
        f"Windows: {len(selected)}"
    )

    print(
        f"Target min:    {min(targets):.3f}"
    )
    print(
        f"Target median: "
        f"{sorted(targets)[len(targets) // 2]:.3f}"
    )
    print(
        f"Target max:    {max(targets):.3f}"
    )

    print()
    print(
        f"{'Threshold':>12} "
        f"{'Count':>8} "
        f"{'Percent':>10}"
    )
    print("-" * 45)

    for threshold in thresholds:
        count = sum(
            1
            for target in targets
            if target >= threshold
        )

        print(
            f">= {threshold:>6.2f} "
            f"{count:>8} "
            f"{format_pct(count, len(targets)):>10}"
        )


def print_mtb3_analysis(
    analyses: list[WindowAnalysis],
) -> None:
    mtb3_windows = [
        a
        for a in analyses
        if a.has_mtb3
    ]

    print()
    print("=" * 110)
    print("WINDOWS CONTAINING MTB3")
    print("=" * 110)

    if not mtb3_windows:
        print("No windows contain MTB3.")
        return

    below_3 = [
        a
        for a in mtb3_windows
        if a.window.target < 3.0
    ]

    exactly_3 = [
        a
        for a in mtb3_windows
        if math.isclose(
            a.window.target,
            3.0,
            rel_tol=0.0,
            abs_tol=1e-9,
        )
    ]

    print(
        f"Windows containing MTB3: {len(mtb3_windows)}"
    )
    print(
        f"Target < 3:             {len(below_3)} "
        f"({format_pct(len(below_3), len(mtb3_windows)).strip()})"
    )
    print(
        f"Target == 3:            {len(exactly_3)} "
        f"({format_pct(len(exactly_3), len(mtb3_windows)).strip()})"
    )

    print()
    print(
        "MTB3 overlap versus target"
    )
    print("-" * 70)

    print(
        f"{'MTB3 overlap':>15} "
        f"{'Windows':>10} "
        f"{'Target <3':>12} "
        f"{'Target ==3':>12}"
    )

    buckets = [
        ("0-25%", 0.0, 0.25),
        ("25-50%", 0.25, 0.50),
        ("50-75%", 0.50, 0.75),
        ("75-100%", 0.75, 1.0000001),
    ]

    for name, lower, upper in buckets:
        selected = []

        for analysis in mtb3_windows:
            fraction = (
                analysis.mtb3_overlap
                / analysis.window.duration
            )

            if (
                fraction >= lower
                and fraction < upper
            ):
                selected.append(analysis)

        if not selected:
            continue

        below = sum(
            1
            for a in selected
            if a.window.target < 3.0
        )

        exact = sum(
            1
            for a in selected
            if math.isclose(
                a.window.target,
                3.0,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
        )

        print(
            f"{name:>15} "
            f"{len(selected):>10} "
            f"{below:>12} "
            f"{exact:>12}"
        )


def print_mtb3_high_coverage_sub3(
    analyses: list[WindowAnalysis],
    coverage_threshold: float,
) -> None:
    """
    Show windows with substantial MTB3 coverage but target < 3.

    These are the MTB3 boundary/edge windows that are worth inspecting
    because most of the window is annotated MTB3, yet the generated
    regression target is below 3.
    """

    selected = []

    for analysis in analyses:
        if not analysis.has_mtb3:
            continue

        window_duration = analysis.window.duration

        if window_duration <= 0:
            continue

        mtb3_fraction = (
            analysis.mtb3_overlap
            / window_duration
        )

        if mtb3_fraction < coverage_threshold:
            continue

        if analysis.window.target >= 3.0:
            continue

        selected.append(
            (
                analysis,
                mtb3_fraction,
            )
        )

    selected.sort(
        key=lambda item: (
            item[0].window.source_file.stem.lower(),
            item[0].window.start,
        )
    )
    
    print()
    print("=" * 120)
    print(
        "HIGH-COVERAGE MTB3 WINDOWS WITH TARGET < 3"
    )
    print("=" * 120)

    print(
        f"MTB3 coverage threshold: "
        f"{coverage_threshold:.1%}"
    )
    print(
        f"Matching windows: {len(selected)}"
    )

    if not selected:
        print("None.")
        return

    print()
    print(
        f"{'Video':<16} "
        f"{'Window':>17} "
        f"{'Target':>8} "
        f"{'MTB3 ov':>9} "
        f"{'MTB3 %':>8} "
        f"{'Other':>8} "
        f"Remarks"
    )
    print("-" * 120)

    for analysis, mtb3_fraction in selected:
        window = analysis.window

        other_overlap = (
            window.duration
            - analysis.mtb3_overlap
        )

        # Collect remarks from annotations that overlap the
        # non-MTB3 portion of the window.
        other_annotations = [
            annotation
            for annotation in analysis.overlapping
            if annotation.score != 3
        ]

        remarks = "; ".join(
            f"MTB{annotation.score}: "
            f"{annotation.remarks}"
            for annotation in other_annotations
        )

        if not remarks:
            remarks = "(no other annotation)"

        print(
            f"{window.source_file.stem:<16} "
            f"{window.start:7.2f}-{window.end:<7.2f} "
            f"{window.target:>8.3f} "
            f"{analysis.mtb3_overlap:>9.2f} "
            f"{mtb3_fraction:>7.1%} "
            f"{other_overlap:>8.2f} "
            f"{remarks}"
        )

        
def print_short_event_windows(
    annotations: list[Annotation],
    analyses: list[WindowAnalysis],
    threshold: float,
    context: float,
) -> None:
    short_events = [
        a
        for a in annotations
        if a.score in (2, 3)
        and a.duration < threshold
    ]

    print()
    print("=" * 120)
    print(
        f"SHORT MTB2/MTB3 EVENTS AND THEIR GENERATED WINDOWS "
        f"(< {threshold:g}s)"
    )
    print("=" * 120)

    if not short_events:
        print("None.")
        return

    for annotation in short_events:
        print()
        print(
            f"{annotation.source_file.stem}: "
            f"MTB{annotation.score} "
            f"{annotation.start:.2f}-{annotation.end:.2f}s "
            f"({annotation.duration:.2f}s)"
        )
        print(
            f"  {annotation.remarks}"
        )

        region_start = (
            annotation.start - context
        )
        region_end = (
            annotation.end + context
        )

        matching = [
            analysis
            for analysis in analyses
            if overlap_duration(
                analysis.window.start,
                analysis.window.end,
                region_start,
                region_end,
            ) > 0.0
        ]

        matching.sort(
            key=lambda a: a.window.start
        )

        print(
            f"  {'Window':>17} "
            f"{'Target':>8} "
            f"{'MTB2 ov':>9} "
            f"{'MTB3 ov':>9}"
        )
        print(
            "  " + "-" * 55
        )

        for analysis in matching:
            print(
                f"  "
                f"{analysis.window.start:7.2f}-"
                f"{analysis.window.end:<7.2f} "
                f"{analysis.window.target:>8.3f} "
                f"{analysis.mtb2_overlap:>9.2f} "
                f"{analysis.mtb3_overlap:>9.2f}"
            )


def print_dataset_summary(
    annotations: list[Annotation],
    analyses: list[WindowAnalysis],
    path: Path,
) -> None:
    print()
    print("=" * 110)
    print(
        f"DATASET: {path.stem}"
    )
    print("=" * 110)

    print(
        f"Annotations:        {len(annotations)}"
    )
    print(
        f"Valid windows:      {len(analyses)}"
    )

    for score in (2, 3):
        count = sum(
            1
            for a in analyses
            if any(
                annotation.score == score
                for annotation in a.overlapping
            )
        )

        print(
            f"Windows containing MTB{score}: "
            f"{count}"
        )


def print_global_summary(
    all_analyses: list[WindowAnalysis],
    thresholds: list[float],
    mtb3_coverage_threshold: float,
) -> None:
    print()
    print("=" * 110)
    print("GLOBAL 4-SECOND WINDOW TARGET AUDIT")
    print("=" * 110)

    print(
        f"Valid generated windows: "
        f"{len(all_analyses)}"
    )

    print_target_distribution(
        all_analyses,
        "WINDOWS CONTAINING MTB2",
        lambda a: a.has_mtb2,
        thresholds,
    )

    print_target_distribution(
        all_analyses,
        "WINDOWS CONTAINING MTB3",
        lambda a: a.has_mtb3,
        thresholds,
    )

    print_target_distribution(
        all_analyses,
        "WINDOWS CONTAINING MTB2 OR MTB3",
        lambda a: a.has_mtb23,
        thresholds,
    )

    print_mtb3_analysis(
        all_analyses
    )

    print_mtb3_high_coverage_sub3(
        all_analyses,
        mtb3_coverage_threshold,
    )

def main() -> None:
    args = parse_args()

    if args.short_threshold <= 0:
        raise ValueError(
            "--short-threshold must be > 0"
        )

    if args.context < 0:
        raise ValueError(
            "--context must be >= 0"
        )

    if not 0.0 < args.mtb3_coverage_threshold <= 1.0:
        raise ValueError(
            "--mtb3-coverage-threshold must be > 0 and <= 1"
        )
    
    if any(
        threshold < 0
        for threshold in args.target_thresholds
    ):
        raise ValueError(
            "--target-thresholds must be >= 0"
        )

    all_analyses: list[WindowAnalysis] = []
    all_annotations: list[Annotation] = []

    for annotation_name, audit_name in args.pair:
        annotation_path = Path(annotation_name)
        audit_path = Path(audit_name)

        if not annotation_path.is_file():
            raise FileNotFoundError(
                f"Annotation file not found: "
                f"{annotation_path}"
            )

        if not audit_path.is_file():
            raise FileNotFoundError(
                f"Audit CSV not found: "
                f"{audit_path}"
            )

        annotations = parse_annotation_file(
            annotation_path
        )

        windows = load_audit_csv(
            audit_path
        )

        analyses = analyze_dataset(
            annotations,
            windows,
        )

        print_dataset_summary(
            annotations,
            analyses,
            audit_path,
        )

        all_annotations.extend(
            annotations
        )
        all_analyses.extend(
            analyses
        )

        print_short_event_windows(
            annotations,
            analyses,
            args.short_threshold,
            args.context,
        )

    print_global_summary(
        all_analyses,
        sorted(set(args.target_thresholds)),
        args.mtb3_coverage_threshold,
    )

    print()
    print("AUDIT COMPLETE")


if __name__ == "__main__":
    main()