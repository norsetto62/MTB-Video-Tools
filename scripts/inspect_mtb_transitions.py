#!/usr/bin/env python3
"""
Inspect generated MTB dataset windows around annotation score transitions.

Usage:

    python scripts/inspect_mtb_transitions.py \
        --pair data/annotations/Ascoli.txt output/datasets/ascoli_dataset_audit.csv \
        --pair data/annotations/Capranica.txt output/datasets/capranica_dataset_audit.csv \
        --context 4

The audit CSV must have been produced by build_mtb_dataset.py after
independent validation.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Annotation:
    start: float
    end: float
    score: int
    remarks: str


def parse_time(value: str) -> float:
    """Convert seconds, MM:SS or HH:MM:SS to seconds."""

    value = value.strip()
    parts = value.split(":")

    if len(parts) == 1:
        return float(parts[0])

    if len(parts) == 2:
        return (
            float(parts[0]) * 60.0
            + float(parts[1])
        )

    if len(parts) == 3:
        return (
            float(parts[0]) * 3600.0
            + float(parts[1]) * 60.0
            + float(parts[2])
        )

    raise ValueError(
        f"invalid time: {value!r}"
    )


def is_windows_path(value: str) -> bool:
    """Return True for a normal Windows absolute path."""

    return (
        len(value) >= 3
        and value[1] == ":"
        and value[2] in ("\\", "/")
    )


def load_annotations(
    path: Path,
) -> list[Annotation]:
    """Load the simplified MTB annotation file."""

    lines = path.read_text(
        encoding="utf-8-sig"
    ).splitlines()

    annotations: list[Annotation] = []
    video_path_found = False

    for line_no, raw in enumerate(
        lines,
        start=1,
    ):
        line = raw.strip()

        if not line or line.startswith("#"):
            continue

        lower = line.lower()

        if (
            "start" in lower
            and "end" in lower
            and "mtb" in lower
        ):
            continue

        if not video_path_found and is_windows_path(line):
            video_path_found = True
            continue

        parts = line.split(
            maxsplit=3
        )

        if len(parts) < 3:
            raise ValueError(
                f"{path}:{line_no}: expected "
                "Start End MTB [Remarks]"
            )

        try:
            start = parse_time(parts[0])
            end = parse_time(parts[1])
            score = int(parts[2])
        except ValueError as exc:
            raise ValueError(
                f"{path}:{line_no}: {exc}"
            ) from exc

        if end <= start:
            raise ValueError(
                f"{path}:{line_no}: "
                "End must be greater than Start"
            )

        if not 0 <= score <= 3:
            raise ValueError(
                f"{path}:{line_no}: "
                "MTB score must be 0..3"
            )

        remarks = (
            parts[3].strip()
            if len(parts) == 4
            else ""
        )

        annotations.append(
            Annotation(
                start=start,
                end=end,
                score=score,
                remarks=remarks,
            )
        )

    if not video_path_found:
        raise ValueError(
            f"{path}: source video path not found"
        )

    annotations.sort(
        key=lambda annotation: annotation.start
    )

    return annotations


def load_audit_csv(
    path: Path,
) -> list[dict[str, str]]:
    """Load the validated generated-window audit CSV."""

    with path.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        reader = csv.DictReader(file)

        if reader.fieldnames is None:
            raise ValueError(
                f"{path}: missing CSV header"
            )

        required = {
            "example_id",
            "start",
            "end",
            "target",
            "expected_target",
            "valid",
        }

        missing = required.difference(
            reader.fieldnames
        )

        if missing:
            raise ValueError(
                f"{path}: missing required columns: "
                f"{', '.join(sorted(missing))}"
            )

        return list(reader)


def format_clock(seconds: float) -> str:
    """Format seconds as MM:SS.sss."""

    total_minutes = int(seconds // 60)
    remainder = seconds - (
        total_minutes * 60
    )

    return (
        f"{total_minutes:02d}:"
        f"{remainder:06.3f}"
    )


def find_transitions(
    annotations: list[Annotation],
) -> list[tuple[Annotation, Annotation]]:
    """Return adjacent annotations whose scores differ."""

    transitions: list[
        tuple[Annotation, Annotation]
    ] = []

    for previous, current in zip(
        annotations,
        annotations[1:],
    ):
        if previous.score != current.score:
            transitions.append(
                (previous, current)
            )

    return transitions


def windows_near_transition(
    rows: list[dict[str, str]],
    boundary: float,
    context: float,
) -> list[dict[str, str]]:
    """
    Return generated windows whose represented interval intersects
    the requested time range around the transition.
    """

    context_start = boundary - context
    context_end = boundary + context

    selected: list[dict[str, str]] = []

    for row in rows:
        start = float(row["start"])
        end = float(row["end"])

        if (
            end > context_start
            and start < context_end
        ):
            selected.append(row)

    selected.sort(
        key=lambda row: (
            float(row["start"]),
            int(row["example_id"]),
        )
    )

    return selected


def print_transition(
    previous: Annotation,
    current: Annotation,
    rows: list[dict[str, str]],
    context: float,
) -> None:
    """Print one annotation transition and its generated windows."""

    boundary = current.start

    print()
    print(
        f"{boundary:.3f}s  "
        f"MTB{previous.score} -> MTB{current.score}"
    )

    if previous.remarks:
        print(
            f"  previous: {format_clock(previous.start)}-"
            f"{format_clock(previous.end)}  "
            f"MTB{previous.score}  "
            f"{previous.remarks}"
        )

    if current.remarks:
        print(
            f"  current:  {format_clock(current.start)}-"
            f"{format_clock(current.end)}  "
            f"MTB{current.score}  "
            f"{current.remarks}"
        )

    selected = windows_near_transition(
        rows,
        boundary,
        context,
    )

    if not selected:
        print(
            "  No generated windows intersect "
            f"{boundary - context:.3f}s - "
            f"{boundary + context:.3f}s"
        )
        return

    for row in selected:
        start = float(row["start"])
        end = float(row["end"])
        target = float(row["target"])
        expected = float(
            row["expected_target"]
        )

        valid = (
            row["valid"].strip()
            in {"1", "true", "True"}
        )

        status = (
            "PASS"
            if valid
            else "FAIL"
        )

        print(
            f"  {start:8.3f} - {end:8.3f}  "
            f"target={target:.3f}  "
            f"expected={expected:.3f}  "
            f"{status}"
        )


def inspect_pair(
    annotations_path: Path,
    audit_path: Path,
    context: float,
) -> None:
    """Inspect all score transitions for one pair."""

    annotations = load_annotations(
        annotations_path
    )

    rows = load_audit_csv(
        audit_path
    )

    transitions = find_transitions(
        annotations
    )

    print()
    print("=" * 80)
    print(
        annotations_path.stem.upper()
    )
    print("=" * 80)

    print(
        f"Annotations:  {len(annotations)}"
    )

    print(
        f"Generated windows: {len(rows)}"
    )

    print(
        f"Score transitions: {len(transitions)}"
    )

    print(
        f"Context:       +/- {context:.3f}s"
    )

    if not transitions:
        print()
        print("No score transitions.")
        return

    for previous, current in transitions:
        print_transition(
            previous,
            current,
            rows,
            context,
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect generated MTB dataset windows "
            "around annotation score transitions."
        )
    )

    parser.add_argument(
        "--pair",
        nargs=2,
        action="append",
        required=True,
        metavar=("ANNOTATIONS", "AUDIT_CSV"),
        help=(
            "Annotation file and corresponding "
            "validated audit CSV. May be repeated."
        ),
    )

    parser.add_argument(
        "--context",
        type=float,
        required=True,
        help=(
            "Seconds before and after each "
            "annotation score transition to inspect."
        ),
    )

    args = parser.parse_args()

    if args.context < 0:
        parser.error(
            "--context must be >= 0"
        )

    for annotations, audit_csv in args.pair:
        inspect_pair(
            annotations_path=Path(
                annotations
            ),
            audit_path=Path(
                audit_csv
            ),
            context=args.context,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())