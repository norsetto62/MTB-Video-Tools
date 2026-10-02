#!/usr/bin/env python3
"""
Audit generated MTB dataset examples from *_dataset_audit.csv files.

Reads the audit CSVs produced by build_mtb_dataset.py and prints a compact,
human-readable report focused on:
  - per-video summary
  - target distribution
  - all intermediate-target examples
  - all multiple-annotation examples
  - suspicious examples
  - neighborhoods around annotation boundaries

No input files are modified.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


def parse_float(value: str) -> float:
    return float(value.strip())


def parse_int(value: str) -> int:
    return int(value.strip())


def fmt(value: float) -> str:
    return f"{value:.3f}"


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def row_text(row: dict) -> str:
    return (
        f"  {fmt(parse_float(row['start'])):>8} - "
        f"{fmt(parse_float(row['end'])):<8} "
        f"target={fmt(parse_float(row['target'])):<7} "
        f"coverage={fmt(parse_float(row['coverage'])):<6} "
        f"ann={row.get('annotation_id', '') or '-':<12} "
        f"remarks={row.get('remarks', '') or '-'}"
    )


def print_section(title: str) -> None:
    print()
    print(title)
    print("-" * len(title))


def analyze_file(path: Path, neighborhood: int) -> dict:
    rows = read_csv(path)

    if not rows:
        print_section(path.stem.upper())
        print("NO ROWS")
        return {
            "examples": 0,
            "intermediate": 0,
            "multiple": 0,
            "suspicious": 0,
        }

    rows.sort(key=lambda r: parse_float(r["start"]))

    exact = Counter()
    intermediate_rows = []
    multiple_rows = []
    suspicious_rows = []

    for row in rows:
        target = parse_float(row["target"])

        if abs(target - round(target)) < 1e-6 and 0 <= round(target) <= 3:
            exact[int(round(target))] += 1
        else:
            intermediate_rows.append(row)

        if parse_int(row.get("multiple_annotations", "0")):
            multiple_rows.append(row)

        if parse_int(row.get("suspicious", "0")):
            suspicious_rows.append(row)

    print_section(path.stem.upper())
    print(f"File:                  {path}")
    print(f"Examples:              {len(rows)}")
    print(f"Exact 0:               {exact[0]}")
    print(f"Exact 1:               {exact[1]}")
    print(f"Exact 2:               {exact[2]}")
    print(f"Exact 3:               {exact[3]}")
    print(f"Intermediate:          {len(intermediate_rows)}")
    print(f"Multiple annotations:  {len(multiple_rows)}")
    print(f"Suspicious:            {len(suspicious_rows)}")

    print_section("INTERMEDIATE TARGETS")
    if intermediate_rows:
        for row in intermediate_rows:
            print(row_text(row))
    else:
        print("  none")

    print_section("MULTIPLE-ANNOTATION WINDOWS")
    if multiple_rows:
        for row in multiple_rows:
            print(row_text(row))
    else:
        print("  none")

    print_section("SUSPICIOUS WINDOWS")
    if suspicious_rows:
        for row in suspicious_rows:
            print(row_text(row))
    else:
        print("  none")

    # Find contiguous groups of windows marked as near an annotation boundary.
    boundary_rows = [
        i
        for i, row in enumerate(rows)
        if parse_int(row.get("near_annotation_boundary", "0"))
    ]

    groups: list[list[int]] = []

    for idx in boundary_rows:
        if not groups or idx > groups[-1][-1] + 1:
            groups.append([idx])
        else:
            groups[-1].append(idx)

    print_section("BOUNDARY NEIGHBORHOODS")

    if not groups:
        print("  none")
    else:
        for group_no, group in enumerate(groups, 1):
            center = group[len(group) // 2]

            lo = max(0, center - neighborhood)
            hi = min(len(rows), center + neighborhood + 1)

            print()
            print(
                f"  Group {group_no}: rows {lo}-{hi - 1}, "
                f"time {fmt(parse_float(rows[lo]['start']))} - "
                f"{fmt(parse_float(rows[hi - 1]['end']))}"
            )

            for i in range(lo, hi):
                marker = "  *" if i in group else "   "
                print(f"{marker}{row_text(rows[i])}")

    return {
        "examples": len(rows),
        "intermediate": len(intermediate_rows),
        "multiple": len(multiple_rows),
        "suspicious": len(suspicious_rows),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Audit *_dataset_audit.csv files produced by "
            "build_mtb_dataset.py."
        )
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("output/datasets"),
        help="Directory containing *_dataset_audit.csv files.",
    )

    parser.add_argument(
        "--neighborhood",
        type=int,
        default=3,
        help=(
            "Number of generated windows shown before/after "
            "each boundary group."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional text file to receive the report.",
    )

    args = parser.parse_args()

    if args.neighborhood < 0:
        parser.error("--neighborhood must be >= 0")

    files = sorted(args.input_dir.glob("*_dataset_audit.csv"))

    if not files:
        print(
            f"No *_dataset_audit.csv files found in: "
            f"{args.input_dir}"
        )
        return 1

    import contextlib
    import io

    buffer = io.StringIO()

    with contextlib.redirect_stdout(buffer):
        print("=" * 80)
        print("GENERATED MTB EXAMPLE AUDIT")
        print("=" * 80)
        print(f"Input directory: {args.input_dir}")
        print(f"Files found:     {len(files)}")
        print(
            f"Neighborhood:    +/- "
            f"{args.neighborhood} generated windows"
        )

        totals = Counter()

        for path in files:
            result = analyze_file(path, args.neighborhood)
            totals.update(result)

        print_section("GLOBAL SUMMARY")
        print(f"Files:                 {len(files)}")
        print(f"Generated windows:     {totals['examples']}")
        print(f"Intermediate targets:  {totals['intermediate']}")
        print(f"Multiple annotations:  {totals['multiple']}")
        print(f"Suspicious windows:    {totals['suspicious']}")

        print()
        print("=" * 80)
        print("END OF AUDIT")
        print("=" * 80)

    report = buffer.getvalue()

    print(report, end="")

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
        print(f"Report written to: {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())