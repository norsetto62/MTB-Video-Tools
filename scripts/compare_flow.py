#!/usr/bin/env python3

import argparse
import csv
import math
import re
import statistics
from pathlib import Path

import numpy as np


DEFAULT_FLOW_DIR = Path("data") / "flow_scores"


# ---------------------------------------------------------------------------
# Time parsing
# ---------------------------------------------------------------------------

def parse_time(value: str) -> float:
    value = value.strip()

    if not value:
        raise ValueError("Empty time value")

    if ":" not in value:
        return float(value)

    parts = value.split(":")

    if len(parts) == 2:
        minutes, seconds = parts
        return int(minutes) * 60 + float(seconds)

    if len(parts) == 3:
        hours, minutes, seconds = parts
        return (
            int(hours) * 3600
            + int(minutes) * 60
            + float(seconds)
        )

    raise ValueError(f"Invalid time: {value}")


def format_time(seconds: float) -> str:
    seconds = max(0.0, seconds)

    minutes = int(seconds // 60)
    secs = seconds % 60

    if minutes >= 60:
        hours = minutes // 60
        minutes %= 60
        return f"{hours:02d}:{minutes:02d}:{secs:05.2f}"

    return f"{minutes:02d}:{secs:05.2f}"


# ---------------------------------------------------------------------------
# Annotation parsing
# ---------------------------------------------------------------------------

def looks_like_video_path(line: str) -> bool:
    lower = line.lower()

    if lower.endswith(
        (".mp4", ".mov", ".mkv", ".avi", ".m4v", ".mts", ".m2ts")
    ):
        return True

    if re.match(r"^[A-Za-z]:[\\/]", line):
        return True

    if line.startswith(("./", "../", ".\\", "..\\", "/")):
        return True

    return False


def is_header(line: str) -> bool:
    tokens = line.lower().split()

    return (
        len(tokens) >= 2
        and tokens[0] == "start"
        and tokens[1] == "end"
    )


def parse_annotations(path: Path):
    """
    Parse:

        Start    End   MTB   Video   Remarks
        D:\\video.mp4

        00:00   01:19   4   2   description
    """

    annotations = []

    current_video = None

    with path.open("r", encoding="utf-8-sig") as f:
        for line_number, raw_line in enumerate(f, start=1):
            line = raw_line.strip()

            if not line:
                continue

            if line.startswith("#"):
                continue

            if is_header(line):
                continue

            if looks_like_video_path(line):
                current_video = line
                continue

            parts = line.split()

            if len(parts) < 4:
                print(
                    f"Warning: ignoring line {line_number}: {line}"
                )
                continue

            if current_video is None:
                print(
                    f"Warning: annotation without video on line "
                    f"{line_number}: {line}"
                )
                continue

            try:
                start = parse_time(parts[0])
                end = parse_time(parts[1])
                mtb = int(parts[2])
                video_rating = int(parts[3])
            except ValueError as exc:
                print(
                    f"Warning: ignoring line {line_number}: {exc}"
                )
                continue

            if end <= start:
                print(
                    f"Warning: invalid interval on line "
                    f"{line_number}: {line}"
                )
                continue

            if not 1 <= mtb <= 5:
                print(
                    f"Warning: MTB rating outside 1-5 on line "
                    f"{line_number}: {line}"
                )
                continue

            if not 1 <= video_rating <= 5:
                print(
                    f"Warning: Video rating outside 1-5 on line "
                    f"{line_number}: {line}"
                )
                continue

            remarks = " ".join(parts[4:]) if len(parts) > 4 else ""

            annotations.append(
                {
                    "video": current_video,
                    "start": start,
                    "end": end,
                    "mtb": mtb,
                    "video_rating": video_rating,
                    "remarks": remarks,
                }
            )

    return annotations


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------

def load_flow_csv(path: Path):
    timestamps = []
    columns = {}

    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as f:
        reader = csv.DictReader(f)

        if reader.fieldnames is None:
            raise ValueError("CSV has no header")

        fieldnames = reader.fieldnames

        for field in fieldnames:
            if field != "timestamp":
                columns[field] = []

        for row in reader:
            try:
                timestamp = float(row["timestamp"])
            except (ValueError, TypeError):
                continue

            timestamps.append(timestamp)

            for field in columns:
                try:
                    value = float(row[field])
                except (ValueError, TypeError):
                    value = math.nan

                columns[field].append(value)

    timestamps = np.asarray(
        timestamps,
        dtype=np.float64,
    )

    for field in columns:
        columns[field] = np.asarray(
            columns[field],
            dtype=np.float64,
        )

    return timestamps, columns


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def finite_values(values):
    values = np.asarray(values, dtype=np.float64)

    return values[np.isfinite(values)]


def stats(values):
    values = finite_values(values)

    if len(values) == 0:
        return None

    return {
        "n": len(values),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
        "p10": float(np.percentile(values, 10)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def format_stats(s):
    if s is None:
        return "no samples"

    return (
        f"n={s['n']:4d} "
        f"mean={s['mean']:8.3f} "
        f"median={s['median']:8.3f} "
        f"std={s['std']:8.3f} "
        f"p90={s['p90']:8.3f} "
        f"p95={s['p95']:8.3f} "
        f"max={s['max']:8.3f}"
    )


# ---------------------------------------------------------------------------
# Correlation
# ---------------------------------------------------------------------------

def correlation(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    mask = np.isfinite(x) & np.isfinite(y)

    x = x[mask]
    y = y[mask]

    if len(x) < 3:
        return math.nan

    if np.std(x) == 0 or np.std(y) == 0:
        return math.nan

    return float(np.corrcoef(x, y)[0, 1])


# ---------------------------------------------------------------------------
# Feature definitions
# ---------------------------------------------------------------------------

CORE_FEATURES = [
    "flow_x",
    "flow_y",
    "flow_abs_x",
    "flow_abs_y",
    "flow_magnitude",
    "flow_median_magnitude",
    "flow_p90_magnitude",
    "flow_p95_magnitude",
    "flow_std_x",
    "flow_std_y",
    "flow_std_magnitude",
    "flow_coherence",
    "flow_angle",
]


# ---------------------------------------------------------------------------
# Video matching
# ---------------------------------------------------------------------------

def resolve_flow_csv(annotation_video, flow_dir):
    """
    Match annotation video basename to flow CSV basename.

    This avoids requiring the exact D:\\... path to be inside the project.
    """

    video_path = Path(annotation_video)

    candidate = flow_dir / f"{video_path.stem}.csv"

    if candidate.exists():
        return candidate

    # Case-insensitive fallback.
    target = candidate.name.lower()

    for path in flow_dir.glob("*.csv"):
        if path.name.lower() == target:
            return path

    return None


# ---------------------------------------------------------------------------
# Interval selection
# ---------------------------------------------------------------------------

def select_interval(
    timestamps,
    start,
    end,
):
    return (
        (timestamps >= start)
        & (timestamps < end)
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_feature_table(
    title,
    columns,
    mask,
):
    print()
    print(title)
    print("-" * len(title))

    for feature in CORE_FEATURES:
        if feature not in columns:
            continue

        s = stats(columns[feature][mask])

        print(
            f"{feature:25s} "
            f"{format_stats(s)}"
        )


def print_rating_summary(
    annotations,
    timestamps,
    columns,
    rating_name,
):
    print()
    print("=" * 70)
    print(f"{rating_name} rating summary")
    print("=" * 70)

    ratings = sorted(
        set(
            annotation[rating_name]
            for annotation in annotations
        )
    )

    for rating in ratings:
        mask = np.zeros(
            len(timestamps),
            dtype=bool,
        )

        for annotation in annotations:
            if annotation[rating_name] != rating:
                continue

            mask |= select_interval(
                timestamps,
                annotation["start"],
                annotation["end"],
            )

        print()
        print(
            f"{rating_name.upper()} {rating}: "
            f"{np.sum(mask)} samples"
        )

        for feature in CORE_FEATURES:
            if feature not in columns:
                continue

            s = stats(columns[feature][mask])

            if s is not None:
                print(
                    f"  {feature:23s} "
                    f"mean={s['mean']:8.3f} "
                    f"median={s['median']:8.3f} "
                    f"p90={s['p90']:8.3f}"
                )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compare optical-flow features against manual "
            "MTB/Video annotations."
        )
    )

    parser.add_argument(
        "annotation_file",
        help="Annotation file, e.g. data/annotations/Mentorella.txt",
    )

    parser.add_argument(
        "--flow-dir",
        default=str(DEFAULT_FLOW_DIR),
        help="Flow CSV directory",
    )

    parser.add_argument(
        "--flow-csv",
        help="Explicit flow CSV to use",
    )

    args = parser.parse_args()

    annotation_path = Path(args.annotation_file)
    flow_dir = Path(args.flow_dir)

    annotations = parse_annotations(
        annotation_path
    )

    if not annotations:
        print("No annotations found.")
        return 1

    # Group annotations by video.
    videos = {}

    for annotation in annotations:
        videos.setdefault(
            annotation["video"],
            [],
        ).append(annotation)

    print("=" * 70)
    print("Optical Flow / Manual Annotation Comparison")
    print("=" * 70)

    print(f"Annotation file: {annotation_path}")
    print(f"Annotations:     {len(annotations)}")

    for video, video_annotations in videos.items():

        if args.flow_csv:
            flow_path = Path(args.flow_csv)
        else:
            flow_path = resolve_flow_csv(
                video,
                flow_dir,
            )

        if flow_path is None:
            print()
            print(
                f"ERROR: no flow CSV found for:\n"
                f"  {video}"
            )
            continue

        if not flow_path.exists():
            print(
                f"ERROR: flow CSV not found: {flow_path}"
            )
            continue

        print()
        print("=" * 70)
        print(f"Video:      {video}")
        print(f"Flow CSV:   {flow_path}")
        print(
            f"Annotations for video: "
            f"{len(video_annotations)}"
        )
        print("=" * 70)

        timestamps, columns = load_flow_csv(
            flow_path
        )

        print(
            f"Flow samples: {len(timestamps)}"
        )

        if len(timestamps) == 0:
            continue

        # ---------------------------------------------------------------
        # Overall statistics
        # ---------------------------------------------------------------

        print()
        print("Overall flow statistics")
        print("-----------------------")

        for feature in CORE_FEATURES:
            if feature not in columns:
                continue

            s = stats(columns[feature])

            print(
                f"{feature:25s} "
                f"{format_stats(s)}"
            )

        # ---------------------------------------------------------------
        # Annotated vs outside
        # ---------------------------------------------------------------

        annotated_mask = np.zeros(
            len(timestamps),
            dtype=bool,
        )

        for annotation in video_annotations:
            annotated_mask |= select_interval(
                timestamps,
                annotation["start"],
                annotation["end"],
            )

        outside_mask = ~annotated_mask

        print()
        print("Annotated vs outside")
        print("--------------------")

        for feature in CORE_FEATURES:
            if feature not in columns:
                continue

            annotated_stats = stats(
                columns[feature][annotated_mask]
            )

            outside_stats = stats(
                columns[feature][outside_mask]
            )

            if (
                annotated_stats is None
                or outside_stats is None
            ):
                continue

            print()
            print(feature)

            print(
                f"  Annotated: "
                f"{format_stats(annotated_stats)}"
            )

            print(
                f"  Outside:   "
                f"{format_stats(outside_stats)}"
            )

            difference = (
                annotated_stats["mean"]
                - outside_stats["mean"]
            )

            print(
                f"  Mean diff: "
                f"{difference:+.3f}"
            )

        # ---------------------------------------------------------------
        # Individual annotations
        # ---------------------------------------------------------------

        print()
        print("=" * 70)
        print("Manual annotations")
        print("=" * 70)

        for index, annotation in enumerate(
            video_annotations,
            start=1,
        ):
            mask = select_interval(
                timestamps,
                annotation["start"],
                annotation["end"],
            )

            print()
            print(
                f"{index:2d}. "
                f"{format_time(annotation['start'])} - "
                f"{format_time(annotation['end'])} "
                f"MTB={annotation['mtb']} "
                f"Video={annotation['video_rating']}"
            )

            if annotation["remarks"]:
                print(
                    f"    {annotation['remarks']}"
                )

            print(
                f"    samples={np.sum(mask)}"
            )

            for feature in CORE_FEATURES:
                if feature not in columns:
                    continue

                s = stats(
                    columns[feature][mask]
                )

                if s is None:
                    continue

                print(
                    f"    {feature:23s} "
                    f"mean={s['mean']:8.3f} "
                    f"median={s['median']:8.3f} "
                    f"p90={s['p90']:8.3f} "
                    f"max={s['max']:8.3f}"
                )

        # ---------------------------------------------------------------
        # Rating summaries
        # ---------------------------------------------------------------

        print_rating_summary(
            video_annotations,
            timestamps,
            columns,
            "mtb",
        )

        print_rating_summary(
            video_annotations,
            timestamps,
            columns,
            "video_rating",
        )

        # ---------------------------------------------------------------
        # Correlation with ratings
        # ---------------------------------------------------------------

        print()
        print("=" * 70)
        print("Feature / rating correlation")
        print("=" * 70)

        # Each flow sample receives the rating of the annotation
        # covering that timestamp.
        sample_mtb = np.full(
            len(timestamps),
            np.nan,
            dtype=np.float64,
        )

        sample_video = np.full(
            len(timestamps),
            np.nan,
            dtype=np.float64,
        )

        for annotation in video_annotations:
            mask = select_interval(
                timestamps,
                annotation["start"],
                annotation["end"],
            )

            sample_mtb[mask] = annotation["mtb"]
            sample_video[mask] = annotation["video_rating"]

        print()
        print(
            f"{'Feature':25s} "
            f"{'MTB r':>10s} "
            f"{'Video r':>10s}"
        )
        print("-" * 50)

        correlations = []

        for feature in CORE_FEATURES:
            if feature not in columns:
                continue

            r_mtb = correlation(
                columns[feature],
                sample_mtb,
            )

            r_video = correlation(
                columns[feature],
                sample_video,
            )

            correlations.append(
                (
                    feature,
                    r_mtb,
                    r_video,
                )
            )

            print(
                f"{feature:25s} "
                f"{r_mtb:10.3f} "
                f"{r_video:10.3f}"
            )

        # ---------------------------------------------------------------
        # Spatial grid comparison
        # ---------------------------------------------------------------

        grid_features = [
            feature
            for feature in columns
            if re.match(
                r"^g[1-3][1-3]_mag$",
                feature,
            )
        ]

        if grid_features:
            print()
            print("=" * 70)
            print("3x3 spatial flow magnitude")
            print("=" * 70)

            print()
            print(
                "Grid layout:"
            )
            print(
                "  g11   g12   g13"
            )
            print(
                "  g21   g22   g23"
            )
            print(
                "  g31   g32   g33"
            )

            print()
            print(
                f"{'Region':10s} "
                f"{'Annotated':>12s} "
                f"{'Outside':>12s} "
                f"{'Diff':>12s}"
            )

            print("-" * 50)

            for feature in sorted(grid_features):
                a = stats(
                    columns[feature][annotated_mask]
                )

                o = stats(
                    columns[feature][outside_mask]
                )

                if a is None or o is None:
                    continue

                print(
                    f"{feature:10s} "
                    f"{a['mean']:12.3f} "
                    f"{o['mean']:12.3f} "
                    f"{a['mean'] - o['mean']:+12.3f}"
                )

        # ---------------------------------------------------------------
        # Top correlated features
        # ---------------------------------------------------------------

        valid_correlations = [
            item
            for item in correlations
            if math.isfinite(item[1])
        ]

        valid_correlations.sort(
            key=lambda item: abs(item[1]),
            reverse=True,
        )

        if valid_correlations:
            print()
            print("=" * 70)
            print("Features with strongest MTB correlation")
            print("=" * 70)

            for feature, r_mtb, r_video in valid_correlations[:10]:
                print(
                    f"{feature:25s} "
                    f"MTB r={r_mtb:+.3f} "
                    f"Video r={r_video:+.3f}"
                )

    print()
    print("Done.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())