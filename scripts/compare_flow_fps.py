#!/usr/bin/env python3

import argparse
import csv
import math
from pathlib import Path

import numpy as np


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
            raise ValueError(f"{path}: CSV has no header")

        if "time" not in reader.fieldnames:
            raise ValueError(
                f"{path}: CSV has no 'time' column"
            )

        for field in reader.fieldnames:
            if field != "time":
                columns[field] = []

        for row in reader:
            try:
                timestamp = float(row["time"])
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
# Helpers
# ---------------------------------------------------------------------------

def finite(values):
    values = np.asarray(values, dtype=np.float64)
    return values[np.isfinite(values)]


def stats(values):
    values = finite(values)

    if len(values) == 0:
        return None

    return {
        "n": len(values),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "p10": float(np.percentile(values, 10)),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "max": float(np.max(values)),
    }


def format_stats(values):
    s = stats(values)

    if s is None:
        return "no finite samples"

    return (
        f"n={s['n']:6d} "
        f"mean={s['mean']:10.4f} "
        f"std={s['std']:10.4f} "
        f"p10={s['p10']:10.4f} "
        f"median={s['median']:10.4f} "
        f"p90={s['p90']:10.4f} "
        f"max={s['max']:10.4f}"
    )


def timestamp_stats(timestamps):
    if len(timestamps) < 2:
        return None

    dt = np.diff(timestamps)

    return {
        "count": len(timestamps),
        "start": float(timestamps[0]),
        "end": float(timestamps[-1]),
        "duration": float(timestamps[-1] - timestamps[0]),
        "dt_mean": float(np.mean(dt)),
        "dt_median": float(np.median(dt)),
        "dt_min": float(np.min(dt)),
        "dt_max": float(np.max(dt)),
    }


def file_size_mb(path):
    return path.stat().st_size / (1024.0 * 1024.0)


# ---------------------------------------------------------------------------
# Feature comparison
# ---------------------------------------------------------------------------

def print_feature_comparison(
    columns_a,
    columns_b,
    name_a,
    name_b,
):
    common = sorted(
        set(columns_a) & set(columns_b)
    )

    only_a = sorted(
        set(columns_a) - set(columns_b)
    )

    only_b = sorted(
        set(columns_b) - set(columns_a)
    )

    print()
    print("=" * 100)
    print("FEATURE COLUMNS")
    print("=" * 100)

    print(f"Common columns: {len(common)}")

    if only_a:
        print()
        print(f"Only in {name_a}:")
        for feature in only_a:
            print(f"  {feature}")

    if only_b:
        print()
        print(f"Only in {name_b}:")
        for feature in only_b:
            print(f"  {feature}")

    return common


# ---------------------------------------------------------------------------
# Feature statistics
# ---------------------------------------------------------------------------

def print_feature_statistics(
    columns_a,
    columns_b,
    common,
    name_a,
    name_b,
):
    print()
    print("=" * 100)
    print("FEATURE STATISTICS")
    print("=" * 100)

    print()
    print(
        f"{'Feature':30s} "
        f"{name_a + ' mean':>16s} "
        f"{name_b + ' mean':>16s} "
        f"{'mean diff':>14s}"
    )

    print("-" * 100)

    for feature in common:
        a = stats(columns_a[feature])
        b = stats(columns_b[feature])

        if a is None or b is None:
            continue

        diff = b["mean"] - a["mean"]

        print(
            f"{feature:30s} "
            f"{a['mean']:16.4f} "
            f"{b['mean']:16.4f} "
            f"{diff:+14.4f}"
        )


# ---------------------------------------------------------------------------
# 2 FPS -> 10 FPS temporal density
# ---------------------------------------------------------------------------

def print_temporal_density(
    timestamps_a,
    timestamps_b,
    name_a,
    name_b,
):
    print()
    print("=" * 100)
    print("TEMPORAL SAMPLING")
    print("=" * 100)

    a = timestamp_stats(timestamps_a)
    b = timestamp_stats(timestamps_b)

    if a is None or b is None:
        print("Not enough timestamps for comparison.")
        return

    print()
    print(f"{name_a}:")
    print(f"  Samples:          {a['count']}")
    print(f"  First timestamp:  {a['start']:.6f} s")
    print(f"  Last timestamp:   {a['end']:.6f} s")
    print(f"  Duration:         {a['duration']:.3f} s")
    print(f"  Mean dt:          {a['dt_mean']:.6f} s")
    print(f"  Median dt:        {a['dt_median']:.6f} s")
    print(f"  Min dt:           {a['dt_min']:.6f} s")
    print(f"  Max dt:           {a['dt_max']:.6f} s")

    print()
    print(f"{name_b}:")
    print(f"  Samples:          {b['count']}")
    print(f"  First timestamp:  {b['start']:.6f} s")
    print(f"  Last timestamp:   {b['end']:.6f} s")
    print(f"  Duration:         {b['duration']:.3f} s")
    print(f"  Mean dt:          {b['dt_mean']:.6f} s")
    print(f"  Median dt:        {b['dt_median']:.6f} s")
    print(f"  Min dt:           {b['dt_min']:.6f} s")
    print(f"  Max dt:           {b['dt_max']:.6f} s")

    ratio = b["count"] / a["count"]

    print()
    print(f"Sample-count ratio ({name_b}/{name_a}): {ratio:.3f}x")

    print()
    print("Samples in a 4-second model window:")
    print(
        f"  {name_a}: "
        f"{round(4.0 / a['dt_median']):d}"
    )
    print(
        f"  {name_b}: "
        f"{round(4.0 / b['dt_median']):d}"
    )


# ---------------------------------------------------------------------------
# Timestamp alignment
# ---------------------------------------------------------------------------

def print_timestamp_alignment(
    timestamps_a,
    timestamps_b,
    name_a,
    name_b,
):
    print()
    print("=" * 100)
    print("TIMESTAMP ALIGNMENT")
    print("=" * 100)

    if len(timestamps_a) == 0 or len(timestamps_b) == 0:
        print("No timestamps available.")
        return

    common_end = min(
        timestamps_a[-1],
        timestamps_b[-1],
    )

    a = timestamps_a[
        timestamps_a <= common_end
    ]

    b = timestamps_b[
        timestamps_b <= common_end
    ]

    print()
    print(
        f"Common time range: "
        f"0.000 - {common_end:.3f} s"
    )

    print(
        f"{name_a} samples in common range: "
        f"{len(a)}"
    )

    print(
        f"{name_b} samples in common range: "
        f"{len(b)}"
    )

    # For every low-rate timestamp, find the nearest high-rate timestamp.
    indices = np.searchsorted(
        b,
        a,
        side="left",
    )

    indices = np.clip(
        indices,
        0,
        len(b) - 1,
    )

    left = np.maximum(
        indices - 1,
        0,
    )

    right = indices

    left_dist = np.abs(
        b[left] - a
    )

    right_dist = np.abs(
        b[right] - a
    )

    nearest_dist = np.minimum(
        left_dist,
        right_dist,
    )

    print()
    print(
        "Nearest timestamp difference "
        f"({name_a} -> {name_b}):"
    )

    print(
        f"  mean:   {np.mean(nearest_dist):.6f} s"
    )
    print(
        f"  median: {np.median(nearest_dist):.6f} s"
    )
    print(
        f"  max:    {np.max(nearest_dist):.6f} s"
    )


# ---------------------------------------------------------------------------
# Within-0.5-second variation
# ---------------------------------------------------------------------------

def print_subinterval_variation(
    timestamps_10,
    columns_10,
    common_features,
):
    """
    Measure how much the 10 FPS signal changes inside each
    0.5-second interval.

    This is the key question for whether the extra temporal
    resolution contains information that a 2 FPS representation
    could not capture.
    """

    if len(timestamps_10) < 2:
        return

    print()
    print("=" * 100)
    print("10 FPS WITHIN-0.5-SECOND VARIATION")
    print("=" * 100)

    print(
        "For each feature, calculate the mean range of the 10 FPS "
        "samples inside each 0.5 s interval."
    )

    print()

    interval_index = np.floor(
        timestamps_10 / 0.5
    ).astype(np.int64)

    unique_intervals = np.unique(
        interval_index
    )

    for feature in common_features:
        values = columns_10[feature]

        ranges = []

        for interval in unique_intervals:
            mask = (
                interval_index == interval
            )

            interval_values = finite(
                values[mask]
            )

            if len(interval_values) < 2:
                continue

            ranges.append(
                np.max(interval_values)
                - np.min(interval_values)
            )

        if not ranges:
            continue

        ranges = np.asarray(
            ranges,
            dtype=np.float64,
        )

        print(
            f"{feature:30s} "
            f"intervals={len(ranges):6d} "
            f"mean_range={np.mean(ranges):10.4f} "
            f"median_range={np.median(ranges):10.4f} "
            f"p90_range={np.percentile(ranges, 90):10.4f} "
            f"max_range={np.max(ranges):10.4f}"
        )


# ---------------------------------------------------------------------------
# 4-second window representation
# ---------------------------------------------------------------------------

def print_window_representation():
    print()
    print("=" * 100)
    print("4-SECOND MODEL INPUT")
    print("=" * 100)

    print()
    print("With the current dataset design:")
    print()
    print("  Window: 4.0 s")
    print("  Dataset stride: 0.5 s")
    print()
    print("Flow samples per input window:")
    print("  2 FPS:  8 samples")
    print("  10 FPS: 40 samples")
    print()
    print(
        "The 10 FPS representation therefore gives the temporal model "
        "5x more flow time steps for the same 4-second window."
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compare two optical-flow CSVs, typically the same video "
            "generated at different flow FPS values."
        )
    )

    parser.add_argument(
        "flow_a",
        help="First flow CSV, e.g. the existing 2 FPS CSV",
    )

    parser.add_argument(
        "flow_b",
        help="Second flow CSV, e.g. the 10 FPS CSV",
    )

    args = parser.parse_args()

    path_a = Path(args.flow_a)
    path_b = Path(args.flow_b)

    if not path_a.exists():
        raise FileNotFoundError(
            f"Flow CSV not found: {path_a}"
        )

    if not path_b.exists():
        raise FileNotFoundError(
            f"Flow CSV not found: {path_b}"
        )

    name_a = path_a.stem
    name_b = path_b.stem

    print("=" * 100)
    print("OPTICAL FLOW FPS COMPARISON")
    print("=" * 100)

    print()
    print(f"File A: {path_a}")
    print(f"File B: {path_b}")

    print()
    print("File sizes:")
    print(
        f"  {name_a:30s} "
        f"{file_size_mb(path_a):10.2f} MB"
    )
    print(
        f"  {name_b:30s} "
        f"{file_size_mb(path_b):10.2f} MB"
    )

    timestamps_a, columns_a = load_flow_csv(
        path_a
    )

    timestamps_b, columns_b = load_flow_csv(
        path_b
    )

    print()
    print("Rows:")
    print(
        f"  {name_a:30s} "
        f"{len(timestamps_a):10d}"
    )
    print(
        f"  {name_b:30s} "
        f"{len(timestamps_b):10d}"
    )

    common_features = print_feature_comparison(
        columns_a,
        columns_b,
        name_a,
        name_b,
    )

    print_temporal_density(
        timestamps_a,
        timestamps_b,
        name_a,
        name_b,
    )

    print_timestamp_alignment(
        timestamps_a,
        timestamps_b,
        name_a,
        name_b,
    )

    print_feature_statistics(
        columns_a,
        columns_b,
        common_features,
        name_a,
        name_b,
    )

    print_subinterval_variation(
        timestamps_b,
        columns_b,
        common_features,
    )

    print_window_representation()

    print()
    print("=" * 100)
    print("COMPARISON COMPLETE")
    print("=" * 100)


if __name__ == "__main__":
    raise SystemExit(main())