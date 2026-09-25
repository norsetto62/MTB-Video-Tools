#!/usr/bin/env python3
"""
detect_flow_events.py

Detect candidate events from optical-flow feature CSV files.

Version 4
---------
- Detect elevated regions independently for each feature.
- Coherence is analysed and written to the score trace, but does NOT
  generate candidate events.
- Candidate regions from the other features are merged temporally,
  preserving the v3 "nearby signals belong to the same sequence" logic.
- Individual contributing feature regions are preserved in the event CSV.
- Events are padded after merging.
- Produces:
    output/candidate_events/<input>_candidate_events.csv
    output/candidate_events/<input>_event_scores.csv

Expected input columns (aliases are supported):
    time
    flow_mean
    flow_coherence
    angle_change_abs
    div_abs_mean
    curl_abs_mean
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# ============================================================================
# Configuration
# ============================================================================

SMOOTHING_SECONDS = 2.0
THRESHOLD_PERCENTILE = 92.0

MIN_REGION_DURATION_SECONDS = 0.8
MAX_INTERNAL_GAP_SECONDS = 1.0

EVENT_PADDING_SECONDS = 1.0
MERGE_GAP_SECONDS = 1.5

# Coherence remains in the analysis/score trace, but does not create events.
COHERENCE_GENERATES_EVENTS = False

FEATURES = {
    "magnitude": "flow_mean",
    "angle": "angle_change_abs",
    "divergence": "div_abs_mean",
    "curl": "curl_abs_mean",
    "coherence": "flow_coherence",
}


# ============================================================================
# Utility functions
# ============================================================================

def resolve_column(df: pd.DataFrame, aliases: List[str], required: bool = True) -> str | None:
    """
    Resolve a column name from a list of possible aliases.
    Matching is case-insensitive.
    """
    normalized = {str(c).strip().lower(): c for c in df.columns}

    for alias in aliases:
        key = alias.strip().lower()
        if key in normalized:
            return normalized[key]

    if required:
        raise ValueError(
            f"Could not find any of these columns: {', '.join(aliases)}\n"
            f"Available columns:\n  " + "\n  ".join(map(str, df.columns))
        )

    return None


def robust_zscore(series: pd.Series) -> pd.Series:
    """
    Robust z-score using median and MAD.

    z = (x - median) / (1.4826 * MAD)

    Negative values are clipped to zero because we are interested in
    unusually HIGH values.
    """
    x = pd.to_numeric(series, errors="coerce").astype(float)

    median = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - median))

    if not np.isfinite(mad) or mad <= 1e-12:
        std = np.nanstd(x)

        if not np.isfinite(std) or std <= 1e-12:
            return pd.Series(np.zeros(len(series)), index=series.index)

        z = (x - median) / std
    else:
        z = (x - median) / (1.4826 * mad)

    z = pd.Series(z, index=series.index)
    z = z.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return z.clip(lower=0.0)


def smooth_series(series: pd.Series, sample_period: float) -> pd.Series:
    """
    Centered rolling median smoothing.
    """
    window = max(1, int(round(SMOOTHING_SECONDS / sample_period)))

    # Prefer an odd window so the centered median is symmetric.
    if window % 2 == 0:
        window += 1

    return (
        series
        .rolling(window=window, center=True, min_periods=1)
        .median()
    )


def fill_small_gaps(
    above: np.ndarray,
    max_gap_samples: int,
) -> np.ndarray:
    """
    Fill False gaps between True regions when the gap is short enough.
    """
    result = above.copy()

    n = len(result)
    i = 0

    while i < n:
        if result[i]:
            i += 1
            continue

        start = i

        while i < n and not result[i]:
            i += 1

        end = i

        gap_length = end - start

        left_true = start > 0 and result[start - 1]
        right_true = end < n and result[end]

        if left_true and right_true and gap_length <= max_gap_samples:
            result[start:end] = True

    return result


def find_elevated_regions(
    time: np.ndarray,
    score: np.ndarray,
    threshold: float,
    sample_period: float,
) -> List[dict]:
    """
    Find sustained regions where score >= threshold.

    Returns dictionaries containing:
        start
        end
        duration
        peak
        peak_time
        mean
    """
    above = np.asarray(score >= threshold, dtype=bool)

    max_gap_samples = max(
        0,
        int(round(MAX_INTERNAL_GAP_SECONDS / sample_period))
    )

    above = fill_small_gaps(above, max_gap_samples)

    regions: List[dict] = []

    n = len(above)
    i = 0

    while i < n:
        if not above[i]:
            i += 1
            continue

        start_idx = i

        while i < n and above[i]:
            i += 1

        end_idx = i - 1

        start_time = float(time[start_idx])
        end_time = float(time[end_idx])

        duration = end_time - start_time + sample_period

        if duration < MIN_REGION_DURATION_SECONDS:
            continue

        region_scores = score[start_idx:end_idx + 1]

        local_peak_idx = int(np.nanargmax(region_scores))
        peak_idx = start_idx + local_peak_idx

        regions.append(
            {
                "start": start_time,
                "end": end_time,
                "duration": duration,
                "peak": float(score[peak_idx]),
                "peak_time": float(time[peak_idx]),
                "mean": float(np.nanmean(region_scores)),
            }
        )

    return regions


def format_region(region: dict) -> str:
    """
    Compact human-readable representation of one feature region.
    """
    return (
        f"{region['start']:.2f}-{region['end']:.2f}"
    )


# ============================================================================
# Event merging
# ============================================================================

def merge_feature_regions(
    feature_regions: Dict[str, List[dict]],
) -> List[dict]:
    """
    Merge elevated regions from different features into temporal candidate
    events.

    IMPORTANT:
    This intentionally retains the v3 temporal merging philosophy.

    Two regions are connected when:
        next.start <= current_end + MERGE_GAP_SECONDS

    This means a chain such as:

        angle      33.4-35.6
        curl       33.3-36.3
        magnitude  39.0-41.3
        curl       43.8-45.0

    can become ONE candidate sequence.

    Coherence is excluded from candidate generation.
    """

    candidates = []

    for feature_name, regions in feature_regions.items():

        if feature_name == "coherence" and not COHERENCE_GENERATES_EVENTS:
            continue

        for region in regions:
            candidates.append(
                {
                    "feature": feature_name,
                    "region": region,
                }
            )

    candidates.sort(
        key=lambda x: (
            x["region"]["start"],
            x["region"]["end"],
        )
    )

    if not candidates:
        return []

    merged: List[dict] = []

    current = {
        "start": candidates[0]["region"]["start"],
        "end": candidates[0]["region"]["end"],
        "regions": [candidates[0]],
    }

    for candidate in candidates[1:]:

        region = candidate["region"]

        # Same v3 philosophy:
        # if the next region starts before the current sequence has been
        # separated by more than MERGE_GAP_SECONDS, merge it.
        if region["start"] <= current["end"] + MERGE_GAP_SECONDS:

            current["end"] = max(
                current["end"],
                region["end"],
            )

            current["regions"].append(candidate)

        else:
            merged.append(current)

            current = {
                "start": region["start"],
                "end": region["end"],
                "regions": [candidate],
            }

    merged.append(current)

    return merged


def calculate_event_properties(
    event: dict,
    all_feature_regions: Dict[str, List[dict]],
    score_columns: Dict[str, pd.Series],
    time: np.ndarray,
) -> dict:
    """
    Calculate summary information for one merged candidate event.

    The event's SCORE is the maximum robust score observed among the
    feature regions that contributed to the event.

    The individual feature regions are retained in text form.
    """

    contributing_features = sorted(
        set(r["feature"] for r in event["regions"])
    )

    # Keep a deterministic feature order rather than alphabetical order.
    preferred_order = [
        "magnitude",
        "angle",
        "divergence",
        "curl",
        "coherence",
    ]

    ordered_features = [
        f for f in preferred_order
        if f in contributing_features
    ]

    feature_strings = []

    for feature_name in ordered_features:

        feature_specific = [
            r["region"]
            for r in event["regions"]
            if r["feature"] == feature_name
        ]

        for region in feature_specific:
            feature_strings.append(
                f"{feature_name}[{format_region(region)}]"
            )

    # Determine strongest contributing feature region.
    strongest = max(
        event["regions"],
        key=lambda x: x["region"]["peak"]
    )

    return {
        "start": event["start"],
        "end": event["end"],
        "features": "+".join(ordered_features),
        "feature_regions": ";".join(feature_strings),
        "peak": strongest["region"]["peak_time"],
        "score": strongest["region"]["peak"],
        "strongest_feature": strongest["feature"],
        "num_feature_regions": len(event["regions"]),
    }


# ============================================================================
# Main analysis
# ============================================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description="Detect candidate MTB events from optical-flow features."
    )

    parser.add_argument(
        "input_csv",
        type=Path,
        help="Input optical-flow CSV file",
    )

    args = parser.parse_args()

    input_csv = args.input_csv

    if not input_csv.exists():
        raise FileNotFoundError(
            f"Input CSV not found:\n{input_csv}"
        )

    print("=" * 72)
    print(f"Input: {input_csv}")
    print("=" * 72)

    df = pd.read_csv(input_csv)

    if df.empty:
        raise ValueError("Input CSV is empty.")

    # ------------------------------------------------------------------------
    # Resolve columns
    # ------------------------------------------------------------------------

    aliases = {
        "time": [
            "time",
            "timestamp",
            "t",
        ],
        "flow_mean": [
            "flow_mean",
            "mean_flow",
            "flow_magnitude_mean",
        ],
        "flow_coherence": [
            "flow_coherence",
            "coherence",
        ],
        "angle_change_abs": [
            "angle_change_abs",
            "angle_change",
            "flow_angle_change_abs",
        ],
        "div_abs_mean": [
            "div_abs_mean",
            "divergence_abs_mean",
            "divergence",
        ],
        "curl_abs_mean": [
            "curl_abs_mean",
            "curl_abs",
            "curl",
        ],
    }

    resolved = {}

    for key, possible in aliases.items():
        required = True

        # All five features are expected.
        resolved[key] = resolve_column(
            df,
            possible,
            required=required,
        )

    print("Rows:", len(df))

    print("Resolved columns:")
    for key, column in resolved.items():
        print(f"  {key:<20} -> {column}")

    # ------------------------------------------------------------------------
    # Time / sample period
    # ------------------------------------------------------------------------

    time = pd.to_numeric(
        df[resolved["time"]],
        errors="coerce",
    ).to_numpy(dtype=float)

    valid_time = np.isfinite(time)

    if valid_time.sum() < 2:
        raise ValueError("Not enough valid time samples.")

    time_diffs = np.diff(time[valid_time])

    sample_period = float(np.median(time_diffs))

    if not np.isfinite(sample_period) or sample_period <= 0:
        raise ValueError(
            f"Invalid sample period: {sample_period}"
        )

    print(f"Sample period:          {sample_period:.3f} s")
    print(f"Smoothing:              {SMOOTHING_SECONDS:.2f} s")
    print(f"Feature threshold:      {THRESHOLD_PERCENTILE:.1f} percentile")
    print(f"Min region duration:    {MIN_REGION_DURATION_SECONDS:.2f} s")
    print(f"Max internal gap:       {MAX_INTERNAL_GAP_SECONDS:.2f} s")
    print(f"Event padding:          {EVENT_PADDING_SECONDS:.2f} s")
    print(f"Merge gap:              {MERGE_GAP_SECONDS:.2f} s")
    print(
        "Coherence candidates:   "
        + ("YES" if COHERENCE_GENERATES_EVENTS else "NO")
    )

    # ------------------------------------------------------------------------
    # Feature analysis
    # ------------------------------------------------------------------------

    feature_regions: Dict[str, List[dict]] = {}
    score_trace = pd.DataFrame()

    score_trace["time"] = time

    print()
    print("Feature-specific elevated regions:")

    for feature_name, column_name in FEATURES.items():

        raw = pd.to_numeric(
            df[resolved[column_name]],
            errors="coerce",
        ).astype(float)

        smoothed = smooth_series(
            raw,
            sample_period,
        )

        score = robust_zscore(smoothed)

        threshold = float(
            np.nanpercentile(
                score.to_numpy(),
                THRESHOLD_PERCENTILE,
            )
        )

        regions = find_elevated_regions(
            time=time,
            score=score.to_numpy(),
            threshold=threshold,
            sample_period=sample_period,
        )

        feature_regions[feature_name] = regions

        # --------------------------------------------------------------------
        # Score trace
        # --------------------------------------------------------------------

        score_trace[f"{feature_name}_raw"] = raw.to_numpy()
        score_trace[f"{feature_name}_smooth"] = smoothed.to_numpy()
        score_trace[f"{feature_name}_score"] = score.to_numpy()
        score_trace[f"{feature_name}_threshold"] = threshold
        score_trace[f"{feature_name}_above"] = (
            score.to_numpy() >= threshold
        )

        print()
        print(
            f"  {feature_name:<12}: "
            f"{len(regions)} region(s)"
        )

        if not regions:
            print("       none")
            continue

        for index, region in enumerate(regions, start=1):
            print(
                f"       {index:>2}. "
                f"{region['start']:7.2f} - "
                f"{region['end']:7.2f}s "
                f"dur={region['duration']:5.2f}s "
                f"peak={region['peak_time']:7.2f}s "
                f"score={region['peak']:6.3f} "
                f"mean={region['mean']:6.3f} "
                f"threshold={threshold:5.3f}"
            )

    # ------------------------------------------------------------------------
    # Merge candidates
    # ------------------------------------------------------------------------

    merged_events = merge_feature_regions(
        feature_regions
    )

    print()
    print(f"Candidate events: {len(merged_events)}")

    # ------------------------------------------------------------------------
    # Build final event table
    # ------------------------------------------------------------------------

    event_rows = []

    for event_id, event in enumerate(merged_events, start=1):

        props = calculate_event_properties(
            event=event,
            all_feature_regions=feature_regions,
            score_columns={},
            time=time,
        )

        # Apply padding ONLY after the merged event has been established.
        padded_start = max(
            float(time[0]),
            props["start"] - EVENT_PADDING_SECONDS,
        )

        padded_end = min(
            float(time[-1]),
            props["end"] + EVENT_PADDING_SECONDS,
        )

        duration = padded_end - padded_start

        event_rows.append(
            {
                "id": event_id,
                "start": padded_start,
                "end": padded_end,
                "duration": duration,
                "peak": props["peak"],
                "score": props["score"],
                "features": props["features"],
                "strongest_feature": props["strongest_feature"],
                "num_feature_regions": props["num_feature_regions"],
                "feature_regions": props["feature_regions"],
            }
        )

    events_df = pd.DataFrame(event_rows)

    # ------------------------------------------------------------------------
    # Add final event markers to score trace
    # ------------------------------------------------------------------------

    score_trace["candidate_event_id"] = 0

    for row in event_rows:

        mask = (
            (score_trace["time"] >= row["start"])
            & (score_trace["time"] <= row["end"])
        )

        score_trace.loc[
            mask,
            "candidate_event_id"
        ] = row["id"]

    # ------------------------------------------------------------------------
    # Output paths
    # ------------------------------------------------------------------------

    output_dir = (
        Path("output")
        / "candidate_events"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    stem = input_csv.stem

    events_path = (
        output_dir
        / f"{stem}_candidate_events.csv"
    )

    scores_path = (
        output_dir
        / f"{stem}_event_scores.csv"
    )

    events_df.to_csv(
        events_path,
        index=False,
        float_format="%.3f",
    )

    score_trace.to_csv(
        scores_path,
        index=False,
        float_format="%.6f",
    )

    # ------------------------------------------------------------------------
    # Console summary
    # ------------------------------------------------------------------------

    print(f"Events output: {events_path}")
    print(f"Score trace:   {scores_path}")

    print()
    print("Candidate events:")
    print()

    if events_df.empty:
        print("  None")
    else:
        print(
            " ID    START      END     DUR     PEAK    "
            "SCORE FEATURES"
        )

        for _, row in events_df.iterrows():

            print(
                f"{int(row['id']):3d} "
                f"{row['start']:9.2f} "
                f"{row['end']:9.2f} "
                f"{row['duration']:7.2f} "
                f"{row['peak']:9.2f} "
                f"{row['score']:8.3f} "
                f"{row['features']}"
            )

            print(
                f"       regions: {row['feature_regions']}"
            )

    print()
    print("Finished: 1/1 successful.")


if __name__ == "__main__":
    main()