"""Compute interpretable MTB interest scores from cached optical-flow features.

Examples:
    python tools/score_interest.py data/annotations/mentorella.txt --start 300 --end 350
    python tools/score_interest.py data/videos/mentorella.mp4 --window-sizes 4 6 8

The tool is exploratory: it does not change training features or model behavior.
It accepts either a video file or an annotation TXT file. Annotations are optional
and are only overlaid in the HTML; they do not affect score calculation. It reuses
the canonical feature sequence/cache and writes CSV + interactive HTML under
data/plots by default.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Sequence

import numpy as np

from autocut.annotations import load_training_annotations
from autocut.config_loader import load_config
from autocut.data_models import Clip
from autocut.dataset.sequence import build_feature_sequence
from autocut.motion.features import FEATURE_NAMES
from autocut.pipeline.training import _data_values
from autocut.video.probe import get_video_info


def _column(name: str) -> int:
    try:
        return FEATURE_NAMES.index(name)
    except ValueError as exc:
        raise RuntimeError(f"Required canonical feature {name!r} is missing from the schema.") from exc


def _derive_raw_signals(features: np.ndarray, *, epsilon: float = 1e-5) -> dict[str, np.ndarray]:
    """Derive engineered signals from the canonical 54-column feature matrix."""
    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 2 or features.shape[1] != len(FEATURE_NAMES):
        raise ValueError(
            f"Expected a 2D feature array with {len(FEATURE_NAMES)} columns; "
            f"got shape {features.shape}."
        )
    if not np.isfinite(features).all():
        raise ValueError("Feature array contains non-finite values.")

    def col(name: str) -> np.ndarray:
        return features[:, _column(name)]

    vectors: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
    magnitudes: dict[tuple[int, int], np.ndarray] = {}
    angles: list[np.ndarray] = []
    residuals: dict[tuple[int, int], np.ndarray] = {}
    global_u, global_v = col("flow_u"), col("flow_v")
    for row in range(3):
        for column in range(3):
            u = col(f"grid_{row}_{column}_u")
            v = col(f"grid_{row}_{column}_v")
            vectors[(row, column)] = (u, v)
            magnitudes[(row, column)] = np.hypot(u, v)
            angles.append(np.arctan2(v, u))
            residuals[(row, column)] = np.hypot(u - global_u, v - global_v)

    top = np.mean(np.stack([magnitudes[(0, c)] for c in range(3)]), axis=0)
    bottom = np.mean(np.stack([magnitudes[(2, c)] for c in range(3)]), axis=0)
    left = np.mean(np.stack([magnitudes[(r, 0)] for r in range(3)]), axis=0)
    right = np.mean(np.stack([magnitudes[(r, 2)] for r in range(3)]), axis=0)
    center = magnitudes[(1, 1)]
    peripheral = np.mean(np.stack([
        magnitudes[(r, c)] for r in range(3) for c in range(3)
        if (r, c) != (1, 1)
    ]), axis=0)
    mean_cos = np.mean(np.stack([np.cos(a) for a in angles]), axis=0)
    mean_sin = np.mean(np.stack([np.sin(a) for a in angles]), axis=0)

    return {
        "directional_turbulence": 1.0 - np.hypot(mean_cos, mean_sin),
        "lateral_flow_asymmetry": (right - left) / (center + epsilon),
        "roll_index": (
            magnitudes[(0, 0)] + magnitudes[(2, 2)]
            - magnitudes[(0, 2)] - magnitudes[(2, 0)]
        ) / (center + epsilon),
        "vertical_flow_gradient": (top - bottom) / (center + epsilon),
        "expansion_index": (peripheral - center) / (center + epsilon),
        "global_flow_magnitude": np.hypot(global_u, global_v),
        "center_residual": residuals[(1, 1)],
    }


def rolling_median(values: np.ndarray, timestamps: np.ndarray, duration: float) -> np.ndarray:
    """Centered time-based rolling median; edge windows are clipped to available data."""
    values = np.asarray(values, dtype=np.float64)
    timestamps = np.asarray(timestamps, dtype=np.float64)
    if values.ndim != 1 or timestamps.ndim != 1 or len(values) != len(timestamps):
        raise ValueError("values and timestamps must be 1D arrays of equal length.")
    if duration < 0 or not np.isfinite(duration):
        raise ValueError("smoothing duration must be finite and >= 0.")
    if len(timestamps) == 0:
        return values.copy()
    if not np.isfinite(values).all() or not np.isfinite(timestamps).all():
        raise ValueError("values and timestamps must be finite.")
    if np.any(np.diff(timestamps) <= 0):
        raise ValueError("timestamps must be strictly increasing.")
    if duration == 0:
        return values.copy()
    half = duration / 2.0
    result = np.empty_like(values)
    left = 0
    right = 0
    for i, time in enumerate(timestamps):
        while left < len(timestamps) and timestamps[left] < time - half:
            left += 1
        right = max(right, i)
        while right + 1 < len(timestamps) and timestamps[right + 1] <= time + half:
            right += 1
        result[i] = np.median(values[left:right + 1])
    return result


def robust_unit_scale(values: np.ndarray, *, low: float = 5.0, high: float = 95.0,
                      absolute: bool = False) -> np.ndarray:
    """Map a signal to [0, 1] using its robust percentile range."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("normalization expects a finite 1D signal.")
    if not 0 <= low < high <= 100:
        raise ValueError("percentiles must satisfy 0 <= low < high <= 100.")
    signal = np.abs(values) if absolute else values
    lo, hi = np.percentile(signal, [low, high])
    if hi - lo <= np.finfo(np.float64).eps:
        return np.zeros_like(signal)
    return np.clip((signal - lo) / (hi - lo), 0.0, 1.0)


def _sigmoid_gate(values: np.ndarray, low: float, high: float) -> np.ndarray:
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        raise ValueError("motion gate high threshold must be greater than low threshold.")
    x = np.clip((values - (low + high) / 2.0) / ((high - low) / 8.0), -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-x))


def derive_interest_channels(
    features: np.ndarray,
    timestamps: np.ndarray,
    *,
    smoothing_seconds: float = 0.5,
    gate_low: float | None = None,
    gate_high: float | None = None,
) -> dict[str, np.ndarray]:
    """Calculate separate technical, maneuver, apparent-motion and final channels.

    Normalization percentiles are computed over the requested sequence, so scores
    are relative to that sequence, not globally calibrated between videos.
    """
    features = np.asarray(features, dtype=np.float64)
    timestamps = np.asarray(timestamps, dtype=np.float64)
    if features.ndim != 2 or features.shape != (len(timestamps), len(FEATURE_NAMES)):
        raise ValueError(f"Expected features shape (n, {len(FEATURE_NAMES)}), got {features.shape}.")
    if len(timestamps) < 2 or not np.isfinite(timestamps).all() or np.any(np.diff(timestamps) <= 0):
        raise ValueError("At least two finite, strictly increasing timestamps are required.")
    if not np.isfinite(features).all():
        raise ValueError("Feature array contains non-finite values.")

    # Smooth the source channels before any temporal derivative.
    raw = _derive_raw_signals(features)
    smoothed = {name: rolling_median(value, timestamps, smoothing_seconds)
                for name, value in raw.items()}

    tech = robust_unit_scale(smoothed["directional_turbulence"])
    lfa = robust_unit_scale(smoothed["lateral_flow_asymmetry"], absolute=True)
    roll = robust_unit_scale(smoothed["roll_index"], absolute=True)
    vfg = robust_unit_scale(smoothed["vertical_flow_gradient"], absolute=True)
    expansion_rate = np.gradient(smoothed["expansion_index"], timestamps)
    expansion_change = robust_unit_scale(expansion_rate, absolute=True)

    # Components are intentionally explicit and separately inspectable.
    score_tech = tech
    score_maneuver = (lfa + roll + vfg + expansion_change) / 4.0

    motion = smoothed["global_flow_magnitude"]
    if gate_low is None:
        gate_low = float(np.percentile(motion, 20))
    if gate_high is None:
        gate_high = float(np.percentile(motion, 70))
        if gate_high <= gate_low:
            gate_high = gate_low + max(float(np.std(motion)), 1e-6)
    gate_motion = _sigmoid_gate(motion, gate_low, gate_high)
    combined = gate_motion * (0.5 * score_tech + 0.5 * score_maneuver)
    score_final = robust_unit_scale(combined)

    return {
        "Score_Tech": score_tech,
        "Score_Maneuver": score_maneuver,
        "Gate_Motion": gate_motion,
        "Score_Final": score_final,
        "Expansion_Rate": expansion_rate,
        "LFA_Normalized": lfa,
        "Roll_Normalized": roll,
        "VFG_Normalized": vfg,
    }


def aggregate_windows(
    timestamps: np.ndarray,
    channels: dict[str, np.ndarray],
    *,
    window_seconds: float = 4.0,
    stride_seconds: float = 2.0,
) -> list[dict[str, float]]:
    """Aggregate channels into overlapping windows with max/mean emphasis."""
    timestamps = np.asarray(timestamps, dtype=np.float64)
    if window_seconds <= 0 or stride_seconds <= 0:
        raise ValueError("window and stride must be greater than zero.")
    if not len(timestamps) or np.any(np.diff(timestamps) <= 0):
        raise ValueError("timestamps must be non-empty and strictly increasing.")
    for name, values in channels.items():
        if np.asarray(values).shape != timestamps.shape:
            raise ValueError(f"Channel {name!r} length does not match timestamps.")

    results: list[dict[str, float]] = []
    start = float(timestamps[0])
    end = float(timestamps[-1])
    while start <= end:
        stop = start + window_seconds
        mask = (timestamps >= start) & (timestamps < stop)
        if mask.any():
            row: dict[str, float] = {"window_start": start, "window_end": min(stop, end)}
            for name, values in channels.items():
                selected = np.asarray(values)[mask]
                row[name] = float(0.6 * np.max(selected) + 0.4 * np.mean(selected))
            results.append(row)
        start += stride_seconds
    return results


def _write_csv(path: Path, rows: list[dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("No windows were generated.")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_html(path: Path, rows_by_window: dict[float, list[dict[str, float]]],
                annotations: Sequence[object], video_name: str) -> None:
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError as exc:
        raise ImportError("HTML plotting requires Plotly: pip install plotly") from exc
    channel_names = ("Score_Tech", "Score_Maneuver", "Gate_Motion", "Score_Final")
    fig = make_subplots(rows=len(channel_names), cols=1, shared_xaxes=True,
                        subplot_titles=channel_names, vertical_spacing=0.04)
    colors = {0: "#8A8A8A", 1: "#B0D7D7", 2: "#E69F00", 3: "#D55E00"}
    for row_idx, channel in enumerate(channel_names, start=1):
        for window, rows in rows_by_window.items():
            centers = [(r["window_start"] + r["window_end"]) / 2 for r in rows]
            fig.add_trace(go.Scatter(x=centers, y=[r[channel] for r in rows], mode="lines+markers",
                                     name=f"{channel} — {window:g}s", legendgroup=f"{window:g}",
                                     line={"width": 1.5}), row=row_idx, col=1)
        for ann in annotations:
            fig.add_vrect(x0=ann.start, x1=ann.end, fillcolor=colors.get(int(ann.score), "#CCCCCC"),
                          opacity=0.15, line_width=0, layer="below", row=row_idx, col=1)
    fig.update_layout(title=f"Interest score channels — {video_name}", template="plotly_white",
                      hovermode="x unified", height=1000)
    fig.update_xaxes(title_text="Video time (seconds)")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(path, include_plotlyjs="cdn", full_html=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path,
                        help="Video file to score, or an annotation TXT file that identifies a video.")
    parser.add_argument("--config", type=Path, default=Path("config/train.yaml"))
    parser.add_argument("--start", type=float, help="Optional start time in seconds.")
    parser.add_argument("--end", type=float, help="Optional end time in seconds.")
    parser.add_argument("--window-sizes", type=float, nargs="+", default=[2.0, 4.0, 6.0, 8.0])
    parser.add_argument("--stride", type=float, default=2.0, help="Window stride in seconds (default: 2).")
    parser.add_argument("--smoothing-seconds", type=float, default=0.5)
    parser.add_argument("--gate-low", type=float, help="Optional explicit apparent-motion gate lower threshold.")
    parser.add_argument("--gate-high", type=float, help="Optional explicit apparent-motion gate upper threshold.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/plots"))
    parser.add_argument("--allow-extraction", action="store_true",
                        help="Allow expensive feature extraction if the compatible cache is missing.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.start is not None and args.start < 0:
        raise ValueError("--start must be >= 0.")
    if args.end is not None and args.end <= 0:
        raise ValueError("--end must be > 0.")
    if any(window <= 0 for window in args.window_sizes) or args.stride <= 0:
        raise ValueError("window sizes and stride must be greater than zero.")
    if args.gate_low is not None and args.gate_high is None or args.gate_high is not None and args.gate_low is None:
        raise ValueError("--gate-low and --gate-high must be supplied together.")
    if not args.source.is_file():
        raise FileNotFoundError(f"Input file not found: {args.source}")

    # Annotations are optional context for the plot, not an input to the scorer.
    # TXT inputs retain the existing annotation-file workflow; other files are
    # treated as videos and can be scored without labels.
    if args.source.suffix.lower() == ".txt":
        video_path, annotations = load_training_annotations(args.source)
        if not video_path.is_file():
            raise FileNotFoundError(f"Video referenced by annotation file not found: {video_path}")
    else:
        video_path = args.source
        annotations = []
    settings = load_config(args.config)
    (_, sample_fps, processing_width, _, _, roi_left, roi_right, roi_bottom, roi_top) = _data_values(settings)
    paths = settings.get("paths", {})
    cache_dir = Path(paths.get("cache_dir", "data/cache")).expanduser()
    info = get_video_info(video_path)
    clip_start = max(0.0, args.start if args.start is not None else 0.0)
    default_end = max((a.end for a in annotations), default=float(info.duration))
    clip_end = min(float(info.duration), args.end if args.end is not None else default_end)
    if clip_end <= clip_start:
        raise ValueError("Requested interval does not overlap the video.")
    clip = Clip(video_name=video_path.stem, start=clip_start, end=clip_end)

    # Avoid accidental expensive extraction; the explicit flag is required on cache miss.
    if not args.allow_extraction:
        from autocut.dataset.feature_cache import load_feature_sequence
        sequence = load_feature_sequence(
            video_path, cache_dir, clip, sample_fps=sample_fps,
            processing_width=processing_width, roi_left=roi_left, roi_right=roi_right,
            roi_bottom=roi_bottom, roi_top=roi_top,
        )
        if sequence is None:
            raise RuntimeError(
                "No exact compatible feature cache entry was found. Re-run with "
                "--allow-extraction to permit video decoding and extraction."
            )
    else:
        sequence = build_feature_sequence(
            video_path, clip, sample_fps=sample_fps, processing_width=processing_width,
            cache_dir=cache_dir, roi_left=roi_left, roi_right=roi_right,
            roi_bottom=roi_bottom, roi_top=roi_top,
        )

    channels = derive_interest_channels(
        sequence.features, sequence.timestamps, smoothing_seconds=args.smoothing_seconds,
        gate_low=args.gate_low, gate_high=args.gate_high,
    )
    rows_by_window = {
        float(window): aggregate_windows(sequence.timestamps, channels,
                                         window_seconds=float(window), stride_seconds=args.stride)
        for window in args.window_sizes
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for window, rows in rows_by_window.items():
        stem = f"{video_path.stem}_interest_{window:g}s"
        csv_path = args.output_dir / f"{stem}.csv"
        _write_csv(csv_path, rows)
        print(f"Saved {csv_path}")
    html_path = args.output_dir / f"{video_path.stem}_interest_comparison.html"
    _write_html(html_path, rows_by_window, annotations, video_path.name)
    print(f"Saved {html_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
