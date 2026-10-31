"""Plot one canonical motion feature over a video's training annotations.

Examples:
    python tools/plot_feature.py annotations.txt flow_mean
    python tools/plot_feature.py annotations.txt 0 --config config/train.yaml --show
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np

from autocut.annotations import load_training_annotations
from autocut.config_loader import load_config
from autocut.data_models import Clip
from autocut.dataset.sequence import build_feature_sequence
from autocut.dataset.cache_paths import feature_cache_dir
from autocut.motion.features import FEATURE_NAMES
from autocut.pipeline.training import _data_values
from autocut.video.probe import get_video_info


# Saturated, visually distinct colours. Shading is deliberately stronger than
# the probability plot so annotation classes remain easy to distinguish.
ANNOTATION_COLORS = {
    0: "#8A8A8A",  # uninteresting
    1: "#0072B2",  # blue
    2: "#E69F00",  # orange
    3: "#D55E00",  # vermillion / P
}


def resolve_feature(feature: str | int) -> tuple[int, str]:
    """Resolve a canonical feature name or zero-based feature index."""
    value = str(feature).strip()
    if not value:
        raise ValueError("Feature must be a canonical feature name or zero-based index.")

    try:
        index = int(value)
    except ValueError:
        try:
            index = FEATURE_NAMES.index(value)
        except ValueError as exc:
            raise ValueError(
                f"Unknown feature {value!r}. Use one of: {', '.join(FEATURE_NAMES)}"
            ) from exc

    if index < 0 or index >= len(FEATURE_NAMES):
        raise ValueError(
            f"Feature index must be between 0 and {len(FEATURE_NAMES) - 1}; got {index}."
        )
    return index, FEATURE_NAMES[index]


def apply_savgol_filter(
    values: np.ndarray,
    *,
    window_length: int = 9,
    polyorder: int = 2,
) -> np.ndarray:
    """Return a Savitzky-Golay-smoothed copy of a 1D feature signal.

    Window length is measured in samples and must be odd; polyorder must be
    smaller than the window length. Values are not modified in place.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("feature values must be a 1D array.")
    if window_length < 3 or window_length % 2 == 0:
        raise ValueError("Savitzky-Golay window length must be an odd integer >= 3.")
    if polyorder < 0 or polyorder >= window_length:
        raise ValueError("Savitzky-Golay polynomial degree must be >= 0 and smaller than the window length.")
    if window_length > len(values):
        raise ValueError(f"Savitzky-Golay window length ({window_length}) exceeds the number of samples ({len(values)}).")
    try:
        from scipy.signal import savgol_filter
    except ImportError as exc:
        raise ImportError(
            "Savitzky-Golay filtering requires SciPy. Install it with: "
            "python -m pip install -e '.[plot]'"
        ) from exc
    return savgol_filter(values, window_length=window_length, polyorder=polyorder)


def plot_feature(
    timestamps: np.ndarray,
    values: np.ndarray,
    annotations: Sequence[object],
    *,
    feature_name: str,
    video_name: str,
    output_path: str | Path,
    show: bool = False,
    filtered_values: np.ndarray | None = None,
    filter_label: str = "Savitzky–Golay",
) -> Path:
    """Save a feature-vs-time plot with strongly coloured annotation intervals."""
    timestamps = np.asarray(timestamps, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    if timestamps.ndim != 1 or values.shape != timestamps.shape:
        raise ValueError("timestamps and feature values must be 1D arrays with the same shape.")
    if not len(timestamps):
        raise ValueError("No feature samples are available for this video.")
    if filtered_values is not None:
        filtered_values = np.asarray(filtered_values, dtype=np.float64)
        if filtered_values.shape != values.shape or not np.isfinite(filtered_values).all():
            raise ValueError("filtered_values must match values in shape and contain only finite values.")
    if not np.isfinite(timestamps).all() or not np.isfinite(values).all():
        raise ValueError("timestamps and feature values must contain only finite values.")
    if len(timestamps) > 1 and np.any(np.diff(timestamps) <= 0):
        raise ValueError("timestamps must be strictly increasing.")

    try:
        import matplotlib
        # This utility always writes a static PNG. Interactive viewing is
        # handled separately by a browser-based Plotly HTML page.
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
        from matplotlib.ticker import FuncFormatter
    except ImportError as exc:
        raise ImportError(
            "Plotting requires matplotlib. Install it with: "
            "python -m pip install -e '.[plot]'"
        ) from exc

    figure, ax = plt.subplots(figsize=(16, 6))
    for annotation in annotations:
        ax.axvspan(
            annotation.start,
            annotation.end,
            facecolor=ANNOTATION_COLORS[annotation.score],
            alpha=0.34,
            edgecolor=ANNOTATION_COLORS[annotation.score],
            linewidth=0.7,
            zorder=0,
        )

    ax.plot(
        timestamps,
        values,
        color="#111111",
        linewidth=1.6,
        label=feature_name,
        zorder=3,
    )
    if filtered_values is not None:
        ax.plot(
            timestamps,
            filtered_values,
            color="#0072B2",
            linewidth=2.0,
            label=filter_label,
            zorder=4,
        )
    ax.set_xlim(0, max(float(timestamps[-1]), max((a.end for a in annotations), default=0.0)))
    ax.set_xlabel("Video time (min:sec)")
    ax.set_ylabel(feature_name)
    ax.set_title(f"{feature_name} over time — {video_name}", loc="left", fontweight="bold")
    ax.grid(True, color="#777777", alpha=0.25, linewidth=0.6, zorder=1)
    ax.xaxis.set_major_formatter(
        FuncFormatter(lambda value, _position: f"{int(value // 60):02d}:{int(value % 60):02d}")
    )
    annotation_handles = [
        Patch(
            facecolor=ANNOTATION_COLORS[score],
            edgecolor=ANNOTATION_COLORS[score],
            alpha=0.75,
            label=f"Training MTB {score}" + (" (P)" if score == 3 else " (U)" if score in (0, 1, 2) else ""),
        )
        for score in range(4)
    ]
    ax.legend(handles=[*annotation_handles, *ax.get_legend_handles_labels()[0]], loc="upper right", ncol=5)
    figure.tight_layout()

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=160, bbox_inches="tight")
    plt.close(figure)

    if show:
        html_destination = destination.with_suffix(".html")
        save_interactive_html(
            timestamps,
            values,
            annotations,
            feature_name=feature_name,
            video_name=video_name,
            output_path=html_destination,
            filtered_values=filtered_values,
            filter_label=filter_label,
        )
        import webbrowser

        webbrowser.open(html_destination.resolve().as_uri())
        print(f"Opened interactive feature plot: {html_destination.resolve()}")

    return destination


def save_interactive_html(
    timestamps: np.ndarray,
    values: np.ndarray,
    annotations: Sequence[object],
    *,
    feature_name: str,
    video_name: str,
    output_path: str | Path,
    filtered_values: np.ndarray | None = None,
    filter_label: str = "Savitzky–Golay",
) -> Path:
    """Write a zoomable, hoverable Plotly chart and open it in a browser."""
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise ImportError(
            "Browser-based interactive plotting requires Plotly. Install it with: "
            "python -m pip install plotly"
        ) from exc

    figure = go.Figure()
    for annotation in annotations:
        score = int(annotation.score)
        figure.add_vrect(
            x0=annotation.start,
            x1=annotation.end,
            fillcolor=ANNOTATION_COLORS[score],
            opacity=0.30,
            line_width=0,
            layer="below",
        )
    figure.add_trace(
        go.Scatter(
            x=timestamps,
            y=values,
            mode="lines",
            name=feature_name,
            line={"color": "#111111", "width": 1.5},
            hovertemplate="Time: %{x:.2f}s<br>Value: %{y:.6g}<extra></extra>",
        )
    )
    if filtered_values is not None:
        figure.add_trace(
            go.Scatter(
                x=timestamps,
                y=filtered_values,
                mode="lines",
                name=filter_label,
                line={"color": "#0072B2", "width": 2.2},
                hovertemplate="Time: %{x:.2f}s<br>Filtered value: %{y:.6g}<extra></extra>",
            )
        )
    for score, color in ANNOTATION_COLORS.items():
        label = f"MTB {score}" + (" (P)" if score == 3 else " (U)" if score in (0, 1, 2) else "")
        figure.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker={"size": 10, "color": color, "symbol": "square"},
                name=label,
                hoverinfo="skip",
            )
        )
    figure.update_layout(
        title={"text": f"{feature_name} over time — {video_name}", "x": 0.01},
        xaxis_title="Video time (seconds)",
        yaxis_title=feature_name,
        hovermode="x",
        template="plotly_white",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0},
        margin={"l": 70, "r": 30, "t": 110, "b": 70},
    )
    figure.update_xaxes(showgrid=True)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(destination, include_plotlyjs="cdn", full_html=True)
    return destination


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plot one canonical motion feature against video time, with high-contrast "
            "background shading for the video's training annotation scores."
        )
    )
    parser.add_argument("annotations", type=Path, nargs="?", help="Training annotation TXT file.")
    parser.add_argument("feature", nargs="?", help="Canonical feature name or zero-based feature index.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/train.yaml"),
        help="Training YAML; its data sampling, ROI, and cache settings are used.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output image path (default: <annotations-stem>_<feature>_feature.png).",
    )
    parser.add_argument("--show", action="store_true", help="Also open the plot interactively.")
    parser.add_argument(
        "--savgol-window-length", type=int, default=9,
        help="Savitzky-Golay window length in samples (odd integer >= 3; default: 9).",
    )
    parser.add_argument(
        "--savgol-polyorder", type=int, default=2,
        help="Savitzky-Golay polynomial degree, smaller than the window length (default: 2).",
    )
    parser.add_argument("--list-features", action="store_true", help="List feature indices and names, then exit.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.list_features:
        for index, name in enumerate(FEATURE_NAMES):
            print(f"{index:2d}  {name}")
        return 0

    if args.annotations is None or args.feature is None:
        parser.error("annotations and feature are required unless --list-features is used")
    if not args.annotations.is_file():
        raise FileNotFoundError(f"Annotation file not found: {args.annotations}")
    feature_index, feature_name = resolve_feature(args.feature)
    video_path, annotations = load_training_annotations(args.annotations)
    if not annotations:
        raise ValueError(f"Annotation file contains no training intervals: {args.annotations}")

    settings = load_config(args.config)
    (
        _manifest,
        sample_fps,
        processing_width,
        _window_duration,
        _window_stride,
        roi_left,
        roi_right,
        roi_bottom,
        roi_top,
    ) = _data_values(settings)
    cache_dir = feature_cache_dir(settings)
    video_info = get_video_info(video_path)
    # Match training's exact cache identity: the extraction clip ends at the
    # last annotation end (clamped to the probed video duration).
    clip = Clip(
        video_name=video_path.stem,
        start=0.0,
        end=min(float(annotations[-1].end), float(video_info.duration)),
    )
    sequence = build_feature_sequence(
        video_path,
        clip,
        sample_fps=sample_fps,
        processing_width=processing_width,
        cache_dir=cache_dir,
        roi_left=roi_left,
        roi_right=roi_right,
        roi_bottom=roi_bottom,
        roi_top=roi_top,
    )

    filtered_values = apply_savgol_filter(
        sequence.features[:, feature_index],
        window_length=args.savgol_window_length,
        polyorder=args.savgol_polyorder,
    )

    output_path = args.output or args.annotations.with_name(
        f"{args.annotations.stem}_{feature_name}_feature.png"
    )
    saved_path = plot_feature(
        sequence.timestamps,
        sequence.features[:, feature_index],
        annotations,
        feature_name=feature_name,
        video_name=video_path.name,
        output_path=output_path,
        show=args.show,
        filtered_values=filtered_values,
        filter_label=f"Savitzky–Golay (window={args.savgol_window_length}, degree={args.savgol_polyorder})",
    )
    print(f"Saved feature plot: {saved_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
