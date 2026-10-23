"""Plot engineered optical-flow signals against MTB training annotations.

Examples:
    python tools/plot_motion_experiments.py annotations.txt --show
    python tools/plot_motion_experiments.py annotations.txt --signals edge_residual,tunnel_vision --show
    python tools/plot_motion_experiments.py annotations.txt --edge-cells 0,0 0,2 --center-cell 1,1
    python tools/plot_motion_experiments.py --list-signals

Signals are derived from the canonical 54-column feature sequence. No model
training or feature-schema changes are performed by this tool.
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
from autocut.motion.features import FEATURE_NAMES
from autocut.pipeline.training import _data_values
from autocut.video.probe import get_video_info

ANNOTATION_COLORS = {
    0: "#8A8A8A",
    1: "#0072B2",
    2: "#E69F00",
    3: "#D55E00",
}
DEFAULT_SIGNALS = (
    "edge_left_residual", "edge_right_residual", "edge_residual",
    "center_residual", "tunnel_vision",
)


def _column(name: str) -> int:
    """Return a canonical feature column index, failing loudly on schema drift."""
    try:
        return FEATURE_NAMES.index(name)
    except ValueError as exc:
        raise RuntimeError(f"Required canonical feature {name!r} is missing.") from exc


def parse_cell(value: str) -> tuple[int, int]:
    """Parse a 3x3 grid cell specified as ROW,COL."""
    try:
        row_text, col_text = value.split(",", maxsplit=1)
        row, col = int(row_text), int(col_text)
    except (ValueError, AttributeError) as exc:
        raise argparse.ArgumentTypeError("Cell must be ROW,COL, e.g. 0,2.") from exc
    if row not in range(3) or col not in range(3):
        raise argparse.ArgumentTypeError("Cell row and column must each be 0, 1, or 2.")
    return row, col


def _relative_magnitude(features: np.ndarray, row: int, col: int) -> np.ndarray:
    global_u = features[:, _column("flow_u")]
    global_v = features[:, _column("flow_v")]
    cell_u = features[:, _column(f"grid_{row}_{col}_u")]
    cell_v = features[:, _column(f"grid_{row}_{col}_v")]
    return np.hypot(cell_u - global_u, cell_v - global_v)


def derive_signals(
    features: np.ndarray,
    *,
    epsilon: float = 1e-5,
    edge_cells: Sequence[tuple[int, int]] = ((0, 0), (0, 2)),
    center_cell: tuple[int, int] = (1, 1),
) -> dict[str, np.ndarray]:
    """Calculate cell residuals, edge activity, center activity and their ratio.

    A cell residual is the magnitude of its mean flow vector minus the global
    mean flow vector. The global subtraction removes common translation only;
    rotation, zoom, parallax and flow-estimation errors can remain.
    """
    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 2 or features.shape[1] != len(FEATURE_NAMES):
        raise ValueError(
            f"Expected a 2D feature array with {len(FEATURE_NAMES)} columns; "
            f"got shape {features.shape}."
        )
    if not np.isfinite(features).all():
        raise ValueError("Feature array contains non-finite values.")
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be a finite number greater than zero.")
    if not edge_cells:
        raise ValueError("At least one edge cell is required.")

    for row, col in [*edge_cells, center_cell]:
        if row not in range(3) or col not in range(3):
            raise ValueError(f"Invalid grid cell {(row, col)}; row and col must be 0, 1, or 2.")

    signals: dict[str, np.ndarray] = {}
    residuals: dict[tuple[int, int], np.ndarray] = {}
    for row in range(3):
        for col in range(3):
            value = _relative_magnitude(features, row, col)
            residuals[(row, col)] = value
            signals[f"residual_{row}{col}"] = value

    signals["edge_left_residual"] = residuals[(0, 0)]
    signals["edge_right_residual"] = residuals[(0, 2)]
    signals["edge_residual"] = np.mean(
        np.stack([residuals[cell] for cell in edge_cells], axis=0), axis=0
    )
    signals["center_residual"] = residuals[center_cell]
    signals["tunnel_vision"] = signals["edge_residual"] / (
        signals["center_residual"] + epsilon
    )
    signals["global_flow_magnitude"] = np.hypot(
        features[:, _column("flow_u")], features[:, _column("flow_v")]
    )
    return signals


def apply_savgol(values: np.ndarray, window_length: int, polyorder: int) -> np.ndarray:
    """Return a Savitzky-Golay-smoothed signal without modifying its input."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("Signal must be one-dimensional.")
    if window_length < 3 or window_length % 2 == 0:
        raise ValueError("Savitzky-Golay window length must be an odd integer >= 3.")
    if polyorder < 0 or polyorder >= window_length:
        raise ValueError("Polynomial degree must be >= 0 and smaller than window length.")
    if window_length > len(values):
        raise ValueError(
            f"Savitzky-Golay window length ({window_length}) exceeds "
            f"number of samples ({len(values)}); use --no-smoothing or a smaller window."
        )
    try:
        from scipy.signal import savgol_filter
    except ImportError as exc:
        raise ImportError(
            "Savitzky-Golay filtering requires SciPy. Install with: "
            "python -m pip install -e '.[plot]'"
        ) from exc
    return savgol_filter(values, window_length=window_length, polyorder=polyorder)


def _annotation_label(score: int) -> str:
    suffix = " (P)" if score == 3 else " (U)" if score in (0, 1, 2) else ""
    return f"MTB {score}{suffix}"


def plot_signals(
    timestamps: np.ndarray,
    signals: dict[str, np.ndarray],
    annotations: Sequence[object],
    *,
    video_name: str,
    output_path: Path,
    smooth: bool,
    window_length: int,
    polyorder: int,
    show: bool,
) -> Path:
    """Write stacked per-signal plots with annotation shading."""
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        from matplotlib.ticker import FuncFormatter
    except ImportError as exc:
        raise ImportError(
            "Plotting requires matplotlib. Install with: "
            "python -m pip install -e '.[plot]'"
        ) from exc

    smoothed = {
        name: apply_savgol(values, window_length, polyorder)
        for name, values in signals.items()
    } if smooth else {}

    figure, axes = plt.subplots(
        len(signals), 1, figsize=(16, max(3.0, 2.7 * len(signals))),
        sharex=True, squeeze=False,
    )
    axes = axes[:, 0]
    for ax, (name, values) in zip(axes, signals.items()):
        for annotation in annotations:
            ax.axvspan(
                annotation.start, annotation.end,
                facecolor=ANNOTATION_COLORS[annotation.score],
                alpha=0.30, edgecolor=ANNOTATION_COLORS[annotation.score],
                linewidth=0.5, zorder=0,
            )
        ax.plot(timestamps, values, color="#222222", linewidth=1.0, label="raw", zorder=3)
        if smooth:
            ax.plot(
                timestamps, smoothed[name], color="#0072B2", linewidth=1.6,
                label=f"Savitzky–Golay ({window_length}, {polyorder})", zorder=4,
            )
        ax.set_ylabel(name.replace("_", "\n"), rotation=0, ha="right", va="center", labelpad=30)
        ax.grid(True, color="#777777", alpha=0.23, linewidth=0.5, zorder=1)
        ax.legend(loc="upper right", fontsize="small")
    axes[-1].set_xlabel("Video time (min:sec)")
    axes[-1].xaxis.set_major_formatter(
        FuncFormatter(lambda value, _position: f"{int(value // 60):02d}:{int(value % 60):02d}")
    )
    axes[0].set_title(f"Motion engineering experiments — {video_name}", loc="left", fontweight="bold")
    axes[-1].set_xlim(0, max(float(timestamps[-1]), max((a.end for a in annotations), default=0.0)))

    from matplotlib.patches import Patch
    handles = [
        Patch(facecolor=ANNOTATION_COLORS[score], edgecolor=ANNOTATION_COLORS[score],
              alpha=0.75, label=_annotation_label(score))
        for score in range(4)
    ]
    figure.legend(handles=handles, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.0))
    figure.tight_layout(rect=(0, 0, 1, 0.985))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)

    if show:
        html_path = output_path.with_suffix(".html")
        save_interactive_html(
            timestamps, signals, annotations, video_name=video_name,
            output_path=html_path, smooth=smooth, smoothed=smoothed,
            window_length=window_length, polyorder=polyorder,
        )
        import webbrowser
        webbrowser.open(html_path.resolve().as_uri())
        print(f"Opened interactive plot: {html_path.resolve()}")
    return output_path


def save_interactive_html(
    timestamps: np.ndarray,
    signals: dict[str, np.ndarray],
    annotations: Sequence[object],
    *,
    video_name: str,
    output_path: Path,
    smooth: bool,
    smoothed: dict[str, np.ndarray],
    window_length: int,
    polyorder: int,
) -> Path:
    """Write zoomable Plotly subplots with synchronized time axes."""
    try:
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go
    except ImportError as exc:
        raise ImportError(
            "Interactive plotting requires Plotly. Install with: "
            "python -m pip install -e '.[plot]'"
        ) from exc

    names = list(signals)
    figure = make_subplots(
        rows=len(names), cols=1, shared_xaxes=True, vertical_spacing=0.025,
        subplot_titles=names,
    )
    for row, name in enumerate(names, start=1):
        for annotation in annotations:
            figure.add_vrect(
                x0=annotation.start, x1=annotation.end,
                fillcolor=ANNOTATION_COLORS[annotation.score],
                opacity=0.25, line_width=0, layer="below", row=row, col=1,
            )
        figure.add_trace(
            go.Scatter(
                x=timestamps, y=signals[name], mode="lines", name=f"{name} raw",
                line={"color": "#222222", "width": 1.2},
                hovertemplate="Time: %{x:.2f}s<br>Raw: %{y:.6g}<extra></extra>",
            ), row=row, col=1,
        )
        if smooth:
            figure.add_trace(
                go.Scatter(
                    x=timestamps, y=smoothed[name], mode="lines",
                    name=f"{name} smoothed",
                    line={"color": "#0072B2", "width": 1.6},
                    hovertemplate="Time: %{x:.2f}s<br>Smoothed: %{y:.6g}<extra></extra>",
                ), row=row, col=1,
            )
    for score, color in ANNOTATION_COLORS.items():
        figure.add_trace(
            go.Scatter(
                x=[None], y=[None], mode="markers",
                marker={"size": 9, "color": color, "symbol": "square"},
                name=_annotation_label(score), hoverinfo="skip",
            ), row=1, col=1,
        )
    figure.update_layout(
        title={"text": f"Motion engineering experiments — {video_name}", "x": 0.01},
        template="plotly_white", hovermode="x unified",
        height=max(450, 260 * len(names)),
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0},
        margin={"l": 100, "r": 30, "t": 120, "b": 65},
    )
    figure.update_xaxes(title_text="Video time (seconds)", showgrid=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(output_path, include_plotlyjs="cdn", full_html=True)
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot engineered optical-flow residual signals against MTB annotations."
    )
    parser.add_argument("annotations", type=Path, nargs="?", help="Training annotation TXT file.")
    parser.add_argument(
        "--config", type=Path, default=Path("config/train.yaml"),
        help="Training YAML; canonical sampling, ROI, and feature-cache settings are used.",
    )
    parser.add_argument(
        "--signals", default=",".join(DEFAULT_SIGNALS),
        help="Comma-separated signal names, or 'all'. Default: the five main experiment signals.",
    )
    parser.add_argument("--list-signals", action="store_true", help="List available engineered signals and exit.")
    parser.add_argument("--edge-cells", nargs="+", type=parse_cell, default=[(0, 0), (0, 2)],
                        metavar="ROW,COL", help="Cells averaged for edge_residual and tunnel_vision; default: 0,0 0,2.")
    parser.add_argument("--center-cell", type=parse_cell, default=(1, 1), metavar="ROW,COL",
                        help="Cell used as center_residual denominator; default: 1,1.")
    parser.add_argument("--epsilon", type=float, default=1e-5,
                        help="Positive denominator stabilizer for tunnel_vision (default: 1e-5).")
    parser.add_argument("--no-smoothing", action="store_true", help="Plot raw signals only.")
    parser.add_argument("--savgol-window-length", type=int, default=9,
                        help="Odd smoothing window in samples (default: 9).")
    parser.add_argument("--savgol-polyorder", type=int, default=2,
                        help="Smoothing polynomial degree (default: 2).")
    parser.add_argument("--output", type=Path,
                        help="PNG output path (default: <annotation-stem>_motion_experiments.png).")
    parser.add_argument("--show", action="store_true",
                        help="Also open a zoomable Plotly HTML plot in the browser.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list_signals:
        print("Main signals:")
        for name in DEFAULT_SIGNALS:
            print(f"  {name}")
        print("Cell residuals:")
        for row in range(3):
            for col in range(3):
                print(f"  residual_{row}{col}")
        print("  global_flow_magnitude")
        print("Use --signals all to plot every signal.")
        return 0

    if args.annotations is None:
        parser.error("annotations is required unless --list-signals is used")
    if not args.annotations.is_file():
        raise FileNotFoundError(f"Annotation file not found: {args.annotations}")
    if not np.isfinite(args.epsilon) or args.epsilon <= 0:
        parser.error("--epsilon must be finite and greater than zero")

    video_path, annotations = load_training_annotations(args.annotations)
    if not annotations:
        raise ValueError(f"Annotation file contains no training intervals: {args.annotations}")

    settings = load_config(args.config)
    (
        _manifest, sample_fps, processing_width, _window_duration, _window_stride,
        roi_left, roi_right, roi_bottom, roi_top,
    ) = _data_values(settings)
    paths = settings.get("paths", {})
    cache_dir = Path(paths.get("cache_dir", "data/cache")).expanduser()
    video_info = get_video_info(video_path)
    clip = Clip(
        video_name=video_path.stem, start=0.0,
        end=min(float(annotations[-1].end), float(video_info.duration)),
    )
    sequence = build_feature_sequence(
        video_path, clip, sample_fps=sample_fps,
        processing_width=processing_width, cache_dir=cache_dir,
        roi_left=roi_left, roi_right=roi_right,
        roi_bottom=roi_bottom, roi_top=roi_top,
    )
    all_signals = derive_signals(
        sequence.features, epsilon=args.epsilon,
        edge_cells=args.edge_cells, center_cell=args.center_cell,
    )

    if args.signals.strip().lower() == "all":
        selected_names = list(all_signals)
    else:
        selected_names = [item.strip() for item in args.signals.split(",") if item.strip()]
        if not selected_names:
            parser.error("--signals must name at least one signal, or be 'all'")
        unknown = [name for name in selected_names if name not in all_signals]
        if unknown:
            parser.error(f"Unknown signal(s): {', '.join(unknown)}. Use --list-signals.")
        if len(set(selected_names)) != len(selected_names):
            parser.error("--signals contains duplicate names")
    signals = {name: all_signals[name] for name in selected_names}

    output_path = args.output or args.annotations.with_name(
        f"{args.annotations.stem}_motion_experiments.png"
    )
    saved_path = plot_signals(
        sequence.timestamps, signals, annotations,
        video_name=video_path.name, output_path=output_path,
        smooth=not args.no_smoothing,
        window_length=args.savgol_window_length, polyorder=args.savgol_polyorder,
        show=args.show,
    )
    print(f"Saved motion experiment plot: {saved_path.resolve()}")
    print(f"Signals plotted: {', '.join(selected_names)}")
    print(f"Edge cells averaged: {args.edge_cells}; center cell: {args.center_cell}")
    print(f"Tunnel Vision epsilon: {args.epsilon:g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
