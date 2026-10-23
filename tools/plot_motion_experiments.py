"""Plot engineered optical-flow signals against MTB training annotations.

Examples:
    python tools/plot_motion_experiments.py annotations.txt --show
    python tools/plot_motion_experiments.py annotations.txt --signals edge_residual,tunnel_vision --show
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

# Keep the default view focused on the original hypothesis. More detailed
# cell-level residuals are available via --signals all or --list-signals.
DEFAULT_SIGNALS = (
    "edge_left_residual",
    "edge_right_residual",
    "edge_residual",
    "center_residual",
    "tunnel_vision",
)


def _column(name: str) -> int:
    """Return a canonical feature column index, failing loudly on schema drift."""
    try:
        return FEATURE_NAMES.index(name)
    except ValueError as exc:
        raise RuntimeError(f"Required canonical feature {name!r} is missing.") from exc


def derive_signals(features: np.ndarray, *, epsilon: float = 1e-5) -> dict[str, np.ndarray]:
    """Calculate residual-motion signals from canonical feature rows.

    Residual vectors subtract the global mean flow vector (flow_u, flow_v)
    from each cell's mean flow vector. This removes common translation, but
    does not guarantee removal of camera rotation, zoom, parallax, etc.
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

    global_u = features[:, _column("flow_u")]
    global_v = features[:, _column("flow_v")]
    signals: dict[str, np.ndarray] = {}

    residuals: dict[str, np.ndarray] = {}
    for row in range(3):
        for col in range(3):
            prefix = f"grid_{row}_{col}"
            rel_u = features[:, _column(f"{prefix}_u")] - global_u
            rel_v = features[:, _column(f"{prefix}_v")] - global_v
            magnitude = np.hypot(rel_u, rel_v)
            residuals[f"{row}{col}"] = magnitude
            signals[f"residual_{row}{col}"] = magnitude

    signals["edge_left_residual"] = residuals["00"]
    signals["edge_right_residual"] = residuals["02"]
    signals["edge_residual"] = (
        signals["edge_left_residual"] + signals["edge_right_residual"]
    ) / 2.0
    signals["center_residual"] = residuals["11"]
    signals["tunnel_vision"] = signals["edge_residual"] / (
        signals["center_residual"] + epsilon
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
            f"the number of samples ({len(values)}); use --no-smoothing or a smaller window."
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
        save_interactive_html(
            timestamps, signals, annotations, video_name=video_name,
            output_path=output_path.with_suffix(".html"),
            smooth=smooth, smoothed=smoothed,
            window_length=window_length, polyorder=polyorder,
        )
        import webbrowser
        webbrowser.open(output_path.with_suffix(".html").resolve().as_uri())
        print(f"Opened interactive plot: {output_path.with_suffix('.html').resolve()}")
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
    parser.add_argument("--epsilon", type=float, default=1e-5, help="Positive denominator stabilizer for tunnel_vision (default: 1e-5).")
    parser.add_argument("--no-smoothing", action="store_true", help="Plot raw signals only.")
    parser.add_argument("--savgol-window-length", type=int, default=9, help="Odd smoothing window in samples (default: 9).")
    parser.add_argument("--savgol-polyorder", type=int, default=2, help="Smoothing polynomial degree (default: 2).")
    parser.add_argument("--output", type=Path, help="PNG output path (default: <annotation-stem>_motion_experiments.png).")
    parser.add_argument("--show", action="store_true", help="Also open a zoomable Plotly HTML plot in the browser.")
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
        video_name=video_path.stem,
        start=0.0,
        end=min(float(annotations[-1].end), float(video_info.duration)),
    )
    sequence = build_feature_sequence(
        video_path, clip,
        sample_fps=sample_fps,
        processing_width=processing_width,
        cache_dir=cache_dir,
        roi_left=roi_left, roi_right=roi_right,
        roi_bottom=roi_bottom, roi_top=roi_top,
    )
    all_signals = derive_signals(sequence.features, epsilon=args.epsilon)

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
        video_name=video_path.name,
        output_path=output_path,
        smooth=not args.no_smoothing,
        window_length=args.savgol_window_length,
        polyorder=args.savgol_polyorder,
        show=args.show,
    )
    print(f"Saved motion experiment plot: {saved_path.resolve()}")
    print(f"Signals plotted: {', '.join(selected_names)}")
    print(f"Tunnel Vision epsilon: {args.epsilon:g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
