"""Plot model window probabilities over the human annotations.

Example:
    python tools/plot_window_probabilities.py annotations.txt
    python tools/plot_window_probabilities.py annotations.txt --config config/autocut.yaml --show
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np

from autocut.annotations import load_training_annotations
from autocut.config_loader import load_config
from autocut.ml.inference import infer_windows
from autocut.pipeline.run import _extract_windows, _run_config_sections
from autocut.video.probe import get_video_info


# Consistent annotation score colours: 0 = irrelevant, 1 = ordinary,
# 2 = borderline, 3 = interesting.
ANNOTATION_COLORS = {
    0: "#bdbdbd",
    1: "#80b1d3",
    2: "#fdb462",
    3: "#8dd3c7",
}


def annotation_color(score: int) -> str:
    """Return the plot colour for a human annotation score (0..3)."""
    try:
        return ANNOTATION_COLORS[score]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Annotation score must be one of 0, 1, 2, 3; got {score!r}.") from exc


def validate_plot_data(timestamps: np.ndarray, probabilities: np.ndarray) -> None:
    """Validate the arrays needed for a three-class probability plot."""
    if timestamps.ndim != 2 or timestamps.shape[1] != 2:
        raise ValueError("timestamps must have shape (N, 2).")
    if probabilities.shape != (len(timestamps), 3):
        raise ValueError("probabilities must have shape (N, 3), for p_0, p_1 and p_2.")
    if len(timestamps) == 0:
        raise ValueError("No windows were generated for this video.")
    if not np.isfinite(timestamps).all() or not np.isfinite(probabilities).all():
        raise ValueError("timestamps and probabilities must contain only finite values.")
    if np.any(timestamps[:, 1] <= timestamps[:, 0]):
        raise ValueError("Every window end must be after its start.")
    if np.any(probabilities < 0) or np.any(probabilities > 1):
        raise ValueError("Class probabilities must be between 0 and 1.")


def plot_probabilities(
    timestamps: np.ndarray,
    probabilities: np.ndarray,
    annotations: Sequence[object],
    *,
    video_name: str,
    output_path: str | Path,
    show: bool = False,
) -> Path:
    """Save a single-axis probability plot with background annotation shading."""
    timestamps = np.asarray(timestamps, dtype=np.float64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    validate_plot_data(timestamps, probabilities)

    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
        from matplotlib.ticker import FuncFormatter
    except ImportError as exc:
        raise ImportError(
            "Plotting requires matplotlib. Install it with: "
            "python -m pip install -e '.[plot]'"
        ) from exc

    centres = timestamps.mean(axis=1)
    duration = max(float(timestamps[:, 1].max()), max(a.end for a in annotations))
    
    # 1. Single plot setup instead of two subplots
    figure, ax = plt.subplots(figsize=(16, 6))

    # Draw background annotation shading (alpha=0.25 keeps it behind lines nicely)
    for annotation in annotations:
        ax.axvspan(
            annotation.start,
            annotation.end,
            color=annotation_color(annotation.score),
            alpha=0.25,
            linewidth=0,
            zorder=1,
        )
        # 3. Improved remarks: placed near the top with a white bbox for readability
        if annotation.remarks:
            midpoint = (annotation.start + annotation.end) / 2
            ax.text(
                midpoint,
                0.88,
                annotation.remarks,
                ha="center",
                va="center",
                fontsize=9,
                fontweight="medium",
                rotation=0,
                clip_on=True,
                zorder=3,
                bbox=dict(
                    facecolor="white",
                    edgecolor="none",
                    alpha=0.85,
                    boxstyle="round,pad=0.3",
                ),
            )

    # 2. High-contrast line colors and explicit zorder so lines sit cleanly on top
    # (Using distinct, saturated colors that pop against pastel background bands)
    line_colors = ["#1f77b4", "#d62728", "#2ca02c"]  # Blue, Red, Green
    for class_id in range(3):
        ax.plot(
            centres,
            probabilities[:, class_id],
            label=f"p_{class_id}",
            color=line_colors[class_id],
            linewidth=2.0,
            zorder=2,
        )

    ax.set_xlim(0, duration)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Predicted probability", fontsize=10)
    ax.set_xlabel("Video time (min:sec)", fontsize=10)
    ax.set_title(f"Window probabilities and annotations — {video_name}", loc="left", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3, zorder=0)
    
    # Combined legend for both probability lines and annotation scores
    line_handles, line_labels = ax.get_legend_handles_labels()
    annotation_handles = [
        Patch(facecolor=ANNOTATION_COLORS[score], alpha=0.5, label=f"MTB score {score}")
        for score in range(4)
    ]
    ax.legend(
        handles=line_handles + annotation_handles,
        loc="upper right",
        ncol=len(line_handles) + len(annotation_handles),
        frameon=True,
        facecolor="white",
        framealpha=0.9,
    )

    ax.xaxis.set_major_formatter(
        FuncFormatter(lambda value, _position: f"{int(value // 60):02d}:{int(value % 60):02d}")
    )

    figure.tight_layout()
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=160, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(figure)
    return destination

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plot p_0, p_1 and p_2 for each video window, with human MTB "
            "annotation intervals shown as a coloured strip."
        )
    )
    parser.add_argument("annotations", type=Path, help="Training annotations file (Start End MTB [Remarks]).")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/autocut.yaml"),
        help="AutoCut YAML config; its checkpoint and feature settings are used.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output image path (default: <annotations-stem>_probabilities.png).",
    )
    parser.add_argument("--show", action="store_true", help="Also open the plot in an interactive window.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.annotations.is_file():
        raise FileNotFoundError(f"Annotation file not found: {args.annotations}")

    video_path, annotations = load_training_annotations(args.annotations)
    settings, _candidate_config, _audio_config = _run_config_sections(load_config(args.config))
    if not settings.model.checkpoint.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {settings.model.checkpoint}")

    video_info = get_video_info(video_path)
    windows = _extract_windows(video_path, 0.0, video_info.duration, settings)
    if windows is None:
        raise ValueError(
            f"Video is too short to produce windows of "
            f"{settings.features.window_duration:g} seconds: {video_path}"
        )

    result = infer_windows(
        settings.model.checkpoint,
        windows,
        batch_size=settings.model.inference_batch_size,
    )
    output_path = args.output or args.annotations.with_name(
        f"{args.annotations.stem}_probabilities.png"
    )
    saved_path = plot_probabilities(
        result.timestamps,
        result.probabilities,
        annotations,
        video_name=video_path.name,
        output_path=output_path,
        show=args.show,
    )
    print(f"Saved probability plot: {saved_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
