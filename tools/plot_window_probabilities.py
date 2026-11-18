"""Plot LightGBM LambdaRank window scores over human MTB annotations.

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
from autocut.data_models import Clip
from autocut.dataset.cache_paths import feature_cache_dir
from autocut.dataset.normalization import normalize_per_video_features
from autocut.dataset.sequence import build_feature_sequence
from autocut.dataset.windows import build_lightgbm_windows_from_features
from autocut.ml.lightgbm_checkpoint import load_lightgbm_checkpoint
from autocut.ml.lightgbm_model import LightGBMRanker
from autocut.motion.engineered import engineer_lightgbm_sequence
from autocut.pipeline.lightgbm_training import _data_config
from autocut.video.probe import get_video_info


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


def validate_plot_data(timestamps: np.ndarray, scores: np.ndarray) -> None:
    """Validate window timestamps and finite raw LambdaRank scores."""
    timestamps = np.asarray(timestamps)
    scores = np.asarray(scores)
    if timestamps.ndim != 2 or timestamps.shape[1] != 2:
        raise ValueError("timestamps must have shape (N, 2).")
    if scores.shape != (len(timestamps),):
        raise ValueError("scores must have shape (N,).")
    if len(timestamps) == 0:
        raise ValueError("No windows were generated for this video.")
    if not np.isfinite(timestamps).all() or not np.isfinite(scores).all():
        raise ValueError("timestamps and scores must contain only finite values.")
    if np.any(timestamps[:, 1] <= timestamps[:, 0]):
        raise ValueError("Every window end must be after its start.")


def plot_scores(
    timestamps: np.ndarray,
    scores: np.ndarray,
    annotations: Sequence[object],
    *,
    video_name: str,
    output_path: str | Path,
    show: bool = False,
) -> Path:
    """Save raw LightGBM ranking scores with background annotation shading."""
    timestamps = np.asarray(timestamps, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    validate_plot_data(timestamps, scores)

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
    duration = max(float(timestamps[:, 1].max()), max((a.end for a in annotations), default=0.0))
    figure, ax = plt.subplots(figsize=(16, 6))
    for annotation in annotations:
        ax.axvspan(
            annotation.start, annotation.end,
            color=annotation_color(annotation.score), alpha=0.25,
            linewidth=0, zorder=1,
        )
        if getattr(annotation, "remarks", None):
            midpoint = (annotation.start + annotation.end) / 2
            ax.text(
                midpoint, 0.88, annotation.remarks,
                transform=ax.get_xaxis_transform(),
                ha="center", va="center", fontsize=9, fontweight="medium",
                rotation=0, clip_on=True, zorder=3,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.85,
                          boxstyle="round,pad=0.3"),
            )

    ax.plot(
        centres, scores, label="LightGBM raw ranking score",
        color="#2ca02c", linewidth=2.0, zorder=2,
    )
    ax.set_xlim(0, duration)
    score_min, score_max = float(scores.min()), float(scores.max())
    if score_min == score_max:
        padding = max(abs(score_min) * 0.05, 0.1)
    else:
        padding = (score_max - score_min) * 0.08
    ax.set_ylim(score_min - padding, score_max + padding)
    ax.set_ylabel("Raw LambdaRank score", fontsize=10)
    ax.set_xlabel("Video time (min:sec)", fontsize=10)
    ax.set_title(f"Window ranking scores and annotations — {video_name}",
                 loc="left", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3, zorder=0)
    line_handles, _ = ax.get_legend_handles_labels()
    annotation_handles = [
        Patch(facecolor=ANNOTATION_COLORS[score], alpha=0.5,
              label=f"MTB score {score}")
        for score in range(4)
    ]
    ax.legend(
        handles=line_handles + annotation_handles,
        loc="upper right", ncol=len(line_handles) + len(annotation_handles),
        frameon=True, facecolor="white", framealpha=0.9,
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


# Kept as a source-level alias so external plotting notebooks do not break.
plot_probabilities = plot_scores


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot LightGBM raw ranking scores for video windows with MTB annotations."
    )
    parser.add_argument("annotations", type=Path, help="Training annotation TXT file.")
    parser.add_argument(
        "--config", type=Path, default=Path("config/autocut.yaml"),
        help="AutoCut YAML config; its data and checkpoint settings are used.",
    )
    parser.add_argument("--checkpoint", type=Path, help="Override the LightGBM ranker checkpoint.")
    parser.add_argument(
        "--output", type=Path,
        help="Output image path (default: <annotations-stem>_ranking_scores.png).",
    )
    parser.add_argument("--show", action="store_true", help="Also open the plot interactively.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.annotations.is_file():
        raise FileNotFoundError(f"Annotation file not found: {args.annotations}")

    video_path, annotations = load_training_annotations(args.annotations)
    settings = load_config(args.config)
    raw_settings = settings.to_dict() if hasattr(settings, "to_dict") else settings
    experiment = raw_settings.get("experiment", {})
    model_settings = raw_settings.get("model", {})
    checkpoint_path = args.checkpoint or experiment.get(
        "lightgbm_model_path", model_settings.get("checkpoint", "models/lightgbm/ranker.json")
    )
    payload = load_lightgbm_checkpoint(checkpoint_path)
    saved = payload.get("data_config", {})
    if "data" in raw_settings:
        data = _data_config(raw_settings)
    else:
        video = raw_settings.get("video", {})
        features = raw_settings.get("features", {})
        data = {
            "sampling_fps": float(saved.get("sampling_fps", video.get("sampling_fps", 2.0))),
            "processing_width": int(saved.get("processing_width", video.get("processing_width", 480))),
            "window": float(saved.get("window", saved.get("window_duration", features.get("window_duration", 4.0)))),
            "stride": float(saved.get("stride", saved.get("window_stride", features.get("window_stride", 2.0)))),
            "roi_left": saved.get("roi_left", video.get("roi_left", 0.15)),
            "roi_right": saved.get("roi_right", video.get("roi_right", 0.85)),
            "roi_bottom": saved.get("roi_bottom", video.get("roi_bottom", 0.50)),
            "roi_top": saved.get("roi_top", video.get("roi_top", 0.10)),
        }
    if payload.get("model_class") != "LightGBMRanker":
        raise ValueError("The selected checkpoint is not a LightGBMRanker.")
    from autocut.dataset.windows import LIGHTGBM_TABULAR_FEATURE_NAMES
    if payload.get("feature_names") != list(LIGHTGBM_TABULAR_FEATURE_NAMES):
        raise ValueError("Checkpoint feature schema does not match current LightGBM windows.")

    info = get_video_info(video_path)
    clip = Clip(video_name=video_path.stem, start=0.0, end=min(float(annotations[-1].end), float(info.duration)))
    sequence = build_feature_sequence(
        video_path, clip,
        sample_fps=data["sampling_fps"],
        processing_width=data["processing_width"],
        cache_dir=feature_cache_dir(raw_settings),
        roi_left=data["roi_left"], roi_right=data["roi_right"],
        roi_bottom=data["roi_bottom"], roi_top=data["roi_top"],
        window_duration=data["window"], window_stride=data["stride"],
    )
    signals = engineer_lightgbm_sequence(sequence.features)
    windows = build_lightgbm_windows_from_features(
        signals, sequence.timestamps, window=data["window"], stride=data["stride"]
    )
    if len(windows.X) == 0:
        raise ValueError("Video is too short to produce LightGBM windows.")
    normalized = normalize_per_video_features(windows.X)
    ranker = LightGBMRanker(
        payload["_booster"], feature_names=tuple(payload["feature_names"])
    )
    scores = ranker.predict(normalized)

    output_path = args.output or args.annotations.with_name(
        f"{args.annotations.stem}_ranking_scores.png"
    )
    saved_path = plot_scores(
        windows.timestamps, scores, annotations,
        video_name=video_path.name, output_path=output_path, show=args.show,
    )
    print(f"Saved ranking-score plot: {saved_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
