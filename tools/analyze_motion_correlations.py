"""Explore Pearson and Spearman correlations between engineered motion signals.

This is an exploratory utility, separate from the production pipeline. It reuses
an exact feature-cache entry by default and refuses to trigger optical-flow
extraction unless --allow-extraction is explicitly supplied.

Example:
    python tools/analyze_motion_correlations.py path/to/video.mp4
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.stats import spearmanr

from autocut.config_loader import load_config
from autocut.data_models import Clip
from autocut.dataset.feature_cache import load_feature_sequence
from autocut.dataset.sequence import build_feature_sequence
from autocut.pipeline.training import _data_values
from autocut.video.probe import get_video_info
from plot_motion_experiments import derive_signals


DEFAULT_SIGNALS = (
    "vertical_flow_gradient",
    "lateral_flow_asymmetry",
    "expansion_index",
    "roll_index",
    "directional_turbulence",
    "global_flow_magnitude",
    "center_residual",
    "tunnel_vision",
)


def correlation_matrices(
    signals: dict[str, np.ndarray],
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Return pairwise Pearson and Spearman matrices using finite paired samples."""
    names = list(signals)
    count = len(names)
    pearson = np.eye(count, dtype=np.float64)
    spearman = np.eye(count, dtype=np.float64)

    for i in range(count):
        x = np.asarray(signals[names[i]], dtype=np.float64)
        for j in range(i):
            y = np.asarray(signals[names[j]], dtype=np.float64)
            valid = np.isfinite(x) & np.isfinite(y)
            xv, yv = x[valid], y[valid]
            if len(xv) < 3 or np.std(xv) == 0 or np.std(yv) == 0:
                p, s = np.nan, np.nan
            else:
                p = float(np.corrcoef(xv, yv)[0, 1])
                s = float(spearmanr(xv, yv).statistic)
            pearson[i, j] = pearson[j, i] = p
            spearman[i, j] = spearman[j, i] = s
    return names, pearson, spearman


def save_heatmaps(
    names: list[str],
    pearson: np.ndarray,
    spearman: np.ndarray,
    output_path: Path,
    *,
    show: bool,
) -> None:
    try:
        import matplotlib
        if not show:
            matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "Plotting requires matplotlib. Install with: python -m pip install -e '.[plot]'"
        ) from exc

    figure, axes = plt.subplots(
        1, 2,
        figsize=(max(13, len(names) * 1.25), max(6, len(names) * 0.8)),
        constrained_layout=True,
    )
    for ax, matrix, title in (
        (axes[0], pearson, "Pearson (linear association)"),
        (axes[1], spearman, "Spearman (monotonic association)"),
    ):
        image = ax.imshow(matrix, vmin=-1, vmax=1, cmap="coolwarm")
        ax.set_title(title)
        ax.set_xticks(range(len(names)), labels=names, rotation=60, ha="right")
        ax.set_yticks(range(len(names)), labels=names)
        for row in range(len(names)):
            for col in range(len(names)):
                value = matrix[row, col]
                label = "n/a" if not np.isfinite(value) else f"{value:.2f}"
                ax.text(col, row, label, ha="center", va="center", fontsize=8)
    figure.colorbar(image, ax=axes, shrink=0.8, pad=0.04, label="Correlation")
    figure.suptitle("Motion feature correlations", x=0.02, ha="left", fontweight="bold")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(figure)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Calculate Pearson and Spearman correlations for engineered motion signals."
    )
    parser.add_argument("video", type=Path, help="Source video to analyze (no annotations are used).")
    parser.add_argument("--config", type=Path, default=Path("config/train.yaml"),
                        help="Config containing canonical extraction/cache settings.")
    parser.add_argument("--signals", default=",".join(DEFAULT_SIGNALS),
                        help="Comma-separated signal names, or 'all'.")
    parser.add_argument("--output", type=Path,
                        help="Output PNG path (default: data/plots/<video-stem>_correlations.png).")
    parser.add_argument("--allow-extraction", action="store_true",
                        help="Allow expensive optical-flow extraction on an exact cache miss.")
    parser.add_argument("--show", action="store_true", help="Open the plot interactively.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    video_path = args.video.expanduser().resolve()
    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")

    settings = load_config(args.config)
    (_, sample_fps, processing_width, _, _, roi_left, roi_right,
     roi_bottom, roi_top) = _data_values(settings)
    cache_dir = Path(settings.get("paths", {}).get("cache_dir", "data/cache")).expanduser()
    video_info = get_video_info(video_path)
    clip = Clip(video_name=video_path.stem, start=0.0, end=float(video_info.duration))

    sequence = load_feature_sequence(
        video_path, cache_dir, clip,
        sample_fps=sample_fps, processing_width=processing_width,
        roi_left=roi_left, roi_right=roi_right,
        roi_bottom=roi_bottom, roi_top=roi_top,
    )
    if sequence is None:
        if not args.allow_extraction:
            raise RuntimeError(
                "No exact full-video feature-cache entry matches this video, sampling, width, and ROI. "
                "No extraction was started. Add --allow-extraction to compute and cache the features."
            )
        print("Cache miss: extracting optical-flow features because --allow-extraction was set.")
        sequence = build_feature_sequence(
            video_path, clip, sample_fps=sample_fps,
            processing_width=processing_width, cache_dir=cache_dir,
            roi_left=roi_left, roi_right=roi_right,
            roi_bottom=roi_bottom, roi_top=roi_top,
        )
    else:
        print(f"Reusing exact cached feature sequence ({len(sequence.features)} samples).")

    available = derive_signals(sequence.features)
    if args.signals.strip().lower() == "all":
        selected = available
    else:
        requested = [part.strip() for part in args.signals.split(",") if part.strip()]
        if not requested:
            raise ValueError("--signals must contain at least one signal name or 'all'.")
        unknown = sorted(set(requested) - set(available))
        if unknown:
            raise ValueError(f"Unknown signals: {', '.join(unknown)}. Available: {', '.join(available)}")
        if len(set(requested)) != len(requested):
            raise ValueError("--signals contains duplicate names.")
        selected = {name: available[name] for name in requested}

    names, pearson, spearman = correlation_matrices(selected)
    output = args.output or Path("data/plots") / f"{video_path.stem}_correlations.png"
    save_heatmaps(names, pearson, spearman, output, show=args.show)
    pearson_path = output.with_name(output.stem + "_pearson.csv")
    spearman_path = output.with_name(output.stem + "_spearman.csv")
    np.savetxt(pearson_path, pearson, delimiter=",",
               header=",".join(names), comments="", fmt="%.6f")
    np.savetxt(spearman_path, spearman, delimiter=",",
               header=",".join(names), comments="", fmt="%.6f")
    print(f"Saved correlation heatmaps: {output.resolve()}")
    print(f"Saved matrices: {pearson_path.resolve()} and {spearman_path.resolve()}")
    print(f"Samples: {len(sequence.features)}; signals: {', '.join(names)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
