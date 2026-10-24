"""Synthetic sanity check for the four engineered-signal PU training path.

Positive events contain a known bias. Only a fraction of their windows are
labelled P; the remaining positive windows are hidden inside U alongside
unbiased background windows. This tests nnPU with a genuine P/U setup.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import tempfile

import numpy as np
import torch

from autocut.annotations import load_training_annotations
from autocut.data_models import FeatureWindows
from autocut.dataset.targets import build_targets
from autocut.ml.training import TrainingConfig, train_single_fold


def _timecode(seconds: int) -> str:
    hours, remainder = divmod(int(seconds), 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _write_annotations(path: Path, video_path: Path, duration: int, event_starts: list[int], event_duration: int) -> None:
    """Write contiguous score-0/score-3 annotations across the full duration."""
    boundaries = {0, duration}
    for start in event_starts:
        boundaries.add(start)
        boundaries.add(start + event_duration)
    ordered = sorted(boundaries)
    lines = [str(video_path), "", "Start End MTB Remarks"]
    for start, end in zip(ordered[:-1], ordered[1:]):
        score = 3 if any(start >= event and end <= event + event_duration for event in event_starts) else 0
        remark = "synthetic_signal_bias" if score == 3 else "synthetic_unlabeled"
        lines.append(f"{_timecode(start)} {_timecode(end)} {score} {remark}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _make_split(
    *,
    seed: int,
    duration: int,
    fps: float,
    window: float,
    stride: float,
    noise_std: float,
    bias: float,
    event_duration: int,
    event_starts: list[int],
    workdir: Path,
    split_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    samples_float = duration * fps
    samples = int(round(samples_float))
    if not np.isclose(samples_float, samples):
        raise ValueError("duration * fps must be an integer.")
    rng = np.random.default_rng(seed)
    # Four independent, zero-mean noise signals. Bias exists only during events.
    signals = rng.normal(0.0, noise_std, size=(samples, 4)).astype(np.float32)
    for event_index, start in enumerate(event_starts):
        first = int(round(start * fps))
        last = int(round((start + event_duration) * fps))
        signals[first:last, event_index % signals.shape[1]] += bias

    annotation_path = workdir / f"{split_name}_annotations.txt"
    video_path = workdir / f"{split_name}_synthetic.mp4"
    # The annotation parser validates that its source path exists; no video is read.
    video_path.touch()
    _write_annotations(annotation_path, video_path, duration, event_starts, event_duration)
    _, annotations = load_training_annotations(annotation_path)

    window_samples_float = window * fps
    stride_samples_float = stride * fps
    window_samples = int(round(window_samples_float))
    stride_samples = int(round(stride_samples_float))
    if not np.isclose(window_samples_float, window_samples) or not np.isclose(stride_samples_float, stride_samples):
        raise ValueError("window and stride must be integer multiples of 1/fps.")
    starts = np.arange(0, samples - window_samples + 1, stride_samples, dtype=int)
    X_windows = np.stack([signals[i:i + window_samples] for i in starts]).astype(np.float32)
    timestamps = np.column_stack((starts / fps, starts / fps + window)).astype(np.float64)
    windows = FeatureWindows(X=X_windows, timestamps=timestamps)
    X, y = build_targets(windows, annotations)
    if len(X) == 0 or not np.any(y == 3) or not np.any(y == 0):
        raise ValueError(f"{split_name} split did not produce both positive and background windows.")
    return X, y


def _make_pu_split(
    X: np.ndarray,
    annotation_scores: np.ndarray,
    *,
    labelled_positive_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, int, int]:
    """Return shuffled P/U examples, training labels, hidden truth, and U prior."""
    rng = np.random.default_rng(seed)
    true_positive = annotation_scores == 3
    positive_indices = np.flatnonzero(true_positive)
    negative_indices = np.flatnonzero(~true_positive)
    rng.shuffle(positive_indices)
    labelled_count = int(round(len(positive_indices) * labelled_positive_fraction))
    labelled_count = min(max(labelled_count, 1), len(positive_indices) - 1)
    p_indices = positive_indices[:labelled_count]
    hidden_positive_indices = positive_indices[labelled_count:]
    # U contains hidden positives plus all unbiased background windows.
    u_indices = np.concatenate((hidden_positive_indices, negative_indices))
    u_prior = len(hidden_positive_indices) / len(u_indices)
    indices = np.concatenate((p_indices, u_indices))
    labels = np.concatenate((
        np.ones(len(p_indices), dtype=np.int64),
        np.zeros(len(u_indices), dtype=np.int64),
    ))
    truth = true_positive[indices]
    order = rng.permutation(len(indices))
    # The model sees only P/U labels. Ground truth is retained only for evaluation.
    return X[indices][order], np.where(labels[order] == 1, 3, 0), truth[order], float(u_prior), len(p_indices), len(u_indices)


def run(args: argparse.Namespace) -> int:
    event_duration = args.event_duration
    event_starts = list(range(args.first_event, args.duration - event_duration + 1,
                              event_duration + args.gap_after_event))
    if len(event_starts) < 2:
        raise ValueError("Duration must allow at least two synthetic positive events.")
    if any(start < 0 or start + event_duration > args.duration for start in event_starts):
        raise ValueError("Synthetic event lies outside the sequence.")
    if not 0 < args.labelled_positive_fraction < 1:
        raise ValueError("--labelled-positive-fraction must be strictly between 0 and 1.")

    with tempfile.TemporaryDirectory(prefix="autocut_synthetic_") as temporary:
        workdir = Path(args.output_dir).expanduser().resolve() if args.output_dir else Path(temporary).resolve()
        workdir.mkdir(parents=True, exist_ok=True)
        raw_train_X, raw_train_scores = _make_split(
            seed=args.seed, duration=args.duration, fps=args.fps,
            window=args.window, stride=args.stride, noise_std=args.noise_std,
            bias=args.bias, event_duration=event_duration, event_starts=event_starts,
            workdir=workdir, split_name="train",
        )
        raw_val_X, raw_val_scores = _make_split(
            seed=args.seed + 1, duration=args.duration, fps=args.fps,
            window=args.window, stride=args.stride, noise_std=args.noise_std,
            bias=args.bias, event_duration=event_duration, event_starts=event_starts,
            workdir=workdir, split_name="validation",
        )
        train_X, train_y, _, positive_prior, train_p_count, train_u_count = _make_pu_split(
            raw_train_X, raw_train_scores,
            labelled_positive_fraction=args.labelled_positive_fraction, seed=args.seed + 2,
        )
        val_X, val_y, val_truth, val_prior, val_p_count, val_u_count = _make_pu_split(
            raw_val_X, raw_val_scores,
            labelled_positive_fraction=args.labelled_positive_fraction, seed=args.seed + 3,
        )
        # The two independently generated splits should have the same designed prior.
        if not np.isclose(positive_prior, val_prior):
            raise ValueError(f"Train/validation U priors differ ({positive_prior:.4f} vs {val_prior:.4f}).")

        checkpoint_path = workdir / "synthetic_engineered_best.pt"
        config = TrainingConfig(
            epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
            hidden_dim=32, classifier_mid_dim=16, dropout=0.1,
            use_scaling=True, scaler_strategy=args.scaler,
            patience=max(5, args.epochs // 4), seed=args.seed,
            positive_prior=positive_prior,
        )
        model, scaler, summary = train_single_fold(
            train_X, train_y, val_X, val_y, config,
            checkpoint_path=checkpoint_path,
            data_config={"flow_fps": args.fps, "window": args.window, "stride": args.stride},
            fold_video_id="synthetic-train",
        )

        model.eval()
        val_tensor = torch.from_numpy(np.ascontiguousarray(scaler.transform(val_X), dtype=np.float32))
        with torch.inference_mode():
            probabilities = torch.sigmoid(model(val_tensor)[:, 0]).cpu().numpy()
        is_positive = val_truth
        is_negative = ~val_truth
        mean_p = float(probabilities[is_positive].mean())
        mean_n = float(probabilities[is_negative].mean())
        threshold = 0.5
        recall = float(np.mean(probabilities[is_positive] >= threshold))
        background_alarm_rate = float(np.mean(probabilities[is_negative] >= threshold))
        hidden_in_u = int(np.sum((val_y == 0) & val_truth))
        val_u_positive_prior = hidden_in_u / val_u_count

        print(f"Signals: 4; duration: {args.duration}s; fps: {args.fps:g}; samples/signal: {int(args.duration * args.fps)}")
        print(f"Window geometry: {args.window:g}s / stride {args.stride:g}s")
        print(f"Events: {len(event_starts)} x {event_duration}s; bias={args.bias:g}; noise std={args.noise_std:g}")
        print(f"Labelled fraction of positive windows: {args.labelled_positive_fraction:.0%}")
        print(f"Train windows: {len(train_X)} (P={train_p_count}, U={train_u_count})")
        print(f"Validation windows: {len(val_X)} (P={val_p_count}, U={val_u_count})")
        print(f"Known positive prior in U: {positive_prior:.3f}")
        print(f"Mean true-positive probability: {mean_p:.3f}")
        print(f"Mean background probability: {mean_n:.3f}")
        print(f"Mean separation (positive-background): {mean_p - mean_n:+.3f}")
        print(f"Recall at 0.5: {recall:.1%}; background alarm rate at 0.5: {background_alarm_rate:.1%}")
        print(f"Best validation nnPU risk: {summary['best_val_pu_risk']}")
        print(f"Checkpoint: {checkpoint_path}")
        print(f"Annotations: {workdir / 'validation_annotations.txt' if args.output_dir else '(temporary)'}")
        if args.output_dir:
            print(f"Artifacts retained in: {workdir}")
        else:
            print("Temporary artifacts will be removed on exit.")
        return 0 if mean_p > mean_n else 2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=int, default=900, help="Sequence duration in seconds (default: 900 = 15 min).")
    parser.add_argument("--fps", type=float, default=2.0, help="Signal sampling frequency (default: 2 Hz).")
    parser.add_argument("--window", type=float, default=4.0, help="Training window duration in seconds.")
    parser.add_argument("--stride", type=float, default=2.0, help="Distance between window starts in seconds.")
    parser.add_argument("--event-duration", type=int, default=30, help="Duration of each positive event in seconds.")
    parser.add_argument("--first-event", type=int, default=60, help="Time of the first event in seconds.")
    parser.add_argument("--gap-after-event", type=int, default=60, help="Quiet time after each event before the next event.")
    parser.add_argument("--noise-std", type=float, default=0.35, help="White-noise standard deviation.")
    parser.add_argument("--bias", type=float, default=2.0, help="Added bias during each event.")
    parser.add_argument("--labelled-positive-fraction", type=float, default=0.5, help="Fraction of true positive windows whose labels are revealed as P.")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--scaler", choices=("robust", "standard"), default="robust")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", help="Optional directory to retain generated annotations and checkpoint.")
    args = parser.parse_args()
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()
