import argparse
from pathlib import Path
import csv
import json
import numpy as np
import torch
import torch.nn as nn


# ==============================================================================
# MODEL ARCHITECTURE
# ==============================================================================
class MTBClassifier(nn.Module):
    def __init__(
        self,
        feature_dim,
        hidden_dim=128,
        classifier_mid_dim=64,
        num_classes=3,
        dropout=0.4,
    ):
        super().__init__()
        self.backbone = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, classifier_mid_dim),
            nn.ReLU(),
            nn.Linear(classifier_mid_dim, num_classes),
        )

    def forward(self, x):
        out, _ = self.backbone(x)
        pooled = torch.mean(out, dim=1)
        logits = self.classifier(pooled)
        return logits


# ==============================================================================
# TIERED SELECTION & TIMELINE ASSEMBLY
# ==============================================================================
def assemble_tiered_timeline(
    probs,
    window_duration=4.0,
    window_stride=2.0,
    target_duration_sec=120,
    l3_threshold=0.60,
    l2_threshold=0.40,
    l0_threshold=0.20,
    pad_windows=2,
    max_gap=3.0,
    min_clip=3.0,
):
    """
    Tier 1: Level 3 technical detections + temporal padding.
    Tier 2: Highest-scoring Level 1/2 flow windows fill remaining target.

    Temporal geometry:
      prediction i -> [i * window_stride,
                       i * window_stride + window_duration]

    The target-duration constraint is applied to the ACTUAL resulting
    video duration after interval construction, gap merging and
    micro-clip filtering.
    """

    total_windows = len(probs)

    p_level0 = probs[:, 0]
    p_flow = probs[:, 1]
    p_level3 = probs[:, 2]

    top_class = np.argmax(probs, axis=1)

    selected_indices = set()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def indices_to_intervals(indices):
        """Convert selected prediction indices to physical video intervals."""
        if not indices:
            return []

        sorted_indices = sorted(indices)

        intervals = []
        curr_start = sorted_indices[0]
        curr_end = sorted_indices[0]

        for idx in sorted_indices[1:]:
            if idx == curr_end + 1:
                curr_end = idx
            else:
                start = curr_start * window_stride
                end = curr_end * window_stride + window_duration
                intervals.append((start, end))

                curr_start = idx
                curr_end = idx

        start = curr_start * window_stride
        end = curr_end * window_stride + window_duration
        intervals.append((start, end))

        return intervals

    def merge_intervals(intervals):
        """Merge intervals whose gap is <= max_gap."""
        if not intervals:
            return []

        merged = [intervals[0]]

        for current in intervals[1:]:
            prev_start, prev_end = merged[-1]
            curr_start, curr_end = current

            if curr_start - prev_end <= max_gap:
                merged[-1] = (
                    prev_start,
                    max(prev_end, curr_end),
                )
            else:
                merged.append(current)

        return merged

    def apply_min_clip(intervals, announce=False):
        """Remove clips shorter than min_clip."""
        final_clips = []

        for start, end in intervals:
            duration = end - start

            if duration >= min_clip:
                final_clips.append((start, end))
            elif announce:
                print(
                    f"[!] Dropped micro-clip: "
                    f"{start:.1f}s -> {end:.1f}s "
                    f"({duration:.1f}s)"
                )

        return final_clips

    def actual_clips(indices, announce=False):
        """
        Convert selected indices to the actual video clips that would
        be produced by the extractor.
        """
        intervals = indices_to_intervals(indices)
        merged = merge_intervals(intervals)
        return apply_min_clip(merged, announce=announce)

    def actual_duration(indices):
        """Return actual generated video duration for selected indices."""
        clips = actual_clips(indices)
        return sum(end - start for start, end in clips)

    # ------------------------------------------------------------------
    # STEP 1: TIER 1 - LEVEL 3 + PADDING
    # ------------------------------------------------------------------

    l3_raw_indices = np.where(
        (p_level3 >= l3_threshold)
        & (p_level0 < l0_threshold)
        & (top_class == 2)
    )[0]

    sorted_l3_indices = l3_raw_indices[
        np.argsort(-p_level3[l3_raw_indices])
    ]

    for idx in sorted_l3_indices:

        start_pad = max(0, idx - pad_windows)
        end_pad = min(total_windows - 1, idx + pad_windows)

        pad_range = range(start_pad, end_pad + 1)

        valid_pad_indices = {
            i
            for i in pad_range
            if p_level0[i] < l0_threshold
        }

        new_indices = valid_pad_indices - selected_indices

        if not new_indices:
            continue

        candidate_indices = selected_indices | new_indices

        projected_duration = actual_duration(candidate_indices)

        if (
            projected_duration > target_duration_sec
            and selected_indices
        ):
            continue

        selected_indices = candidate_indices

    t1_clips = actual_clips(selected_indices)
    t1_duration = sum(end - start for start, end in t1_clips)

    print(
        f"[+] Tier 1 (Level 3 Technical Features): "
        f"Selected {len(selected_indices)} windows "
        f"-> {t1_duration:.1f}s / {target_duration_sec}s target"
    )

    # ------------------------------------------------------------------
    # STEP 2: TIER 2 - FLOW FILLER
    # ------------------------------------------------------------------

    if t1_duration < target_duration_sec:

        flow_candidates = [
            idx
            for idx in np.argsort(-p_flow)
            if idx not in selected_indices
            and p_level0[idx] < l0_threshold
            and p_flow[idx] >= l2_threshold
            and (
                top_class[idx] == 1
                or p_flow[idx] > p_level3[idx]
            )
        ]

        added_t2 = 0

        for idx in flow_candidates:

            candidate_indices = selected_indices | {idx}

            projected_duration = actual_duration(candidate_indices)

            if projected_duration > target_duration_sec:
                continue

            selected_indices.add(idx)
            added_t2 += 1

            if projected_duration >= target_duration_sec:
                break

        final_t2_duration = actual_duration(selected_indices) - t1_duration

        print(
            f"[+] Tier 2 (Level 1/2 Flow Filler): "
            f"Added {added_t2} windows "
            f"({final_t2_duration:.1f}s)"
        )

    # ------------------------------------------------------------------
    # STEP 3: FINAL INTERVAL ASSEMBLY
    # ------------------------------------------------------------------

    final_clips = actual_clips(
        selected_indices,
        announce=True,
    )

    return final_clips

# ==============================================================================
# MAIN EXTRACTION PIPELINE
# ==============================================================================
def extract_highlights(args):
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"[+] Active Device: {device}")

    # --------------------------------------------------------------------------
    # 1. LOAD MODEL CHECKPOINT
    # --------------------------------------------------------------------------
    checkpoint_path = args.model_dir / args.model_name

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found at: {checkpoint_path}"
        )

    print(f"[+] Loading model checkpoint: {checkpoint_path}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    checkpoint_version = checkpoint.get(
        "checkpoint_version",
        "unknown",
    )

    if checkpoint_version == "unknown":
        raise ValueError(
            "Checkpoint does not contain checkpoint_version."
        )

    if "model_config" not in checkpoint:
        raise ValueError(
            "Checkpoint does not contain model_config."
        )

    if "data_config" not in checkpoint:
        raise ValueError(
            "Checkpoint does not contain data_config. "
            "This extractor requires checkpoint version 1.1 or newer."
        )

    feature_mask = checkpoint["feature_mask"]
    feature_dim = checkpoint["feature_dim"]
    model_config = checkpoint["model_config"]
    data_config = checkpoint["data_config"]

    required_data_config = (
        "flow_fps",
        "window",
        "stride",
    )

    missing_data_config = [
        key for key in required_data_config
        if key not in data_config
    ]

    if missing_data_config:
        raise ValueError(
            "Checkpoint data_config is missing required fields: "
            f"{missing_data_config}"
        )

    flow_fps = float(data_config["flow_fps"])
    window_duration = float(data_config["window"])
    window_stride = float(data_config["stride"])

    print(f"[+] Checkpoint version: {checkpoint_version}")
    print(f"[+] Feature dimension: {feature_dim}")
    print(f"[+] Feature mask: {len(feature_mask)} features")
    print(f"[+] Model configuration: {model_config}")
    print(
        f"[+] Data configuration: "
        f"{flow_fps:g} FPS, "
        f"{window_duration:g}s window, "
        f"{window_stride:g}s stride"
    )

    # --------------------------------------------------------------------------
    # 2. RECONSTRUCT MODEL FROM CHECKPOINT CONFIGURATION
    # --------------------------------------------------------------------------
    required_model_config = (
        "hidden_dim",
        "classifier_mid_dim",
        "num_classes",
        "dropout",
    )

    missing_model_config = [
        key for key in required_model_config
        if key not in model_config
    ]

    if missing_model_config:
        raise ValueError(
            "Checkpoint model_config is missing required fields: "
            f"{missing_model_config}"
        )

    model = MTBClassifier(
        feature_dim=feature_dim,
        hidden_dim=model_config["hidden_dim"],
        classifier_mid_dim=model_config["classifier_mid_dim"],
        num_classes=model_config["num_classes"],
        dropout=model_config["dropout"],
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    # --------------------------------------------------------------------------
    # 3. LOAD NPZ DATASET FOR INFERENCE
    # --------------------------------------------------------------------------
    npz_path = Path(args.input_npz)

    if not npz_path.exists():
        raise FileNotFoundError(
            f"Input NPZ dataset not found at: {npz_path}"
        )

    print(f"[+] Loading input features: {npz_path}")

    data = np.load(
        npz_path,
        allow_pickle=True,
    )

    if "X" not in data:
        raise ValueError(
            f"Input NPZ does not contain X: {npz_path}"
        )

    X = data["X"]

    if X.ndim != 3:
        raise ValueError(
            f"Expected X to have shape [N, T, F], "
            f"but got shape {X.shape}"
        )

    # --------------------------------------------------------------------------
    # 4. FILTER FEATURES USING CHECKPOINT FEATURE MASK
    # --------------------------------------------------------------------------
    if feature_mask is not None:
        if "feature_names" not in data:
            raise ValueError(
                "Checkpoint contains a feature_mask, but input NPZ "
                "does not contain feature_names."
            )

        all_names = list(data["feature_names"])

        missing_features = [
            f for f in feature_mask
            if f not in all_names
        ]

        if missing_features:
            raise ValueError(
                f"Input NPZ is missing {len(missing_features)} required "
                f"features: {missing_features}"
            )

        valid_indices = [
            all_names.index(f)
            for f in feature_mask
        ]

        X = X[:, :, valid_indices]

    # --------------------------------------------------------------------------
    # 5. VALIDATE FEATURE DIMENSION
    # --------------------------------------------------------------------------
    input_feature_dim = X.shape[2]

    if input_feature_dim != feature_dim:
        raise ValueError(
            f"Feature dimension mismatch: checkpoint expects "
            f"{feature_dim} features, but input NPZ produced "
            f"{input_feature_dim}"
        )

    X_tensor = torch.tensor(
        X,
        dtype=torch.float32,
    ).to(device)

    # --------------------------------------------------------------------------
    # 6. PREDICT 3-CLASS PROBABILITIES
    # --------------------------------------------------------------------------
    print(
        f"[+] Running inference across "
        f"{len(X_tensor)} sequence windows..."
    )

    with torch.no_grad():
        logits = model(X_tensor)

        probs = torch.softmax(
            logits,
            dim=-1,
        ).cpu().numpy()

    # --------------------------------------------------------------------------
    # 7. ASSEMBLE HIGHLIGHT INTERVALS
    # --------------------------------------------------------------------------
    intervals = assemble_tiered_timeline(
        probs=probs,
        window_duration=window_duration,
        window_stride=window_stride,
        target_duration_sec=args.target_duration,
        l3_threshold=args.l3_threshold,
        l2_threshold=args.l2_threshold,
        l0_threshold=args.l0_threshold,
        pad_windows=args.pad_windows,
        max_gap=args.max_gap,
        min_clip=args.min_clip,
    )

    # --------------------------------------------------------------------------
    # 8. OUTPUT SUMMARY
    # --------------------------------------------------------------------------
    total_selected_sec = sum(
        end - start
        for start, end in intervals
    )

    print("\n" + "=" * 60)
    print("[SUCCESS] Highlight Assembly Complete")
    print(
        f"[SUCCESS] Total Generated Highlight Duration: "
        f"{total_selected_sec:.1f}s / Target: "
        f"{args.target_duration}s"
    )
    print(
        f"[SUCCESS] Total Continuous Video Clips: "
        f"{len(intervals)}"
    )
    print("=" * 60)

    print("\nGenerated Timestamp Clips (Start -> End):")

    for i, (start, end) in enumerate(
        intervals,
        1,
    ):
        print(
            f"  Clip {i:02d}: "
            f"{start:06.1f}s  -->  "
            f"{end:06.1f}s  "
            f"(Duration: {end - start:.1f}s)"
        )

    if args.output_clips:
        output_path = Path(args.output_clips)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        clips = [
            {
                "clip": i,
                "start": round(start, 3),
                "end": round(end, 3),
                "duration": round(end - start, 3),
            }
            for i, (start, end) in enumerate(intervals, 1)
        ]

        suffix = output_path.suffix.lower()

        if suffix == ".csv":
            with output_path.open(
                "w",
                newline="",
                encoding="utf-8",
            ) as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=("clip", "start", "end", "duration"),
                )
                writer.writeheader()
                writer.writerows(clips)

        elif suffix == ".json":
            with output_path.open(
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(clips, f, indent=2)

        else:
            raise ValueError(
                "--output-clips must use a .csv or .json extension."
            )

        print(f"[+] Saved clip timestamps: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Extract MTB Video Highlights using "
            "3-Class Tiered Logic"
        )
    )

    parser.add_argument(
        "--input-npz",
        type=str,
        required=True,
        help="Path to input video .npz feature file",
    )

    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path(
            r"C:\VideoTools\MTB-Video-Tools\output\checkpoints"
        ),
        help="Directory containing model checkpoints",
    )

    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Checkpoint filename to load",
    )

    parser.add_argument(
        "--target-duration",
        type=int,
        default=120,
        help="Target intro duration in seconds",
    )

    parser.add_argument(
        "--l3-threshold",
        type=float,
        default=0.60,
        help=(
            "Probability threshold for Tier 1 "
            "Level 3 selection"
        ),
    )

    parser.add_argument(
        "--l2-threshold",
        type=float,
        default=0.40,
        help=(
            "Probability threshold for Tier 2 "
            "Level 1-2 selection"
        ),
    )

    parser.add_argument(
        "--l0-threshold",
        type=float,
        default=0.35,
        help=(
            "Upper limit for Level 0 confidence"
        ),
    )

    parser.add_argument(
        "--pad-windows",
        type=int,
        default=2,
        help=(
            "Number of windows to pad around "
            "Level 3 features"
        ),
    )

    parser.add_argument(
        "--max-gap",
        type=float,
        default=3.0,
        help="Max gap between clips for merging",
    )

    parser.add_argument(
        "--min-clip",
        type=float,
        default=3.0,
        help="Drops micro-clips shorter than this",
    )

    parser.add_argument(
        "--output-clips",
        type=str,
        default=None,
        help=(
            "Optional output file for final clip timestamps (.csv or .json)"
        ),
    )

    args = parser.parse_args()

    extract_highlights(args)