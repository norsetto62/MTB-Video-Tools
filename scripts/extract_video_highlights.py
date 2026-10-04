import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

# ==============================================================================
# MODEL CONFIGURATION CONSTANTS
# ==============================================================================
DEFAULT_HIDDEN_DIM = 128
CLASSIFIER_MID_DIM = 64
DEFAULT_NUM_CLASSES = 3
DEFAULT_DROPOUT_RATE = 0.35


# ==============================================================================
# MODEL ARCHITECTURE (Matches Training Definition)
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
    window_duration=1.0, 
    target_duration_sec=120, 
    l3_threshold=0.60,
    l2_threshold=0.40,
    l0_threshold=0.20,  # Upper limit for Level 0 dismount confidence
    pad_windows=2,
    max_gap=3.0,
    min_clip=3.0, # Drops micro-clips shorter than 3s
):
    """
    Tier 1: All Level 3 (technical) windows exceeding threshold (+ padding).
    Tier 2: Highest scoring Level 1/2 (flow) windows to fill remaining target deficit (if any).
    Excludes Level 0 (pauses/idle) entirely.
    """
    total_windows = len(probs)
    p_level0 = probs[:, 0]   # Probabilities for Level 0 (Pauses/Dismounts)
    p_flow   = probs[:, 1]   # Probabilities for Level 1/2
    p_level3 = probs[:, 2]   # Probabilities for Level 3

    # Calculate top predicted class index per window (0, 1, or 2)
    top_class = np.argmax(probs, axis=1)

    # Track selected window indices
    selected_indices = set()

    # --------------------------------------------------------------------------
    # STEP 1: TIER 1 - HARD SELECT LEVEL 3 + TEMPORAL PADDING
    # --------------------------------------------------------------------------
    # 1. Get all indices exceeding the threshold
    l3_raw_indices = np.where(
        (p_level3 >= l3_threshold)
        & (p_level0 < l0_threshold)
        & (top_class == 2)
    )[0]

    # 2. Sort the detected indices descending by their p_level3 probability score
    sorted_l3_indices = l3_raw_indices[np.argsort(-p_level3[l3_raw_indices])]

    # 3. Add detections + padding up to target_duration_sec
    for idx in sorted_l3_indices:
        # Add window along with preceding and following padding windows
        start_pad = max(0, idx - pad_windows)
        end_pad = min(total_windows - 1, idx + pad_windows)

        # Check for Level 0 contamination within the padded region
        # Optional: ensure padding doesn't heavily overlap a high L0 zone
        pad_range = range(start_pad, end_pad + 1)
        valid_pad_indices = [
            i for i in pad_range if p_level0[i] < l0_threshold
        ]

        # Calculate how many NEW unique windows this detection + padding would add
        new_windows = set(valid_pad_indices) - selected_indices
        projected_duration = (
            len(selected_indices) + len(new_windows)
        ) * window_duration

        # Stop if adding this detection exceeds the target duration
        if (
            projected_duration > target_duration_sec
            and len(selected_indices) > 0
        ):
            break

        # Otherwise, add the windows
        selected_indices.update(valid_pad_indices)

    t1_duration = len(selected_indices) * window_duration
    print(
        f"[+] Tier 1 (Level 3 Technical Features): Selected {len(selected_indices)} windows "
        f"-> {t1_duration:.1f}s / {target_duration_sec}s target"
    )

    # --------------------------------------------------------------------------
    # STEP 2: TIER 2 - FILL DEFICIT WITH TOP LEVEL 1/2 FLOW
    # --------------------------------------------------------------------------
    if t1_duration < target_duration_sec:
        # Sort flow candidates descending by p_flow, excluding already selected or high L0 windows
        flow_candidates = [
            idx
            for idx in np.argsort(-p_flow)
            if idx not in selected_indices
            and p_level0[idx] < l0_threshold
            and p_flow[idx] >= l2_threshold
            and (
                top_class[idx] == 1 or p_flow[idx] > p_level3[idx]
            )  # Level 1/2 outranks Level 3
        ]

        added_t2 = 0
        for idx in flow_candidates:
            if (
                len(selected_indices) + 1
            ) * window_duration > target_duration_sec:
                break
            selected_indices.add(idx)
            added_t2 += 1

        print(
            f"[+] Tier 2 (Level 1/2 Flow Filler): Added {added_t2} windows ({added_t2 * window_duration:.1f}s)"
        )

    # Convert to a sorted list for downstream merging/segment generation
    sorted_indices = sorted(list(selected_indices))
    if not sorted_indices:
        return []

    # --------------------------------------------------------------------------
    # STEP 3: CONVERT WINDOW INDICES TO CONTINUOUS TIMESTAMP INTERVALS
    # --------------------------------------------------------------------------
    intervals = []
    curr_start = sorted_indices[0]
    curr_end = sorted_indices[0]

    for idx in sorted_indices[1:]:
        if idx == curr_end + 1:
            curr_end = idx
        else:
            intervals.append((curr_start * window_duration, (curr_end + 1) * window_duration))
            curr_start = idx
            curr_end = idx
            
    intervals.append((curr_start * window_duration, (curr_end + 1) * window_duration))

    # Merge adjacent clips separated by <= 3.0s
    merged = [intervals[0]]
    for current in intervals[1:]:
        prev_start, prev_end = merged[-1]
        curr_start, curr_end = current
        
        # If the gap between previous end and current start is <= max_gap_sec, merge them
        if curr_start - prev_end <= max_gap:
            merged[-1] = (prev_start, max(prev_end, curr_end))
        else:
            merged.append(current)

    # Drop standalone micro-clips shorter than min_clip_duration (e.g., < 3.0s)
    final_clips = []
    for start, end in merged:
        duration = end - start
        if duration >= min_clip:
            final_clips.append((start,end))
        else:
            print(
                f"[!] Dropped micro-clip: {start:.1f}s -> {end:.1f}s ({duration:.1f}s)"
            )

    return final_clips

# ==============================================================================
# MAIN EXTRACTION PIPELINE
# ==============================================================================
def extract_highlights(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[+] Active Device: {device}")

    # 1. Load Model Checkpoint
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

    feature_mask = checkpoint["feature_mask"]
    feature_dim = checkpoint["feature_dim"]
    model_config = checkpoint["model_config"]

    print(f"[+] Checkpoint version: {checkpoint.get('checkpoint_version', 'unknown')}")
    print(f"[+] Feature dimension: {feature_dim}")
    print(f"[+] Feature mask: {len(feature_mask)} features")
    print(f"[+] Model configuration: {model_config}")

    model = MTBClassifier(
        feature_dim=feature_dim,
        hidden_dim=model_config["hidden_dim"],
        classifier_mid_dim=model_config["classifier_mid_dim"],
        num_classes=model_config["num_classes"],
        dropout=model_config["dropout"],
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # 2. Load NPZ Dataset for Inference
    npz_path = Path(args.input_npz)
    if not npz_path.exists():
        raise FileNotFoundError(f"Input NPZ dataset not found at: {npz_path}")

    print(f"[+] Loading input features: {npz_path}")
    data = np.load(npz_path, allow_pickle=True)
    X = data["X"]  # Shape: [N, T, F]

    # Filter features using the model's saved feature_mask whitelist
    if feature_mask is not None and "feature_names" in data:
        all_names = list(data["feature_names"])

        missing_features = [
            f for f in feature_mask if f not in all_names
        ]
        if missing_features:
            raise ValueError(
                f"Input NPZ is missing {len(missing_features)} required "
                f"features: {missing_features}"
            )
                
        valid_indices = [all_names.index(f) for f in feature_mask if f in all_names]
        X = X[:, :, valid_indices]

    if X.shape[2] != feature_dim:
        raise ValueError(
            f"Feature dimension mismatch: checkpoint expects {feature_dim} "
            f"features, but input NPZ produced {X.shape[2]}"
        )
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)

    # 3. Predict 3-Class Probabilities
    print(f"[+] Running inference across {len(X_tensor)} sequence windows...")
    with torch.no_grad():
        logits = model(X_tensor)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()  # [N, 3] -> P0, P1/2, P3

    # 4. Assemble Highlight Intervals
    intervals = assemble_tiered_timeline(
        probs=probs,
        window_duration=args.window_stride,
        target_duration_sec=args.target_duration,
        l3_threshold=args.l3_threshold,
        l2_threshold=args.l2_threshold,
        l0_threshold=args.l0_threshold,
        pad_windows=args.pad_windows,
        max_gap=args.max_gap,
        min_clip=args.min_clip,
    )

    # 5. Output Summary
    total_selected_sec = sum(end - start for start, end in intervals)
    print("\n" + "=" * 60)
    print(f"[SUCCESS] Highlight Assembly Complete")
    print(f"[SUCCESS] Total Generated Highlight Duration: {total_selected_sec:.1f}s / Target: {args.target_duration}s")
    print(f"[SUCCESS] Total Continuous Video Clips: {len(intervals)}")
    print("=" * 60)
    print("\nGenerated Timestamp Clips (Start -> End):")
    for i, (start, end) in enumerate(intervals, 1):
        print(f"  Clip {i:02d}: {start:06.1f}s  -->  {end:06.1f}s  (Duration: {end - start:.1f}s)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract MTB Video Highlights using 3-Class Tiered Logic")
    parser.add_argument("--input-npz", type=str, required=True, help="Path to input video .npz feature file")
    parser.add_argument("--model-dir", type=Path, default=Path(r"C:\VideoTools\MTB-Video-Tools\output\checkpoints"), help="Directory containing model checkpoints")
    parser.add_argument("--model-name", type=str, required=True,  help="Checkpoint filename to load")
    parser.add_argument("--target-duration", type=int, default=120, help="Target intro duration in seconds")
    parser.add_argument("--l3-threshold", type=float, default=0.60, help="Probability threshold for Tier 1 Level 3 selection")
    parser.add_argument("--l2-threshold", type=float, default=0.40, help="Probability threshold for Tier 2 Level 1-2 selection")
    parser.add_argument("--l0-threshold", type=float, default=0.35, help="Upper limit for Level 0 confidence")
    parser.add_argument("--window-stride", type=float, default=1.0, help="Time duration per window in seconds")
    parser.add_argument("--pad-windows", type=int, default=2, help="Number of windows to pad around Level 3 features")
    parser.add_argument("--max-gap", type=float, default=3.0, help="Max gap between clips for merging")
    parser.add_argument("--min-clip", type=float, default=3.0, help="Drops micro-clips shorter than this")
    args = parser.parse_args()
    extract_highlights(args)