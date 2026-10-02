import argparse
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn


# ==============================================================================
# 1. MODEL ARCHITECTURE & SCALER MATCHING TRAINING PIPELINE
# ==============================================================================
class MTBConvNet(nn.Module):
    def __init__(self, in_channels, dropout=0.4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=3, padding=1, dilation=1),
            nn.GroupNorm(num_groups=8, num_channels=64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(64, 32, kernel_size=3, padding=2, dilation=2),
            nn.GroupNorm(num_groups=4, num_channels=32),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        return self.net(x).squeeze(-1)


class TargetScaler:
    def __init__(self):
        self.mean = 0.0
        self.std = 1.0

    def load_state_dict(self, state_dict):
        self.mean = state_dict["mean"]
        self.std = state_dict["std"]

    def inverse_transform(self, y_scaled):
        if isinstance(y_scaled, torch.Tensor):
            return y_scaled * self.std + self.mean
        return y_scaled * self.std + self.mean


# ==============================================================================
# 2. HIGHLIGHT EXTRACTION HELPER
# ==============================================================================
def find_top_highlights(scores, window_fps=0.5, top_k=5, min_clip_sec=10, max_clip_sec=30, min_gap_sec=5):
    """
    Extracts top non-overlapping highlight time windows based on predicted interest scores.
    
    Args:
        scores (np.ndarray): 1D array of predicted interest scores.
        window_fps (float): Frequency of predictions per second (e.g., 0.5 = 1 prediction every 2 sec).
        top_k (int): Maximum number of highlight clips to extract.
        min_clip_sec (int): Minimum clip duration in seconds.
        max_clip_sec (int): Maximum clip duration in seconds.
        min_gap_sec (int): Minimum time gap in seconds between extracted clips.
    """
    sample_interval_sec = 1.0 / window_fps if window_fps > 0 else 1.0
    min_samples = int(min_clip_sec / sample_interval_sec)
    max_samples = int(max_clip_sec / sample_interval_sec)
    gap_samples = int(min_gap_sec / sample_interval_sec)

    # Smooth raw predictions with moving average to avoid jittery peaks
    kernel_size = max(3, min_samples // 2)
    smoothed_scores = np.convolve(scores, np.ones(kernel_size) / kernel_size, mode='same')

    candidate_windows = []
    
    # Evaluate rolling window regions
    for start_idx in range(0, len(smoothed_scores) - min_samples):
        for length in range(min_samples, min(max_samples, len(smoothed_scores) - start_idx)):
            end_idx = start_idx + length
            avg_score = np.mean(smoothed_scores[start_idx:end_idx])
            candidate_windows.append((avg_score, start_idx, end_idx))

    # Sort candidates by highest score
    candidate_windows.sort(key=lambda x: x[0], reverse=True)

    selected_highlights = []
    
    # Non-maximum suppression to prevent overlapping highlight selections
    for score, start_idx, end_idx in candidate_windows:
        overlap = False
        for _, s_start, s_end in selected_highlights:
            if not (end_idx + gap_samples <= s_start or start_idx >= s_end + gap_samples):
                overlap = True
                break
        
        if not overlap:
            selected_highlights.append((score, start_idx, end_idx))
            if len(selected_highlights) >= top_k:
                break

    # Format selected windows to timestamp ranges
    highlights = []
    for rank, (score, start_idx, end_idx) in enumerate(selected_highlights, start=1):
        start_sec = start_idx * sample_interval_sec
        end_sec = end_idx * sample_interval_sec
        highlights.append({
            "rank": rank,
            "score": float(score),
            "start_sec": round(start_sec, 2),
            "end_sec": round(end_sec, 2),
            "duration_sec": round(end_sec - start_sec, 2)
        })

    return highlights, smoothed_scores


# ==============================================================================
# 3. INFERENCE PIPELINE
# ==============================================================================
def run_inference(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[+] Using Device: {device}")

    model_path = Path(args.model_dir) / "mtb_interest_model.pt"
    scaler_path = Path(args.model_dir) / "mtb_scaler.pt"

    if not model_path.exists() or not scaler_path.exists():
        raise FileNotFoundError(f"Model or Scaler checkpoint missing in: {args.model_dir}")

    # Load Model Checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    in_channels = checkpoint["in_channels"]
    feature_mask = checkpoint.get("feature_mask", None)

    model = MTBConvNet(in_channels=in_channels, dropout=0.0).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Load Scaler Checkpoint
    scaler_dict = torch.load(scaler_path, map_location=device)
    scaler = TargetScaler()
    scaler.load_state_dict(scaler_dict)

    print(f"[+] Model and Target Scaler loaded successfully.")

    # Load Target Input Feature NPZ
    input_file = Path(args.input_npz)
    if not input_file.exists():
        raise FileNotFoundError(f"Input feature file not found: {input_file}")

    data = np.load(input_file, allow_pickle=True)
    X = data["X"]  # Shape: (Num_Windows, Sequence_Length, Features)

    # Filter features if whitelist mask exists in checkpoint
    if feature_mask is not None and "feature_names" in data:
        all_names = list(data["feature_names"])
        valid_indices = [all_names.index(f) for f in feature_mask if f in all_names]
        if valid_indices:
            X = X[:, :, valid_indices]

    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)

    # Predict in Batches
    batch_size = args.batch_size
    raw_preds = []

    with torch.no_grad():
        for i in range(0, len(X_tensor), batch_size):
            batch_X = X_tensor[i:i + batch_size]
            preds_scaled = model(batch_X)
            preds_orig = scaler.inverse_transform(preds_scaled.cpu()).numpy()
            raw_preds.extend(preds_orig)

    raw_preds = np.array(raw_preds)

    # Convert predictions to highlights
    highlights, smoothed_preds = find_top_highlights(
        scores=raw_preds,
        window_fps=args.window_fps,
        top_k=args.top_k,
        min_clip_sec=args.min_clip_sec,
        max_clip_sec=args.max_clip_sec,
        min_gap_sec=args.min_gap_sec
    )

    # Print Highlight Summary
    print("\n" + "=" * 80)
    print(f"TOP {len(highlights)} HIGHLIGHT CLIPS FOR: {input_file.name}")
    print("=" * 80)
    print(f"{'Rank':<6} | {'Start Time':<12} | {'End Time':<12} | {'Duration (s)':<12} | {'Score':<8}")
    print("-" * 80)

    for h in highlights:
        start_min, start_sec = divmod(h['start_sec'], 60)
        end_min, end_sec = divmod(h['end_sec'], 60)
        
        start_fmt = f"{int(start_min):02d}:{start_sec:05.2f}"
        end_fmt = f"{int(end_min):02d}:{end_sec:05.2f}"

        print(f"#{h['rank']:<5} | {start_fmt:<12} | {end_fmt:<12} | {h['duration_sec']:<12.1f} | {h['score']:<8.4f}")

    print("=" * 80)

    # Export CSV / JSON results if requested
    if args.output_dir:
        out_path = Path(args.output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        
        npz_stem = input_file.stem
        csv_file = out_path / f"{npz_stem}_predictions.csv"

        # Save second-by-second predicted curve
        timestamps = np.arange(len(raw_preds)) / args.window_fps
        out_data = np.column_stack((timestamps, raw_preds, smoothed_preds))
        np.savetxt(
            csv_file,
            out_data,
            delimiter=",",
            header="timestamp_sec,raw_interest_score,smoothed_interest_score",
            comments="",
            fmt="%.2f,%.4f,%.4f"
        )
        print(f"[+] Prediction timeline saved to: {csv_file}")


# ==============================================================================
# 4. CLI ENTRYPOINT
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict second-by-second MTB interest scores and extract highlight clips")
    parser.add_argument("--input-npz", type=str, required=True, help="Path to preprocessed feature .npz file")
    parser.add_argument("--model-dir", type=str, default=r"C:\VideoTools\MTB-Video-Tools\output\models", help="Directory containing mtb_interest_model.pt and mtb_scaler.pt")
    parser.add_argument("--output-dir", type=str, default=r"C:\VideoTools\MTB-Video-Tools\output\highlights", help="Output directory for predictions CSV")
    
    # Highlight selection parameters
    parser.add_argument("--window-fps", type=float, default=0.5, help="Sampling frequency of window features per second (default: 0.5 = 1 window every 2s)")
    parser.add_argument("--top-k", type=int, default=5, help="Number of top highlight clips to select")
    parser.add_argument("--min-clip-sec", type=int, default=10, help="Minimum clip duration in seconds")
    parser.add_argument("--max-clip-sec", type=int, default=25, help="Maximum clip duration in seconds")
    parser.add_argument("--min-gap-sec", type=int, default=5, help="Minimum gap in seconds between clips")
    parser.add_argument("--batch-size", type=int, default=64)

    args = parser.parse_args()

    try:
        run_inference(args)
    except KeyboardInterrupt:
        print("\n[!] Inference canceled by user.")
        sys.exit(0)