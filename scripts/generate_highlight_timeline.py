import torch
import numpy as np
from pathlib import Path

def generate_highlight_timeline(model_path, test_npz_path, target_duration_sec=90, l3_threshold=0.35, window_stride_sec=1.0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    feature_mask = checkpoint["feature_mask"]
    
    # 2. Load Test Dataset
    data = np.load(test_npz_path)
    X = data["X"]  # [N, T, F]
    
    if feature_mask and "feature_names" in data:
        all_names = list(data["feature_names"])
        valid_indices = [all_names.index(f) for f in feature_mask if f in all_names]
        X = X[:, :, valid_indices]
        
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    
    # 3. Load Model
    from train_production_model import MTBClassifier
    model = MTBClassifier(feature_dim=X_tensor.shape[-1], hidden_dim=128, num_classes=3).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    # 4. Predict Class Probabilities
    with torch.no_grad():
        logits = model(X_tensor)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()  # [N, 3] -> P0, P1/2, P3
        
    p_level3 = probs[:, 2]
    p_flow   = probs[:, 1]
    
    # 5. Tiered Selection Logic
    selected_indices = []
    
    # Tier 1: All Level 3 clips exceeding threshold
    l3_indices = np.where(p_level3 >= l3_threshold)[0]
    selected_indices.extend(l3_indices)
    
    current_duration = len(selected_indices) * window_stride_sec
    print(f"[+] Tier 1 (Level 3 Features Detected): {len(l3_indices)} clips ({current_duration:.1f}s)")
    
    # Tier 2: Fill remaining deficit with top Level 1/2 Flow clips
    deficit_sec = target_duration_sec - current_duration
    if deficit_sec > 0:
        # Rank remaining indices (excluding L3 already picked) by P1/2 score
        remaining_indices = [i for i in range(len(probs)) if i not in selected_indices]
        remaining_indices_sorted = sorted(remaining_indices, key=lambda i: p_flow[i], reverse=True)
        
        num_filler_clips = int(np.ceil(deficit_sec / window_stride_sec))
        filler_indices = remaining_indices_sorted[:num_filler_clips]
        
        selected_indices.extend(filler_indices)
        print(f"[+] Tier 2 (Level 1/2 Flow Filler Added): {len(filler_indices)} clips ({len(filler_indices) * window_stride_sec:.1f}s)")
    
    # Sort chronologically for video export
    selected_indices = sorted(selected_indices)
    total_intro_duration = len(selected_indices) * window_stride_sec
    
    print("=" * 60)
    print(f"[SUCCESS] Final Intro Edit Duration: {total_intro_duration:.1f}s / Target: {target_duration_sec}s")
    print(f"[SUCCESS] Total Level 3 Clips Included: {len(l3_indices)}")
    print("=" * 60)
    
    return selected_indices

# Example Usage:
# generate_highlight_timeline(
#     model_path=r"C:\VideoTools\MTB-Video-Tools\output\models\mtb_interest_model.pt",
#     test_npz_path=r"C:\VideoTools\MTB-Video-Tools\output\datasets\Mentorella_dataset.npz",
#     target_duration_sec=120,
#     l3_threshold=0.35
# )