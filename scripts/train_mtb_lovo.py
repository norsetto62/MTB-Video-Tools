import argparse
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, ConcatDataset


# ==============================================================================
# 1. TARGET SCALER (Per-Fold Standardization)
# ==============================================================================
class TargetScaler:
    """Zero-mean, unit-variance scaler for regression targets to prevent mean bias."""
    def __init__(self):
        self.mean = 0.0
        self.std = 1.0

    def fit(self, y_tensor):
        y_np = y_tensor.numpy()
        self.mean = float(np.mean(y_np))
        self.std = float(np.std(y_np))
        if self.std < 1e-6:
            self.std = 1.0  # Prevent division by zero

    def transform(self, y_tensor):
        return (y_tensor - self.mean) / self.std

    def inverse_transform(self, y_scaled):
        if isinstance(y_scaled, torch.Tensor):
            return y_scaled * self.std + self.mean
        return y_scaled * self.std + self.mean


# ==============================================================================
# 2. DATASET DEFINITION
# ==============================================================================
class MTBDataset(Dataset):
    def __init__(self, npz_file, feature_mask=None):
        data = np.load(npz_file, allow_pickle=True)
        
        X = data["X"]  # Shape: (Samples, Window_Length, Features)
        y = data["y"]  # Shape: (Samples,)

        if feature_mask is not None and "feature_names" in data:
            all_names = list(data["feature_names"])
            valid_indices = [all_names.index(f) for f in feature_mask if f in all_names]
            if valid_indices:
                X = X[:, :, valid_indices]

        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ==============================================================================
# 3. MODEL ARCHITECTURE (Dilated 1D-TCN)
# ==============================================================================
class MTBConvNet(nn.Module):
    """1D Dilated Temporal Convolutional Network with GroupNorm."""
    def __init__(self, in_channels, dropout=0.4):
        super().__init__()
        self.net = nn.Sequential(
            # Layer 1: Standard Receptive Field
            nn.Conv1d(in_channels, 64, kernel_size=3, padding=1, dilation=1),
            nn.GroupNorm(num_groups=8, num_channels=64),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            # Layer 2: Expanded Dilated Receptive Field (Dilation = 2)
            nn.Conv1d(64, 32, kernel_size=3, padding=2, dilation=2),
            nn.GroupNorm(num_groups=4, num_channels=32),
            nn.ReLU(),
            
            # Global Temporal Pooling
            nn.AdaptiveAvgPool1d(1),  # Pools (Batch, 32, Time) -> (Batch, 32, 1)
            nn.Flatten(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        # Input shape: (Batch, Sequence_Len, Features) -> (Batch, Features, Sequence_Len)
        x = x.transpose(1, 2)
        return self.net(x).squeeze(-1)


# ==============================================================================
# 4. TRAINING AND EVALUATION HELPERS
# ==============================================================================
def train_one_epoch(model, dataloader, optimizer, criterion, scaler, device):
    model.train()
    total_loss = 0.0

    for X_batch, y_batch in dataloader:
        X_batch = X_batch.to(device)
        # Transform target to scaled space for gradient step
        y_scaled = scaler.transform(y_batch).to(device)

        optimizer.zero_grad()
        predictions = model(X_batch)
        loss = criterion(predictions, y_scaled)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(y_batch)

    return total_loss / len(dataloader.dataset)


def evaluate(model, dataloader, criterion, scaler, device):
    model.eval()
    total_loss = 0.0
    all_preds_orig = []
    all_targets_orig = []

    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            y_scaled = scaler.transform(y_batch).to(device)

            preds_scaled = model(X_batch)
            loss = criterion(preds_scaled, y_scaled)

            total_loss += loss.item() * len(y_batch)

            # Inverse-transform predictions back to true target scale for metrics
            preds_orig = scaler.inverse_transform(preds_scaled.cpu()).numpy()
            targets_orig = y_batch.numpy()

            all_preds_orig.extend(preds_orig)
            all_targets_orig.extend(targets_orig)

    all_preds_orig = np.array(all_preds_orig)
    all_targets_orig = np.array(all_targets_orig)

    val_loss = total_loss / len(dataloader.dataset)
    mae = np.mean(np.abs(all_preds_orig - all_targets_orig))
    rmse = np.sqrt(np.mean((all_preds_orig - all_targets_orig) ** 2))

    # Variance-safe R2 calculation in original target units
    target_variance = np.var(all_targets_orig)
    if target_variance < 1e-6:
        r2 = 0.0
    else:
        ss_res = np.sum((all_targets_orig - all_preds_orig) ** 2)
        ss_tot = np.sum((all_targets_orig - np.mean(all_targets_orig)) ** 2)
        r2 = 1.0 - (ss_res / ss_tot)

    return val_loss, mae, rmse, r2


# ==============================================================================
# 5. LEAVE-ONE-VIDEO-OUT CROSS VALIDATION HARNESS
# ==============================================================================
def run_lovo_cv(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[+] Using Device: {device}")

    data_path = Path(args.data_dir)
    dataset_files = sorted(list(data_path.glob("*.npz")))

    if not dataset_files:
        raise FileNotFoundError(f"No .npz files found in: {args.data_dir}")

    print(f"[+] Found {len(dataset_files)} video datasets for LOVO CV.")

    feature_mask = None
    if args.feature_file and Path(args.feature_file).exists():
        with open(args.feature_file, "r") as f:
            feature_mask = [line.strip() for line in f if line.strip()]
        print(f"[+] Loaded feature mask with {len(feature_mask)} whitelist features.")

    fold_results = []

    for fold_idx, val_path in enumerate(dataset_files):
        val_name = val_path.stem
        train_paths = [p for p in dataset_files if p != val_path]

        # 1. Load training datasets per video
        train_datasets = [MTBDataset(p, feature_mask) for p in train_paths]

        # 2. Calculate sample weights for dataset size balancing
        sample_weights = []
        for ds in train_datasets:
            ds_len = len(ds)
            weight_per_sample = 1.0 / ds_len
            sample_weights.extend([weight_per_sample] * ds_len)

        train_dataset = ConcatDataset(train_datasets)
        val_dataset = MTBDataset(val_path, feature_mask)

        # 3. Fit Target Scaler strictly on training targets (no data leakage)
        train_y_concat = torch.cat([ds.y for ds in train_datasets])
        scaler = TargetScaler()
        scaler.fit(train_y_concat)

        num_features = train_datasets[0].X.shape[-1]

        sampler = WeightedRandomSampler(
            weights=torch.DoubleTensor(sample_weights),
            num_samples=len(sample_weights),
            replacement=True
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            shuffle=False
        )
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

        # 4. Instantiate Model & Optimizer
        model = MTBConvNet(in_channels=num_features, dropout=args.dropout).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
        criterion = nn.HuberLoss(delta=1.0)

        best_val_loss = float("inf")
        best_metrics = (0.0, 0.0, 0.0)
        best_epoch = 0

        # 5. Training Loop
        for epoch in range(1, args.epochs + 1):
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, scaler, device)
            val_loss, mae, rmse, r2 = evaluate(model, val_loader, criterion, scaler, device)
            scheduler.step()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_metrics = (mae, rmse, r2)
                best_epoch = epoch

        print(
            f"Fold {fold_idx + 1:2d}/{len(dataset_files)} | "
            f"Held-out: {val_name:<22} | Best Loss: {best_val_loss:.4f} @ Ep {best_epoch:2d}/{args.epochs:2d} | "
            f"MAE: {best_metrics[0]:.4f} | RMSE: {best_metrics[1]:.4f} | R2: {best_metrics[2]:.4f}"
        )
        fold_results.append((val_name, best_val_loss, *best_metrics, best_epoch))

    # Print Summary Table
    print("\n" + "=" * 92)
    print("LEAVE-ONE-VIDEO-OUT CROSS-VALIDATION SUMMARY (WITH TARGET SCALING)")
    print("=" * 92)
    print(f"{'Held-Out Video':<24} | {'Val Loss':<10} | {'Best Epoch':<10} | {'MAE':<10} | {'RMSE':<10} | {'R2':<8}")
    print("-" * 92)
    for name, loss, mae, rmse, r2, epoch in fold_results:
        print(f"{name:<24} | {loss:<10.4f} | {epoch:<10} | {mae:<10.4f} | {rmse:<10.4f} | {r2:<8.4f}")

    avg_loss = np.mean([r[1] for r in fold_results])
    avg_mae = np.mean([r[2] for r in fold_results])
    avg_rmse = np.mean([r[3] for r in fold_results])
    avg_r2 = np.mean([r[4] for r in fold_results])
    avg_epoch = np.mean([r[5] for r in fold_results])

    print("-" * 92)
    print(f"{'MEAN ACROSS ALL FOLDS':<24} | {avg_loss:<10.4f} | {int(avg_epoch):<10} | {avg_mae:<10.4f} | {avg_rmse:<10.4f} | {avg_r2:<8.4f}")
    print("=" * 92)


# ==============================================================================
# 6. CLI ENTRYPOINT
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LOVO Cross-Validation with Target Scaling and Dilated Conv1D")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=r"C:\VideoTools\MTB-Video-Tools\output\datasets",
        help="Path to directory containing .npz dataset files",
    )
    parser.add_argument(
        "--feature-file",
        type=str,
        default=None,
        help="Path to txt file containing feature whitelist",
    )
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs per fold")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-2, help="Weight decay for AdamW")
    parser.add_argument("--dropout", type=float, default=0.4, help="Dropout probability")

    args = parser.parse_args()

    try:
        run_lovo_cv(args)
    except KeyboardInterrupt:
        print("\n[!] Execution interrupted by user.")
        sys.exit(0)