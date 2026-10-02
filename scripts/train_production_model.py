import argparse
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, ConcatDataset


# ==============================================================================
# 1. TARGET SCALER
# ==============================================================================
class TargetScaler:
    def __init__(self):
        self.mean = 0.0
        self.std = 1.0

    def fit(self, y_tensor):
        y_np = y_tensor.numpy()
        self.mean = float(np.mean(y_np))
        self.std = float(np.std(y_np))
        if self.std < 1e-6:
            self.std = 1.0

    def transform(self, y_tensor):
        return (y_tensor - self.mean) / self.std

    def inverse_transform(self, y_scaled):
        if isinstance(y_scaled, torch.Tensor):
            return y_scaled * self.std + self.mean
        return y_scaled * self.std + self.mean

    def state_dict(self):
        return {"mean": self.mean, "std": self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict["mean"]
        self.std = state_dict["std"]


# ==============================================================================
# 2. DATASET DEFINITION
# ==============================================================================
class MTBDataset(Dataset):
    def __init__(self, npz_file, feature_mask=None):
        data = np.load(npz_file, allow_pickle=True)
        X = data["X"]
        y = data["y"]

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
# 3. MODEL ARCHITECTURE
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


# ==============================================================================
# 4. PRODUCTION TRAINING FUNCTION
# ==============================================================================
def train_production_model(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[+] Using Device: {device}")

    data_path = Path(args.data_dir)
    dataset_files = sorted(list(data_path.glob("*.npz")))

    if not dataset_files:
        raise FileNotFoundError(f"No .npz files found in: {args.data_dir}")

    print(f"[+] Training production model on ALL {len(dataset_files)} datasets...")

    feature_mask = None
    if args.feature_file and Path(args.feature_file).exists():
        with open(args.feature_file, "r") as f:
            feature_mask = [line.strip() for line in f if line.strip()]
        print(f"[+] Loaded feature whitelist ({len(feature_mask)} features).")

    # Load all datasets
    all_datasets = [MTBDataset(p, feature_mask) for p in dataset_files]

    # Video balancing via WeightedRandomSampler
    sample_weights = []
    for ds in all_datasets:
        ds_len = len(ds)
        weight_per_sample = 1.0 / ds_len
        sample_weights.extend([weight_per_sample] * ds_len)

    combined_dataset = ConcatDataset(all_datasets)

    # Fit Target Scaler on full combined dataset
    full_y = torch.cat([ds.y for ds in all_datasets])
    scaler = TargetScaler()
    scaler.fit(full_y)
    print(f"[+] Target Scaler Fitted -> Mean: {scaler.mean:.4f}, Std: {scaler.std:.4f}")

    num_features = all_datasets[0].X.shape[-1]

    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True
    )

    loader = DataLoader(combined_dataset, batch_size=args.batch_size, sampler=sampler, shuffle=False)

    model = MTBConvNet(in_channels=num_features, dropout=args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    criterion = nn.HuberLoss(delta=1.0)

    print(f"[+] Training for {args.epochs} epochs...")
    model.train()
    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_scaled = scaler.transform(y_batch).to(device)

            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_scaled)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(y_batch)

        scheduler.step()
        avg_loss = total_loss / len(combined_dataset)

        if epoch % 5 == 0 or epoch == args.epochs:
            print(f"Epoch {epoch:2d}/{args.epochs:2d} | Training Huber Loss (Scaled): {avg_loss:.4f}")

    # Save model and scaler metadata
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "mtb_interest_model.pt"
    scaler_path = output_dir / "mtb_scaler.pt"

    torch.save({
        "model_state_dict": model.state_dict(),
        "in_channels": num_features,
        "feature_mask": feature_mask
    }, model_path)

    torch.save(scaler.state_dict(), scaler_path)

    print("\n" + "=" * 60)
    print(f"[SUCCESS] Production model saved to: {model_path}")
    print(f"[SUCCESS] Target scaler saved to:    {scaler_path}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Final MTB Interest Production Model")
    parser.add_argument("--data-dir", type=str, default=r"C:\VideoTools\MTB-Video-Tools\output\datasets")
    parser.add_argument("--feature-file", type=str, default=r"C:\VideoTools\MTB-Video-Tools\data\features\mtb_training_features.txt")
    parser.add_argument("--output-dir", type=str, default=r"C:\VideoTools\MTB-Video-Tools\output\models")
    parser.add_argument("--epochs", type=int, default=15, help="Epoch count (12-15 optimal based on LOVO best epochs)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--dropout", type=float, default=0.4)

    args = parser.parse_args()
    train_production_model(args)