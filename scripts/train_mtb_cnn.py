#!/usr/bin/env python3
"""
train_mtb_cnn.py - End-to-end PyTorch training pipeline for MTB interest regression.

Example usage:
    python train_mtb.py \
        --train-npz data/train_part1.npz data/train_part2.npz \
        --val-npz data/val.npz \
        --epochs 50 \
        --batch-size 32 \
        --lr 1e-3 \
        --loss huber \
        --output-dir ./checkpoints
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


class MTBDataset(Dataset):
    def __init__(self, npz_paths, feature_file=None):
        X_list, y_list = [], []

        # Load feature whitelist if provided
        selected_features = None
        if feature_file and Path(feature_file).exists():
            with open(feature_file, "r") as f:
                selected_features = [
                    line.strip() for line in f if line.strip() and not line.startswith("#")
                ]
            print(f"[Dataset] Loaded {len(selected_features)} features from whitelist: {feature_file}")
        elif feature_file:
            print(f"[Warning] Feature file specified but not found: {feature_file}")

        feature_indices = None

        for path in npz_paths:
            data = np.load(path, allow_pickle=True)
            X = data["X"]  # Shape: (N, seq_len, num_features)
            y = data["y"]  # Shape: (N,)

            if selected_features is not None:
                if "feature_names" in data and feature_indices is None:
                    all_names = list(data["feature_names"])
                    feature_indices = [
                        all_names.index(feat) for feat in selected_features if feat in all_names
                    ]
                    missing = set(selected_features) - set(all_names)
                    if missing:
                        print(f"[Warning] {len(missing)} features in whitelist not found in NPZ.")
                    print(f"[Dataset] Filtering input channels: {X.shape[-1]} -> {len(feature_indices)}")
                
                if feature_indices is not None:
                    X = X[:, :, feature_indices]
                else:
                    # Fallback to positional slicing if feature_names array isn't inside NPZ
                    X = X[:, :, : len(selected_features)]

            X_list.append(X)
            y_list.append(y)

        self.X = torch.tensor(np.concatenate(X_list, axis=0), dtype=torch.float32)
        self.y = torch.tensor(np.concatenate(y_list, axis=0), dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class MTBInterestRegressor(nn.Module):
    def __init__(self, in_channels, dropout=0.3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.gru = nn.GRU(128, 64, batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        # x shape: (B, T, C) -> Conv1d expects (B, C, T)
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)  # (B, T, 128)
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])  # Take last time step representation
        return out.squeeze(-1)


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        preds = model(X)
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
    return total_loss / len(dataloader.dataset)


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        preds = model(X)
        loss = criterion(preds, y)
        total_loss += loss.item() * len(y)
        all_preds.append(preds.cpu().numpy())
        all_targets.append(y.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    mae = np.mean(np.abs(all_preds - all_targets))
    rmse = np.sqrt(np.mean((all_preds - all_targets) ** 2))
    ss_res = np.sum((all_targets - all_preds) ** 2)
    ss_tot = np.sum((all_targets - np.mean(all_targets)) ** 2)
    r2 = 1.0 - (ss_res / (ss_tot + 1e-8))

    return total_loss / len(dataloader.dataset), mae, rmse, r2


def main():
    parser = argparse.ArgumentParser(description="Train 1D-CNN + BiGRU MTB Regressor")
    parser.add_argument("--train-npz", nargs="+", required=True, help="List of training .npz files")
    parser.add_argument("--val-npz", nargs="+", required=True, help="List of validation .npz files")
    parser.add_argument("--feature-file", type=str, default=None, help="Path to feature whitelist text file")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-3, help="Weight decay L2 regularization")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate")
    parser.add_argument("--loss", choices=["huber", "mse"], default="huber", help="Loss function")
    parser.add_argument("--output-dir", type=str, default="./output/checkpoints", help="Output directory")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing on device: {device}")

    # Load datasets with feature whitelist support
    train_ds = MTBDataset(args.train_npz, feature_file=args.feature_file)
    val_ds = MTBDataset(args.val_npz, feature_file=args.feature_file)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    num_features = train_ds.X.shape[-1]
    print(f"Train Dataset: {len(train_ds)} samples | Seq Length: {train_ds.X.shape[1]} | Features: {num_features}")
    print(f"Val Dataset:   {len(val_ds)} samples")

    model = MTBInterestRegressor(in_channels=num_features, dropout=args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.HuberLoss(delta=0.5) if args.loss == "huber" else nn.MSELoss()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint = output_dir / "mtb_temporal_cnn.pt"

    best_val_loss = float("inf")

    print("\nStarting Training Loop...")
    print("-" * 75)
    print(f"{'Epoch':<7} | {'Train Loss':<10} | {'Val Loss':<10} | {'MAE':<10} | {'RMSE':<10} | {'R2':<8}")
    print("-" * 75)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, mae, rmse, r2 = evaluate(model, val_loader, criterion, device)

        print(f"{epoch:<7} | {train_loss:<10.4f} | {val_loss:<10.4f} | {mae:<10.4f} | {rmse:<10.4f} | {r2:<8.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "mae": mae,
                    "r2": r2,
                    "num_features": num_features,
                },
                best_checkpoint,
            )

    print("-" * 75)
    print(f"Training Complete. Best Val Loss: {best_val_loss:.4f} at epoch: {best_epoch:3d}/{epoch:3d} saved to {best_checkpoint}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[!] Training interrupted by user (^C). Cleaning up GPU context...")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        sys.exit(0)