import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset

# ==============================================================================
# MODEL CONFIGURATION CONSTANTS
# ==============================================================================
DEFAULT_HIDDEN_DIM = 128
CLASSIFIER_MID_DIM = 64
DEFAULT_NUM_CLASSES = 3
DEFAULT_DROPOUT_RATE = 0.4

# ==============================================================================
# DATASET DEFINITION
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

        # Convert continuous target scores into 3 class indices: 0, 1, 2
        y_class = np.zeros(len(y), dtype=np.int64)
        y_class[(y >= 0.5) & (y < 2.5)] = 1  # Level 1/2 (Flow)
        y_class[y >= 2.5]               = 2  # Level 3 (Technical)

        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y_class, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ==============================================================================
# MODEL ARCHITECTURE
# ==============================================================================
class MTBClassifier(nn.Module):
    def __init__(self, feature_dim, hidden_dim=DEFAULT_HIDDEN_DIM, num_classes=DEFAULT_NUM_CLASSES, dropout=DEFAULT_DROPOUT_RATE):
        super().__init__()
        self.backbone = nn.LSTM(
            input_size=feature_dim, 
            hidden_size=hidden_dim, 
            batch_first=True, 
            bidirectional=True
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, CLASSIFIER_MID_DIM),
            nn.ReLU(),
            nn.Linear(CLASSIFIER_MID_DIM, num_classes)  # [L0, L1/2, L3]
        )

    def forward(self, x):
        out, _ = self.backbone(x)
        pooled = torch.mean(out, dim=1)  # Mean pooling over sequence time dimension T
        logits = self.classifier(pooled)
        return logits


# ==============================================================================
# PRODUCTION TRAINING FUNCTION
# ==============================================================================
def train_production_model(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[+] Active Device: {device}")

    data_path = Path(args.data_dir)
    dataset_files = sorted(list(data_path.glob("*.npz")))

    if not dataset_files:
        raise FileNotFoundError(f"No .npz files found in: {args.data_dir}")

    print(f"[+] Loading {len(dataset_files)} dataset files...")

    feature_mask = None
    if args.feature_file and Path(args.feature_file).exists():
        with open(args.feature_file, "r") as f:
            feature_mask = [line.strip() for line in f if line.strip()]
        print(f"[+] Feature whitelist loaded ({len(feature_mask)} features).")

    # Combine all .npz datasets into one ConcatDataset
    individual_datasets = [MTBDataset(p, feature_mask) for p in dataset_files]
    dataset = ConcatDataset(individual_datasets)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Infer input feature dimension dynamically
    sample_X, _ = dataset[0]
    feature_dim = sample_X.shape[-1]

    # Class Weights: Penalize missing Level 3 (Class 2) by 5.0x
    class_weights = torch.tensor([0.2, 1.0, 5.0], device=device)

    model = MTBClassifier(
        feature_dim=feature_dim, 
        hidden_dim=128, 
        num_classes=3, 
        dropout=args.dropout
    ).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=args.weight_decay
    )

    best_loss = float('inf')
    best_model_state = None
    best_epoch = 0

    print(f"[+] Training for {args.epochs} epochs...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0

        for batch_X, batch_y in dataloader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
        
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            total_loss += loss.item() * len(batch_y)

        avg_loss = total_loss / len(dataset)

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_state = model.state_dict().copy()
            best_epoch = epoch

        if epoch % 5 == 0 or epoch == args.epochs:
            print(f"Epoch {epoch:2d}/{args.epochs:2d} | Cross-Entropy Loss: {avg_loss:.4f} | Best Loss: {best_loss:.4f} (Epoch {best_epoch})")

    # Save Checkpoint
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "mtb_interest_model.pt"

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"[+] Restored Best Model from Epoch {best_epoch} with Cross-Entropy Loss: {best_loss:.4f}")

    torch.save({
        "model_state_dict": model.state_dict(),
        "feature_dim": feature_dim,
        "feature_mask": feature_mask,
        "num_classes": 3
    }, model_path)

    print("\n" + "=" * 60)
    print(f"[SUCCESS] Model successfully saved to: {model_path}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MTB 3-Class Classifier")
    parser.add_argument("--data-dir", type=str, default=r"C:\VideoTools\MTB-Video-Tools\output\datasets")
    parser.add_argument("--feature-file", type=str, default=r"C:\VideoTools\MTB-Video-Tools\data\features\mtb_training_features.txt")
    parser.add_argument("--output-dir", type=str, default=r"C:\VideoTools\MTB-Video-Tools\output\models")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--dropout", type=float, default=0.4)

    args = parser.parse_args()
    train_production_model(args)