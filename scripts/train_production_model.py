#!/usr/bin/env python3

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset


# ==============================================================================
# CONFIGURATION
# ==============================================================================

DEFAULT_HIDDEN_DIM = 128
CLASSIFIER_MID_DIM = 64
DEFAULT_NUM_CLASSES = 3
DEFAULT_DROPOUT = 0.4

# Production starting point based on the 2 FPS LOVO experiments.
DEFAULT_CLASS_WEIGHTS = [0.2, 1.0, 6.5]
DEFAULT_LABEL_SMOOTHING = 0.1

DEFAULT_EPOCHS = 15
DEFAULT_BATCH_SIZE = 32
DEFAULT_LR = 1e-3
DEFAULT_WEIGHT_DECAY = 1e-2

DEFAULT_DATA_DIR = Path(
    r"C:\VideoTools\MTB-Video-Tools\output\datasets"
)
DEFAULT_FEATURE_FILE = Path(
    r"C:\VideoTools\MTB-Video-Tools\data\features\mtb_training_features.txt"
)
DEFAULT_OUTPUT_DIR = Path(
    r"C:\VideoTools\MTB-Video-Tools\output\models"
)

DEFAULT_SEED = 42


# ==============================================================================
# REPRODUCIBILITY
# ==============================================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ==============================================================================
# DATASET
# ==============================================================================

class MTBDataset(Dataset):
    """
    Load one generated MTB dataset.

    Continuous targets 0..3 are converted to the production
    three-class formulation:

        class 0: target < 0.5
        class 1: 0.5 <= target < 2.5
        class 2: target >= 2.5

    All MTB0 examples remain in training as hard negatives.
    """

    def __init__(self, npz_file, feature_mask=None):
        data = np.load(npz_file, allow_pickle=True)

        X = data["X"]
        y = data["y"]

        if feature_mask is not None and "feature_names" in data:
            all_names = list(data["feature_names"])

            valid_indices = [
                all_names.index(name)
                for name in feature_mask
                if name in all_names
            ]

            if valid_indices:
                X = X[:, :, valid_indices]

        y_class = np.zeros(len(y), dtype=np.int64)

        y_class[
            (y >= 0.5) & (y < 2.5)
        ] = 1

        y_class[
            y >= 2.5
        ] = 2

        self.X = torch.tensor(
            X,
            dtype=torch.float32,
        )

        self.y = torch.tensor(
            y_class,
            dtype=torch.long,
        )

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.X[index], self.y[index]


# ==============================================================================
# MODEL
# ==============================================================================

class MTBClassifier(nn.Module):

    def __init__(
        self,
        feature_dim,
        hidden_dim=DEFAULT_HIDDEN_DIM,
        num_classes=DEFAULT_NUM_CLASSES,
        dropout=DEFAULT_DROPOUT,
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

            nn.Linear(
                hidden_dim * 2,
                CLASSIFIER_MID_DIM,
            ),

            nn.ReLU(),

            nn.Linear(
                CLASSIFIER_MID_DIM,
                num_classes,
            ),
        )

    def forward(self, x):

        out, _ = self.backbone(x)

        pooled = torch.mean(
            out,
            dim=1,
        )

        logits = self.classifier(
            pooled
        )

        return logits


# ==============================================================================
# PRODUCTION TRAINING
# ==============================================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
):
    model.train()

    total_loss = 0.0
    total_samples = 0

    for X, y in loader:

        X = X.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        logits = model(X)

        loss = criterion(
            logits,
            y,
        )

        loss.backward()

        optimizer.step()

        batch_size = len(y)

        total_loss += (
            loss.item()
            * batch_size
        )

        total_samples += batch_size

    return total_loss / total_samples


def train_production_model(args):

    set_seed(args.seed)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"[+] Active Device: {device}"
    )

    # --------------------------------------------------------------------------
    # Data
    # --------------------------------------------------------------------------

    data_path = Path(args.data_dir)

    dataset_files = sorted(
        data_path.glob("*.npz")
    )

    if not dataset_files:
        raise FileNotFoundError(
            f"No .npz files found in: {data_path}"
        )

    print(
        f"[+] Loading {len(dataset_files)} dataset files..."
    )

    feature_mask = None

    if (
        args.feature_file
        and Path(args.feature_file).exists()
    ):
        with open(
            args.feature_file,
            "r",
            encoding="utf-8",
        ) as file:

            feature_mask = [
                line.strip()
                for line in file
                if line.strip()
            ]

        print(
            f"[+] Feature whitelist loaded "
            f"({len(feature_mask)} features)."
        )

    else:
        print(
            "[!] Feature whitelist not found; "
            "using all features."
        )

    individual_datasets = [
        MTBDataset(
            path,
            feature_mask,
        )
        for path in dataset_files
    ]

    dataset = ConcatDataset(
        individual_datasets
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )

    sample_X, _ = dataset[0]

    feature_dim = sample_X.shape[-1]

    print(
        f"[+] Total training examples: "
        f"{len(dataset)}"
    )

    print(
        f"[+] Feature dimension: "
        f"{feature_dim}"
    )

    # --------------------------------------------------------------------------
    # Model
    # --------------------------------------------------------------------------

    model = MTBClassifier(
        feature_dim=feature_dim,
        hidden_dim=args.hidden_dim,
        num_classes=args.num_classes,
        dropout=args.dropout,
    ).to(device)

    class_weights = torch.tensor(
        args.class_weights,
        dtype=torch.float32,
        device=device,
    )

    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=args.label_smoothing,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # --------------------------------------------------------------------------
    # Training
    #
    # There is deliberately no validation split here. Production training uses
    # all available annotated videos. Best epoch is selected by training loss
    # only, matching the established LOVO checkpoint convention without using
    # any held-out data.
    # --------------------------------------------------------------------------

    best_train_loss = float("inf")
    best_state = None
    best_epoch = 0

    print(
        f"[+] Training for {args.epochs} epochs..."
    )

    for epoch in range(
        1,
        args.epochs + 1,
    ):

        train_loss = train_one_epoch(
            model,
            dataloader,
            optimizer,
            criterion,
            device,
        )

        if train_loss < best_train_loss:

            best_train_loss = train_loss
            best_epoch = epoch

            best_state = {
                key: value.detach().cpu().clone()
                for key, value
                in model.state_dict().items()
            }

        if (
            epoch == 1
            or epoch % 5 == 0
            or epoch == args.epochs
        ):
            print(
                f"    Epoch "
                f"{epoch:2d}/{args.epochs} | "
                f"Train loss: "
                f"{train_loss:.4f} | "
                f"Best loss: "
                f"{best_train_loss:.4f} "
                f"(Epoch {best_epoch})"
            )

    if best_state is None:
        raise RuntimeError(
            "Training completed without a best model state."
        )

    model.load_state_dict(
        best_state
    )

    print(
        f"[+] Restored best model from "
        f"Epoch {best_epoch} with training loss "
        f"{best_train_loss:.4f}"
    )

    # --------------------------------------------------------------------------
    # Checkpoint
    #
    # Schema 1.1 is directly consumable by extract_video_highlights.py.
    # Production training is fixed to the 2 FPS dataset, but the temporal
    # values are read from dataset metadata so the checkpoint records the
    # actual generated dataset configuration.
    # --------------------------------------------------------------------------

    first_dataset = np.load(
        dataset_files[0],
        allow_pickle=True,
    )

    if "metadata" not in first_dataset:
        raise ValueError(
            f"Dataset {dataset_files[0]} does not contain metadata."
        )

    metadata = first_dataset["metadata"].item()

    if (
        "window" not in metadata
        or "stride" not in metadata
    ):
        raise ValueError(
            f"Dataset {dataset_files[0]} is missing "
            "window/stride metadata."
        )

    window_duration = float(
        metadata["window"]
    )

    window_stride = float(
        metadata["stride"]
    )

    flow_fps = float(
        metadata.get("flow_fps", 2.0)
    )

    if abs(flow_fps - 2.0) > 1e-6:
        raise ValueError(
            f"Production training expects 2 FPS data, "
            f"but dataset metadata reports {flow_fps} FPS."
        )

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_path = (
        output_dir
        / "mtb_interest_model.pt"
    )

    checkpoint = {
        "checkpoint_version": "1.1",

        "model_state_dict": model.state_dict(),

        "feature_dim": feature_dim,

        "feature_mask": feature_mask,

        "model_config": {
            "hidden_dim": args.hidden_dim,
            "classifier_mid_dim": CLASSIFIER_MID_DIM,
            "num_classes": args.num_classes,
            "dropout": args.dropout,
        },

        "training_config": {
            "mode": "production",
            "class_weights": list(
                args.class_weights
            ),
            "label_smoothing": args.label_smoothing,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "seed": args.seed,
            "best_epoch": best_epoch,
            "best_train_loss": best_train_loss,
            "dataset_count": len(dataset_files),
            "example_count": len(dataset),
        },

        "data_config": {
            "flow_fps": flow_fps,
            "window": window_duration,
            "stride": window_stride,
        },
    }

    torch.save(
        checkpoint,
        model_path,
    )

    print()
    print("=" * 70)
    print(
        f"[SUCCESS] Production model saved to: "
        f"{model_path}"
    )
    print("=" * 70)


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Train the production MTB 3-class "
            "interest classifier on all annotated "
            "2 FPS datasets."
        )
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
    )

    parser.add_argument(
        "--feature-file",
        type=Path,
        default=DEFAULT_FEATURE_FILE,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=DEFAULT_HIDDEN_DIM,
    )

    parser.add_argument(
        "--num-classes",
        type=int,
        default=DEFAULT_NUM_CLASSES,
    )

    parser.add_argument(
        "--class-weights",
        type=float,
        nargs=3,
        default=DEFAULT_CLASS_WEIGHTS,
        metavar=("W0", "W1", "W2"),
    )

    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=DEFAULT_LABEL_SMOOTHING,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=DEFAULT_LR,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=DEFAULT_WEIGHT_DECAY,
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=DEFAULT_DROPOUT,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
    )

    args = parser.parse_args()

    train_production_model(args)