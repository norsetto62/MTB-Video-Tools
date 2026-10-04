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

DEFAULT_CLASS_WEIGHTS = [0.2, 1.0, 5.0]
DEFAULT_LABEL_SMOOTHING = 0.1

DEFAULT_EPOCHS = 15
DEFAULT_BATCH_SIZE = 32
DEFAULT_LR = 1e-3
DEFAULT_WEIGHT_DECAY = 1e-2

DEFAULT_MODEL_DIR = Path(
    r"C:\VideoTools\MTB-Video-Tools\output\checkpoints"
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

    # Deterministic behavior where possible.
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

        # x:
        #   [batch, time, features]

        out, _ = self.backbone(x)

        # Mean temporal pooling.
        pooled = torch.mean(
            out,
            dim=1,
        )

        logits = self.classifier(
            pooled
        )

        return logits


# ==============================================================================
# METRICS
# ==============================================================================

def calculate_metrics(
    targets,
    predictions,
    num_classes=3,
):
    targets = np.asarray(targets)
    predictions = np.asarray(predictions)

    accuracy = float(
        np.mean(predictions == targets)
    )

    confusion = np.zeros(
        (num_classes, num_classes),
        dtype=np.int64,
    )

    for true, pred in zip(
        targets,
        predictions,
    ):
        confusion[true, pred] += 1

    recalls = []
    precisions = []
    f1s = []

    for cls in range(num_classes):

        tp = confusion[cls, cls]

        fn = (
            np.sum(confusion[cls, :])
            - tp
        )

        fp = (
            np.sum(confusion[:, cls])
            - tp
        )

        if tp + fn > 0:
            recall = tp / (tp + fn)
        else:
            recall = 0.0

        if tp + fp > 0:
            precision = tp / (tp + fp)
        else:
            precision = 0.0

        if precision + recall > 0:
            f1 = (
                2.0
                * precision
                * recall
                / (precision + recall)
            )
        else:
            f1 = 0.0

        recalls.append(recall)
        precisions.append(precision)
        f1s.append(f1)

    balanced_accuracy = float(
        np.mean(recalls)
    )

    macro_f1 = float(
        np.mean(f1s)
    )

    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "macro_f1": macro_f1,
        "precision": precisions,
        "recall": recalls,
        "f1": f1s,
        "confusion": confusion,
    }


# ==============================================================================
# TRAINING
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


# ==============================================================================
# EVALUATION
# ==============================================================================

@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion,
    device,
):
    model.eval()

    total_loss = 0.0
    total_samples = 0

    all_targets = []
    all_predictions = []

    for X, y in loader:

        X = X.to(device)
        y = y.to(device)

        logits = model(X)

        loss = criterion(
            logits,
            y,
        )

        batch_size = len(y)

        total_loss += (
            loss.item()
            * batch_size
        )

        total_samples += batch_size

        predictions = torch.argmax(
            logits,
            dim=1,
        )

        all_targets.extend(
            y.cpu().numpy()
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

    metrics = calculate_metrics(
        all_targets,
        all_predictions,
    )

    metrics["loss"] = (
        total_loss
        / total_samples
    )

    return metrics


# ==============================================================================
# ONE LOVO FOLD
# ==============================================================================

def run_fold(
    dataset_files,
    val_path,
    args,
    device,
    label,
):

    train_paths = [
        path
        for path in dataset_files
        if path != val_path
    ]

    train_datasets = [
        MTBDataset(
            path,
            args.feature_mask,
        )
        for path in train_paths
    ]

    val_dataset = MTBDataset(
        val_path,
        args.feature_mask,
    )

    train_dataset = ConcatDataset(
        train_datasets
    )

    # --------------------------------------------------------------------------
    # Data loaders
    # --------------------------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )

    feature_dim = (
        train_datasets[0]
        .X
        .shape[-1]
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
    # IMPORTANT:
    # Best epoch is selected using TRAINING loss only.
    #
    # The held-out video is therefore never used to decide which
    # epoch/model is selected.
    # --------------------------------------------------------------------------

    best_train_loss = float("inf")
    best_state = None
    best_epoch = 0

    for epoch in range(
        1,
        args.epochs + 1,
    ):

        train_loss = train_one_epoch(
            model,
            train_loader,
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
                f"{train_loss:.4f}"
            )

    # --------------------------------------------------------------------------
    # Restore best training-loss model
    # --------------------------------------------------------------------------

    model.load_state_dict(
        best_state
    )

    metrics = evaluate(
        model,
        val_loader,
        criterion,
        device,
    )

    # --------------------------------------------------------------------------
    # Save the exact best model together with its complete configuration.
    # This checkpoint can be used directly by extract_video_highlights.py.
    # --------------------------------------------------------------------------

    held_out_name = val_path.stem.lower()
    fps_name = label.lower().replace(" ", "")
    checkpoint_path = (
        args.model_dir / f"model_lovo_{held_out_name}_{fps_name}.pt"
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "checkpoint_version": 1,

        "model_state_dict": model.state_dict(),

        "feature_dim": feature_dim,
        "feature_mask": args.feature_mask,

        "model_config": {
            "hidden_dim": args.hidden_dim,
            "classifier_mid_dim": CLASSIFIER_MID_DIM,
            "num_classes": args.num_classes,
            "dropout": args.dropout,
        },

        "training_config": {
            "class_weights": list(args.class_weights),
            "label_smoothing": args.label_smoothing,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "seed": args.seed,
        },

        "lovo": {
            "held_out_video": val_path.stem,
            "held_out_dataset": str(val_path),
            "training_datasets": [
                str(path)
                for path in train_paths
            ],
            "best_epoch": best_epoch,
            "best_train_loss": best_train_loss,
        },
    }

    torch.save(
        checkpoint,
        checkpoint_path,
    )

    print(
        f"    Saved checkpoint: "
        f"{checkpoint_path}"
    )

    return (
        best_epoch,
        best_train_loss,
        metrics,
    )


# ==============================================================================
# COMPLETE LOVO RUN
# ==============================================================================

def run_lovo(
    label,
    data_dir,
    args,
    device,
):

    data_dir = Path(data_dir)

    dataset_files = sorted(
        data_dir.glob("*.npz")
    )

    if len(dataset_files) != 8:
        raise RuntimeError(
            f"{label}: expected 8 dataset files, "
            f"found {len(dataset_files)} in "
            f"{data_dir}"
        )

    print()
    print("=" * 90)
    print(
        f"LOVO EXPERIMENT: {label}"
    )
    print("=" * 90)

    print(
        f"Dataset directory: {data_dir}"
    )

    print(
        f"Videos: {len(dataset_files)}"
    )

    results = []

    for fold_index, val_path in enumerate(
        dataset_files,
        start=1,
    ):

        print()
        print("-" * 90)

        print(
            f"Fold {fold_index}/"
            f"{len(dataset_files)}"
        )

        print(
            f"Held out: "
            f"{val_path.stem}"
        )

        # Same seed at every fold and for both FPS experiments.
        set_seed(
            args.seed
        )

        best_epoch, train_loss, metrics = (
            run_fold(
                dataset_files,
                val_path,
                args,
                device,
                label,
            )
        )

        results.append(
            {
                "video": val_path.stem,
                "epoch": best_epoch,
                "train_loss": train_loss,
                **metrics,
            }
        )

        print(
            f"    Accuracy:          "
            f"{metrics['accuracy']:.4f}"
        )

        print(
            f"    Balanced accuracy: "
            f"{metrics['balanced_accuracy']:.4f}"
        )

        print(
            f"    Macro F1:           "
            f"{metrics['macro_f1']:.4f}"
        )

        print(
            f"    Recall "
            f"[0,1,2]: "
            f"{metrics['recall'][0]:.3f}, "
            f"{metrics['recall'][1]:.3f}, "
            f"{metrics['recall'][2]:.3f}"
        )

        print(
            f"    F1 "
            f"[0,1,2]: "
            f"{metrics['f1'][0]:.3f}, "
            f"{metrics['f1'][1]:.3f}, "
            f"{metrics['f1'][2]:.3f}"
        )

        print(
            "    Confusion matrix:"
        )

        print(
            metrics["confusion"]
        )

    # --------------------------------------------------------------------------
    # Summary
    # --------------------------------------------------------------------------

    print()
    print("=" * 90)
    print(
        f"{label} — LOVO SUMMARY"
    )
    print("=" * 90)

    print(
        f"{'Held-out':<20}"
        f"{'Acc':>9}"
        f"{'BalAcc':>9}"
        f"{'MacroF1':>9}"
        f"{'F1-0':>8}"
        f"{'F1-1':>8}"
        f"{'F1-2':>8}"
    )

    print("-" * 90)

    for result in results:

        print(
            f"{result['video']:<20}"
            f"{result['accuracy']:>9.4f}"
            f"{result['balanced_accuracy']:>9.4f}"
            f"{result['macro_f1']:>9.4f}"
            f"{result['f1'][0]:>8.4f}"
            f"{result['f1'][1]:>8.4f}"
            f"{result['f1'][2]:>8.4f}"
        )

    mean_accuracy = np.mean(
        [r["accuracy"] for r in results]
    )

    mean_balanced_accuracy = np.mean(
        [r["balanced_accuracy"] for r in results]
    )

    mean_macro_f1 = np.mean(
        [r["macro_f1"] for r in results]
    )

    mean_f1 = np.mean(
        [
            r["f1"]
            for r in results
        ],
        axis=0,
    )

    print("-" * 90)

    print(
        f"{'MEAN':<20}"
        f"{mean_accuracy:>9.4f}"
        f"{mean_balanced_accuracy:>9.4f}"
        f"{mean_macro_f1:>9.4f}"
        f"{mean_f1[0]:>8.4f}"
        f"{mean_f1[1]:>8.4f}"
        f"{mean_f1[2]:>8.4f}"
    )

    print("=" * 90)

    return {
        "label": label,
        "results": results,
        "mean_accuracy": mean_accuracy,
        "mean_balanced_accuracy": mean_balanced_accuracy,
        "mean_macro_f1": mean_macro_f1,
        "mean_f1": mean_f1,
    }


# ==============================================================================
# FPS COMPARISON
# ==============================================================================

def print_fps_comparison(
    result_2fps,
    result_10fps,
):

    print()
    print()
    print("#" * 100)
    print("2 FPS vs 10 FPS — LOVO COMPARISON")
    print("#" * 100)

    print()
    print(
        f"{'Video':<20}"
        f"{'2FPS Acc':>11}"
        f"{'10FPS Acc':>12}"
        f"{'Delta':>10}"
        f"{'2FPS F1':>11}"
        f"{'10FPS F1':>12}"
        f"{'Delta':>10}"
    )

    print("-" * 100)

    by_video_2 = {
        r["video"]: r
        for r in result_2fps["results"]
    }

    by_video_10 = {
        r["video"]: r
        for r in result_10fps["results"]
    }

    for video in by_video_2:

        a = by_video_2[video]
        b = by_video_10[video]

        acc_delta = (
            b["accuracy"]
            - a["accuracy"]
        )

        f1_delta = (
            b["macro_f1"]
            - a["macro_f1"]
        )

        print(
            f"{video:<20}"
            f"{a['accuracy']:>11.4f}"
            f"{b['accuracy']:>12.4f}"
            f"{acc_delta:>+10.4f}"
            f"{a['macro_f1']:>11.4f}"
            f"{b['macro_f1']:>12.4f}"
            f"{f1_delta:>+10.4f}"
        )

    print("-" * 100)

    print(
        f"{'MEAN':<20}"
        f"{result_2fps['mean_accuracy']:>11.4f}"
        f"{result_10fps['mean_accuracy']:>12.4f}"
        f"{result_10fps['mean_accuracy'] - result_2fps['mean_accuracy']:>+10.4f}"
        f"{result_2fps['mean_macro_f1']:>11.4f}"
        f"{result_10fps['mean_macro_f1']:>12.4f}"
        f"{result_10fps['mean_macro_f1'] - result_2fps['mean_macro_f1']:>+10.4f}"
    )

    print()
    print(
        "Class-2 F1 (technical/interesting) is especially important:"
    )

    print(
        f"  2 FPS : "
        f"{result_2fps['mean_f1'][2]:.4f}"
    )

    print(
        f"  10 FPS: "
        f"{result_10fps['mean_f1'][2]:.4f}"
    )

    print(
        f"  Delta : "
        f"{result_10fps['mean_f1'][2] - result_2fps['mean_f1'][2]:+.4f}"
    )

    print("#" * 100)


# ==============================================================================
# MAIN
# ==============================================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Compare 2 FPS and 10 FPS MTB "
            "interest classifiers using "
            "leave-one-video-out cross-validation."
        )
    )

    parser.add_argument(
        "--data-dir-2fps",
        type=Path,
        default=Path(
            r"C:\VideoTools\MTB-Video-Tools\output\datasets_2fps"
        ),
    )

    parser.add_argument(
        "--data-dir-10fps",
        type=Path,
        default=Path(
            r"C:\VideoTools\MTB-Video-Tools\output\datasets_10fps"
        ),
    )

    parser.add_argument(
        "--feature-file",
        type=Path,
        default=Path(
            r"C:\VideoTools\MTB-Video-Tools\data\features\mtb_training_features.txt"
        ),
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
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
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

    if args.feature_file.exists():

        with open(
            args.feature_file,
            "r",
            encoding="utf-8",
        ) as file:

            args.feature_mask = [
                line.strip()
                for line in file
                if line.strip()
            ]

        print(
            f"[+] Feature whitelist: "
            f"{len(args.feature_mask)} features"
        )

    else:

        args.feature_mask = None

        print(
            "[!] Feature whitelist not found; "
            "using all features."
        )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"[+] Device: {device}"
    )

    result_2fps = run_lovo(
        "2 FPS",
        args.data_dir_2fps,
        args,
        device,
    )

    result_10fps = run_lovo(
        "10 FPS",
        args.data_dir_10fps,
        args,
        device,
    )

    print_fps_comparison(
        result_2fps,
        result_10fps,
    )


if __name__ == "__main__":
    main()