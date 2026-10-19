import sys
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
from torch_lr_finder import LRFinder

# 1. Import your actual model and project utilities
from autocut.ml.model import MTBClassifier
from autocut.dataset.manifest import load_manifest
from autocut.pipeline.training import _combine_records
from autocut.config_compat import load_training_config
from autocut.ml.scaling import FeatureScaler
from autocut.pipeline.training import _combine_records
from autocut.ml.training import _validate_windows, targets_to_classes
from autocut.cli import build_parser

def main(argv: Sequence[str] | None = None):

    print("Loading configuration and training data...")

    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args = parser.parse_args(arguments)
    
    # Load settings and manifest using your project flow
    settings = load_training_config(args.config)
    manifest = args.manifest or settings.manifest
    
    records = load_manifest(manifest, verify_files=True)
    train_groups = set(settings.train_groups)
    train_records = tuple(record for record in records if record.group_id in train_groups)
    
    if not train_records:
        raise ValueError("No manifest records match training groups for LR finder.")

    print(f"Combining {len(train_records)} training records...")
    raw_windows, raw_targets = _combine_records(train_records, settings)

    # Apply the project's official validation and target mapping
    train_X = _validate_windows(raw_windows, name="train_windows")
    train_y_raw = targets_to_classes(raw_targets)

    # 2. Apply your project's exact FeatureScaler and target formatting
    fit_config = settings.training
    scaler = FeatureScaler(enabled=fit_config.use_scaling)
    transformed_train = scaler.fit_transform(train_X)
    
    train_tensor = torch.from_numpy(np.ascontiguousarray(transformed_train, dtype=np.float32))
    train_label_tensor = torch.from_numpy(train_y_raw.astype(np.int64, copy=False))

    print("Setting up model, class weights, and weighted sampler...")
    
    # 3. Instantiate MTBClassifier (locking in hidden_dim=128 as determined in Study #12)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MTBClassifier(
        feature_dim=train_X.shape[2],
        hidden_dim=128,  # Locked in from Study #12
        classifier_mid_dim=fit_config.classifier_mid_dim,
        num_classes=fit_config.num_classes,
        dropout=fit_config.dropout,
    ).to(device)

    # 4. Set up criterion with your class weights
    class_weights = torch.tensor(fit_config.class_weights, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # 5. Set up AdamW (start lr ultra-low for the finder sweep)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=1e-7,
        weight_decay=fit_config.weight_decay,
    )

    # 6. Replicate your exact WeightedRandomSampler logic
    generator = torch.Generator(device="cpu")
    generator.manual_seed(fit_config.seed)

    class_counts = np.bincount(train_y_raw)
    total_samples = len(train_y_raw)
    num_classes = fit_config.num_classes

    computed_class_weights = np.zeros_like(class_counts, dtype=float)
    for i, count in enumerate(class_counts):
        if count > 0:
            computed_class_weights[i] = total_samples / (num_classes * count)

    sample_weights = computed_class_weights[train_y_raw]
    sample_weights_tensor = torch.DoubleTensor(sample_weights)

    sampler = WeightedRandomSampler(
        weights=sample_weights_tensor.tolist(),
        num_samples=len(sample_weights_tensor),
        replacement=True,
        generator=generator,
    )

    loader = DataLoader(
        TensorDataset(train_tensor, train_label_tensor),
        batch_size=fit_config.batch_size,
        sampler=sampler,
        generator=generator,
    )

    print("Initializing LRFinder and running range test...")
    
    # 7. Run torch-lr-finder with the exact same data loader and configuration
    lr_finder = LRFinder(model, optimizer, criterion, device=device)
    
    # Ramps up lr exponentially over 100 iterations
    lr_finder.range_test(loader, end_lr=10, num_iter=100, step_mode="exp")

    # 8. Plot results (opens matplotlib window)
    import matplotlib.pyplot as plt
    lr_finder.plot()
    plt.savefig("lr_finder_plot.png", bbox_inches="tight", dpi=300)
    print("Saved learning rate curve to lr_finder_plot.png")    

    # 9. Reset model weights so nothing changes permanently
    lr_finder.reset()
    print("\nLR Finder complete! Check the pop-up graph for your optimal learning rate drop.")

if __name__ == "__main__":
    main()