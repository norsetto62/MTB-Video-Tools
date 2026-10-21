"""Learning-rate range test using the same non-negative PU risk as training."""
import sys
from typing import Sequence

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from torch_lr_finder import LRFinder

from autocut.ml.model import MTBClassifier
from autocut.dataset.manifest import load_manifest
from autocut.config_loader import load_config
from autocut.pipeline.training import _combine_records, _training_config
from autocut.ml.scaling import FeatureScaler
from autocut.ml.training import _validate_windows, targets_to_classes, _pu_risk
from autocut.cli import build_parser


class PULoss(nn.Module):
    """nnPU criterion; target class 2 is P, classes 0/1 are U."""

    def __init__(self, positive_prior: float) -> None:
        super().__init__()
        self.positive_prior = positive_prior

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return _pu_risk(logits, targets, self.positive_prior)


def main(argv: Sequence[str] | None = None):
    print("Loading PU training data...")
    parser = build_parser()
    args = parser.parse_args(list(sys.argv[1:] if argv is None else argv))
    settings = load_config(args.config)
    data = settings.get("data", {})
    manifest = args.manifest or data.get("manifest", "data/manifest/annotations_manifest.yaml")
    records = load_manifest(manifest, verify_files=True)
    train_groups = set(data.get("train_groups", ()))
    train_records = tuple(record for record in records if record.group_id in train_groups)
    if not train_records:
        raise ValueError("No manifest records match training groups for LR finder.")

    raw_windows, raw_targets = _combine_records(train_records, settings)
    train_X = _validate_windows(raw_windows, name="train_windows")
    train_y = targets_to_classes(raw_targets)
    positive_indices = np.flatnonzero(train_y == 2)
    unlabeled_indices = np.flatnonzero(train_y != 2)
    if not len(positive_indices) or not len(unlabeled_indices):
        raise ValueError("LR finder requires both pure score-3 positives and score-0/1/2 unlabeled windows.")

    fit_config = _training_config(settings)
    scaler = FeatureScaler(enabled=fit_config.use_scaling)
    transformed = scaler.fit_transform(train_X)

    # Interleave class-conditional samples so each even-sized minibatch has
    # both P and U. Sampling within each set is uniform; no weighted sampler.
    rng = np.random.default_rng(fit_config.seed)
    per_set = max(len(positive_indices), len(unlabeled_indices))
    p = rng.choice(positive_indices, size=per_set, replace=len(positive_indices) < per_set)
    u = rng.choice(unlabeled_indices, size=per_set, replace=len(unlabeled_indices) < per_set)
    order = np.empty(2 * per_set, dtype=np.int64)
    order[0::2], order[1::2] = p, u
    X = torch.from_numpy(np.ascontiguousarray(transformed[order], dtype=np.float32))
    y = torch.from_numpy(train_y[order])
    batch_size = max(2, fit_config.batch_size + fit_config.batch_size % 2)
    loader = DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MTBClassifier(
        feature_dim=train_X.shape[2],
        hidden_dim=fit_config.hidden_dim,
        classifier_mid_dim=fit_config.classifier_mid_dim,
        num_classes=3,
        dropout=fit_config.dropout,
    ).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-7, weight_decay=fit_config.weight_decay)
    criterion = PULoss(fit_config.positive_prior)

    finder = LRFinder(model, optimizer, criterion, device=str(device))
    finder.range_test(loader, end_lr=10, num_iter=100, step_mode="exp")
    import matplotlib.pyplot as plt
    finder.plot()
    plt.savefig("lr_finder_plot.png", bbox_inches="tight", dpi=300)
    finder.reset()
    print("Saved PU learning-rate curve to lr_finder_plot.png")


if __name__ == "__main__":
    main()
