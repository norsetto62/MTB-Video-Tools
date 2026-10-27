"""Single-logit MTB interest model for positive-unlabeled learning.

The model accepts feature windows shaped (batch, time, features) and returns
one raw logit per window. Positive logits indicate membership in P.
"""
from __future__ import annotations

import torch
from torch import nn


class MTBClassifier(nn.Module):
    """Bidirectional LSTM that emits one P-vs-U logit per window."""

    def __init__(
        self,
        feature_dim: int = 54,
        hidden_dim: int = 128,
        num_classes: int = 1,
        dropout: float = 0.4,
        classifier_mid_dim: int = 64,
    ) -> None:
        super().__init__()
        for name, value in (
            ("feature_dim", feature_dim),
            ("hidden_dim", hidden_dim),
            ("classifier_mid_dim", classifier_mid_dim),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        if num_classes != 1 or isinstance(num_classes, bool):
            raise ValueError("The P/U model must have exactly one output logit.")
        if isinstance(dropout, bool) or not isinstance(dropout, (int, float)):
            raise ValueError("dropout must be a number in the range [0, 1).")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("dropout must be in the range [0, 1).")
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_classes = 1
        self.dropout_probability = float(dropout)
        self.classifier_mid_dim = classifier_mid_dim
        self.backbone = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(self.dropout_probability),
            nn.Linear(hidden_dim * 2, classifier_mid_dim),
            nn.ReLU(),
            nn.Linear(classifier_mid_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw P-vs-U logits with shape (batch, 1)."""
        if not isinstance(x, torch.Tensor):
            raise TypeError("x must be a torch.Tensor.")
        if x.ndim != 3:
            raise ValueError("x must be a 3D tensor with shape (batch_size, sequence_length, feature_dim).")
        if x.shape[0] == 0:
            raise ValueError("x must contain at least one sequence.")
        if x.shape[1] == 0:
            raise ValueError("x must contain at least one time step.")
        if x.shape[2] != self.feature_dim:
            raise ValueError(f"x must have {self.feature_dim} features per time step; got {x.shape[2]}.")
        if not x.is_floating_point():
            raise TypeError("x must have a floating-point dtype.")
        expected_dtype = self.backbone.weight_ih_l0.dtype
        if x.dtype != expected_dtype:
            raise TypeError(f"x must have dtype {expected_dtype}; got {x.dtype}.")
        if not torch.isfinite(x).all().item():
            raise ValueError("x must contain only finite values (no NaN or Inf).")
        sequence_output, _ = self.backbone(x)
        return self.classifier(torch.mean(sequence_output, dim=1))


# ---------------------------------------------------------------------------
# LightGBM multiclass model configuration. The legacy P/U neural model above
# remains available until production pipelines are migrated.
# ---------------------------------------------------------------------------

from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class LightGBMConfig:
    """Validated parameters for four-class LightGBM training."""

    objective: str = "multiclass"
    task: str = "train"
    num_class: int = 4
    num_leaves: int = 31
    learning_rate: float = 0.03
    max_depth: int = 6
    min_data_in_leaf: int = 100
    num_iterations: int = 2000
    feature_fraction: float = 0.8
    bagging_fraction: float = 0.8
    bagging_frequency: int = 1
    lambda_l1: float = 0.1
    lambda_l2: float = 0.1
    min_split_gain: float = 0.0
    early_stopping_rounds: int = 100
    random_seed: int = 42
    verbosity: int = -1

    def __post_init__(self) -> None:
        if self.objective != "multiclass":
            raise ValueError("objective must be 'multiclass'.")
        if self.task not in {"train", "predict"}:
            raise ValueError("task must be 'train' or 'predict'.")
        if self.num_class != 4:
            raise ValueError("num_class must be 4 for annotation classes 0, 1, 2, 3.")
        for name in (
            "num_leaves", "max_depth", "min_data_in_leaf",
            "num_iterations", "bagging_frequency", "early_stopping_rounds",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        if isinstance(self.random_seed, bool) or not isinstance(self.random_seed, int):
            raise ValueError("random_seed must be an integer.")
        for name in (
            "learning_rate", "feature_fraction", "bagging_fraction",
            "lambda_l1", "lambda_l2", "min_split_gain",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be a finite number.")
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite.")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if not 0 < self.feature_fraction <= 1:
            raise ValueError("feature_fraction must be in (0, 1].")
        if not 0 < self.bagging_fraction <= 1:
            raise ValueError("bagging_fraction must be in (0, 1].")
        if self.lambda_l1 < 0 or self.lambda_l2 < 0 or self.min_split_gain < 0:
            raise ValueError("Regularization and min_split_gain must be non-negative.")

    def to_params(self) -> dict[str, object]:
        """Return LightGBM parameter names accepted by lightgbm.train()."""
        params = asdict(self)
        params.pop("early_stopping_rounds")
        params["metric"] = "multi_logloss"
        params["verbosity"] = self.verbosity
        return params
