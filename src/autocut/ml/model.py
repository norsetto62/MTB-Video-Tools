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
# LightGBM learning-to-rank model configuration.
# ---------------------------------------------------------------------------

from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class LightGBMConfig:
    """Validated parameters for LightGBM ranking over per-video window queries."""

    objective: str = "lambdarank"
    task: str = "train"
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
    ndcg_eval_at: tuple[int, ...] = (10,)
    random_seed: int = 42
    verbosity: int = -1

    def __post_init__(self) -> None:
        if self.objective not in {"lambdarank", "rank_xendcg"}:
            raise ValueError("objective must be 'lambdarank' or 'rank_xendcg'.")
        if self.task not in {"train", "predict"}:
            raise ValueError("task must be 'train' or 'predict'.")
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
        try:
            cutoffs = tuple(self.ndcg_eval_at)
        except TypeError as exc:
            raise ValueError("ndcg_eval_at must be a non-empty sequence of positive integers.") from exc
        if not cutoffs or any(
            isinstance(k, bool) or not isinstance(k, int) or k <= 0 for k in cutoffs
        ) or len(set(cutoffs)) != len(cutoffs):
            raise ValueError("ndcg_eval_at must contain unique positive integers.")
        object.__setattr__(self, "ndcg_eval_at", cutoffs)

    def to_params(self) -> dict[str, object]:
        """Return LightGBM ranking parameters."""
        params = asdict(self)
        params.pop("early_stopping_rounds")
        params.pop("task")
        params.pop("ndcg_eval_at")
        params["metric"] = "ndcg"
        params["ndcg_eval_at"] = list(self.ndcg_eval_at)
        params["verbosity"] = self.verbosity
        return params


class LightGBMRanker:
    """Typed wrapper that returns one raw ranking score per feature row."""

    def __init__(self, booster: object, *, feature_names: tuple[str, ...] | list[str] | None = None) -> None:
        if not callable(getattr(booster, "predict", None)):
            raise TypeError("booster must provide LightGBM's predict method.")
        self.booster = booster
        self.feature_names = tuple(feature_names or ())
        self.feature_dim = len(self.feature_names) if self.feature_names else int(
            getattr(booster, "num_feature")()
        )

    def predict(self, X):
        """Return finite raw ranking scores with shape (N,)."""
        import numpy as np

        values = np.asarray(X, dtype=np.float32)
        if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] != self.feature_dim:
            raise ValueError(f"X must have shape (N, {self.feature_dim}) with N > 0.")
        if not np.isfinite(values).all():
            raise ValueError("X must contain only finite values.")
        scores = np.asarray(self.booster.predict(values), dtype=np.float64).reshape(-1)
        if scores.shape != (len(values),) or not np.isfinite(scores).all():
            raise ValueError("LightGBM ranker must return one finite score per row.")
        return scores.astype(np.float32, copy=False)


class LightGBMClassifier:
    """Compatibility wrapper for legacy four-class LightGBM checkpoints."""

    def __init__(self, booster: object, *, feature_names: tuple[str, ...] | list[str] | None = None) -> None:
        if not callable(getattr(booster, "predict", None)):
            raise TypeError("booster must provide LightGBM's predict method.")
        self.booster = booster
        self.feature_names = tuple(feature_names or ())
        self.num_classes = 4
        self.feature_dim = len(self.feature_names) if self.feature_names else int(
            getattr(booster, "num_feature")()
        )

    def predict_proba(self, X):
        """Return a finite (N, 4) matrix of class probabilities."""
        import numpy as np

        values = np.asarray(X, dtype=np.float32)
        if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] != self.feature_dim:
            raise ValueError(f"X must have shape (N, {self.feature_dim}) with N > 0.")
        if not np.isfinite(values).all():
            raise ValueError("X must contain only finite values.")
        probabilities = np.asarray(self.booster.predict(values), dtype=np.float64)
        if probabilities.ndim == 1 and probabilities.size == len(values) * 4:
            probabilities = probabilities.reshape(len(values), 4)
        if probabilities.shape != (len(values), 4):
            raise ValueError(f"LightGBM booster must return shape (N, 4); got {probabilities.shape}.")
        if not np.isfinite(probabilities).all():
            raise ValueError("LightGBM produced non-finite probabilities.")
        row_sums = probabilities.sum(axis=1)
        if np.any(probabilities < -1e-8) or not np.allclose(row_sums, 1.0, atol=1e-5):
            raise ValueError("LightGBM predictions are not valid class probabilities.")
        return probabilities.astype(np.float32, copy=False)
