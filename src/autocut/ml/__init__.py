"""Machine-learning components for AutoCut.

Backend-specific symbols are loaded lazily so importing the package does not
pull optional frameworks into processes that do not use them.
"""

from __future__ import annotations

from typing import Any

__all__ = ["MTBClassifier", "FeatureScaler"]


def __getattr__(name: str) -> Any:
    if name == "MTBClassifier":
        from autocut.ml.model import MTBClassifier
        return MTBClassifier
    if name == "FeatureScaler":
        from autocut.ml.scaling import FeatureScaler
        return FeatureScaler
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
