from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

# The plotting helper is intentionally a standalone script in tools/, not a
# Python package. Load it by file path rather than importing a tools package.
SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "tools"
    / "plot_window_probabilities.py"
)
SPEC = importlib.util.spec_from_file_location("plot_window_probabilities", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:
    raise ImportError(f"Could not load plotting helper from {SCRIPT_PATH}")
PLOT_HELPER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PLOT_HELPER)

annotation_color = PLOT_HELPER.annotation_color
validate_plot_data = PLOT_HELPER.validate_plot_data


@pytest.mark.parametrize("score", [0, 1, 2, 3])
def test_annotation_color_supports_all_human_scores(score):
    assert annotation_color(score).startswith("#")


@pytest.mark.parametrize("score", [-1, 4, "2"])
def test_annotation_color_rejects_unknown_scores(score):
    with pytest.raises(ValueError, match="Annotation score"):
        annotation_color(score)


def test_validate_plot_data_accepts_raw_ranking_scores():
    timestamps = np.asarray([[0.0, 4.0], [2.0, 6.0]])
    scores = np.asarray([-1.8, 2.4])
    validate_plot_data(timestamps, scores)


def test_validate_plot_data_rejects_wrong_score_shape():
    timestamps = np.asarray([[0.0, 4.0]])
    scores = np.asarray([[0.3, 0.3, 0.4]])
    with pytest.raises(ValueError, match="shape"):
        validate_plot_data(timestamps, scores)


def test_validate_plot_data_accepts_negative_and_unbounded_raw_scores():
    timestamps = np.asarray([[0.0, 4.0], [2.0, 6.0]])
    scores = np.asarray([-3.5, 8.2])
    validate_plot_data(timestamps, scores)
