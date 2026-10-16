from __future__ import annotations

import numpy as np
import pytest

from tools.plot_window_probabilities import annotation_color, validate_plot_data


@pytest.mark.parametrize("score", [0, 1, 2, 3])
def test_annotation_color_supports_all_human_scores(score):
    assert annotation_color(score).startswith("#")


@pytest.mark.parametrize("score", [-1, 4, "2"])
def test_annotation_color_rejects_unknown_scores(score):
    with pytest.raises(ValueError, match="Annotation score"):
        annotation_color(score)


def test_validate_plot_data_accepts_three_class_probabilities():
    timestamps = np.asarray([[0.0, 4.0], [2.0, 6.0]])
    probabilities = np.asarray([[0.7, 0.2, 0.1], [0.1, 0.3, 0.6]])
    validate_plot_data(timestamps, probabilities)


def test_validate_plot_data_rejects_wrong_probability_shape():
    timestamps = np.asarray([[0.0, 4.0]])
    probabilities = np.asarray([[0.5, 0.5]])
    with pytest.raises(ValueError, match="shape"):
        validate_plot_data(timestamps, probabilities)


def test_validate_plot_data_rejects_invalid_probabilities():
    timestamps = np.asarray([[0.0, 4.0]])
    probabilities = np.asarray([[1.2, -0.1, 0.0]])
    with pytest.raises(ValueError, match="between 0 and 1"):
        validate_plot_data(timestamps, probabilities)
