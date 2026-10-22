from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "tools" / "plot_feature.py"
SPEC = importlib.util.spec_from_file_location("plot_feature", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:
    raise ImportError(f"Could not load plotting helper from {SCRIPT_PATH}")
PLOT_HELPER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PLOT_HELPER)


@pytest.mark.parametrize("index", [0, 17, 53])
def test_resolve_feature_accepts_zero_based_indices(index):
    resolved_index, name = PLOT_HELPER.resolve_feature(str(index))
    assert resolved_index == index
    assert name == PLOT_HELPER.FEATURE_NAMES[index]


def test_resolve_feature_accepts_canonical_name():
    index, name = PLOT_HELPER.resolve_feature("flow_mean")
    assert index == PLOT_HELPER.FEATURE_NAMES.index("flow_mean")
    assert name == "flow_mean"


@pytest.mark.parametrize("feature", ["", "-1", "54", "not_a_feature"])
def test_resolve_feature_rejects_invalid_feature(feature):
    with pytest.raises(ValueError):
        PLOT_HELPER.resolve_feature(feature)


def test_plot_feature_writes_image_with_annotation_background(tmp_path):
    pytest.importorskip("matplotlib")
    from autocut.data_models import Annotation

    output = tmp_path / "feature.png"
    saved = PLOT_HELPER.plot_feature(
        np.asarray([1.0, 2.0, 3.0], dtype=np.float64),
        np.asarray([0.2, 0.7, 0.4], dtype=np.float64),
        [
            Annotation(0.0, 2.0, 0, "uninteresting"),
            Annotation(2.0, 4.0, 3, "positive"),
        ],
        feature_name="flow_mean",
        video_name="test.mp4",
        output_path=output,
    )
    assert saved == output
    assert output.is_file()
    assert output.stat().st_size > 0


def test_plot_feature_rejects_mismatched_shapes(tmp_path):
    pytest.importorskip("matplotlib")
    with pytest.raises(ValueError, match="same shape"):
        PLOT_HELPER.plot_feature(
            np.asarray([1.0, 2.0]),
            np.asarray([0.2]),
            [],
            feature_name="flow_mean",
            video_name="test.mp4",
            output_path=tmp_path / "invalid.png",
        )
