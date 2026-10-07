"""Unit tests for FeatureExtractor."""

import numpy as np
import pytest

from autocut.data_models import OpticalFlow
from autocut.motion.features import (
    FEATURE_NAMES,
    FeatureExtractor,
)


# ---------------------------------------------------------------------------
# Fixtures & Helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_flow() -> OpticalFlow:
    """Create a synthetic uniform optical-flow field."""

    height, width = 50, 100

    # Uniform rightward motion u=4.0, upward motion v=-2.0.
    u = np.full((height, width), 4.0, dtype=np.float32)
    v = np.full((height, width), -2.0, dtype=np.float32)

    return OpticalFlow(u=u, v=v)


def create_expanding_flow(
    height: int,
    width: int,
) -> OpticalFlow:
    """Create the same normalized radial expansion at any resolution."""
    y, x = np.mgrid[0:height, 0:width]

    cy = (height - 1) / 2.0
    cx = (width - 1) / 2.0

    u = (x - cx) / (width - 1) * width
    v = (y - cy) / (height - 1) * height

    return OpticalFlow(
        u=u.astype(np.float32),
        v=v.astype(np.float32),
    )


def create_opposing_flow(
    height: int,
    width: int,
) -> OpticalFlow:
    """Create a flow field with opposing horizontal motion."""

    u = np.ones((height, width), dtype=np.float32)

    midpoint = width // 2
    u[:, midpoint:] = -1.0

    v = np.zeros((height, width), dtype=np.float32)

    return OpticalFlow(u=u, v=v)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_feature_vector_shape_and_type(
    sample_flow: OpticalFlow,
) -> None:
    """Verify extract_vector produces a 1D float32 array of length 54."""

    extractor = FeatureExtractor(
        roi_width=100,
        roi_height=50,
    )

    vector = extractor.extract_vector(sample_flow)

    assert isinstance(vector, np.ndarray)
    assert vector.dtype == np.float32
    assert vector.ndim == 1
    assert vector.shape == (54,)
    assert vector.shape[0] == len(FEATURE_NAMES)


def test_feature_names_ordering(
    sample_flow: OpticalFlow,
) -> None:
    """Verify the named feature dictionary follows the canonical schema."""

    extractor = FeatureExtractor(
        roi_width=100,
        roi_height=50,
    )

    features = extractor.extract(sample_flow)

    assert tuple(features.keys()) == FEATURE_NAMES
    assert extractor.get_feature_names() == FEATURE_NAMES
    assert len(FEATURE_NAMES) == 54


def test_extract_vector_matches_named_features(
    sample_flow: OpticalFlow,
) -> None:
    """Verify extract_vector uses exactly the canonical feature ordering."""

    extractor = FeatureExtractor(
        roi_width=100,
        roi_height=50,
    )

    features = extractor.extract(sample_flow)

    # Reset because extract_vector() is itself stateful.
    extractor.reset()

    vector = extractor.extract_vector(sample_flow)

    expected = np.asarray(
        [features[name] for name in FEATURE_NAMES],
        dtype=np.float32,
    )

    np.testing.assert_array_equal(vector, expected)


def test_state_resetting(
    sample_flow: OpticalFlow,
) -> None:
    """Verify temporal deltas reset at the beginning of a new sequence."""

    extractor = FeatureExtractor(
        roi_width=100,
        roi_height=50,
    )

    # First frame: temporal deltas are zero.
    feat1 = extractor.extract(sample_flow)

    assert feat1["delta_flow_u"] == 0.0
    assert feat1["delta_flow_v"] == 0.0
    assert feat1["delta_flow_mean"] == 0.0
    assert feat1["delta_flow_net"] == 0.0
    assert feat1["delta_div_mean"] == 0.0

    # Second frame: change horizontal flow.
    modified_u = sample_flow.u + 5.0
    modified_flow = OpticalFlow(
        u=modified_u,
        v=sample_flow.v,
    )

    feat2 = extractor.extract(modified_flow)

    assert feat2["delta_flow_u"] == pytest.approx(5.0)
    assert feat2["delta_flow_v"] == pytest.approx(0.0)
    assert feat2["delta_flow_mean"] == pytest.approx(
        feat2["flow_mean"] - feat1["flow_mean"]
    )

    # Reset: the next frame is treated as the first frame
    # of an independent sequence.
    extractor.reset()

    feat3 = extractor.extract(modified_flow)

    assert feat3["delta_flow_u"] == 0.0
    assert feat3["delta_flow_v"] == 0.0
    assert feat3["delta_flow_mean"] == 0.0
    assert feat3["delta_flow_net"] == 0.0
    assert feat3["delta_div_mean"] == 0.0


def test_resolution_invariance_divergence_and_curl() -> None:
    """Verify divergence and curl are invariant across resolutions."""
    flow_low = create_expanding_flow(50, 100)
    ext_low = FeatureExtractor(roi_width=100, roi_height=50)
    feat_low = ext_low.extract(flow_low)

    flow_high = create_expanding_flow(100, 200)
    ext_high = FeatureExtractor(roi_width=200, roi_height=100)
    feat_high = ext_high.extract(flow_high)

    assert feat_low["div_normalized_mean"] == pytest.approx(
        feat_high["div_normalized_mean"],
        rel=1e-2,
    )
    assert feat_low["div_abs_p90"] == pytest.approx(
        feat_high["div_abs_p90"],
        rel=1e-2,
    )
    assert feat_low["curl_std"] == pytest.approx(
        feat_high["curl_std"],
        abs=1e-3,
    )


def test_uniform_flow_global_and_grid_features(
    sample_flow: OpticalFlow,
) -> None:
    """Verify expected values for a uniform flow field."""

    extractor = FeatureExtractor(
        roi_width=100,
        roi_height=50,
    )

    features = extractor.extract(sample_flow)

    expected_magnitude = np.hypot(4.0, -2.0)

    assert features["flow_u"] == pytest.approx(4.0)
    assert features["flow_v"] == pytest.approx(-2.0)
    assert features["flow_mean"] == pytest.approx(expected_magnitude)
    assert features["flow_net"] == pytest.approx(expected_magnitude)

    for row in range(3):
        for col in range(3):
            prefix = f"grid_{row}_{col}"

            assert features[f"{prefix}_u"] == pytest.approx(4.0)
            assert features[f"{prefix}_v"] == pytest.approx(-2.0)
            assert features[f"{prefix}_magnitude"] == pytest.approx(
                expected_magnitude
            )
            assert features[f"{prefix}_net"] == pytest.approx(
                expected_magnitude
            )


def test_grid_magnitude_differs_from_net() -> None:
    """Verify mean magnitude and net magnitude capture different signals."""

    height, width = 30, 60

    flow = create_opposing_flow(
        height=height,
        width=width,
    )

    extractor = FeatureExtractor(
        roi_width=width,
        roi_height=height,
    )

    features = extractor.extract(flow)

    # The left and right cells are each internally uniform.
    assert features["grid_0_0_magnitude"] == pytest.approx(1.0)
    assert features["grid_0_0_net"] == pytest.approx(1.0)

    assert features["grid_0_2_magnitude"] == pytest.approx(1.0)
    assert features["grid_0_2_net"] == pytest.approx(1.0)


def test_grid_feature_keys(
    sample_flow: OpticalFlow,
) -> None:
    """Verify all 36 spatial features are present."""

    extractor = FeatureExtractor(
        roi_width=100,
        roi_height=50,
    )

    features = extractor.extract(sample_flow)

    grid_keys = [
        key
        for key in features
        if key.startswith("grid_")
    ]

    assert len(grid_keys) == 36

    assert "grid_0_0_u" in features
    assert "grid_0_0_net" in features
    assert "grid_2_2_magnitude" in features