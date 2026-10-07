"""Feature extraction from dense optical flow."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..data_models import OpticalFlow


# ---------------------------------------------------------------------------
# Feature schema
# ---------------------------------------------------------------------------

GLOBAL_FEATURES = (
    "flow_mean",
    "flow_p90",
    "flow_std",
    "flow_u",
    "flow_v",
    "flow_net",
)

STRUCTURE_FEATURES = (
    "div_normalized_mean",
    "div_abs_p90",
    "div_std",
    "div_pos_fraction",
    "curl_normalized_mean",
    "curl_abs_p90",
    "curl_std",
)

TEMPORAL_FEATURES = (
    "delta_flow_u",
    "delta_flow_v",
    "delta_flow_mean",
    "delta_flow_net",
    "delta_div_mean",
)

GRID_FEATURES = (
    "u",
    "v",
    "magnitude",
    "net",
)

GRID_ROWS = 3
GRID_COLS = 3

FEATURE_NAMES = (
    GLOBAL_FEATURES
    + STRUCTURE_FEATURES
    + TEMPORAL_FEATURES
    + tuple(
        f"grid_{row}_{col}_{name}"
        for row in range(GRID_ROWS)
        for col in range(GRID_COLS)
        for name in GRID_FEATURES
    )
)

EPSILON = 1e-6


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


@dataclass
class FeatureExtractor:
    """Extract the AutoCut motion feature vector from optical flow.

    Optical flow is assumed to have been calculated upstream on the
    already-selected ROI.

    Global and 3×3 spatial-grid flow features remain in Farneback's native
    pixel-displacement units. Divergence and curl are calculated from flow
    normalized by the ROI dimensions and expressed in normalized ROI
    coordinates.

    The extractor retains the previous feature vector to calculate temporal
    deltas. Call :meth:`reset` before processing each independent sequence.
    """
    roi_width: int
    roi_height: int

    _previous: dict[str, float] | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _grid_slices: tuple[tuple[int, int, int, int], ...] = field(
        default=(),
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        """Precompute the fixed 3×3 grid boundaries."""

        row_edges = np.linspace(
            0,
            self.roi_height,
            GRID_ROWS + 1,
            dtype=int,
        )
        col_edges = np.linspace(
            0,
            self.roi_width,
            GRID_COLS + 1,
            dtype=int,
        )

        self._grid_slices = tuple(
            (
                row_edges[row],
                row_edges[row + 1],
                col_edges[col],
                col_edges[col + 1],
            )
            for row in range(GRID_ROWS)
            for col in range(GRID_COLS)
        )

    def reset(self) -> None:
        """Reset temporal state before processing a new independent sequence."""

        self._previous = None

    def get_feature_names(self) -> tuple[str, ...]:
        """Return the canonical feature ordering used by model input vectors."""

        return FEATURE_NAMES

    def extract(self, flow: OpticalFlow) -> dict[str, float]:
        """Extract one complete feature vector as a named dictionary."""

        features = self._extract_features(flow)
        return features

    def extract_vector(self, flow: OpticalFlow) -> np.ndarray:
        """Extract one complete feature vector in canonical column order."""

        features = self._extract_features(flow)

        return np.asarray(
            [features[name] for name in FEATURE_NAMES],
            dtype=np.float32,
        )

    # -----------------------------------------------------------------------
    # Feature calculation
    # -----------------------------------------------------------------------

    def _extract_features(
        self,
        flow: OpticalFlow,
    ) -> dict[str, float]:
        """Calculate all features and update temporal state."""

        u = flow.u
        v = flow.v

        magnitude = np.hypot(u, v)

        flow_u = float(np.mean(u))
        flow_v = float(np.mean(v))
        flow_mean = float(np.mean(magnitude))
        flow_p90 = float(np.percentile(magnitude, 90))
        flow_std = float(np.std(magnitude))
        flow_net = float(np.hypot(flow_u, flow_v))

        divergence, curl, normalized_flow_mean = (
            self._divergence_and_curl(u, v)
        )

        div_abs = np.abs(divergence)
        curl_abs = np.abs(curl)

        div_abs_mean = float(np.mean(div_abs))

        features: dict[str, float] = {
            # Global flow
            "flow_mean": flow_mean,
            "flow_p90": flow_p90,
            "flow_std": flow_std,
            "flow_u": flow_u,
            "flow_v": flow_v,
            "flow_net": flow_net,

            # Divergence
            "div_normalized_mean": (
                div_abs_mean / (normalized_flow_mean + EPSILON)
            ),
            "div_abs_p90": float(np.percentile(div_abs, 90)),
            "div_std": float(np.std(divergence)),
            "div_pos_fraction": float(np.mean(divergence > 0)),

            # Curl
            "curl_normalized_mean": (
                float(np.mean(curl_abs)) / (normalized_flow_mean + EPSILON)
            ),
            "curl_abs_p90": float(np.percentile(curl_abs, 90)),
            "curl_std": float(np.std(curl)),

            # Temporal features are added below.
        }

        self._add_temporal_features(features)

        # Spatial 3×3 grid.
        features.update(
            self._grid_features(u, v, magnitude)
        )

        self._previous = features.copy()

        return features

    # -----------------------------------------------------------------------
    # Spatial derivatives
    # -----------------------------------------------------------------------

    def _divergence_and_curl(
        self,
        u: np.ndarray,
        v: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Calculate normalized divergence and curl."""

        u_normalized = u / self.roi_width
        v_normalized = v / self.roi_height

        dx = 1.0 / (self.roi_width - 1)
        dy = 1.0 / (self.roi_height - 1)

        du_dy, du_dx = np.gradient(
            u_normalized,
            dy,
            dx,
        )
        dv_dy, dv_dx = np.gradient(
            v_normalized,
            dy,
            dx,
        )

        divergence = du_dx + dv_dy
        curl = dv_dx - du_dy

        normalized_flow_mean = float(
            np.mean(np.hypot(u_normalized, v_normalized))
        )

        return divergence, curl, normalized_flow_mean

    # -----------------------------------------------------------------------
    # Temporal features
    # -----------------------------------------------------------------------

    def _add_temporal_features(
        self,
        features: dict[str, float],
    ) -> None:
        """Add temporal differences from the previous feature vector."""

        if self._previous is None:
            features.update(
                {
                    "delta_flow_u": 0.0,
                    "delta_flow_v": 0.0,
                    "delta_flow_mean": 0.0,
                    "delta_flow_net": 0.0,
                    "delta_div_mean": 0.0,
                }
            )
            return

        features.update(
            {
                "delta_flow_u": (
                    features["flow_u"]
                    - self._previous["flow_u"]
                ),
                "delta_flow_v": (
                    features["flow_v"]
                    - self._previous["flow_v"]
                ),
                "delta_flow_mean": (
                    features["flow_mean"]
                    - self._previous["flow_mean"]
                ),
                "delta_flow_net": (
                    features["flow_net"]
                    - self._previous["flow_net"]
                ),
                "delta_div_mean": (
                    features["div_normalized_mean"]
                    - self._previous["div_normalized_mean"]
                ),
            }
        )

    # -----------------------------------------------------------------------
    # Spatial grid
    # -----------------------------------------------------------------------

    def _grid_features(
        self,
        u: np.ndarray,
        v: np.ndarray,
        magnitude: np.ndarray,
    ) -> dict[str, float]:
        """Calculate four flow statistics for each 3×3 ROI cell."""

        features: dict[str, float] = {}

        for index, (
            y0,
            y1,
            x0,
            x1,
        ) in enumerate(self._grid_slices):
            row = index // GRID_COLS
            col = index % GRID_COLS

            cell_u = u[y0:y1, x0:x1]
            cell_v = v[y0:y1, x0:x1]
            cell_magnitude = magnitude[y0:y1, x0:x1]

            mean_u = float(np.mean(cell_u))
            mean_v = float(np.mean(cell_v))

            prefix = f"grid_{row}_{col}"

            features[f"{prefix}_u"] = mean_u
            features[f"{prefix}_v"] = mean_v
            features[f"{prefix}_magnitude"] = float(
                np.mean(cell_magnitude)
            )
            features[f"{prefix}_net"] = float(
                np.hypot(mean_u, mean_v)
            )

        return features