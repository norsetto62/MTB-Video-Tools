"""Dense optical-flow calculation."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ..models import OpticalFlow


@dataclass(frozen=True)
class FarnebackConfig:
    """Parameters used by OpenCV's Farneback optical-flow algorithm."""

    pyr_scale: float = 0.5
    levels: int = 3
    winsize: int = 15
    iterations: int = 3
    poly_n: int = 5
    poly_sigma: float = 1.2
    flags: int = 0


DEFAULT_FARNEBACK = FarnebackConfig()

# ---------------------------------------------------------------------------
# Optical flow
# ---------------------------------------------------------------------------

def calculate_optical_flow(
    previous_gray: np.ndarray,
    current_gray: np.ndarray,
    config: FarnebackConfig = DEFAULT_FARNEBACK,
) -> OpticalFlow:
    
    """Calculate dense optical flow between two BGR frames.

    Parameters
    ----------
    previous_frame:
        Previous BGR frame as a uint8 NumPy array.

    current_frame:
        Current BGR frame as a uint8 NumPy array.

    config:
        Farneback optical-flow parameters.

    Returns
    -------
    OpticalFlow
        Dense horizontal and vertical flow for the selected ROI.
    """

    flow = cv2.calcOpticalFlowFarneback(
        previous_gray,
        current_gray,
        None,
        config.pyr_scale,
        config.levels,
        config.winsize,
        config.iterations,
        config.poly_n,
        config.poly_sigma,
        config.flags,
    )

    return OpticalFlow(
        u=flow[..., 0],
        v=flow[..., 1],
    )