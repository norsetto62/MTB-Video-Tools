"""Domain data models used throughout AutoCut."""

from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class Clip:
    """A video interval selected or requested for processing."""

    video_name: str
    start: float
    end: float
    mandatory: bool = False

    @property
    def duration(self) -> float:
        """Return the clip duration in seconds."""
        return self.end - self.start


@dataclass(frozen=True)
class AudioConfig:
    """Configuration for an optional audio track."""

    path: str
    mix: bool = False

@dataclass(frozen=True)
class VideoInfo:
    """Basic metadata describing a video."""

    width: int
    height: int
    fps: float
    duration: float

@dataclass(frozen=True)
class OpticalFlow:
    """Dense optical flow."""

    u: np.ndarray
    v: np.ndarray

@dataclass(frozen=True)
class FeatureSequence:
    """Time-ordered motion-feature vectors for a video clip."""

    features: np.ndarray
    timestamps: np.ndarray

    def __post_init__(self) -> None:
        if self.features.ndim != 2:
            raise ValueError("features must be a 2D array.")

        if self.features.shape[0] < 1:
            raise ValueError("FeatureSequence must contain at least one feature.")

        if self.timestamps.ndim != 1:
            raise ValueError("timestamps must be a 1D array.")

        if len(self.timestamps) != len(self.features):
            raise ValueError(
                "features and timestamps must contain the same number of samples."
            )

        if self.features.shape[1] != 54:
            raise ValueError(
                f"features must contain 54 values per sample, "
                f"got {self.features.shape[1]}."
            )

        if self.features.dtype != np.float32:
            raise ValueError("features must have dtype float32.")

        if self.timestamps.dtype != np.float64:
            raise ValueError("timestamps must have dtype float64.")

        if not np.all(np.isfinite(self.timestamps)):
            raise ValueError("timestamps must contain only finite values.")

        if len(self.timestamps) > 1 and not np.all(
            np.diff(self.timestamps) > 0
        ):
            raise ValueError("timestamps must be strictly increasing.")


@dataclass(frozen=True)
class FeatureWindows:
    """Fixed-size temporal windows extracted from a feature sequence.

    Attributes:
        X: Windowed feature matrix with shape (N, T, F).
        timestamps: Window start/end times with shape (N, 2).
    """

    X: np.ndarray
    timestamps: np.ndarray


@dataclass(frozen=True)
class Annotation:
    """A manually assigned interest interval."""
    
    start: float
    end: float
    score: int
    remarks: str
