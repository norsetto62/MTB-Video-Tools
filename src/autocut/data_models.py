"""Domain data models used throughout AutoCut."""

from dataclasses import dataclass


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