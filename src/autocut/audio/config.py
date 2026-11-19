"""Configuration and synchronization modes for optional audio processing."""

from dataclasses import dataclass
from enum import Enum
import math
from numbers import Real


class SyncMode(str, Enum):
    """How clip boundaries are aligned to detected music events."""

    OFF = "off"
    BEAT = "beat"
    MEASURE = "measure"
    FORWARD_BEAT = "forward-beat"
    ONSET = "onset"
    COMBINED = "combined"
    BASS = "bass"


def _finite_nonnegative(name: str, value: Real) -> float:
    """Validate a duration/limit and normalize it to a float."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite non-negative number.")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0:
        raise ValueError(f"{name} must be a finite non-negative number.")
    return normalized


@dataclass(frozen=True)
class AudioConfig:
    """Validated settings for synchronization and final music fade.

    A max_synced_clip_duration of 0 disables the optional-clip duration limit.
    A fade_duration of 0 disables the fade.
    """

    sync_mode: SyncMode = SyncMode.OFF
    max_synced_clip_duration: float = 0.0
    fade_duration: float = 2.0
    bass_sensitivity: float = 0.15
    bass_min_gap_sec: float = 0.30

    def __post_init__(self) -> None:
        mode = self.sync_mode
        if isinstance(mode, str) and not isinstance(mode, SyncMode):
            try:
                mode = SyncMode(mode)
            except ValueError as exc:
                choices = ", ".join(item.value for item in SyncMode)
                raise ValueError(
                    f"sync_mode must be one of: {choices}; got {self.sync_mode!r}."
                ) from exc
        elif not isinstance(mode, SyncMode):
            raise TypeError("sync_mode must be a SyncMode or a valid mode string.")

        object.__setattr__(self, "sync_mode", mode)
        object.__setattr__(
            self, "max_synced_clip_duration",
            _finite_nonnegative("max_synced_clip_duration", self.max_synced_clip_duration),
        )
        object.__setattr__(
            self, "fade_duration",
            _finite_nonnegative("fade_duration", self.fade_duration),
        )
        sensitivity = _finite_nonnegative("bass_sensitivity", self.bass_sensitivity)
        if sensitivity > 1.0:
            raise ValueError("bass_sensitivity must be between 0.0 and 1.0.")
        object.__setattr__(self, "bass_sensitivity", sensitivity)
        gap = _finite_nonnegative("bass_min_gap_sec", self.bass_min_gap_sec)
        if gap <= 0:
            raise ValueError("bass_min_gap_sec must be greater than zero.")
        object.__setattr__(self, "bass_min_gap_sec", gap)
