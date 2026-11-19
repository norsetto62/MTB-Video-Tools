"""Music analysis and validated beat/onset timestamp grids."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import re
from numbers import Real

import librosa
import numpy as np
from scipy import signal


_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_DEFAULT_HOP_LENGTH = 512
_BASS_LOW_CUTOFF_HZ = 150
_DEFAULT_BASS_SENSITIVITY = 0.15
_DEFAULT_BASS_MIN_GAP_SEC = 0.30


def _finite_number(name: str, value: Real, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite number.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    if positive and result <= 0:
        raise ValueError(f"{name} must be greater than zero.")
    return result


def _timestamp_tuple(name: str, values: tuple[float, ...], duration: float) -> tuple[float, ...]:
    try:
        result = tuple(_finite_number(f"{name} timestamp", item) for item in values)
    except TypeError as exc:
        raise TypeError(f"{name} must be an iterable of timestamps.") from exc
    if any(value < 0 or value > duration for value in result):
        raise ValueError(f"{name} timestamps must be within [0, duration].")
    if any(right <= left for left, right in zip(result, result[1:])):
        raise ValueError(f"{name} timestamps must be strictly increasing.")
    return result


@dataclass(frozen=True)
class AudioAnalysis:
    """Immutable analysis results tied to the exact source music file."""

    source_path: Path
    source_hash: str
    duration: float
    tempo_bpm: float
    beats: tuple[float, ...]
    measures: tuple[float, ...]
    onsets: tuple[float, ...]
    combined: tuple[float, ...]
    bass: tuple[float, ...] = ()
    bass_sensitivity: float = _DEFAULT_BASS_SENSITIVITY
    bass_min_gap_sec: float = _DEFAULT_BASS_MIN_GAP_SEC

    def __post_init__(self) -> None:
        if not isinstance(self.source_path, Path):
            try:
                object.__setattr__(self, "source_path", Path(self.source_path))
            except (TypeError, ValueError) as exc:
                raise TypeError("source_path must be path-like.") from exc
        if not isinstance(self.source_hash, str) or not _SHA256_RE.fullmatch(self.source_hash):
            raise ValueError("source_hash must be a 64-character SHA-256 hex digest.")
        object.__setattr__(self, "source_hash", self.source_hash.lower())
        duration = _finite_number("duration", self.duration, positive=True)
        tempo = _finite_number("tempo_bpm", self.tempo_bpm)
        if tempo < 0:
            raise ValueError("tempo_bpm must be non-negative.")
        object.__setattr__(self, "duration", duration)
        object.__setattr__(self, "tempo_bpm", tempo)
        sensitivity = _finite_number("bass_sensitivity", self.bass_sensitivity)
        if not 0.0 <= sensitivity <= 1.0:
            raise ValueError("bass_sensitivity must be between 0.0 and 1.0.")
        gap = _finite_number("bass_min_gap_sec", self.bass_min_gap_sec, positive=True)
        object.__setattr__(self, "bass_sensitivity", sensitivity)
        object.__setattr__(self, "bass_min_gap_sec", gap)
        for name in ("beats", "measures", "onsets", "combined", "bass"):
            values = getattr(self, name)
            if not isinstance(values, (tuple, list, np.ndarray)):
                raise TypeError(f"{name} must be a sequence of timestamps.")
            object.__setattr__(self, name, _timestamp_tuple(name, values, duration))

        expected_combined = tuple(sorted(set(
            round(value, 3) for value in (*self.beats, *self.onsets)
            if 0.0 <= round(value, 3) <= duration
        )))
        if self.combined != expected_combined:
            raise ValueError(
                "combined must be the sorted, unique beat/onset grid rounded to 3 decimals."
            )


def _scalar_tempo(value: object) -> float:
    """Normalize librosa versions returning either a scalar or a one-item array."""
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError("Audio tempo analysis returned an unexpected value.")
    return float(array.reshape(-1)[0])


def _frames_to_seconds(frames: np.ndarray, sample_rate: int) -> tuple[float, ...]:
    times = librosa.frames_to_time(
        np.asarray(frames, dtype=np.int64),
        sr=sample_rate,
        hop_length=_DEFAULT_HOP_LENGTH,
    )
    return tuple(float(item) for item in np.asarray(times).reshape(-1))


def analyze_audio(
    music_path: str | Path,
    *,
    cache_dir: str | Path | None = None,
    use_cache: bool = True,
    bass_sensitivity: float = _DEFAULT_BASS_SENSITIVITY,
    bass_min_gap_sec: float = _DEFAULT_BASS_MIN_GAP_SEC,
) -> AudioAnalysis:
    """Analyze a music file into beat, approximate measure, and onset grids.

    Cache support is delegated to :mod:`autocut.audio.audio_cache`. Cache failures
    are non-fatal for analysis; an unreadable source file is not.
    """
    bass_sensitivity = _finite_number("bass_sensitivity", bass_sensitivity)
    if not 0.0 <= bass_sensitivity <= 1.0:
        raise ValueError("bass_sensitivity must be between 0.0 and 1.0.")
    bass_min_gap_sec = _finite_number("bass_min_gap_sec", bass_min_gap_sec, positive=True)
    path = Path(music_path)
    if not path.is_file():
        raise FileNotFoundError(f"Music file does not exist: {path}")

    # A cache read failure should not prevent fresh analysis.
    cache_module = None
    if use_cache and cache_dir is not None:
        try:
            from . import audio_cache as cache_module

            cached = cache_module.load_cached_audio(
                path, cache_dir, bass_sensitivity=bass_sensitivity,
                bass_min_gap_sec=bass_min_gap_sec,
            )
            if cached is not None:
                return cached
        except OSError:
            # A cache read failure should not prevent fresh analysis.
            cache_module = None

    try:
        waveform, sample_rate = librosa.load(path, sr=None, mono=True)
    except Exception as exc:
        raise RuntimeError(f"Could not decode music file {path}: {exc}") from exc
    waveform = np.asarray(waveform)
    if waveform.ndim != 1 or waveform.size == 0:
        raise ValueError(f"Music file has no usable mono audio samples: {path}")
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, Real) or sample_rate <= 0:
        raise ValueError("Audio decoder returned an invalid sample rate.")
    sample_rate = int(sample_rate)
    duration = float(waveform.size / sample_rate)
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("Music duration must be finite and greater than zero.")

    try:
        tempo_raw, beat_frames = librosa.beat.beat_track(
            y=waveform, sr=sample_rate, hop_length=_DEFAULT_HOP_LENGTH
        )
        onset_envelope = librosa.onset.onset_strength(
            y=waveform, sr=sample_rate, hop_length=_DEFAULT_HOP_LENGTH
        )
        onset_frames = librosa.onset.onset_detect(
            onset_envelope=onset_envelope,
            sr=sample_rate,
            hop_length=_DEFAULT_HOP_LENGTH,
            backtrack=False,
            units="frames",
        )

        # Reproduce the dedicated low-end impact detector: low-pass at 150 Hz,
        # combine normalized onset strength and RMS, then retain prominent peaks.
        sos = signal.butter(
            4, _BASS_LOW_CUTOFF_HZ, btype="lowpass", fs=sample_rate, output="sos"
        )
        low_waveform = np.asarray(signal.sosfilt(sos, waveform))
        bass_onset = librosa.onset.onset_strength(
            y=low_waveform, sr=sample_rate, hop_length=_DEFAULT_HOP_LENGTH,
            fmax=_BASS_LOW_CUTOFF_HZ, n_mels=8,
        )
        bass_rms = librosa.feature.rms(
            y=low_waveform, hop_length=_DEFAULT_HOP_LENGTH
        ).squeeze()
        bass_onset_norm = librosa.util.normalize(np.asarray(bass_onset).reshape(-1))
        bass_rms_norm = librosa.util.normalize(np.asarray(bass_rms).reshape(-1))
        bass_envelope = librosa.util.normalize(bass_onset_norm * bass_rms_norm)
        bass_wait = max(1, int(sample_rate / _DEFAULT_HOP_LENGTH * bass_min_gap_sec))
        bass_frames = librosa.onset.onset_detect(
            onset_envelope=bass_envelope, sr=sample_rate,
            hop_length=_DEFAULT_HOP_LENGTH, delta=bass_sensitivity,
            wait=bass_wait, backtrack=True, units="frames",
        )
        beat_times = _frames_to_seconds(np.asarray(beat_frames), sample_rate)
        onset_times = _frames_to_seconds(np.asarray(onset_frames), sample_rate)
        bass_times = _frames_to_seconds(np.asarray(bass_frames), sample_rate)
    except Exception as exc:
        raise RuntimeError(f"Could not analyze music file {path}: {exc}") from exc

    # Filter rounding/decoder edge cases to the actual source duration, then
    # retain the legacy three-decimal precision for the combined event grid.
    beats = tuple(sorted(set(t for t in beat_times if 0.0 <= t <= duration)))
    onsets = tuple(sorted(set(t for t in onset_times if 0.0 <= t <= duration)))
    bass = tuple(sorted(set(t for t in bass_times if 0.0 <= t <= duration)))
    measures = beats[::4]
    combined = tuple(sorted(set(
        rounded for t in (*beats, *onsets)
        if 0.0 <= t <= duration
        for rounded in (round(t, 3),)
        if 0.0 <= rounded <= duration
    )))

    digest = None
    if cache_dir is not None:
        try:
            if cache_module is None:
                from . import audio_cache as cache_module
            digest = cache_module.music_file_hash(path)
        except (ImportError, OSError):
            digest = None
    if digest is None:
        # A local hash fallback keeps AudioAnalysis source identity meaningful
        # even when optional cache persistence is unavailable.
        import hashlib

        hasher = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                hasher.update(chunk)
        digest = hasher.hexdigest()

    result = AudioAnalysis(
        source_path=path,
        source_hash=digest,
        duration=duration,
        tempo_bpm=_scalar_tempo(tempo_raw),
        beats=beats,
        measures=measures,
        onsets=onsets,
        combined=combined,
        bass=bass,
        bass_sensitivity=bass_sensitivity,
        bass_min_gap_sec=bass_min_gap_sec,
    )

    if use_cache and cache_dir is not None:
        try:
            if cache_module is None:
                from . import audio_cache as cache_module
            cache_module.save_cached_audio(result, cache_dir)
        except (ImportError, OSError):
            # The analysis remains valid even if cache storage is unavailable.
            pass
    return result
