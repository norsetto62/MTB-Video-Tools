"""Contract tests for music analysis; librosa/media decoding are mocked."""
from pathlib import Path

import numpy as np
import pytest

from autocut.audio.analysis import AudioAnalysis, analyze_audio


def test_audio_analysis_validates_timestamp_arrays():
    analysis = AudioAnalysis(
        source_path=Path("music.mp3"),
        source_hash="a" * 64,
        duration=4.0,
        tempo_bpm=120.0,
        beats=(0.5, 1.0, 1.5),
        measures=(0.5,),
        onsets=(0.75, 1.25),
        combined=(0.5, 0.75, 1.0, 1.25, 1.5),
    )
    assert analysis.duration == 4.0
    assert analysis.combined == tuple(sorted(set(analysis.combined)))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"duration": 0.0},
        {"duration": float("nan")},
        {"tempo_bpm": float("inf")},
        {"beats": (0.5, 0.4)},
        {"beats": (-0.1, 0.5)},
        {"beats": (0.5, 4.1)},
        {"source_hash": "not-a-sha256"},
    ],
)
def test_audio_analysis_rejects_invalid_values(kwargs):
    values = dict(
        source_path=Path("music.mp3"),
        source_hash="a" * 64,
        duration=4.0,
        tempo_bpm=120.0,
        beats=(0.5, 1.0),
        measures=(0.5,),
        onsets=(0.75,),
        combined=(0.5, 0.75, 1.0),
    )
    values.update(kwargs)
    with pytest.raises((ValueError, TypeError)):
        AudioAnalysis(**values)


def test_analyze_audio_builds_combined_grid_and_uses_cache(tmp_path, monkeypatch):
    import autocut.audio.analysis as module

    music = tmp_path / "music.mp3"
    music.write_bytes(b"mock media")
    calls = {"load": 0}

    def fake_load(*args, **kwargs):
        calls["load"] += 1
        # Native-rate mono waveform; analysis implementation owns interpretation.
        return np.ones(22050 * 4, dtype=np.float32), 22050

    monkeypatch.setattr(module.librosa, "load", fake_load)
    monkeypatch.setattr(module.librosa.beat, "beat_track",
                        lambda **kwargs: (120.0, np.array([10, 20, 30])))
    monkeypatch.setattr(module.librosa, "frames_to_time",
                        lambda frames, sr, hop_length: np.asarray(frames) * 0.1)
    monkeypatch.setattr(module.librosa.onset, "onset_strength",
                        lambda **kwargs: np.array([0.0, 1.0, 0.0, 1.0]))
    monkeypatch.setattr(module.librosa.onset, "onset_detect",
                        lambda **kwargs: np.array([1, 3]))
    result = analyze_audio(music, cache_dir=tmp_path / "cache", use_cache=True)
    assert result.duration == pytest.approx(4.0)
    assert result.combined == tuple(sorted(set(result.combined)))
    assert calls["load"] == 1
    cached = analyze_audio(music, cache_dir=tmp_path / "cache", use_cache=True)
    assert cached == result
    assert calls["load"] == 1


def test_unreadable_music_raises_clear_error(tmp_path):
    missing = tmp_path / "missing.mp3"
    with pytest.raises((OSError, ValueError, RuntimeError)):
        analyze_audio(missing, cache_dir=tmp_path / "cache")
