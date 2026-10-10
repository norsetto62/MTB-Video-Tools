"""Tests for Phase 11 audio configuration."""

import pytest

from autocut.audio.config import AudioConfig, SyncMode


def test_sync_modes_are_explicit_string_enums():
    assert SyncMode.OFF.value == "off"
    assert SyncMode.BEAT.value == "beat"
    assert SyncMode.MEASURE.value == "measure"
    assert SyncMode.FORWARD_BEAT.value == "forward-beat"
    assert SyncMode.ONSET.value == "onset"
    assert SyncMode.COMBINED.value == "combined"
    assert str(SyncMode.BEAT) == "SyncMode.BEAT"


def test_audio_config_defaults_are_safe_and_documented():
    config = AudioConfig()
    assert config.sync_mode is SyncMode.OFF
    assert config.max_clip_duration == 0.0
    assert config.fade_duration == 2.0


@pytest.mark.parametrize("mode", list(SyncMode))
def test_audio_config_accepts_each_sync_mode(mode):
    assert AudioConfig(sync_mode=mode).sync_mode is mode


@pytest.mark.parametrize("mode", ["beat", "forward-beat", "off"])
def test_audio_config_normalizes_valid_mode_strings(mode):
    assert AudioConfig(sync_mode=mode).sync_mode is SyncMode(mode)


@pytest.mark.parametrize("mode", ["nearest", "", None, 1])
def test_audio_config_rejects_unknown_sync_modes(mode):
    with pytest.raises((ValueError, TypeError)):
        AudioConfig(sync_mode=mode)


@pytest.mark.parametrize("field", ["max_clip_duration", "fade_duration"])
@pytest.mark.parametrize("value", [-1, float("nan"), float("inf"), -float("inf")])
def test_audio_config_rejects_invalid_durations(field, value):
    with pytest.raises((ValueError, TypeError)):
        AudioConfig(**{field: value})


def test_audio_config_accepts_zero_clip_limit_and_zero_fade():
    config = AudioConfig(max_clip_duration=0, fade_duration=0)
    assert config.max_clip_duration == 0.0
    assert config.fade_duration == 0.0


def test_audio_config_is_immutable():
    config = AudioConfig()
    with pytest.raises((AttributeError, TypeError)):
        config.fade_duration = 3.0
