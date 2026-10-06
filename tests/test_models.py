import pytest

from autocut.data_models import AudioConfig, Clip


def test_clip_duration():
    clip = Clip(
        video_name="ride.mp4",
        start=10.0,
        end=25.5,
    )

    assert clip.duration == 15.5


def test_clip_defaults_mandatory_to_false():
    clip = Clip(
        video_name="ride.mp4",
        start=10.0,
        end=20.0,
    )

    assert clip.mandatory is False


def test_clip_can_be_mandatory():
    clip = Clip(
        video_name="ride.mp4",
        start=10.0,
        end=11.0,
        mandatory=True,
    )

    assert clip.mandatory is True


def test_clip_is_immutable():
    clip = Clip(
        video_name="ride.mp4",
        start=10.0,
        end=20.0,
    )

    with pytest.raises(AttributeError):
        clip.start = 15.0


def test_audio_config_defaults_mix_to_false():
    audio = AudioConfig(path="music.mp3")

    assert audio.path == "music.mp3"
    assert audio.mix is False


def test_audio_config_can_enable_mix():
    audio = AudioConfig(
        path="music.mp3",
        mix=True,
    )

    assert audio.path == "music.mp3"
    assert audio.mix is True


def test_audio_config_is_immutable():
    audio = AudioConfig(path="music.mp3")

    with pytest.raises(AttributeError):
        audio.mix = True