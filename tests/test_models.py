import pytest

from autocut.data_models import AudioConfig, Clip, VideoInfo

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

def test_video_info_stores_metadata():
    info = VideoInfo(
        width=1920,
        height=1080,
        fps=29.97,
        duration=123.45,
    )

    assert info.width == 1920
    assert info.height == 1080
    assert info.fps == 29.97
    assert info.duration == 123.45


def test_video_info_is_immutable():
    info = VideoInfo(
        width=1920,
        height=1080,
        fps=30.0,
        duration=60.0,
    )

    with pytest.raises(AttributeError):
        info.width = 1280


def test_video_info_equality():
    info1 = VideoInfo(
        width=1920,
        height=1080,
        fps=30.0,
        duration=60.0,
    )
    info2 = VideoInfo(
        width=1920,
        height=1080,
        fps=30.0,
        duration=60.0,
    )

    assert info1 == info2