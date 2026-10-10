"""Contract tests for FFmpeg music mixing; no real FFmpeg process is required."""
from pathlib import Path
import subprocess

import pytest

from autocut.audio.config import AudioConfig
from autocut.editing.audio import build_music_mix_command, mix_music


def test_mix_command_maps_video_from_first_input_and_music_from_second(tmp_path):
    command = build_music_mix_command(
        tmp_path / "silent.mp4", tmp_path / "music.mp3", tmp_path / "final.mp4",
        duration=20.0, fade_duration=2.0,
    )
    assert command[command.index("-map") + 1] == "0:v:0"
    map_indices = [i for i, value in enumerate(command) if value == "-map"]
    assert [command[i + 1] for i in map_indices] == ["0:v:0", "1:a:0"]
    assert command[command.index("-c:v") + 1] == "copy"
    assert command[command.index("-c:a") + 1] == "aac"
    assert command[command.index("-b:a") + 1] == "320k"
    assert command[command.index("-t") + 1] == "20.0"
    assert command[-1] == str(tmp_path / "final.mp4")
    assert "-movflags" in command and "+faststart" in command


def test_zero_fade_omits_afade_filter(tmp_path):
    command = build_music_mix_command(
        tmp_path / "video.mp4", tmp_path / "music.mp3", tmp_path / "out.mp4",
        duration=8.0, fade_duration=0.0,
    )
    assert not any(value.startswith("afade=") for value in command)


def test_fade_duration_is_clamped_to_short_output(tmp_path):
    command = build_music_mix_command(
        tmp_path / "video.mp4", tmp_path / "music.mp3", tmp_path / "out.mp4",
        duration=0.5, fade_duration=2.0,
    )
    filters = command[command.index("-af") + 1]
    assert "afade=t=out:st=0.0:d=0.5" in filters


def test_mix_music_uses_argument_list_and_returns_measured_duration(tmp_path, monkeypatch):
    video = tmp_path / "silent.mp4"
    music = tmp_path / "music.mp3"
    output = tmp_path / "out.mp4"
    video.write_bytes(b"video")
    music.write_bytes(b"music")
    commands = []

    def fake_run(command, **kwargs):
        assert isinstance(command, list)
        commands.append(command)
        output.write_bytes(b"rendered")

    monkeypatch.setattr("autocut.editing.audio.subprocess.run", fake_run)
    monkeypatch.setattr("autocut.editing.audio.probe_duration", lambda path: 7.99)
    result = mix_music(
        video, music, output,
        duration=8.0,
        config=AudioConfig(fade_duration=2.0),
    )
    assert commands
    assert result.output_path == output
    assert result.duration == pytest.approx(7.99)


def test_ffmpeg_failure_has_actionable_context(tmp_path, monkeypatch):
    def fake_run(command, **kwargs):
        raise subprocess.CalledProcessError(1, command, stderr="codec error")

    monkeypatch.setattr("autocut.editing.audio.subprocess.run", fake_run)
    with pytest.raises(RuntimeError, match="codec error"):
        mix_music(
            tmp_path / "video.mp4", tmp_path / "music.mp3", tmp_path / "out.mp4",
            duration=5.0, config=AudioConfig(),
        )


@pytest.mark.parametrize(
    ("duration", "fade", "valid"),
    [(5.0, 0.0, True), (5.0, 2.0, True), (0.5, 2.0, True),
     (5.0, -1.0, False), (0.0, 0.0, False)],
)
def test_mix_command_validates_durations(tmp_path, duration, fade, valid):
    args = (
        tmp_path / "video.mp4", tmp_path / "music.mp3", tmp_path / "out.mp4"
    )
    if valid:
        assert build_music_mix_command(
            *args, duration=duration, fade_duration=fade
        )
    else:
        with pytest.raises(ValueError):
            build_music_mix_command(
                *args, duration=duration, fade_duration=fade
            )
