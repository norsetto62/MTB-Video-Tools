import pytest

import subprocess
from pathlib import Path

from autocut.video import probe

# ---------------------------------------------------------------------------
# Basic parsing
# ---------------------------------------------------------------------------

def test_find_ffprobe_from_path(monkeypatch):
    monkeypatch.setattr(
        probe.shutil,
        "which",
        lambda name: r"C:\ffmpeg\bin\ffprobe.exe",
    )

    assert probe._find_ffprobe() == r"C:\ffmpeg\bin\ffprobe.exe"

def test_find_ffprobe_from_default_location(monkeypatch, tmp_path):
    ffprobe = tmp_path / "ffprobe.exe"
    ffprobe.touch()

    monkeypatch.setattr(probe.shutil, "which", lambda name: None)
    monkeypatch.setattr(probe, "DEFAULT_FFPROBE", ffprobe)

    assert probe._find_ffprobe() == str(ffprobe)

def test_find_ffprobe_returns_none_when_unavailable(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(probe.shutil, "which", lambda name: None)

    missing = tmp_path / "ffprobe.exe"
    monkeypatch.setattr(probe, "DEFAULT_FFPROBE", missing)

    assert probe._find_ffprobe() is None

def test_probe_with_ffprobe(monkeypatch, tmp_path):
    video = tmp_path / "ride.mp4"
    video.touch()

    class FakeResult:
        stdout = "1920\n1080\n30000/1001\n123.456\n"

    def fake_run(*args, **kwargs):
        return FakeResult()

    monkeypatch.setattr(probe.subprocess, "run", fake_run)

    info = probe._probe_with_ffprobe(
        video,
        "ffprobe",
    )

    assert info == probe.VideoInfo(
        width=1920,
        height=1080,
        fps=30000 / 1001,
        duration=123.456,
    )

# ---------------------------------------------------------------------------
# Malformed ffprobe output
# ---------------------------------------------------------------------------

def test_probe_with_ffprobe_rejects_insufficient_output(
    monkeypatch,
    tmp_path,
):
    video = tmp_path / "ride.mp4"
    video.touch()

    class FakeResult:
        stdout = "1920\n1080\n"

    monkeypatch.setattr(
        probe.subprocess,
        "run",
        lambda *args, **kwargs: FakeResult(),
    )

    with pytest.raises(RuntimeError, match="insufficient video information"):
        probe._probe_with_ffprobe(video, "ffprobe")

# ---------------------------------------------------------------------------
# OpenCV path
# ---------------------------------------------------------------------------

def test_probe_with_opencv(monkeypatch, tmp_path):
    video = tmp_path / "ride.mp4"
    video.touch()

    class FakeCapture:
        def isOpened(self):
            return True

        def get(self, property_id):
            values = {
                probe.cv2.CAP_PROP_FRAME_WIDTH: 1920,
                probe.cv2.CAP_PROP_FRAME_HEIGHT: 1080,
                probe.cv2.CAP_PROP_FPS: 30.0,
                probe.cv2.CAP_PROP_FRAME_COUNT: 3000,
            }
            return values[property_id]

        def release(self):
            pass

    monkeypatch.setattr(
        probe.cv2,
        "VideoCapture",
        lambda path: FakeCapture(),
    )

    info = probe._probe_with_opencv(video)

    assert info == probe.VideoInfo(
        width=1920,
        height=1080,
        fps=30.0,
        duration=100.0,
    )

def test_probe_with_opencv_rejects_unopened_video(
    monkeypatch,
    tmp_path,
):
    video = tmp_path / "ride.mp4"
    video.touch()

    class FakeCapture:
        def isOpened(self):
            return False

    monkeypatch.setattr(
        probe.cv2,
        "VideoCapture",
        lambda path: FakeCapture(),
    )

    with pytest.raises(RuntimeError, match="failed to open video"):
        probe._probe_with_opencv(video)

# ---------------------------------------------------------------------------
# OpenCV fall-back logic
# ---------------------------------------------------------------------------

def test_get_video_info_falls_back_to_opencv(
    monkeypatch,
    tmp_path,
):
    video = tmp_path / "ride.mp4"
    video.touch()

    monkeypatch.setattr(
        probe,
        "_find_ffprobe",
        lambda: None,
    )

    expected = probe.VideoInfo(
        width=1280,
        height=720,
        fps=30.0,
        duration=60.0,
    )

    monkeypatch.setattr(
        probe,
        "_probe_with_opencv",
        lambda path: expected,
    )

    assert probe.get_video_info(video) == expected

def test_get_video_info_prefers_ffprobe(
    monkeypatch,
    tmp_path,
):
    video = tmp_path / "ride.mp4"
    video.touch()

    expected = probe.VideoInfo(
        width=1920,
        height=1080,
        fps=29.97,
        duration=120.0,
    )

    monkeypatch.setattr(
        probe,
        "_find_ffprobe",
        lambda: "ffprobe",
    )

    monkeypatch.setattr(
        probe,
        "_probe_with_ffprobe",
        lambda path, executable: expected,
    )

    def fail_if_called(path):
        raise AssertionError("OpenCV should not be used")

    monkeypatch.setattr(
        probe,
        "_probe_with_opencv",
        fail_if_called,
    )

    assert probe.get_video_info(video) == expected

def test_get_video_info_falls_back_when_ffprobe_fails(
    monkeypatch,
    tmp_path,
):
    video = tmp_path / "ride.mp4"
    video.touch()

    expected = probe.VideoInfo(
        width=1280,
        height=720,
        fps=30.0,
        duration=60.0,
    )

    monkeypatch.setattr(
        probe,
        "_find_ffprobe",
        lambda: "ffprobe",
    )

    def fail_ffprobe(path, executable):
        raise subprocess.CalledProcessError(
            returncode=1,
            cmd=[executable],
        )

    monkeypatch.setattr(
        probe,
        "_probe_with_ffprobe",
        fail_ffprobe,
    )

    monkeypatch.setattr(
        probe,
        "_probe_with_opencv",
        lambda path: expected,
    )

    assert probe.get_video_info(video) == expected