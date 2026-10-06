import json
import subprocess

import pytest

from autocut.video import probe


# ---------------------------------------------------------------------------
# FFprobe discovery
# ---------------------------------------------------------------------------


def test_find_ffprobe_from_path(monkeypatch):
    monkeypatch.setattr(
        probe.shutil,
        "which",
        lambda name: r"C:\ffmpeg\bin\ffprobe.exe",
    )

    assert probe._find_ffprobe() == r"C:\ffmpeg\bin\ffprobe.exe"


def test_find_ffprobe_from_default_location(
    monkeypatch,
    tmp_path,
):
    ffprobe = tmp_path / "ffprobe.exe"
    ffprobe.touch()

    monkeypatch.setattr(
        probe.shutil,
        "which",
        lambda name: None,
    )
    monkeypatch.setattr(
        probe,
        "DEFAULT_FFPROBE",
        ffprobe,
    )

    assert probe._find_ffprobe() == str(ffprobe)


def test_find_ffprobe_returns_none_when_unavailable(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(
        probe.shutil,
        "which",
        lambda name: None,
    )

    missing = tmp_path / "ffprobe.exe"

    monkeypatch.setattr(
        probe,
        "DEFAULT_FFPROBE",
        missing,
    )

    assert probe._find_ffprobe() is None


# ---------------------------------------------------------------------------
# ffprobe JSON parsing
# ---------------------------------------------------------------------------


def test_probe_with_ffprobe(monkeypatch, tmp_path):
    video = tmp_path / "ride.mp4"
    video.touch()

    payload = {
        "streams": [
            {
                "codec_type": "video",
                "width": 1920,
                "height": 1080,
                "avg_frame_rate": "30000/1001",
                "r_frame_rate": "30000/1001",
                "duration": "123.456",
            }
        ],
        "format": {
            "duration": "123.456",
        },
    }

    class FakeResult:
        stdout = json.dumps(payload)

    def fake_run(*args, **kwargs):
        return FakeResult()

    monkeypatch.setattr(
        probe.subprocess,
        "run",
        fake_run,
    )

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


def test_probe_with_ffprobe_prefers_avg_frame_rate(
    monkeypatch,
    tmp_path,
):
    video = tmp_path / "ride.mp4"
    video.touch()

    payload = {
        "streams": [
            {
                "codec_type": "video",
                "width": 1920,
                "height": 1080,
                "avg_frame_rate": "30000/1001",
                "r_frame_rate": "60/1",
                "duration": "10.0",
            }
        ],
        "format": {
            "duration": "10.0",
        },
    }

    class FakeResult:
        stdout = json.dumps(payload)

    monkeypatch.setattr(
        probe.subprocess,
        "run",
        lambda *args, **kwargs: FakeResult(),
    )

    info = probe._probe_with_ffprobe(video, "ffprobe")

    assert info.fps == pytest.approx(30000 / 1001)


def test_probe_with_ffprobe_falls_back_to_r_frame_rate(
    monkeypatch,
    tmp_path,
):
    video = tmp_path / "ride.mp4"
    video.touch()

    payload = {
        "streams": [
            {
                "codec_type": "video",
                "width": 1920,
                "height": 1080,
                "avg_frame_rate": "0/0",
                "r_frame_rate": "25/1",
                "duration": "10.0",
            }
        ],
        "format": {
            "duration": "10.0",
        },
    }

    class FakeResult:
        stdout = json.dumps(payload)

    monkeypatch.setattr(
        probe.subprocess,
        "run",
        lambda *args, **kwargs: FakeResult(),
    )

    info = probe._probe_with_ffprobe(video, "ffprobe")

    assert info.fps == 25.0


def test_probe_with_ffprobe_falls_back_to_format_duration(
    monkeypatch,
    tmp_path,
):
    video = tmp_path / "ride.mp4"
    video.touch()

    payload = {
        "streams": [
            {
                "codec_type": "video",
                "width": 1920,
                "height": 1080,
                "avg_frame_rate": "30/1",
                "r_frame_rate": "30/1",
            }
        ],
        "format": {
            "duration": "42.5",
        },
    }

    class FakeResult:
        stdout = json.dumps(payload)

    monkeypatch.setattr(
        probe.subprocess,
        "run",
        lambda *args, **kwargs: FakeResult(),
    )

    info = probe._probe_with_ffprobe(video, "ffprobe")

    assert info.duration == 42.5


# ---------------------------------------------------------------------------
# Invalid ffprobe data
# ---------------------------------------------------------------------------


def test_probe_with_ffprobe_rejects_invalid_json(
    monkeypatch,
    tmp_path,
):
    video = tmp_path / "ride.mp4"
    video.touch()

    class FakeResult:
        stdout = "not valid json"

    monkeypatch.setattr(
        probe.subprocess,
        "run",
        lambda *args, **kwargs: FakeResult(),
    )

    with pytest.raises(
        (ValueError, RuntimeError),
    ):
        probe._probe_with_ffprobe(video, "ffprobe")


def test_probe_with_ffprobe_rejects_missing_video_stream(
    monkeypatch,
    tmp_path,
):
    video = tmp_path / "ride.mp4"
    video.touch()

    payload = {
        "streams": [
            {
                "codec_type": "audio",
            }
        ],
        "format": {
            "duration": "10.0",
        },
    }

    class FakeResult:
        stdout = json.dumps(payload)

    monkeypatch.setattr(
        probe.subprocess,
        "run",
        lambda *args, **kwargs: FakeResult(),
    )

    with pytest.raises(RuntimeError):
        probe._probe_with_ffprobe(video, "ffprobe")


@pytest.mark.parametrize(
    "field, value, format_duration, match",
    [
        ("width", 0, "10.0", "width"),
        ("height", 0, "10.0", "height"),
        ("avg_frame_rate", "0/0", "10.0", "FPS"),
        ("duration", "0", "0", "duration"),
    ],
)
def test_probe_with_ffprobe_rejects_invalid_metadata(
    monkeypatch,
    tmp_path,
    field,
    value,
    format_duration,
    match,
):
    video = tmp_path / "ride.mp4"
    video.touch()

    stream = {
        "codec_type": "video",
        "width": 1920,
        "height": 1080,
        "avg_frame_rate": "30/1",
        "r_frame_rate": "30/1",
        "duration": "10.0",
    }

    stream[field] = value

    if field == "avg_frame_rate":
        stream["r_frame_rate"] = "0/0"

    payload = {
        "streams": [stream],
        "format": {
            "duration": format_duration,
        },
    }

    class FakeResult:
        stdout = json.dumps(payload)

    monkeypatch.setattr(
        probe.subprocess,
        "run",
        lambda *args, **kwargs: FakeResult(),
    )

    with pytest.raises(RuntimeError, match=match):
        probe._probe_with_ffprobe(video, "ffprobe")

def test_probe_with_ffprobe_uses_r_frame_rate_when_avg_frame_rate_invalid(
    monkeypatch,
    tmp_path,
):
    video = tmp_path / "ride.mp4"
    video.touch()

    payload = {
        "streams": [
            {
                "codec_type": "video",
                "width": 1920,
                "height": 1080,
                "avg_frame_rate": "0/0",
                "r_frame_rate": "25/1",
                "duration": "10.0",
            }
        ],
        "format": {
            "duration": "10.0",
        },
    }

    class FakeResult:
        stdout = json.dumps(payload)

    monkeypatch.setattr(
        probe.subprocess,
        "run",
        lambda *args, **kwargs: FakeResult(),
    )

    info = probe._probe_with_ffprobe(video, "ffprobe")

    assert info.fps == 25.0

def test_probe_with_ffprobe_uses_format_duration_when_stream_duration_invalid(
    monkeypatch,
    tmp_path,
):
    video = tmp_path / "ride.mp4"
    video.touch()

    payload = {
        "streams": [
            {
                "codec_type": "video",
                "width": 1920,
                "height": 1080,
                "avg_frame_rate": "30/1",
                "r_frame_rate": "30/1",
                "duration": "0",
            }
        ],
        "format": {
            "duration": "42.5",
        },
    }

    class FakeResult:
        stdout = json.dumps(payload)

    monkeypatch.setattr(
        probe.subprocess,
        "run",
        lambda *args, **kwargs: FakeResult(),
    )

    info = probe._probe_with_ffprobe(video, "ffprobe")

    assert info.duration == 42.5

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
# get_video_info fallback logic
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


def test_get_video_info_does_not_fallback_on_invalid_metadata(
    monkeypatch,
    tmp_path,
):
    video = tmp_path / "ride.mp4"
    video.touch()

    monkeypatch.setattr(
        probe,
        "_find_ffprobe",
        lambda: "ffprobe",
    )

    def fail_ffprobe(path, executable):
        raise RuntimeError("Invalid video metadata: fps must be > 0")

    monkeypatch.setattr(
        probe,
        "_probe_with_ffprobe",
        fail_ffprobe,
    )

    def fail_if_called(path):
        raise AssertionError(
            "OpenCV should not be used for invalid ffprobe metadata"
        )

    monkeypatch.setattr(
        probe,
        "_probe_with_opencv",
        fail_if_called,
    )

    with pytest.raises(RuntimeError, match="fps"):
        probe.get_video_info(video)