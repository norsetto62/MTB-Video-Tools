import pytest

from pathlib import Path

from autocut.data_models import VideoInfo
from autocut.video.reader import VideoReader


def test_reader_does_not_start_decoder_on_construction():
    reader = VideoReader("ride.mp4")

    assert reader._process is None
    assert reader._frame_index == 0

def test_find_ffmpeg_from_path(monkeypatch):
    monkeypatch.setattr(
        "autocut.video.reader.shutil.which",
        lambda name: r"C:\ffmpeg\bin\ffmpeg.exe",
    )

    assert VideoReader._find_ffmpeg() == r"C:\ffmpeg\bin\ffmpeg.exe"

def test_find_ffmpeg_from_default_location(
    monkeypatch,
    tmp_path,
):
    ffmpeg = tmp_path / "ffmpeg.exe"
    ffmpeg.touch()

    monkeypatch.setattr(
        "autocut.video.reader.shutil.which",
        lambda name: None,
    )
    monkeypatch.setattr(
        "autocut.video.reader.DEFAULT_FFMPEG",
        ffmpeg,
    )

    assert VideoReader._find_ffmpeg() == str(ffmpeg)

def test_find_ffmpeg_raises_when_unavailable(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(
        "autocut.video.reader.shutil.which",
        lambda name: None,
    )
    monkeypatch.setattr(
        "autocut.video.reader.DEFAULT_FFMPEG",
        tmp_path / "missing.exe",
    )

    with pytest.raises(
        FileNotFoundError,
        match="FFmpeg executable not found",
    ):
        VideoReader._find_ffmpeg()

def test_processing_duration_is_remaining_video(
    monkeypatch,
):
    reader = VideoReader(
        "ride.mp4",
        start=10.0,
    )

    reader.info = VideoInfo(
        width=1920,
        height=1080,
        fps=30.0,
        duration=100.0,
    )

    assert reader._get_processing_duration() == 90.0

def test_processing_duration_is_limited_by_requested_duration():
    reader = VideoReader(
        "ride.mp4",
        start=10.0,
        duration=20.0,
    )

    reader.info = VideoInfo(
        width=1920,
        height=1080,
        fps=30.0,
        duration=100.0,
    )

    assert reader._get_processing_duration() == 20.0

def test_processing_duration_is_limited_by_remaining_video():
    reader = VideoReader(
        "ride.mp4",
        start=90.0,
        duration=30.0,
    )

    reader.info = VideoInfo(
        width=1920,
        height=1080,
        fps=30.0,
        duration=100.0,
    )

    assert reader._get_processing_duration() == 10.0

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "kwargs",
    [
        {"start": -1.0},
        {"duration": 0.0},
        {"duration": -1.0},
        {"sample_fps": 0.0},
        {"sample_fps": -1.0},
    ],
)
def test_reader_rejects_invalid_parameters(kwargs, tmp_path):
    video_path = tmp_path / "ride.mp4"
    video_path.touch()

    reader = VideoReader(video_path, **kwargs)

    with pytest.raises(ValueError):
        reader._start_decoder()

# ---------------------------------------------------------------------------
# Fake FFmpeg's stdout with three raw frames.
# ---------------------------------------------------------------------------

import numpy as np

def test_reader_yields_one_frame_per_iteration(monkeypatch, tmp_path):
    info = VideoInfo(
        width=2,
        height=1,
        fps=2.0,
        duration=2.0,
    )

    monkeypatch.setattr(
        "autocut.video.reader.get_video_info",
        lambda path: info,
    )
    monkeypatch.setattr(
        VideoReader,
        "_find_ffmpeg",
        staticmethod(lambda: "ffmpeg"),
    )

    frame1 = bytes([1, 2, 3, 4, 5, 6])
    frame2 = bytes([7, 8, 9, 10, 11, 12])

    class FakeStdout:
        def __init__(self):
            self.data = frame1 + frame2

        def read(self, size):
            if not self.data:
                return b""

            result = self.data[:size]
            self.data = self.data[size:]
            return result

        def close(self):
            pass

    class FakeStderr:
        def close(self):
            pass

    class FakeProcess:
        def __init__(self):
            self.stdout = FakeStdout()
            self.stderr = FakeStderr()

        def poll(self):
            return 0

        def terminate(self):
            pass

        def wait(self, timeout=None):
            return 0

    monkeypatch.setattr(
        "autocut.video.reader.subprocess.Popen",
        lambda *args, **kwargs: FakeProcess(),
    )

    video_path = tmp_path / "ride.mp4"
    video_path.touch()

    with VideoReader(
        video_path,
        sample_fps=2.0,
    ) as reader:
        frames = list(reader)

    assert len(frames) == 2

    index, timestamp, frame = frames[0]

    assert index == 0
    assert timestamp == 0.0
    assert frame.shape == (1, 2, 3)
    assert frame.dtype == np.uint8

    np.testing.assert_array_equal(
        frame,
        np.array(
            [[[1, 2, 3], [4, 5, 6]]],
            dtype=np.uint8,
        ),
    )

    assert frames[1][0] == 1
    assert frames[1][1] == 0.5

# ---------------------------------------------------------------------------
# Verify ffmpeg command
# ---------------------------------------------------------------------------

def test_reader_builds_expected_ffmpeg_command(monkeypatch, tmp_path):
    info = VideoInfo(
        width=1920,
        height=1080,
        fps=30.0,
        duration=100.0,
    )

    monkeypatch.setattr(
        "autocut.video.reader.get_video_info",
        lambda path: info,
    )
    monkeypatch.setattr(
        VideoReader,
        "_find_ffmpeg",
        staticmethod(lambda: "ffmpeg"),
    )

    captured = {}

    class FakeProcess:
        stdout = None
        stderr = None

    def fake_popen(cmd, **kwargs):
        captured["cmd"] = cmd
        return FakeProcess()

    monkeypatch.setattr(
        "autocut.video.reader.subprocess.Popen",
        fake_popen,
    )

    video_path = tmp_path / "ride.mp4"
    video_path.touch()

    reader = VideoReader(
        video_path,
        start=10.0,
        duration=20.0,
        sample_fps=2.0,
    )

    reader._start_decoder()

    assert captured["cmd"] == [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        "10.000000",
        "-i",
        str(video_path),
        "-t",
        "20.000000",
        "-vf",
        "fps=2.00000000",
        "-an",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "pipe:1",
    ]

#TBD
"""
def test_reader_rejects_incomplete_frame(monkeypatch):
    # Fake a video whose first frame is incomplete.
    reader = 

    with pytest.raises(
        RuntimeError,
        match="Incomplete frame received",
    ):
        list(reader)
"""
def test_reader_requires_context_manager():
    reader = VideoReader("ride.mp4")

    with pytest.raises(
        RuntimeError,
        match="must be used inside a context manager",
    ):
        list(reader)

def test_reader_closes_ffmpeg_process(monkeypatch):
    calls = []

    class FakeProcess:
        stdout = None
        stderr = None

        def poll(self):
            return None

        def terminate(self):
            calls.append("terminate")

        def wait(self, timeout=None):
            calls.append(("wait", timeout))

    reader = VideoReader("ride.mp4")
    reader._process = FakeProcess()

    reader.close()

    assert calls == [
        "terminate",
        ("wait", 10),
    ]
    assert reader._process is None

def test_reader_close_is_idempotent():
    reader = VideoReader("ride.mp4")

    reader.close()
    reader.close()

    assert reader._process is None

#TBD
"""
def test_reader_reports_ffmpeg_failure(monkeypatch):
    reader = 

    with pytest.raises(
        RuntimeError,
        match="FFmpeg decoder failed",
    ):
        list(reader)
"""