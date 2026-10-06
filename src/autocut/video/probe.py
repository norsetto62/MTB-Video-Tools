"""Video metadata probing."""

import shutil
import subprocess
from pathlib import Path

import cv2

from ..data_models import VideoInfo


DEFAULT_FFPROBE = Path(
    r"C:\Program Files\ffmpeg\bin\ffprobe.exe"
)


def _find_ffprobe() -> str | None:
    """Return the ffprobe executable path, if available."""

    ffprobe = shutil.which("ffprobe")
    if ffprobe is not None:
        return ffprobe

    if DEFAULT_FFPROBE.is_file():
        return str(DEFAULT_FFPROBE)

    return None


def _probe_with_ffprobe(video_path: Path, ffprobe: str) -> VideoInfo:
    """Read video metadata using ffprobe."""

    cmd = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate,duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=True,
    )

    values = [
        line.strip()
        for line in result.stdout.splitlines()
        if line.strip()
    ]

    if len(values) < 4:
        raise RuntimeError(
            "ffprobe returned insufficient video information:\n"
            + result.stdout
        )

    width = int(values[0])
    height = int(values[1])

    num, den = values[2].split("/")
    fps = float(num) / float(den)

    duration = float(values[3])

    return VideoInfo(
        width=width,
        height=height,
        fps=fps,
        duration=duration,
    )


def _probe_with_opencv(video_path: Path) -> VideoInfo:
    """Read video metadata using OpenCV."""

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(
            f"OpenCV failed to open video: {video_path}"
        )

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    cap.release()

    duration = frame_count / fps if fps > 0 else 0.0

    return VideoInfo(
        width=width,
        height=height,
        fps=fps,
        duration=duration,
    )


def get_video_info(video_path: Path) -> VideoInfo:
    """Return video metadata using ffprobe or OpenCV."""

    ffprobe = _find_ffprobe()

    if ffprobe is not None:
        try:
            return _probe_with_ffprobe(video_path, ffprobe)
        except (OSError, subprocess.CalledProcessError, ValueError):
            pass

    return _probe_with_opencv(video_path)