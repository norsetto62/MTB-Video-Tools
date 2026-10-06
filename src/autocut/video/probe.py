"""Video metadata probing."""

from __future__ import annotations

import json
import math
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


def _parse_rate(value: object) -> float:
    """Convert an FFmpeg frame-rate value to Hz.

    FFmpeg normally reports rates as rational strings such as
    ``30000/1001``. Invalid, missing, or zero rates return 0.0.
    """

    if value is None:
        return 0.0

    text = str(value).strip()

    if not text:
        return 0.0

    try:
        if "/" in text:
            numerator, denominator = text.split("/", 1)
            denominator_value = float(denominator)

            if denominator_value == 0:
                return 0.0

            return float(numerator) / denominator_value

        return float(text)

    except (TypeError, ValueError, ZeroDivisionError):
        return 0.0


def _parse_duration(value: object) -> float:
    """Convert an FFmpeg duration value to seconds.

    Missing or non-numeric durations return 0.0.
    """

    if value is None:
        return 0.0

    try:
        duration = float(value)
    except (TypeError, ValueError):
        return 0.0

    if not math.isfinite(duration):
        return 0.0

    return duration


def _validate_video_info(
    info: VideoInfo,
    video_path: Path,
) -> VideoInfo:
    """Validate the invariants required by the video pipeline."""

    if info.width <= 0:
        raise RuntimeError(
            f"Invalid video width for {video_path}: "
            f"{info.width}"
        )

    if info.height <= 0:
        raise RuntimeError(
            f"Invalid video height for {video_path}: "
            f"{info.height}"
        )

    if not math.isfinite(info.fps) or info.fps <= 0:
        raise RuntimeError(
            f"Invalid video FPS for {video_path}: "
            f"{info.fps}"
        )

    if not math.isfinite(info.duration) or info.duration <= 0:
        raise RuntimeError(
            f"Invalid video duration for {video_path}: "
            f"{info.duration}"
        )

    return info


def _probe_with_ffprobe(
    video_path: Path,
    ffprobe: str,
) -> VideoInfo:
    """Read video metadata using ffprobe JSON output."""

    cmd = [
        ffprobe,
        "-v",
        "error",
        "-show_streams",
        "-show_format",
        "-of",
        "json",
        str(video_path),
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=True,
    )

    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "ffprobe returned invalid JSON:\n"
            + result.stdout
        ) from exc

    streams = data.get("streams")

    if not isinstance(streams, list):
        raise RuntimeError(
            "ffprobe returned no stream information."
        )

    video_stream = next(
        (
            stream
            for stream in streams
            if stream.get("codec_type") == "video"
        ),
        None,
    )

    if video_stream is None:
        raise RuntimeError(
            f"No video stream found in: {video_path}"
        )

    try:
        width = int(video_stream.get("width", 0))
        height = int(video_stream.get("height", 0))
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"Invalid video dimensions reported by ffprobe: "
            f"{video_path}"
        ) from exc

    # avg_frame_rate represents the average presentation rate and is
    # generally the most useful rate for duration/frame calculations.
    # Fall back to r_frame_rate if avg_frame_rate is unavailable or invalid.
    fps = _parse_rate(
        video_stream.get("avg_frame_rate")
    )

    if fps <= 0:
        fps = _parse_rate(
            video_stream.get("r_frame_rate")
        )

    # Prefer stream duration when available. Some containers/codecs do not
    # expose it, in which case the container-level duration is useful.
    duration = _parse_duration(
        video_stream.get("duration")
    )

    if duration <= 0:
        format_info = data.get("format")

        if isinstance(format_info, dict):
            duration = _parse_duration(
                format_info.get("duration")
            )

    info = VideoInfo(
        width=width,
        height=height,
        fps=fps,
        duration=duration,
    )

    return _validate_video_info(info, video_path)


def _probe_with_opencv(video_path: Path) -> VideoInfo:
    """Read video metadata using OpenCV."""

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(
            f"OpenCV failed to open video: {video_path}"
        )

    try:
        width = int(
            cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        )
        height = int(
            cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        )
        fps = float(
            cap.get(cv2.CAP_PROP_FPS)
        )
        frame_count = float(
            cap.get(cv2.CAP_PROP_FRAME_COUNT)
        )
    finally:
        cap.release()

    duration = (
        frame_count / fps
        if fps > 0 and frame_count > 0
        else 0.0
    )

    info = VideoInfo(
        width=width,
        height=height,
        fps=fps,
        duration=duration,
    )

    return _validate_video_info(info, video_path)


def get_video_info(video_path: Path) -> VideoInfo:
    """Return validated video metadata.

    FFprobe is preferred. OpenCV is used only when FFprobe cannot
    be executed successfully. Metadata successfully returned by FFprobe
    but failing validation is treated as a hard error rather than silently
    falling back to another backend.
    """

    video_path = Path(video_path)

    if not video_path.is_file():
        raise FileNotFoundError(
            f"Video file not found: {video_path}"
        )

    ffprobe = _find_ffprobe()

    if ffprobe is not None:
        try:
            return _probe_with_ffprobe(
                video_path,
                ffprobe,
            )
        except OSError:
            # FFprobe could not be executed. OpenCV is a legitimate
            # alternative backend.
            pass
        except subprocess.CalledProcessError:
            # FFprobe executed but failed to probe the file.
            # Allow OpenCV to attempt the fallback.
            pass

    return _probe_with_opencv(video_path)