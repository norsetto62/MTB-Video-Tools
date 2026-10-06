"""Sequential video frame reader backed by FFmpeg."""

import logging
import shutil
import subprocess
from pathlib import Path
from typing import Iterator

import numpy as np

from ..data_models import VideoInfo
from .probe import get_video_info

logger = logging.getLogger(__name__)

DEFAULT_FFMPEG = Path(
    r"C:\Program Files\ffmpeg\bin\ffmpeg.exe"
)


class VideoReader:
    """Read decoded BGR frames from a video using FFmpeg.

    FFmpeg is started when the reader context is entered. Each iteration
    yields one decoded frame together with its index and timestamp.

    Timestamps are relative to the requested start position.
    """

    def __init__(
        self,
        video_path: str | Path,
        start: float = 0.0,
        duration: float | None = None,
        sample_fps: float | None = None,
    ) -> None:
        self.video_path = Path(video_path)
        self.start = float(start)
        self.duration = (
            None if duration is None else float(duration)
        )
        self.sample_fps = (
            None if sample_fps is None else float(sample_fps)
        )

        self.info: VideoInfo | None = None
        self._process: subprocess.Popen | None = None
        self._frame_bytes = 0
        self._output_fps = 0.0
        self._frame_index = 0

    def __enter__(self) -> "VideoReader":
        self._start_decoder()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def __iter__(
        self,
    ) -> Iterator[tuple[int, float, np.ndarray]]:
        if self._process is None:
            raise RuntimeError(
                "VideoReader must be used inside a context manager."
            )

        if self.info is None:
            raise RuntimeError("Video information is unavailable.")

        if self._process.stdout is None:
            raise RuntimeError("FFmpeg stdout is unavailable.")

        while True:
            raw = self._process.stdout.read(self._frame_bytes)

            if len(raw) == 0:
                break

            if len(raw) != self._frame_bytes:
                raise RuntimeError(
                    "Incomplete frame received from FFmpeg."
                )

            frame = np.frombuffer(
                raw,
                dtype=np.uint8,
            ).reshape(
                self.info.height,
                self.info.width,
                3,
            )

            frame_index = self._frame_index
            timestamp = frame_index / self._output_fps

            self._frame_index += 1

            yield frame_index, timestamp, frame

            if (
                self.duration is not None
                and timestamp >= self.duration
            ):
                break

    def close(self) -> None:
        """Stop FFmpeg and release its resources."""

        process = self._process

        if process is None:
            return

        try:
            if process.poll() is None:
                process.terminate()

            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()

        finally:
            if process.stdout is not None:
                process.stdout.close()

            if process.stderr is not None:
                process.stderr.close()

            self._process = None

    def _start_decoder(self) -> None:
        """Probe the video and start the FFmpeg decoder."""

        if self._process is not None:
            raise RuntimeError("VideoReader is already active.")

        if not self.video_path.is_file():
            raise FileNotFoundError(
                f"Video file not found: {self.video_path}"
            )

        if self.start < 0:
            raise ValueError("start must be non-negative.")

        if self.duration is not None and self.duration <= 0:
            raise ValueError("duration must be greater than zero.")

        if self.sample_fps is not None and self.sample_fps <= 0:
            raise ValueError("sample_fps must be greater than zero.")

        self.info = get_video_info(self.video_path)

        processing_duration = self._get_processing_duration()

        if processing_duration <= 0:
            raise RuntimeError(
                "Selected processing interval has zero duration."
            )

        ffmpeg = self._find_ffmpeg()

        self._output_fps = (
            self.sample_fps
            if self.sample_fps is not None
            else self.info.fps
        )

        self._frame_bytes = (
            self.info.width *
            self.info.height *
            3
        )

        cmd = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
        ]

        if self.start > 0:
            cmd += [
                "-ss",
                f"{self.start:.6f}",
            ]

        cmd += [
            "-i",
            str(self.video_path),
        ]

        cmd += [
            "-t",
            f"{processing_duration:.6f}",
        ]

        if self.sample_fps is not None:
            cmd += [
                "-vf",
                f"fps={self.sample_fps:.8f}",
            ]

        cmd += [
            "-an",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "pipe:1",
        ]

        self._frame_index = 0

        try:
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=10**8,
            )
        except OSError:
            self._process = None
            raise

    def _get_processing_duration(self) -> float:
        """Return the actual duration to request from FFmpeg."""

        if self.info is None:
            raise RuntimeError("Video information is unavailable.")

        remaining = self.info.duration - self.start

        if self.duration is None:
            return max(0.0, remaining)

        return max(
            0.0,
            min(self.duration, remaining),
        )

    @staticmethod
    def _find_ffmpeg() -> str:
        """Return the FFmpeg executable path."""

        ffmpeg = shutil.which("ffmpeg")

        if ffmpeg is not None:
            return ffmpeg

        if DEFAULT_FFMPEG.is_file():
            return str(DEFAULT_FFMPEG)

        raise FileNotFoundError(
            "FFmpeg executable not found. "
            "Install FFmpeg or add it to PATH."
        )