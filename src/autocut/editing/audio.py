"""FFmpeg-based soundtrack mixing for a finalized video edit."""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
import shutil
import subprocess
import tempfile
from pathlib import Path

from autocut.audio.config import AudioConfig
from autocut.selection.timeline import Timeline
from autocut.video.probe import _find_ffprobe


def _duration(name: str, value: Real, *, allow_zero: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite number.")
    result = float(value)
    if not math.isfinite(result) or result < 0 or (not allow_zero and result == 0):
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be finite and {qualifier}.")
    return result


def _find_ffmpeg() -> str:
    """Locate FFmpeg on PATH, with a Windows installation fallback."""
    executable = shutil.which("ffmpeg")
    if executable:
        return executable

    fallback = Path(r"C:\Program Files\ffmpeg\bin\ffmpeg.exe")
    if fallback.is_file():
        return str(fallback)
    raise FileNotFoundError(
        "FFmpeg was not found on PATH or at "
        r"C:\Program Files\ffmpeg\bin\ffmpeg.exe."
    )


@dataclass(frozen=True)
class RenderResult:
    """Details about a completed render."""

    output_path: Path
    duration: float


def build_music_mix_command(
    video_path: str | Path,
    music_path: str | Path,
    output_path: str | Path,
    *,
    duration: float,
    fade_duration: float,
    ffmpeg: str | Path | None = None,
    mix_original_audio: bool = False,
) -> list[str]:
    """Build a safe FFmpeg command to copy video and encode the soundtrack."""
    total = _duration("duration", duration)
    fade = _duration("fade_duration", fade_duration, allow_zero=True)
    fade = min(fade, total)

    command = [
        str(ffmpeg) if ffmpeg is not None else _find_ffmpeg(),
        "-hide_banner",
        "-loglevel", "error",
        "-y",
        "-i", str(video_path),
        "-i", str(music_path),
        "-map", "0:v:0",
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "320k",
    ]
    if mix_original_audio:
        audio_filter = "[0:a:0][1:a:0]amix=inputs=2:duration=first:dropout_transition=2"
        if fade > 0:
            fade_start = max(0.0, total - fade)
            audio_filter += f",afade=t=out:st={fade_start}:d={fade}"
        command.extend(["-filter_complex", f"{audio_filter}[aout]", "-map", "[aout]"])
    else:
        command.extend(["-map", "1:a:0"])
        if fade > 0:
            fade_start = max(0.0, total - fade)
            command.extend(["-af", f"afade=t=out:st={fade_start}:d={fade}"])
    command.extend([
        "-t", str(total),
        "-movflags", "+faststart",
        str(output_path),
    ])
    return command


def probe_duration(media_path: str | Path) -> float:
    """Return the container duration reported by ffprobe."""
    ffprobe = _find_ffprobe()
    if ffprobe is None:
        raise FileNotFoundError(
            "ffprobe was not found on PATH or at "
            r"C:\Program Files\ffmpeg\bin\ffprobe.exe."
        )
    command = [
        ffprobe, "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(media_path),
    ]
    result = subprocess.run(
        command, capture_output=True, text=True, check=True
    )
    try:
        duration = float(result.stdout.strip())
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"ffprobe returned an invalid duration for {media_path}: "
            f"{result.stdout!r}"
        ) from exc
    if not math.isfinite(duration) or duration <= 0:
        raise RuntimeError(
            f"ffprobe returned a non-positive duration for {media_path}: "
            f"{duration!r}"
        )
    return duration


def mix_music(
    video_path: str | Path,
    music_path: str | Path,
    output_path: str | Path,
    *,
    duration: float,
    config: AudioConfig,
    mix_original_audio: bool = False,
) -> RenderResult:
    """Add music to a rendered edit, optionally retaining its original audio."""
    video = Path(video_path)
    music = Path(music_path)
    output = Path(output_path)
    if not isinstance(config, AudioConfig):
        raise TypeError("config must be an AudioConfig instance.")
    # Let FFmpeg diagnose missing/unreadable inputs: its stderr includes the
    # decoder and stream context that is most useful when troubleshooting.
    if video.resolve() == output.resolve() or music.resolve() == output.resolve():
        raise ValueError("Output path must differ from the input paths.")
    output.parent.mkdir(parents=True, exist_ok=True)

    command = build_music_mix_command(
        video, music, output,
        duration=duration,
        fade_duration=config.fade_duration,
        mix_original_audio=mix_original_audio,
    )
    try:
        subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.strip() if isinstance(exc.stderr, str) else ""
        detail = stderr or "FFmpeg returned a non-zero exit status."
        raise RuntimeError(
            f"Failed to mix music into {video} (exit status {exc.returncode}): "
            f"{detail}"
        ) from exc
    except OSError as exc:
        raise RuntimeError(f"Could not execute FFmpeg: {exc}") from exc

    try:
        measured_duration = probe_duration(output)
    except (OSError, subprocess.SubprocessError, RuntimeError, ValueError) as exc:
        raise RuntimeError(
            f"Music mix completed but output duration could not be measured: "
            f"{output}: {exc}"
        ) from exc
    return RenderResult(output_path=output, duration=measured_duration)


def _render_reencoded_transport_stream(
    timeline: Timeline,
    temporary_directory: Path,
    *,
    include_audio: bool,
    video_encoder: str,
) -> Path:
    """Re-encode each source interval before concatenating MPEG-TS clips."""
    ffmpeg = _find_ffmpeg()
    clip_files: list[Path] = []
    for index, clip in enumerate(timeline.clips):
        source = Path(clip.video_name).resolve()
        if not source.is_file():
            raise FileNotFoundError(f"Timeline source video not found: {clip.video_name}")
        start = _duration("clip source_start", clip.source_start, allow_zero=True)
        end = _duration("clip source_end", clip.source_end)
        if end <= start:
            raise ValueError(f"Invalid source interval for {clip.video_name}: {start}–{end}.")

        clip_file = temporary_directory / f"clip_{index:04d}.ts"
        command = [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
            "-ss", str(start), "-i", str(source), "-t", str(end - start),
            "-map", "0:v:0",
        ]
        if include_audio:
            command.extend(["-map", "0:a:0?"])
        else:
            command.append("-an")
        if video_encoder == "h264_nvenc":
            command.extend(["-c:v", "h264_nvenc", "-preset", "p4", "-cq", "20"])
        elif video_encoder == "libx264":
            command.extend(["-c:v", "libx264", "-preset", "fast", "-crf", "20"])
        else:
            raise ValueError("video_encoder must be 'h264_nvenc' or 'libx264'.")
        command.extend(["-pix_fmt", "yuv420p"])
        if include_audio:
            command.extend(["-c:a", "aac", "-b:a", "320k"])
        command.extend(["-f", "mpegts", str(clip_file)])
        try:
            subprocess.run(command, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as exc:
            stderr = exc.stderr.strip() if isinstance(exc.stderr, str) else ""
            raise RuntimeError(
                f"Failed to encode timeline clip {index + 1} "
                f"({source}, {start:.3f}–{end:.3f}; exit status {exc.returncode}): "
                f"{stderr or 'FFmpeg returned a non-zero exit status.'}"
            ) from exc
        except OSError as exc:
            raise RuntimeError(f"Could not execute FFmpeg: {exc}") from exc
        clip_files.append(clip_file)

    concat_file = temporary_directory / "clips.ffconcat"
    lines = ["ffconcat version 1.0"]
    for clip_file in clip_files:
        escaped = str(clip_file.resolve()).replace("\\", "/").replace("'", "'\\''")
        lines.append(f"file '{escaped}'")
    concat_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    assembled = temporary_directory / "assembled.ts"
    command = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-f", "concat", "-safe", "0", "-i", str(concat_file),
        "-c", "copy", str(assembled),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.strip() if isinstance(exc.stderr, str) else ""
        raise RuntimeError(
            f"Failed to concatenate re-encoded timeline clips (exit status {exc.returncode}): "
            f"{stderr or 'FFmpeg returned a non-zero exit status.'}"
        ) from exc
    except OSError as exc:
        raise RuntimeError(f"Could not execute FFmpeg: {exc}") from exc
    return assembled


def render_timeline(
    timeline: Timeline,
    output_path: str | Path,
    *,
    video_encoder: str = "h264_nvenc",
) -> RenderResult:
    """Render a timeline by re-encoding each clip before concatenation."""
    if not isinstance(timeline, Timeline):
        raise TypeError("timeline must be a Timeline instance.")
    if not timeline.clips:
        raise ValueError("Cannot render a timeline with no clips.")
    if not math.isfinite(timeline.duration) or timeline.duration <= 0:
        raise ValueError("timeline duration must be finite and positive.")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".autocut-render-", dir=str(output.parent)
    ) as directory:
        temporary = Path(directory)
        assembled = _render_reencoded_transport_stream(
            timeline, temporary, include_audio=True, video_encoder=video_encoder
        )
        command = [
            _find_ffmpeg(), "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(assembled), "-map", "0:v:0", "-map", "0:a?",
            "-c", "copy", "-movflags", "+faststart", str(output),
        ]
        try:
            subprocess.run(command, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as exc:
            stderr = exc.stderr.strip() if isinstance(exc.stderr, str) else ""
            raise RuntimeError(
                f"Failed to finalize rendered timeline (exit status {exc.returncode}): "
                f"{stderr or 'FFmpeg returned a non-zero exit status.'}"
            ) from exc
        except OSError as exc:
            raise RuntimeError(f"Could not execute FFmpeg: {exc}") from exc
    return RenderResult(output_path=output, duration=probe_duration(output))

def render_with_music(
    timeline: Timeline,
    music_path: str | Path,
    output_path: str | Path,
    *,
    config: AudioConfig,
    mix_original_audio: bool = False,
    video_encoder: str = "h264_nvenc",
) -> RenderResult:
    """Re-encode each selected clip, concatenate them, then add the soundtrack."""
    if not isinstance(timeline, Timeline):
        raise TypeError("timeline must be a Timeline instance.")
    if not isinstance(config, AudioConfig):
        raise TypeError("config must be an AudioConfig instance.")
    if not timeline.clips:
        raise ValueError("Cannot render a timeline with no clips.")
    if not math.isfinite(timeline.duration) or timeline.duration <= 0:
        raise ValueError("timeline duration must be finite and positive.")

    music = Path(music_path)
    output = Path(output_path)
    if not music.is_file():
        raise FileNotFoundError(f"Music file not found: {music}")
    output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(
        prefix=".autocut-render-", dir=str(output.parent)
    ) as temporary_directory:
        temporary = Path(temporary_directory)
        assembled = _render_reencoded_transport_stream(
            timeline, temporary, include_audio=mix_original_audio,
            video_encoder=video_encoder,
        )

        return mix_music(
            assembled, music, output,
            duration=timeline.duration, config=config,
            mix_original_audio=mix_original_audio,
        )

