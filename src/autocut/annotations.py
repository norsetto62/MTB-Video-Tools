"""Parsing and validation of AutoCut annotation manifests."""

import logging
from pathlib import Path

from .config import Config
from .utils import convert_hms_to_s

logger = logging.getLogger(__name__)

_AUDIO_EXTENSIONS = {".mp3", ".wav", ".ogg", ".flac"}
_VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi"}


def _resolve_source_path(value: str, annotation_file: Path) -> Path:
    """Resolve a source path relative to the annotation file when needed."""
    path = Path(value)

    if not path.is_absolute():
        path = annotation_file.parent / path

    return path


def _is_audio(value: str) -> bool:
    return Path(value).suffix.lower() in _AUDIO_EXTENSIONS


def _is_video(value: str) -> bool:
    return Path(value).suffix.lower() in _VIDEO_EXTENSIONS


def _looks_numeric(value: str) -> bool:
    try:
        float(value)
    except ValueError:
        return False

    return True


def load_annotations(
    annotation_file: str | Path,
    config: Config | None = None,
):
    """Load and validate an AutoCut annotation manifest.

    Returns:
        A tuple ``(video_clip_list, audio_config)``.

        ``audio_config`` is ``None`` when no music is specified.
    """
    config = config or Config()
    annotation_file = Path(annotation_file)

    if not annotation_file.is_file():
        raise FileNotFoundError(
            f"Annotation file not found: {annotation_file}"
        )

    clips = []
    last_video_name = None
    audio_config = None

    with annotation_file.open(
        "r",
        encoding="utf-8-sig",
    ) as f:
        for line_number, raw_line in enumerate(f, start=1):
            line = raw_line.strip()

            if not line or line.startswith("#"):
                continue

            parts = line.split(maxsplit=4)
            num_parts = len(parts)

            # ---------------------------------------------------------
            # Audio directive
            # ---------------------------------------------------------
            if num_parts in (1, 2) and _is_audio(parts[0]):
                audio_path = _resolve_source_path(
                    parts[0],
                    annotation_file,
                )

                if not audio_path.is_file():
                    raise FileNotFoundError(
                        f"Audio file not found on line "
                        f"{line_number}: {audio_path}"
                    )

                if num_parts == 2 and parts[1].strip() != "*":
                    raise ValueError(
                        f"Invalid audio directive on line "
                        f"{line_number}: '{line}'"
                    )

                mix = num_parts == 2

                if audio_config is not None:
                    logger.warning(
                        "Multiple audio files specified; replacing "
                        "'%s' with '%s'.",
                        audio_config["path"],
                        audio_path,
                    )

                audio_config = {
                    "path": str(audio_path.resolve()),
                    "mix": mix,
                }

                continue

            # ---------------------------------------------------------
            # Standalone video filename
            # ---------------------------------------------------------
            if num_parts == 1 and _is_video(parts[0]):
                video_path = _resolve_source_path(
                    parts[0],
                    annotation_file,
                )

                if not video_path.is_file():
                    raise FileNotFoundError(
                        f"Video file not found on line "
                        f"{line_number}: {video_path}"
                    )

                last_video_name = str(video_path.resolve())

                continue

            # ---------------------------------------------------------
            # Interval using previously declared video
            #
            #   start end
            #   start end *
            # ---------------------------------------------------------
            if ":" in parts[0] or _looks_numeric(parts[0]):
                if last_video_name is None:
                    raise ValueError(
                        f"Interval on line {line_number} omits a "
                        f"video filename before any video was declared: "
                        f"'{line}'"
                    )

                if num_parts not in (2, 3):
                    raise ValueError(
                        f"Invalid annotation row on line "
                        f"{line_number}: '{line}'"
                    )

                video_name = last_video_name
                start_text = parts[0]
                end_text = parts[1]

                mandatory = num_parts == 3

                if mandatory and parts[2].strip() != "*":
                    raise ValueError(
                        f"Invalid annotation directive on line "
                        f"{line_number}: '{line}'"
                    )

            # ---------------------------------------------------------
            # Interval with explicit video
            #
            #   video start end
            #   video start end *
            # ---------------------------------------------------------
            else:
                if num_parts not in (3, 4):
                    raise ValueError(
                        f"Invalid annotation row on line "
                        f"{line_number}: '{line}'"
                    )

                video_path = _resolve_source_path(
                    parts[0],
                    annotation_file,
                )

                if not video_path.is_file():
                    raise FileNotFoundError(
                        f"Video file not found on line "
                        f"{line_number}: {video_path}"
                    )

                video_name = str(video_path.resolve())
                last_video_name = video_name

                start_text = parts[1]
                end_text = parts[2]

                mandatory = num_parts == 4

                if mandatory and parts[3].strip() != "*":
                    raise ValueError(
                        f"Invalid annotation directive on line "
                        f"{line_number}: '{line}'"
                    )

            # ---------------------------------------------------------
            # Time conversion / validation
            # ---------------------------------------------------------
            try:
                start = convert_hms_to_s(start_text)
                end = convert_hms_to_s(end_text)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid timestamp on line "
                    f"{line_number}: '{line}'"
                ) from exc

            if end <= start:
                raise ValueError(
                    f"End time must be greater than start time "
                    f"on line {line_number}: '{line}'"
                )

            duration = end - start

            # ---------------------------------------------------------
            # Minimum clip duration
            # ---------------------------------------------------------
            if duration < config.min_clip:
                if mandatory:
                    logger.warning(
                        "Mandatory interval on line %d is shorter than "
                        "min_clip (%.3fs < %.3fs); keeping it: %s",
                        line_number,
                        duration,
                        config.min_clip,
                        line,
                    )
                else:
                    logger.warning(
                        "Ignoring short interval on line %d "
                        "(%.3fs < %.3fs): %s",
                        line_number,
                        duration,
                        config.min_clip,
                        line,
                    )
                    continue
            # ---------------------------------------------------------
            # Add clip to clips list
            # ---------------------------------------------------------
            clips.append(
                {
                    "video_name": video_name,
                    "start": start,
                    "end": end,
                    "duration": duration,
                    "mandatory": mandatory,
                }
            )

    return clips, audio_config