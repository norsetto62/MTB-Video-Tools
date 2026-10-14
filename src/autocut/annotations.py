"""Parsing and validation of AutoCut annotation manifests."""

import logging
from pathlib import Path

from .utils import convert_hms_to_s
from .data_models import Clip, AudioConfig, Annotation

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


def load_training_annotations(
    path: str | Path,
) -> tuple[Path, list[Annotation]]:
    """
    Load the current training annotation format.

    Expected:

        source-video-path

        Start End MTB Remarks

    Remarks are optional.
    """

    path = Path(path)
    lines = path.read_text(encoding="utf-8-sig").splitlines()

    video_path: Path | None = None
    annotations: list[Annotation] = []

    for line_no, raw in enumerate(lines, start=1):
        line = raw.strip()

        if not line or line.startswith("#"):
            continue

        lower = line.lower()

        # Header.
        if "start" in lower and "end" in lower and "mtb" in lower:
            continue

        # Source video path.
        if video_path is None:
            video_path = _resolve_source_path(line, path)
            if not video_path.is_file():
                raise FileNotFoundError(
                    f"Video file not found on line {line_no}: {video_path}"
                )
            continue

        parts = line.split(maxsplit=3)

        if len(parts) < 3:
            raise ValueError(
                f"{path}:{line_no}: expected "
                "Start End MTB [Remarks]"
            )

        try:
            start = convert_hms_to_s(parts[0])
            end = convert_hms_to_s(parts[1])
            score = int(parts[2])
        except ValueError as exc:
            raise ValueError(
                f"{path}:{line_no}: {exc}"
            ) from exc

        if end <= start:
            raise ValueError(
                f"{path}:{line_no}: End must be greater than Start"
            )

        if not 0 <= score <= 3:
            raise ValueError(
                f"{path}:{line_no}: MTB score must be 0..3"
            )

        remarks = parts[3].strip() if len(parts) == 4 else ""

        annotations.append(
            Annotation(
                start=start,
                end=end,
                score=score,
                remarks=remarks,
            )
        )

    if video_path is None:
        raise ValueError(
            f"{path}: source video path not found"
        )

    annotations.sort(key=lambda annotation: annotation.start)

    # Training intervals must be contiguous
    previous: Annotation | None = None

    for annotation in annotations:
        if (
            previous is not None
            and annotation.start != previous.end
        ):
            raise ValueError(
                f"{path}: non contiguous annotations: "
                f"{previous.start:.3f}-{previous.end:.3f}s and "
                f"{annotation.start:.3f}-{annotation.end:.3f}s"
            )

        previous = annotation

    return video_path, annotations


def load_annotations(
    annotation_file: str | Path,
    *,
    min_clip: float = 3.0,
):
    """Load and validate an AutoCut annotation manifest.

    Returns:
        A tuple ``(video_clip_list, audio_config)``.

        ``audio_config`` is ``None`` when no music is specified.
    """
    if isinstance(min_clip, bool) or not isinstance(min_clip, (int, float)):
        raise ValueError("min_clip must be a positive finite number.")
    min_clip = float(min_clip)
    if min_clip <= 0 or min_clip == float("inf") or min_clip != min_clip:
        raise ValueError("min_clip must be a positive finite number.")

    annotation_file = Path(annotation_file)

    if not annotation_file.is_file():
        raise FileNotFoundError(
            f"Annotation file not found: {annotation_file}"
        )

    clips = []
    last_video_name = None
    audio_config = None
    declared_videos: set[str] = set()
    declared_video_order: dict[str, int] = {}
    videos_with_intervals: set[str] = set()

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
            #
            # Audio paths are standalone lines and may contain spaces
            # (including apostrophes). Detect the extension on the full
            # line rather than on a whitespace-split token.
            # ---------------------------------------------------------
            audio_text = line
            mix = False
            if audio_text.endswith(" *"):
                audio_text = audio_text[:-2].rstrip()
                mix = True

            if _is_audio(audio_text):
                audio_path = _resolve_source_path(
                    audio_text,
                    annotation_file,
                )

                if not audio_path.is_file():
                    raise FileNotFoundError(
                        f"Audio file not found on line "
                        f"{line_number}: {audio_path}"
                    )

                if audio_config is not None:
                    logger.warning(
                        "Multiple audio files specified; replacing "
                        "'%s' with '%s'.",
                        audio_config.path,
                        audio_path,
                    )

                audio_config = AudioConfig(
                    path = str(audio_path.resolve()),
                    mix = mix,
                )

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
                declared_videos.add(last_video_name)
                declared_video_order.setdefault(last_video_name, line_number)

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
                videos_with_intervals.add(video_name)
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
                declared_videos.add(video_name)
                declared_video_order.setdefault(video_name, line_number)
                videos_with_intervals.add(video_name)

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
            if duration < min_clip:
                if mandatory:
                    logger.warning(
                        "Mandatory interval on line %d is shorter than "
                        "min_clip (%.3fs < %.3fs); keeping it: %s",
                        line_number,
                        duration,
                        min_clip,
                        line,
                    )
                else:
                    logger.warning(
                        "Ignoring short interval on line %d "
                        "(%.3fs < %.3fs): %s",
                        line_number,
                        duration,
                        min_clip,
                        line,
                    )
                    continue
            # ---------------------------------------------------------
            # Add clip to clips list
            # ---------------------------------------------------------
            clips.append(
                Clip(
                    video_name = video_name,
                    start = start,
                    end = end,
                    mandatory = mandatory,
                    annotation_order = line_number,
                )
            )

    # A video-only manifest means "process the whole video". Do this only
    # when no interval row was declared for that video; intervals that were
    # skipped for being too short must not silently turn into full-video runs.
    for video_name in declared_videos - videos_with_intervals:
        from .video.probe import get_video_info

        info = get_video_info(Path(video_name))
        clips.append(
            Clip(
                video_name=video_name,
                start=0.0,
                end=float(info.duration),
                mandatory=False,
                annotation_order=declared_video_order[video_name],
            )
        )

    # A video-only declaration is materialized after parsing because its
    # duration requires probing. Restore manifest order now that all intervals
    # exist, using physical line numbers retained on Clip.
    clips.sort(
        key=lambda clip: (
            clip.annotation_order
            if clip.annotation_order is not None
            else float("inf")
        )
    )
    return clips, audio_config
