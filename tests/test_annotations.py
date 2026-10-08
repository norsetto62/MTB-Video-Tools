import logging

from pathlib import Path

import pytest

from autocut.annotations import (
    load_annotations,
    load_training_annotations,
)

from autocut.config import Config

from autocut.data_models import Annotation, Clip, AudioConfig


def write_annotation_file(
    tmp_path: Path,
    content: str,
) -> Path:
    path = tmp_path / "annotations.txt"
    path.write_text(content, encoding="utf-8")
    return path


def create_video(
    tmp_path: Path,
    name: str = "ride.mp4",
) -> Path:
    path = tmp_path / name
    path.touch()
    return path


def create_audio(
    tmp_path: Path,
    name: str = "music.mp3",
) -> Path:
    path = tmp_path / name
    path.touch()
    return path


# ---------------------------------------------------------------------------
# Basic parsing
# ---------------------------------------------------------------------------


def test_load_annotations_returns_clips_and_no_audio(tmp_path):
    video = create_video(tmp_path)
    annotation_file = write_annotation_file(
        tmp_path,
        f"{video.name}\n"
        "00:10 00:20\n",
    )

    clips, audio = load_annotations(annotation_file)

    assert audio is None
    assert len(clips) == 1
    assert clips[0] == Clip(
        video_name=str(video.resolve()),
        start=10.0,
        end=20.0,
        mandatory=False,
    )


def test_explicit_video_interval(tmp_path):
    video = create_video(tmp_path)
    annotation_file = write_annotation_file(
        tmp_path,
        f"{video.name} 00:10 00:20\n",
    )

    clips, audio = load_annotations(annotation_file)

    assert audio is None
    assert len(clips) == 1
    assert clips[0].video_name == str(video.resolve())
    assert clips[0].start == 10.0
    assert clips[0].end == 20.0


def test_subsequent_intervals_use_previous_video(tmp_path):
    video = create_video(tmp_path)
    annotation_file = write_annotation_file(
        tmp_path,
        f"{video.name}\n"
        "00:10 00:20\n"
        "00:30 00:40\n",
    )

    clips, _ = load_annotations(annotation_file)

    assert len(clips) == 2
    assert all(
        clip.video_name == str(video.resolve())
        for clip in clips
    )


# ---------------------------------------------------------------------------
# Mandatory flag
# ---------------------------------------------------------------------------


def test_mandatory_interval(tmp_path):
    video = create_video(tmp_path)
    annotation_file = write_annotation_file(
        tmp_path,
        f"{video.name}\n"
        "00:10 00:20 *\n",
    )

    clips, _ = load_annotations(annotation_file)

    assert len(clips) == 1
    assert clips[0].mandatory is True


def test_non_mandatory_interval(tmp_path):
    video = create_video(tmp_path)
    annotation_file = write_annotation_file(
        tmp_path,
        f"{video.name}\n"
        "00:10 00:20\n",
    )

    clips, _ = load_annotations(annotation_file)

    assert clips[0].mandatory is False


def test_invalid_mandatory_marker_is_rejected(tmp_path):
    video = create_video(tmp_path)
    annotation_file = write_annotation_file(
        tmp_path,
        f"{video.name}\n"
        "00:10 00:20 X\n",
    )

    with pytest.raises(ValueError):
        load_annotations(annotation_file)


def test_mandatory_short_interval_is_kept(
    tmp_path,
    caplog,
):
    video = tmp_path / "ride.mp4"
    video.touch()

    annotation_file = tmp_path / "annotations.txt"
    annotation_file.write_text(
        "ride.mp4 00:10 00:11 *\n",
        encoding="utf-8",
    )

    config = Config(min_clip=3.0)

    with caplog.at_level(logging.WARNING):
        clips, audio = load_annotations(
            annotation_file,
            config,
        )

    assert audio is None
    assert len(clips) == 1
    assert clips[0].start == 10.0
    assert clips[0].end == 11.0
    assert clips[0].duration == 1.0
    assert clips[0].mandatory is True
    assert "shorter than min_clip" in caplog.text


# ---------------------------------------------------------------------------
# Comments / blank lines
# ---------------------------------------------------------------------------


def test_blank_lines_and_comments_are_ignored(tmp_path):
    video = create_video(tmp_path)
    annotation_file = write_annotation_file(
        tmp_path,
        "# comment\n"
        "\n"
        f"{video.name}\n"
        "\n"
        "00:10 00:20\n"
        "# another comment\n",
    )

    clips, audio = load_annotations(annotation_file)

    assert audio is None
    assert len(clips) == 1


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


def test_relative_video_path_is_resolved_relative_to_annotation_file(
    tmp_path,
):
    video_dir = tmp_path / "videos"
    video_dir.mkdir()

    video = video_dir / "ride.mp4"
    video.touch()

    annotation_file = write_annotation_file(
        tmp_path,
        "videos/ride.mp4 00:10 00:20\n",
    )

    clips, _ = load_annotations(annotation_file)

    assert clips[0].video_name == str(video.resolve())


def test_missing_annotation_file_raises():
    with pytest.raises(FileNotFoundError):
        load_annotations(
            "/definitely/nonexistent/annotations.txt"
        )


def test_missing_video_raises(tmp_path):
    annotation_file = write_annotation_file(
        tmp_path,
        "missing.mp4 00:10 00:20\n",
    )

    with pytest.raises(FileNotFoundError):
        load_annotations(annotation_file)


# ---------------------------------------------------------------------------
# Time validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("start", "end"),
    [
        ("00:20", "00:10"),
        ("00:20", "00:20"),
    ],
)
def test_end_must_be_greater_than_start(
    tmp_path,
    start,
    end,
):
    video = create_video(tmp_path)

    annotation_file = write_annotation_file(
        tmp_path,
        f"{video.name}\n"
        f"{start} {end}\n",
    )

    with pytest.raises(ValueError):
        load_annotations(annotation_file)


def test_invalid_timestamp_is_rejected(tmp_path):
    video = create_video(tmp_path)

    annotation_file = write_annotation_file(
        tmp_path,
        f"{video.name}\n"
        "01:72:00 02:00:00\n",
    )

    with pytest.raises(ValueError):
        load_annotations(annotation_file)


# ---------------------------------------------------------------------------
# Minimum clip duration
# ---------------------------------------------------------------------------


def test_short_clip_is_skipped_and_warning_is_logged(
    tmp_path,
    caplog,
):
    video = create_video(tmp_path)

    annotation_file = write_annotation_file(
        tmp_path,
        f"{video.name}\n"
        "00:10 00:12\n"
        "00:20 00:30\n",
    )

    config = Config(min_clip=3.0)

    with caplog.at_level(logging.WARNING):
        clips, _ = load_annotations(
            annotation_file,
            config=config,
        )

    assert len(clips) == 1
    assert clips[0].start == 20.0
    assert any(
        "Ignoring short interval" in record.message
        for record in caplog.records
    )


def test_clip_at_minimum_duration_is_kept(tmp_path):
    video = create_video(tmp_path)

    annotation_file = write_annotation_file(
        tmp_path,
        f"{video.name}\n"
        "00:10 00:13\n",
    )

    config = Config(min_clip=3.0)

    clips, _ = load_annotations(
        annotation_file,
        config=config,
    )

    assert len(clips) == 1
    assert clips[0].duration == 3.0


# ---------------------------------------------------------------------------
# Audio
# ---------------------------------------------------------------------------


def test_audio_without_mix_flag(tmp_path):
    video = create_video(tmp_path)
    audio = create_audio(tmp_path)

    annotation_file = write_annotation_file(
        tmp_path,
        f"{video.name}\n"
        "00:10 00:20\n"
        f"{audio.name}\n",
    )

    clips, audio_config = load_annotations(annotation_file)

    assert len(clips) == 1
    assert audio_config == AudioConfig(
        path=str(audio.resolve()),
        mix=False,
    )


def test_audio_with_mix_flag(tmp_path):
    video = create_video(tmp_path)
    audio = create_audio(tmp_path)

    annotation_file = write_annotation_file(
        tmp_path,
        f"{video.name}\n"
        "00:10 00:20\n"
        f"{audio.name} *\n",
    )

    _, audio_config = load_annotations(annotation_file)

    assert audio_config == AudioConfig(
        path=str(audio.resolve()),
        mix=True,
    )


def test_missing_audio_raises(tmp_path):
    video = create_video(tmp_path)

    annotation_file = write_annotation_file(
        tmp_path,
        f"{video.name}\n"
        "00:10 00:20\n"
        "missing.mp3\n",
    )

    with pytest.raises(FileNotFoundError):
        load_annotations(annotation_file)


def test_multiple_audio_files_warn_and_last_one_wins(
    tmp_path,
    caplog,
):
    video = create_video(tmp_path)
    audio1 = create_audio(tmp_path, "first.mp3")
    audio2 = create_audio(tmp_path, "second.mp3")

    annotation_file = write_annotation_file(
        tmp_path,
        f"{video.name}\n"
        "00:10 00:20\n"
        f"{audio1.name}\n"
        f"{audio2.name} *\n",
    )

    with caplog.at_level(logging.WARNING):
        _, audio_config = load_annotations(annotation_file)

    assert audio_config == AudioConfig(
        path=str(audio2.resolve()),
        mix=True,
    )

    assert any(
        "Multiple audio files specified" in record.message
        for record in caplog.records
    )


# ===========================================================================
# Training annotations
# ===========================================================================


def test_load_training_annotations_returns_video_and_annotations(
    tmp_path,
):
    video = create_video(tmp_path)

    annotation_file = write_annotation_file(
        tmp_path,
        f"{video.name}\n"
        "\n"
        "Start End MTB Remarks\n"
        "00:10 00:14 1 ordinary riding\n"
        "00:20 00:24 3 good technical section\n",
    )

    loaded_video, annotations = load_training_annotations(
        annotation_file
    )

    assert loaded_video == video.resolve()
    assert len(annotations) == 2

    assert annotations[0] == Annotation(
        start=10.0,
        end=14.0,
        score=1,
        remarks="ordinary riding",
    )

    assert annotations[1] == Annotation(
        start=20.0,
        end=24.0,
        score=3,
        remarks="good technical section",
    )


def test_training_annotation_remarks_are_optional(tmp_path):
    video = create_video(tmp_path)

    annotation_file = write_annotation_file(
        tmp_path,
        f"{video.name}\n"
        "Start End MTB Remarks\n"
        "00:10 00:14 2\n",
    )

    loaded_video, annotations = load_training_annotations(
        annotation_file
    )

    assert loaded_video == video.resolve()
    assert len(annotations) == 1
    assert annotations[0].remarks == ""


def test_training_blank_lines_and_comments_are_ignored(
    tmp_path,
):
    video = create_video(tmp_path)

    annotation_file = write_annotation_file(
        tmp_path,
        "# training annotations\n"
        "\n"
        f"{video.name}\n"
        "\n"
        "# intervals\n"
        "Start End MTB Remarks\n"
        "\n"
        "00:10 00:14 1 ordinary\n"
        "# another comment\n"
        "00:20 00:24 3 technical\n",
    )

    _, annotations = load_training_annotations(
        annotation_file
    )

    assert len(annotations) == 2
    assert annotations[0].score == 1
    assert annotations[1].score == 3


def test_training_annotations_accept_utf8_bom(tmp_path):
    video = create_video(tmp_path)

    annotation_file = tmp_path / "annotations.txt"
    annotation_file.write_text(
        f"{video.name}\n"
        "Start End MTB Remarks\n"
        "00:10 00:14 2 test\n",
        encoding="utf-8-sig",
    )

    loaded_video, annotations = load_training_annotations(
        annotation_file
    )

    assert loaded_video == video.resolve()
    assert len(annotations) == 1
    assert annotations[0].score == 2


def test_training_relative_video_path_is_resolved_relative_to_annotation_file(
    tmp_path,
):
    video_dir = tmp_path / "videos"
    video_dir.mkdir()

    video = video_dir / "ride.mp4"
    video.touch()

    annotation_file = write_annotation_file(
        tmp_path,
        "videos/ride.mp4\n"
        "Start End MTB Remarks\n"
        "00:10 00:14 3 technical\n",
    )

    loaded_video, annotations = load_training_annotations(
        annotation_file
    )

    assert loaded_video == video.resolve()
    assert len(annotations) == 1


def test_training_annotations_are_sorted_by_start_time(
    tmp_path,
):
    video = create_video(tmp_path)

    annotation_file = write_annotation_file(
        tmp_path,
        f"{video.name}\n"
        "Start End MTB Remarks\n"
        "00:20 00:24 3 later\n"
        "00:05 00:10 1 earlier\n"
        "00:12 00:16 2 middle\n",
    )

    _, annotations = load_training_annotations(
        annotation_file
    )

    assert [annotation.start for annotation in annotations] == [
        5.0,
        12.0,
        20.0,
    ]


def test_training_missing_source_video_path_is_rejected(
    tmp_path,
):
    annotation_file = write_annotation_file(
        tmp_path,
        "# no source video\n"
        "\n",
    )

    with pytest.raises(
        ValueError,
        match="source video path not found",
    ):
        load_training_annotations(annotation_file)
        

def test_training_nonexistent_video_is_rejected(tmp_path):
    annotation_file = write_annotation_file(
        tmp_path,
        "missing.mp4\n"
        "Start End MTB Remarks\n"
        "00:10 00:14 1 ordinary\n",
    )

    with pytest.raises(
        FileNotFoundError,
        match="Video file not found",
    ):
        load_training_annotations(annotation_file)


def test_training_malformed_annotation_row_is_rejected(
    tmp_path,
):
    video = create_video(tmp_path)

    annotation_file = write_annotation_file(
        tmp_path,
        f"{video.name}\n"
        "Start End MTB Remarks\n"
        "00:10 1\n",
    )

    with pytest.raises(
        ValueError,
        match="expected Start End MTB",
    ):
        load_training_annotations(annotation_file)


@pytest.mark.parametrize(
    ("start", "end"),
    [
        ("00:10:00", "00:09:00"),
        ("00:10:00", "00:10:00"),
    ],
)
def test_training_end_must_be_greater_than_start(
    tmp_path,
    start,
    end,
):
    video = create_video(tmp_path)

    annotation_file = write_annotation_file(
        tmp_path,
        f"{video.name}\n"
        "Start End MTB Remarks\n"
        f"{start} {end} 2 invalid\n",
    )

    with pytest.raises(
        ValueError,
        match="End must be greater than Start",
    ):
        load_training_annotations(annotation_file)


@pytest.mark.parametrize(
    "score",
    [-1, 4, 99],
)
def test_training_score_must_be_between_zero_and_three(
    tmp_path,
    score,
):
    video = create_video(tmp_path)

    annotation_file = write_annotation_file(
        tmp_path,
        f"{video.name}\n"
        "Start End MTB Remarks\n"
        f"00:10 00:14 {score} invalid\n",
    )

    with pytest.raises(
        ValueError,
        match="MTB score must be 0..3",
    ):
        load_training_annotations(annotation_file)


def test_training_overlapping_annotations_are_rejected(
    tmp_path,
):
    video = create_video(tmp_path)

    annotation_file = write_annotation_file(
        tmp_path,
        f"{video.name}\n"
        "Start End MTB Remarks\n"
        "00:10 00:20 1 first\n"
        "00:19 00:25 3 second\n",
    )

    with pytest.raises(
        ValueError,
        match="overlapping annotations",
    ):
        load_training_annotations(annotation_file)


def test_training_touching_annotations_are_allowed(tmp_path):
    video = create_video(tmp_path)

    annotation_file = write_annotation_file(
        tmp_path,
        f"{video.name}\n"
        "Start End MTB Remarks\n"
        "00:10 00:20 1 first\n"
        "00:20 00:30 3 second\n",
    )

    _, annotations = load_training_annotations(
        annotation_file
    )

    assert len(annotations) == 2
    assert annotations[0].end == 20.0
    assert annotations[1].start == 20.0
