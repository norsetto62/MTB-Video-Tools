"""Tests for dataset feature-sequence building."""

from __future__ import annotations

import numpy as np
import pytest

from autocut.data_models import Clip, FeatureSequence
from autocut.dataset.sequence import (
    DEFAULT_PROCESSING_WIDTH,
    _prepare_frame,
    build_feature_sequence,
)


def test_prepare_frame_when_dimensions_match_and_height_is_even() -> None:
    """A frame already at the target dimensions is returned unchanged."""

    frame = np.zeros((270, 480, 3), dtype=np.uint8)

    result = _prepare_frame(
        frame,
        processing_width=480,
    )

    assert result.shape == (270, 480, 3)
    assert result is frame


def test_prepare_frame_preserves_aspect_ratio_and_forces_even_height() -> None:
    """Resizing preserves aspect ratio as closely as possible and uses even height."""

    # 1920 x 1084 -> nominal height 271 at width 480.
    # The helper must force this to an even height.
    frame = np.zeros((1084, 1920, 3), dtype=np.uint8)

    result = _prepare_frame(
        frame,
        processing_width=480,
    )

    assert result.shape[1] == 480
    assert result.shape[0] % 2 == 0

    source_ratio = 1920 / 1084
    result_ratio = result.shape[1] / result.shape[0]

    assert result_ratio == pytest.approx(
        source_ratio,
        rel=0.01,
    )


def test_prepare_frame_enforces_minimum_height_of_two() -> None:
    """Extremely wide frames still produce a valid minimum height of two."""

    frame = np.zeros((1, 10000, 3), dtype=np.uint8)

    result = _prepare_frame(
        frame,
        processing_width=480,
    )

    assert result.shape == (2, 480, 3)
    assert result.shape[0] >= 2
    assert result.shape[0] % 2 == 0


def test_build_feature_sequence_from_clip(monkeypatch: pytest.MonkeyPatch) -> None:
    """A clip produces one 54-feature vector for every flow pair."""

    frames = [
        np.full((1080, 1920, 3), index, dtype=np.uint8)
        for index in range(4)
    ]

    class FakeReader:
        def __init__(
            self,
            video_path,
            start,
            duration,
            sample_fps,
        ) -> None:
            assert video_path == "test.mp4"
            assert start == pytest.approx(10.0)
            assert duration == pytest.approx(2.0)
            assert sample_fps == pytest.approx(2.0)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback) -> None:
            pass

        def __iter__(self):
            for index, frame in enumerate(frames):
                yield index, index * 0.5, frame

    class FakeExtractor:
        def __init__(self, roi_width, roi_height) -> None:
            assert roi_width == DEFAULT_PROCESSING_WIDTH
            assert roi_height % 2 == 0
            self.calls = 0

        def extract_vector(self, flow):
            self.calls += 1
            return np.full(
                54,
                self.calls,
                dtype=np.float32,
            )

    def fake_calculate_optical_flow(previous_gray, current_gray):
        return object()

    monkeypatch.setattr(
        "autocut.dataset.builder.VideoReader",
        FakeReader,
    )
    monkeypatch.setattr(
        "autocut.dataset.builder.FeatureExtractor",
        FakeExtractor,
    )
    monkeypatch.setattr(
        "autocut.dataset.builder.calculate_optical_flow",
        fake_calculate_optical_flow,
    )

    clip = Clip(
        video_name="test.mp4",
        start=10.0,
        end=12.0,
    )

    result = build_feature_sequence(
        "test.mp4",
        clip,
    )

    assert isinstance(result, FeatureSequence)
    assert result.features.shape == (3, 54)
    assert result.features.dtype == np.float32

    assert result.timestamps.shape == (3,)
    assert result.timestamps.dtype == np.float64

    # Feature timestamps are absolute source-video timestamps.
    assert result.timestamps.tolist() == pytest.approx(
        [10.5, 11.0, 11.5]
    )

    assert np.all(result.features[0] == 1.0)
    assert np.all(result.features[1] == 2.0)
    assert np.all(result.features[2] == 3.0)


@pytest.mark.parametrize(
    "sample_fps",
    [0.0, -1.0],
)
def test_build_feature_sequence_rejects_non_positive_sample_fps(
    sample_fps: float,
) -> None:
    """Sampling FPS must be strictly positive."""

    clip = Clip(
        video_name="test.mp4",
        start=0.0,
        end=2.0,
    )

    with pytest.raises(
        ValueError,
        match="sample_fps must be greater than zero",
    ):
        build_feature_sequence(
            "test.mp4",
            clip,
            sample_fps=sample_fps,
        )


@pytest.mark.parametrize(
    "processing_width",
    [0, 1],
)
def test_build_feature_sequence_rejects_processing_width_below_two(
    processing_width: int,
) -> None:
    """Processing width must be at least two pixels."""

    clip = Clip(
        video_name="test.mp4",
        start=0.0,
        end=2.0,
    )

    with pytest.raises(
        ValueError,
        match="processing_width must be at least 2",
    ):
        build_feature_sequence(
            "test.mp4",
            clip,
            processing_width=processing_width,
        )


@pytest.mark.parametrize(
    ("start", "end"),
    [
        (10.0, 10.0),
        (12.0, 10.0),
    ],
)
def test_build_feature_sequence_rejects_non_positive_clip_duration(
    start: float,
    end: float,
) -> None:
    """A clip must have strictly positive duration."""

    clip = Clip(
        video_name="test.mp4",
        start=start,
        end=end,
    )

    with pytest.raises(
        ValueError,
        match="clip end must be greater than clip start",
    ):
        build_feature_sequence(
            "test.mp4",
            clip,
        )


@pytest.mark.parametrize(
    "frames_to_yield",
    [0, 1],
)
def test_build_feature_sequence_rejects_fewer_than_two_frames(
    monkeypatch: pytest.MonkeyPatch,
    frames_to_yield: int,
) -> None:
    """At least two sampled frames are required to produce optical flow."""

    class FakeReader:
        def __init__(
            self,
            video_path,
            start,
            duration,
            sample_fps,
        ) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback) -> None:
            pass

        def __iter__(self):
            for index in range(frames_to_yield):
                frame = np.zeros(
                    (1080, 1920, 3),
                    dtype=np.uint8,
                )
                yield index, index * 0.5, frame

    monkeypatch.setattr(
        "autocut.dataset.builder.VideoReader",
        FakeReader,
    )

    clip = Clip(
        video_name="test.mp4",
        start=10.0,
        end=12.0,
    )

    with pytest.raises(
        ValueError,
        match="did not contain enough sampled frames",
    ):
        build_feature_sequence(
            "test.mp4",
            clip,
        )


def test_prepare_frame_does_not_modify_input_frame() -> None:
    """Resizing must not modify the decoded source frame."""

    frame = np.zeros(
        (1080, 1920, 3),
        dtype=np.uint8,
    )
    frame[100, 100] = [10, 20, 30]

    original = frame.copy()

    result = _prepare_frame(
        frame,
        processing_width=480,
    )

    assert np.array_equal(frame, original)
    assert result is not frame