"""Build time-ordered motion-feature sequences from video clips."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ..data_models import Clip, FeatureSequence
from ..motion.features import FeatureExtractor
from ..motion.flow import calculate_optical_flow
from ..video.reader import VideoReader


DEFAULT_SAMPLE_FPS = 2.0
DEFAULT_PROCESSING_WIDTH = 480


def build_feature_sequence(
    video_path: str | Path,
    clip: Clip,
    *,
    sample_fps: float = DEFAULT_SAMPLE_FPS,
    processing_width: int = DEFAULT_PROCESSING_WIDTH,
) -> FeatureSequence:
    """Build a motion-feature sequence for a video clip.

    Frames are sampled at ``sample_fps`` and resized to ``processing_width``
    while preserving the source aspect ratio. Optical flow is calculated
    between consecutive sampled frames. Each resulting 54-element feature
    vector is associated with the timestamp of the current frame.

    Feature timestamps are absolute source-video timestamps in seconds.

    The first feature therefore represents the flow from the first sampled
    frame to the second sampled frame.

    Args:
        video_path: Source video.
        clip: Video interval to process.
        sample_fps: Frame sampling rate.
        processing_width: Width used for motion processing.

    Returns:
        A time-ordered FeatureSequence.

    Raises:
        ValueError: If the processing parameters are invalid or the clip
            does not produce any optical-flow features.
    """
    if sample_fps <= 0:
        raise ValueError("sample_fps must be greater than zero.")

    if processing_width < 2:
        raise ValueError("processing_width must be at least 2.")

    if clip.end <= clip.start:
        raise ValueError("clip end must be greater than clip start.")

    feature_vectors: list[np.ndarray] = []
    timestamps: list[float] = []

    extractor: FeatureExtractor | None = None
    previous_gray: np.ndarray | None = None

    with VideoReader(
        video_path,
        start=clip.start,
        duration=clip.duration,
        sample_fps=sample_fps,
    ) as reader:

        for _, relative_timestamp, frame in reader:
            processed = _prepare_frame(
                frame,
                processing_width=processing_width,
            )

            if extractor is None:
                extractor = FeatureExtractor(
                    roi_width=processed.shape[1],
                    roi_height=processed.shape[0],
                )

            gray = cv2.cvtColor(
                processed,
                cv2.COLOR_BGR2GRAY,
            )

            if previous_gray is None:
                previous_gray = gray
                continue

            flow = calculate_optical_flow(
                previous_gray,
                gray,
            )

            vector = extractor.extract_vector(flow)

            feature_vectors.append(vector)
            timestamps.append(
                clip.start + relative_timestamp
            )

            previous_gray = gray

    if not feature_vectors:
        raise ValueError(
            "Clip did not contain enough sampled frames "
            "to produce an optical-flow feature."
        )

    features = np.stack(
        feature_vectors,
        axis=0,
    ).astype(
        np.float32,
        copy=False,
    )

    timestamp_array = np.asarray(
        timestamps,
        dtype=np.float64,
    )

    return FeatureSequence(
        features=features,
        timestamps=timestamp_array,
    )


def _prepare_frame(
    frame: np.ndarray,
    *,
    processing_width: int,
) -> np.ndarray:
    """Resize a decoded frame to the canonical processing width."""

    height, width = frame.shape[:2]

    processing_height = max(
        2,
        round(height * processing_width / width),
    )

    if processing_height % 2:
        processing_height += 1

    if width == processing_width and height == processing_height:
        return frame

    return cv2.resize(
        frame,
        (processing_width, processing_height),
        interpolation=cv2.INTER_AREA,
    )