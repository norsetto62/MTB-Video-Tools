"""Build time-ordered motion-feature sequences from video clips."""

from __future__ import annotations

from pathlib import Path
import logging
import time

import cv2
import numpy as np

from ..data_models import Clip, FeatureSequence, OpticalFlow
from ..motion.features import FeatureExtractor
from ..motion.flow import DEFAULT_FARNEBACK, FarnebackConfig, calculate_optical_flow
from .feature_cache import load_feature_sequence, save_feature_sequence
from ..video.reader import VideoReader


logger = logging.getLogger(__name__)

DEFAULT_SAMPLE_FPS = 2.0
DEFAULT_PROCESSING_WIDTH = 480
DEFAULT_ROI_LEFT = 0.15
DEFAULT_ROI_RIGHT = 0.85
DEFAULT_ROI_BOTTOM = 0.50
DEFAULT_ROI_TOP = 0.10
ROI_CONTEXT_PADDING_X = 0.05
ROI_CONTEXT_PADDING_Y = 0.10


def validate_roi(
    roi_left: float = DEFAULT_ROI_LEFT,
    roi_right: float = DEFAULT_ROI_RIGHT,
    roi_bottom: float = DEFAULT_ROI_BOTTOM,
    roi_top: float = DEFAULT_ROI_TOP,
) -> tuple[float, float, float, float]:
    """Validate and normalize target ROI bounds in [0, 1] frame coordinates."""
    values = {
        "roi_left": roi_left,
        "roi_right": roi_right,
        "roi_bottom": roi_bottom,
        "roi_top": roi_top,
    }
    normalized: dict[str, float] = {}
    for name, value in values.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be a finite number in [0.0, 1.0].")
        value = float(value)
        if not np.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be a finite number in [0.0, 1.0].")
        normalized[name] = value
    if normalized["roi_left"] >= normalized["roi_right"]:
        raise ValueError("roi_left must be less than roi_right.")
    if normalized["roi_top"] >= normalized["roi_bottom"]:
        raise ValueError("roi_top must be less than roi_bottom.")
    return (normalized["roi_left"], normalized["roi_right"], normalized["roi_bottom"], normalized["roi_top"])


def _roi_pixel_bounds(
    width: int,
    height: int,
    *,
    roi_left: float,
    roi_right: float,
    roi_bottom: float,
    roi_top: float,
) -> tuple[int, int, int, int, int, int, int, int]:
    """Return context bounds followed by target bounds in resized-frame pixels."""
    left, right, bottom, top = validate_roi(roi_left, roi_right, roi_bottom, roi_top)
    context_left = max(0, int(np.floor((left - ROI_CONTEXT_PADDING_X) * width)))
    context_right = min(width, int(np.ceil((right + ROI_CONTEXT_PADDING_X) * width)))
    context_top = max(0, int(np.floor((top - ROI_CONTEXT_PADDING_Y) * height)))
    context_bottom = min(height, int(np.ceil((bottom + ROI_CONTEXT_PADDING_Y) * height)))
    target_left = max(context_left, min(context_right - 1, int(np.floor(left * width))))
    target_right = max(target_left + 1, min(context_right, int(np.ceil(right * width))))
    target_top = max(context_top, min(context_bottom - 1, int(np.floor(top * height))))
    target_bottom = max(target_top + 1, min(context_bottom, int(np.ceil(bottom * height))))
    if context_right <= context_left or context_bottom <= context_top:
        raise ValueError("ROI context must contain at least one pixel in each dimension.")
    return (context_left, context_right, context_top, context_bottom,
            target_left, target_right, target_top, target_bottom)


def build_feature_sequence(
    video_path: str | Path,
    clip: Clip,
    *,
    sample_fps: float = DEFAULT_SAMPLE_FPS,
    processing_width: int = DEFAULT_PROCESSING_WIDTH,
    cache_dir: str | Path | None = None,
    flow_config: FarnebackConfig = DEFAULT_FARNEBACK,
    roi_left: float = DEFAULT_ROI_LEFT,
    roi_right: float = DEFAULT_ROI_RIGHT,
    roi_bottom: float = DEFAULT_ROI_BOTTOM,
    roi_top: float = DEFAULT_ROI_TOP,
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

    roi_left, roi_right, roi_bottom, roi_top = validate_roi(
        roi_left, roi_right, roi_bottom, roi_top
    )

    if clip.end <= clip.start:
        raise ValueError("clip end must be greater than clip start.")

    started = time.perf_counter()
    if cache_dir is not None:
        logger.info("Feature cache lookup: %s", video_path)
        cached = load_feature_sequence(
            video_path, cache_dir, clip,
            sample_fps=sample_fps,
            processing_width=processing_width,
            flow_config=flow_config,
            roi_left=roi_left, roi_right=roi_right,
            roi_bottom=roi_bottom, roi_top=roi_top,
        )
        if cached is not None:
            logger.info("Feature cache hit: %s (%d rows)", video_path, len(cached.features))
            return cached
        logger.info("Feature cache miss: %s; decoding video", video_path)

    feature_vectors: list[np.ndarray] = []
    timestamps: list[float] = []

    extractor: FeatureExtractor | None = None
    previous_gray: np.ndarray | None = None
    sampled_frames = 0
    expected_frames = max(1, int(np.ceil(clip.duration * sample_fps)))
    logger.info(
        "Extracting motion features: %s (about %d sampled frames)",
        video_path, expected_frames,
    )

    with VideoReader(
        video_path,
        start=clip.start,
        duration=clip.duration,
        sample_fps=sample_fps,
    ) as reader:

        for _, relative_timestamp, frame in reader:
            sampled_frames += 1
            if sampled_frames == 1 or sampled_frames % 100 == 0:
                logger.info(
                    "Feature extraction %s: %d/%d sampled frames (%.0f%%, %.1fs elapsed)",
                    video_path, sampled_frames, expected_frames,
                    min(100.0, 100.0 * sampled_frames / expected_frames),
                    time.perf_counter() - started,
                )
            processed = _prepare_frame(frame, processing_width=processing_width)
            gray_full = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
            (context_left, context_right, context_top, context_bottom,
             target_left, target_right, target_top, target_bottom) = _roi_pixel_bounds(
                processed.shape[1], processed.shape[0],
                roi_left=roi_left, roi_right=roi_right,
                roi_bottom=roi_bottom, roi_top=roi_top,
            )
            gray = gray_full[context_top:context_bottom, context_left:context_right]

            if extractor is None:
                extractor = FeatureExtractor(
                    roi_width=target_right - target_left,
                    roi_height=target_bottom - target_top,
                )

            if previous_gray is None:
                previous_gray = gray
                continue

            flow = calculate_optical_flow(
                previous_gray,
                gray,
                config=flow_config,
            )

            target_slice_y = slice(target_top - context_top, target_bottom - context_top)
            target_slice_x = slice(target_left - context_left, target_right - context_left)
            target_flow = OpticalFlow(
                u=flow.u[target_slice_y, target_slice_x],
                v=flow.v[target_slice_y, target_slice_x],
            )
            vector = extractor.extract_vector(target_flow)

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

    sequence = FeatureSequence(
        features=features,
        timestamps=timestamp_array,
    )
    logger.info(
        "Feature extraction complete: %s (%d rows from %d sampled frames in %.1fs)",
        video_path, len(features), sampled_frames, time.perf_counter() - started,
    )
    if cache_dir is not None:
        save_feature_sequence(
            sequence, video_path, cache_dir, clip,
            sample_fps=sample_fps,
            processing_width=processing_width,
            flow_config=flow_config,
            roi_left=roi_left, roi_right=roi_right,
            roi_bottom=roi_bottom, roi_top=roi_top,
        )
    return sequence


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