#!/usr/bin/env python3
r"""
Integrated bike/reference tracking + dense optical flow analysis.

Architecture:
    FFmpeg decode
          |
          +--> every original frame --> output video
          |
          +--> selected frames -------> optical flow analysis
                                      |
                                      +--> CSV
                                      +--> visual overlay

Features:
- Manual ROI selection for a bike-mounted/reference object.
- Shi-Tomasi feature detection inside ROI.
- Lucas-Kanade tracking on every source frame.
- Dense Farneback optical flow at configurable sample FPS.
- Flow analysis restricted to the upper portion of the image.
- Colour flow overlay on original video.
- Optional sparse flow arrows.
- Flow magnitude/direction statistics.
- Circular flow-angle change.
- Divergence and curl statistics.
- 3x3 spatial flow statistics.
- NVDEC/CUDA decode through FFmpeg.
- NVENC output encoding.
"""

from __future__ import annotations

import argparse
import csv
import math
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_FLOW_FPS = 10.0
DEFAULT_FLOW_WIDTH = 480
DEFAULT_FLOW_HEIGHT = 50.0
DEFAULT_FLOW_ALPHA = 0.55

# v4 candidate detector
DEFAULT_SCORE_WINDOW = 2.0
DEFAULT_SCORE_PERCENTILE = 92.0
DEFAULT_FILL_GAP = 1.0
DEFAULT_MIN_DURATION = 0.8
DEFAULT_MERGE_GAP = 5.0
DEFAULT_PADDING = 1.0

GRID_ROWS = 3
GRID_COLS = 3

LK_PARAMS = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        30,
        0.01,
    ),
)

FARNEBACK_PARAMS = dict(
    pyr_scale=0.5,
    levels=3,
    winsize=15,
    iterations=3,
    poly_n=5,
    poly_sigma=1.2,
    flags=0,
)


# ---------------------------------------------------------------------------
# Time parsing
# ---------------------------------------------------------------------------

def parse_time(value: str) -> float:
    """
    Parse:
        12.5
        01:23
        01:23:45
    """
    value = value.strip()

    if ":" not in value:
        return float(value)

    parts = value.split(":")

    if len(parts) == 2:
        minutes = float(parts[0])
        seconds = float(parts[1])
        return minutes * 60.0 + seconds

    if len(parts) == 3:
        hours = float(parts[0])
        minutes = float(parts[1])
        seconds = float(parts[2])
        return hours * 3600.0 + minutes * 60.0 + seconds

    raise ValueError(f"Invalid time value: {value}")


# ---------------------------------------------------------------------------
# Video information
# ---------------------------------------------------------------------------

def get_video_info(video_path: Path):
    """Return width, height, FPS and duration using ffprobe."""

    cmd = [
        "ffprobe",
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

    return width, height, fps, duration


# ---------------------------------------------------------------------------
# FFmpeg
# ---------------------------------------------------------------------------

def start_ffmpeg_decode(
    video_path: Path,
    start: float,
    duration: float | None,
    width: int,
    height: int,
):
    """
    Start a single FFmpeg decode process producing raw BGR frames.

    The same decoded frame stream is used both for:
        - output video
        - tracking
        - optical flow
    """

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
    ]

    if start > 0:
        cmd += [
            "-ss",
            f"{start:.6f}",
        ]

    cmd += [
        "-i",
        str(video_path),
    ]

    if duration is not None:
        cmd += [
            "-t",
            f"{duration:.6f}",
        ]

    cmd += [
        "-an",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "pipe:1",
    ]

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=10**8,
    )

    return process


def start_ffmpeg_encode(
    output_path: Path,
    width: int,
    height: int,
    fps: float,
):
    """
    Start FFmpeg NVENC encoder accepting raw BGR frames.
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",

        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{width}x{height}",
        "-r",
        f"{fps:.8f}",
        "-i",
        "pipe:0",

        "-an",

        "-c:v",
        "h264_nvenc",

        "-preset",
        "p5",

        "-cq",
        "18",
        
        "-pix_fmt",
        "yuv420p",

        "-movflags",
        "+faststart",

        "-y",
        str(output_path),
    ]

    process = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=10**8,
    )

    return process


# ---------------------------------------------------------------------------
# Bike/reference tracking
# ---------------------------------------------------------------------------

def initialise_tracker(
    frame: np.ndarray,
):
    """
    Ask the user to select a reference object and detect Shi-Tomasi
    features inside it.
    """
    
    roi = cv2.selectROI(
        "Select bike/reference object",
        frame,
        showCrosshair=True,
        fromCenter=False,
    )

    cv2.destroyWindow("Select bike/reference object")

    x, y, w, h = [int(v) for v in roi]

    if w <= 0 or h <= 0:
        raise RuntimeError(
            "No valid ROI was selected."
        )

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY,
    )

    mask = np.zeros_like(gray)

    mask[y:y + h, x:x + w] = 255

    points = cv2.goodFeaturesToTrack(
        gray,
        mask=mask,
        maxCorners=100,
        qualityLevel=0.01,
        minDistance=7,
        blockSize=7,
    )

    if points is None or len(points) < 3:
        raise RuntimeError(
            "Could not find enough trackable features inside the ROI."
        )

    return {
        "roi": (x, y, w, h),
        "points": points,
        "previous_gray": gray,
        "center_x": x + w / 2.0,
        "center_y": y + h / 2.0,
        "trajectory": [
            (
                x + w / 2.0,
                y + h / 2.0,
            )
        ],
    }


def update_tracker(
    frame: np.ndarray,
    tracker: dict,
):
    """
    Track reference-object features using Lucas-Kanade.

    Returns:
        dx, dy, speed, tracked_points
    """

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY,
    )

    previous_gray = tracker["previous_gray"]
    previous_points = tracker["points"]

    if previous_points is None or len(previous_points) < 3:
        tracker["previous_gray"] = gray
        return 0.0, 0.0, 0.0, 0

    current_points, status, _ = cv2.calcOpticalFlowPyrLK(
        previous_gray,
        gray,
        previous_points,
        None,
        **LK_PARAMS,
    )

    if current_points is None or status is None:
        tracker["previous_gray"] = gray
        tracker["points"] = None

        return 0.0, 0.0, 0.0, 0

    status = status.reshape(-1)

    valid_old = previous_points.reshape(-1, 2)[status == 1]
    valid_new = current_points.reshape(-1, 2)[status == 1]

    if len(valid_new) < 3:
        tracker["previous_gray"] = gray
        tracker["points"] = None

        return 0.0, 0.0, 0.0, len(valid_new)

    displacement = valid_new - valid_old

    median_displacement = np.median(
        displacement,
        axis=0,
    )

    median_displacement = np.asarray(
        median_displacement
    ).reshape(-1)

    dx = float(median_displacement[0])
    dy = float(median_displacement[1])

    speed = math.sqrt(
        dx * dx +
        dy * dy
    )

    tracker["center_x"] += dx
    tracker["center_y"] += dy

    tracker["trajectory"].append(
        (
            tracker["center_x"],
            tracker["center_y"],
        )
    )

    tracker["previous_gray"] = gray
    tracker["points"] = valid_new.reshape(-1, 1, 2)

    return dx, dy, speed, len(valid_new)


def draw_tracker(
    frame: np.ndarray,
    tracker: dict,
):
    """
    Draw tracked feature points, reference rectangle,
    centre and trajectory.
    """

    output = frame.copy()

    points = tracker.get("points")

    if points is not None:
        for point in points.reshape(-1, 2):
            px = int(round(float(point[0])))
            py = int(round(float(point[1])))

            if (
                0 <= px < output.shape[1]
                and 0 <= py < output.shape[0]
            ):
                cv2.circle(
                    output,
                    (px, py),
                    3,
                    (0, 255, 0),
                    -1,
                )

    x, y, w, h = tracker["roi"]

    # The rectangle remains the original visual reference.
    # The feature points are the actual tracker state.
    cv2.rectangle(
        output,
        (x, y),
        (x + w, y + h),
        (0, 255, 0),
        2,
    )

    cx = int(round(tracker["center_x"]))
    cy = int(round(tracker["center_y"]))

    cv2.circle(
        output,
        (cx, cy),
        6,
        (0, 255, 255),
        -1,
    )

    trajectory = tracker["trajectory"]

    if len(trajectory) >= 2:
        for p1, p2 in zip(
            trajectory[:-1],
            trajectory[1:],
        ):
            x1 = int(round(p1[0]))
            y1 = int(round(p1[1]))

            x2 = int(round(p2[0]))
            y2 = int(round(p2[1]))

            cv2.line(
                output,
                (x1, y1),
                (x2, y2),
                (0, 255, 255),
                2,
            )

    return output


# ---------------------------------------------------------------------------
# Optical flow
# ---------------------------------------------------------------------------

def calculate_flow_features(
    prev_frame: np.ndarray,
    current_frame: np.ndarray,
    flow_width: int,
    flow_height_percent: float,
):
    """
    Calculate dense optical flow on the upper portion of the frame.

    Returns:
        flow
        feature dictionary
        resized flow-analysis frame
    """

    source_height, source_width = prev_frame.shape[:2]

    flow_height_percent = max(
        1.0,
        min(100.0, flow_height_percent),
    )

    analysis_height = max(
        1,
        int(
            source_height *
            flow_height_percent /
            100.0
        ),
    )

    scale = flow_width / float(source_width)

    analysis_height_scaled = max(
        1,
        int(round(analysis_height * scale)),
    )

    prev_crop = prev_frame[
        :analysis_height,
        :,
    ]

    current_crop = current_frame[
        :analysis_height,
        :,
    ]

    prev_small = cv2.resize(
        prev_crop,
        (
            flow_width,
            analysis_height_scaled,
        ),
        interpolation=cv2.INTER_AREA,
    )

    current_small = cv2.resize(
        current_crop,
        (
            flow_width,
            analysis_height_scaled,
        ),
        interpolation=cv2.INTER_AREA,
    )

    prev_gray = cv2.cvtColor(
        prev_small,
        cv2.COLOR_BGR2GRAY,
    )

    current_gray = cv2.cvtColor(
        current_small,
        cv2.COLOR_BGR2GRAY,
    )

    flow = cv2.calcOpticalFlowFarneback(
        prev_gray,
        current_gray,
        None,
        **FARNEBACK_PARAMS,
    )

    u = flow[..., 0]
    v = flow[..., 1]

    magnitude, angle = cv2.cartToPolar(
        u,
        v,
        angleInDegrees=True,
    )

    # ---------------------------------------------------------------
    # Global flow statistics
    # ---------------------------------------------------------------

    flow_mean = float(np.mean(magnitude))
    flow_median = float(np.median(magnitude))
    flow_p90 = float(np.percentile(magnitude, 90))
    flow_p95 = float(np.percentile(magnitude, 95))
    flow_std = float(np.std(magnitude))

    flow_x = float(np.mean(u))
    flow_y = float(np.mean(v))

    flow_std_x = float(np.std(u))
    flow_std_y = float(np.std(v))

    # Resultant of the average flow vector.
    #
    # Unlike flow_coherence/flow_angle (which average directions on the
    # unit circle), this uses the actual vector components. This is useful
    # for patterns such as:
    #
    #     < < <   bike   > > >
    #
    # where the left/right motion cancels out and the average vector is
    # close to zero. A coherent upward/downward flow, on the other hand,
    # produces a large resultant.
    flow_resultant = math.sqrt(
        flow_x * flow_x +
        flow_y * flow_y
    )

    # Direction of the average flow vector.
    #
    # Image coordinates are +x = right, +y = down:
    #     right =   0 degrees
    #     down  = +90 degrees
    #     left  = 180 degrees
    #     up    = 270 degrees
    #
    # This is deliberately based on (flow_x, flow_y), not on averaging
    # angles directly.
    flow_resultant_angle = math.degrees(
        math.atan2(
            flow_y,
            flow_x,
        )
    )

    if flow_resultant_angle < 0:
        flow_resultant_angle += 360.0

    # How much of the mean flow magnitude survives directional
    # cancellation. This is in [0, 1] in normal numerical conditions.
    flow_resultant_ratio = (
        flow_resultant / flow_mean
        if flow_mean > 1e-9
        else 0.0
    )

    # Circular directional coherence:
    # length of the mean unit direction vector.
    angle_rad = np.deg2rad(angle)

    mean_cos = float(np.mean(np.cos(angle_rad)))
    mean_sin = float(np.mean(np.sin(angle_rad)))

    flow_coherence = math.sqrt(
        mean_cos * mean_cos +
        mean_sin * mean_sin
    )

    flow_angle = math.degrees(
        math.atan2(
            mean_sin,
            mean_cos,
        )
    )

    if flow_angle < 0:
        flow_angle += 360.0

    # ---------------------------------------------------------------
    # Divergence and curl
    # ---------------------------------------------------------------

    du_dy, du_dx = np.gradient(u)
    dv_dy, dv_dx = np.gradient(v)

    divergence = du_dx + dv_dy

    curl = dv_dx - du_dy

    div_abs = np.abs(divergence)
    curl_abs = np.abs(curl)

    div_abs_mean = float(np.mean(div_abs))
    div_abs_p90 = float(np.percentile(div_abs, 90))
    div_std = float(np.std(divergence))

    div_pos_fraction = float(
        np.mean(divergence > 0)
    )

    div_neg_fraction = float(
        np.mean(divergence < 0)
    )

    curl_abs_mean = float(np.mean(curl_abs))
    curl_abs_p90 = float(np.percentile(curl_abs, 90))
    curl_std = float(np.std(curl))

    # ---------------------------------------------------------------
    # Normalized Divergence & Curl (Relative to Forward Speed)
    # ---------------------------------------------------------------
    # Adding a small epsilon (1e-6) prevents Division-By-Zero when stationary
    eps = 1e-6

    # Unitless expansion rate: How fast the field expands relative to speed
    div_normalized_mean = div_abs_mean / (flow_mean + eps)
    
    # Unitless rotational rate: How strong turns/twists are relative to speed
    curl_normalized_mean = curl_abs_mean / (flow_mean + eps)

    # ---------------------------------------------------------------
    # 3x3 spatial grid
    # ---------------------------------------------------------------

    features = {
        "flow_mean": flow_mean,
        "flow_median": flow_median,
        "flow_p90": flow_p90,
        "flow_p95": flow_p95,
        "flow_std": flow_std,
        "flow_x": flow_x,
        "flow_y": flow_y,
        "flow_std_x": flow_std_x,
        "flow_std_y": flow_std_y,
        "flow_resultant": flow_resultant,
        "flow_resultant_angle": flow_resultant_angle,
        "flow_resultant_ratio": flow_resultant_ratio,
        "flow_coherence": flow_coherence,
        "flow_angle": flow_angle,

        "div_abs_mean": div_abs_mean,
        "div_normalized_mean": div_normalized_mean,
        "div_abs_p90": div_abs_p90,
        "div_std": div_std,
        "div_pos_fraction": div_pos_fraction,
        "div_neg_fraction": div_neg_fraction,

        "curl_abs_mean": curl_abs_mean,
        "curl_normalized_mean": curl_normalized_mean,
        "curl_abs_p90": curl_abs_p90,
        "curl_std": curl_std,
    }

    height, width = magnitude.shape

    for row in range(GRID_ROWS):
        y0 = int(row * height / GRID_ROWS)
        y1 = int((row + 1) * height / GRID_ROWS)

        for col in range(GRID_COLS):
            x0 = int(col * width / GRID_COLS)
            x1 = int((col + 1) * width / GRID_COLS)

            region_u = u[y0:y1, x0:x1]
            region_v = v[y0:y1, x0:x1]
            region_mag = magnitude[y0:y1, x0:x1]

            prefix = f"grid_{row}_{col}"

            features[f"{prefix}_x"] = float(
                np.mean(region_u)
            )

            features[f"{prefix}_y"] = float(
                np.mean(region_v)
            )

            features[f"{prefix}_abs_x"] = float(
                np.mean(np.abs(region_u))
            )

            features[f"{prefix}_abs_y"] = float(
                np.mean(np.abs(region_v))
            )

            features[f"{prefix}_mag"] = float(
                np.mean(region_mag)
            )

            features[f"{prefix}_p90"] = float(
                np.percentile(region_mag, 90)
            )

    return (
        flow,
        features,
        current_small,
    )

def append_temporal_derivatives(sequence_matrix: np.ndarray, target_indices: list[int]) -> np.ndarray:
    """
    Computes frame-to-frame delta features (1st derivatives) for selected column indices 
    using pure NumPy and appends them to the sequence.
    
    Parameters:
        sequence_matrix: 2D array of shape (T, F) where T = time steps, F = feature count
        target_indices: List of column indices for features where delta is calculated 
                       (e.g., [flow_mean_idx, flow_y_idx, div_abs_mean_idx])
                       
    Returns:
        Expanded 2D array of shape (T, F + len(target_indices))
    """
    # Extract only the columns we want derivatives for: shape (T, num_targets)
    targets = sequence_matrix[:, target_indices]
    
    # Compute first difference along the time axis (axis 0): shape (T-1, num_targets)
    deltas = np.diff(targets, axis=0)
    
    # Pad the first row with zeros to maintain length T: shape (T, num_targets)
    deltas_padded = np.pad(deltas, ((1, 0), (0, 0)), mode='constant', constant_values=0.0)
    
    # Concatenate original features with the new delta features along columns (axis 1)
    # Final Shape: (T, F + num_targets)
    return np.hstack([sequence_matrix, deltas_padded])
    
# ---------------------------------------------------------------------------
# Flow visualization
# ---------------------------------------------------------------------------

def create_flow_overlay(
    frame: np.ndarray,
    flow: np.ndarray,
    flow_width: int,
    flow_height_percent: float,
    alpha: float,
    arrows: bool,
):
    """
    Create a semi-transparent HSV flow visualization over the
    upper portion of the original frame.
    """

    output = frame.copy()

    source_height, source_width = frame.shape[:2]

    analysis_height = max(
        1,
        int(
            source_height *
            max(1.0, min(100.0, flow_height_percent))
            / 100.0
        ),
    )

    # Resize flow to the original upper-region dimensions.
    flow_original = cv2.resize(
        flow,
        (
            source_width,
            analysis_height,
        ),
        interpolation=cv2.INTER_LINEAR,
    )

    u = flow_original[..., 0]
    v = flow_original[..., 1]

    magnitude, angle = cv2.cartToPolar(
        u,
        v,
        angleInDegrees=True,
    )

    # Use p95 for visualization normalization.
    p95 = float(
        np.percentile(
            magnitude,
            95,
        )
    )

    if p95 < 1e-6:
        p95 = 1.0

    normalized = np.clip(
        magnitude / p95,
        0.0,
        1.0,
    )

    hsv = np.zeros(
        (
            analysis_height,
            source_width,
            3,
        ),
        dtype=np.uint8,
    )

    # Hue = direction.
    hsv[..., 0] = (
        angle / 2.0
    ).astype(np.uint8)

    # Saturation = flow confidence/intensity.
    hsv[..., 1] = (
        normalized * 255.0
    ).astype(np.uint8)

    # Brightness = magnitude.
    hsv[..., 2] = (
        normalized * 255.0
    ).astype(np.uint8)

    flow_bgr = cv2.cvtColor(
        hsv,
        cv2.COLOR_HSV2BGR,
    )

    # Keep very small flow nearly transparent.
    mask = np.clip(
        normalized * 1.5,
        0.0,
        1.0,
    )

    mask = (
        mask[..., None] *
        alpha
    )

    base = output[
        :analysis_height,
        :,
    ].astype(np.float32)

    overlay = flow_bgr.astype(
        np.float32
    )

    blended = (
        base * (1.0 - mask) +
        overlay * mask
    )

    output[
        :analysis_height,
        :,
    ] = np.clip(
        blended,
        0,
        255,
    ).astype(np.uint8)

    # ---------------------------------------------------------------
    # Sparse arrows
    # ---------------------------------------------------------------

    if arrows:
        step_x = max(
            20,
            source_width // 24,
        )

        step_y = max(
            20,
            analysis_height // 12,
        )

        for y in range(
            step_y // 2,
            analysis_height,
            step_y,
        ):
            for x in range(
                step_x // 2,
                source_width,
                step_x,
            ):
                fx = float(u[y, x])
                fy = float(v[y, x])

                mag = math.sqrt(
                    fx * fx +
                    fy * fy
                )

                if mag < 0.5:
                    continue

                scale = 3.0

                x2 = int(
                    round(x + fx * scale)
                )

                y2 = int(
                    round(y + fy * scale)
                )

                x2 = max(
                    0,
                    min(source_width - 1, x2),
                )

                y2 = max(
                    0,
                    min(analysis_height - 1, y2),
                )

                cv2.arrowedLine(
                    output,
                    (x, y),
                    (x2, y2),
                    (255, 255, 255),
                    1,
                    tipLength=0.25,
                )

    return output


# ---------------------------------------------------------------------------
# CSV columns
# ---------------------------------------------------------------------------

def get_csv_fieldnames():
    fields = [
        "time",

        "bike_x",
        "bike_y",
        "bike_dx",
        "bike_dy",
        "bike_speed",
        "tracked_points",

        "flow_mean",
        "flow_median",
        "flow_p90",
        "flow_p95",
        "flow_std",
        "flow_x",
        "flow_y",
        "flow_std_x",
        "flow_std_y",
        "flow_resultant",
        "flow_resultant_angle",
        "flow_resultant_ratio",
        "flow_coherence",
        "flow_angle",

        "angle_change",
        "angle_change_abs",

        "div_abs_mean",
        "div_norm_mean",
        "div_abs_p90",
        "div_std",
        "div_pos_fraction",
        "div_neg_fraction",

        "curl_abs_mean",
        "curl_norm_mean",
        "curl_abs_p90",
        "curl_std",
    ]

    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):
            prefix = f"grid_{row}_{col}"

            fields.extend([
                f"{prefix}_x",
                f"{prefix}_y",
                f"{prefix}_abs_x",
                f"{prefix}_abs_y",
                f"{prefix}_mag",
                f"{prefix}_p90",
            ])

    return fields


# ---------------------------------------------------------------------------
# Progress
# ---------------------------------------------------------------------------

def print_progress(
    elapsed: float,
    elapsed_wall: float,
    total_duration: float,
    processing_fps: float,
    source_fps: float,
    flow_samples: int,
    csv_rows: int,
):
    if total_duration <= 0:
        percent = 0.0
    else:
        percent = (
            elapsed
            / total_duration
            * 100.0
        )

    percent = max(
        0.0,
        min(100.0, percent),
    )

    if processing_fps > 0 and source_fps > 0:
        realtime = processing_fps / source_fps
    else:
        realtime = 0.0

    # Estimated total processing time.
    if elapsed > 0 and elapsed_wall > 0:
        processing_speed = (
            elapsed /
            elapsed_wall
        )

        eta = max(
            0.0,
            (total_duration - elapsed)
            / processing_speed,
        )
    else:
        eta = 0.0

    eta_minutes, eta_seconds = divmod(
        int(eta),
        60,
    )

    if eta_minutes >= 60:
        eta_hours, eta_minutes = divmod(
            eta_minutes,
            60,
        )
        eta_text = (
            f"{eta_hours:02d}:"
            f"{eta_minutes:02d}:"
            f"{eta_seconds:02d}"
        )
    else:
        eta_text = (
            f"{eta_minutes:02d}:"
            f"{eta_seconds:02d}"
        )

    print(
        f"\r"
        f"{percent:6.1f}%  "
        f"{elapsed:7.1f}s / "
        f"{total_duration:7.1f}s  "
        f"{processing_fps:6.1f} fps  "
        f"{realtime:5.2f}x  "
        f"ETA: {eta_text}  "
        f"flow samples: {flow_samples:6d}  "
        f"CSV rows: {csv_rows:6d}",
        end="",
        flush=True,
    )


# ---------------------------------------------------------------------------
# V4 candidate detector
# ---------------------------------------------------------------------------
"""
Feature Signature   Raw Flow Indicator          Temporal Delta (Δ) Indicator                        Target MTB Feature
-----------------------------------------------------------------------------------------------------------------------
Drop Landing        High flow_y (downward flow) Massive +Δ flow_y (sudden downward spike)           Drop / Jump
Braking into Corner High div_abs_mean           Sharp -Δflow_mean (sudden deceleration)             Tight Switchback
Rock Garden Entry   Moderate flow_mean          Sudden drop in Δ flow_coherence (smooth → chaotic)  Technical / Rocks
Pumping / Flow      Cyclic flow_mean            Rhythmic oscillating Δ div_normalized_mean          Flow Trail
"""
V4_FEATURES = (
    ("magnitude", "flow_mean"),
    ("angle", "angle_change_abs"),
    ("divergence", "div_abs_mean"),
    ("curl", "curl_abs_mean"),
)


def _rolling_median(values: np.ndarray, window_samples: int) -> np.ndarray:
    """Centered rolling median with edge padding."""
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return values.copy()

    window_samples = max(1, int(window_samples))
    if window_samples % 2 == 0:
        window_samples += 1

    radius = window_samples // 2
    padded = np.pad(
        values,
        (radius, radius),
        mode="edge",
    )

    windows = np.lib.stride_tricks.sliding_window_view(
        padded,
        window_samples,
    )

    return np.median(windows, axis=1)


def _robust_zscore(values: np.ndarray, window_samples: int) -> np.ndarray:
    """
    V4 feature normalization:
      1. remove a centered rolling median baseline;
      2. scale the residuals with a robust MAD estimate.
    """
    values = np.asarray(values, dtype=float)

    if len(values) == 0:
        return values.copy()

    baseline = _rolling_median(
        values,
        window_samples,
    )

    residual = values - baseline

    median_residual = float(np.median(residual))
    mad = float(
        np.median(
            np.abs(residual - median_residual)
        )
    )

    scale = 1.4826 * mad

    if not np.isfinite(scale) or scale < 1e-9:
        std = float(np.std(residual))
        scale = std if np.isfinite(std) and std > 1e-9 else 1.0

    return residual / scale


def _regions_from_mask(
    times: np.ndarray,
    mask: np.ndarray,
    fill_gap: float,
    min_duration: float,
):
    """Return contiguous time regions after gap filling and duration filtering."""
    indices = np.flatnonzero(mask)

    if len(indices) == 0:
        return []

    regions = []
    start_idx = int(indices[0])
    end_idx = start_idx

    for index in indices[1:]:
        index = int(index)
        gap = float(times[index] - times[end_idx])

        if gap <= fill_gap + 1e-9:
            end_idx = index
            continue

        start_time = float(times[start_idx])
        if end_idx > 0:
            sample_interval = float(times[end_idx] - times[end_idx - 1])
        else:
            sample_interval = float(times[1] - times[0]) if len(times) > 1 else 0.0

        end_time = float(times[end_idx]) + sample_interval

        if end_time - start_time >= min_duration:
            regions.append((start_time, end_time))

        start_idx = index
        end_idx = index

    start_time = float(times[start_idx])
    end_time = float(times[end_idx])

    if end_time - start_time >= min_duration:
        regions.append((start_time, end_time))

    return regions


def _merge_regions(regions, merge_gap: float):
    """Merge overlapping/nearby regions."""
    if not regions:
        return []

    regions = sorted(
        (float(start), float(end))
        for start, end in regions
    )

    merged = [list(regions[0])]

    for start, end in regions[1:]:
        previous = merged[-1]

        if start - previous[1] <= merge_gap + 1e-9:
            previous[1] = max(previous[1], end)
        else:
            merged.append([start, end])

    return [
        (start, end)
        for start, end in merged
    ]


def _format_autocut_time(seconds: float) -> str:
    """Format seconds as HH:MM:SS.ss for AutoCut."""
    seconds = max(0.0, float(seconds))

    total_hundredths = int(
        round(seconds * 100.0)
    )

    hours, remainder = divmod(
        total_hundredths,
        360000,
    )
    minutes, remainder = divmod(
        remainder,
        6000,
    )
    whole_seconds, hundredths = divmod(
        remainder,
        100,
    )

    return (
        f"{hours:02d}:"
        f"{minutes:02d}:"
        f"{whole_seconds:02d}."
        f"{hundredths:02d}"
    )


def generate_candidate_outputs(
    csv_path: Path,
    video_path: Path,
    source_start: float,
    merge_gap: float,
):
    """
    Run the tested v4 detector on the freshly generated flow CSV.

    Returns:
        candidate_csv_path, autocut_path, candidates
    """
    rows = []

    with open(
        csv_path,
        "r",
        newline="",
        encoding="utf-8",
    ) as handle:
        reader = csv.DictReader(handle)

        for row in reader:
            try:
                time_value = float(row["time"])
            except (KeyError, TypeError, ValueError):
                continue

            parsed = {"time": time_value}

            for _, column in V4_FEATURES:
                try:
                    value = float(row[column])
                except (KeyError, TypeError, ValueError):
                    value = float("nan")

                parsed[column] = value

            rows.append(parsed)

    if not rows:
        raise RuntimeError(
            "Cannot generate candidate events: flow CSV contains no usable rows."
        )

    times = np.asarray(
        [row["time"] for row in rows],
        dtype=float,
    )

    if len(times) > 1:
        sample_intervals = np.diff(times)
        sample_interval = float(
            np.median(sample_intervals[
                sample_intervals > 0
            ])
        ) if np.any(sample_intervals > 0) else 0.1
    else:
        sample_interval = 0.1

    window_samples = max(
        3,
        int(round(
            DEFAULT_SCORE_WINDOW /
            sample_interval
        )),
    )

    if window_samples % 2 == 0:
        window_samples += 1

    feature_scores = {}
    feature_regions = {}

    for feature_name, column in V4_FEATURES:
        values = np.asarray(
            [row[column] for row in rows],
            dtype=float,
        )

        finite = np.isfinite(values)

        if not np.any(finite):
            scores = np.zeros_like(values)
        else:
            fill_value = float(
                np.nanmedian(values)
            )
            values = np.where(
                finite,
                values,
                fill_value,
            )

            scores = _robust_zscore(
                values,
                window_samples,
            )

            scores = np.maximum(
                scores,
                0.0,
            )

        feature_scores[feature_name] = scores

        threshold = float(
            np.percentile(
                scores,
                DEFAULT_SCORE_PERCENTILE,
            )
        )

        mask = scores >= threshold

        feature_regions[feature_name] = _regions_from_mask(
            times,
            mask,
            DEFAULT_FILL_GAP,
            DEFAULT_MIN_DURATION,
        )

    # Union all feature regions, then merge them into candidate events.
    all_regions = [
        region
        for regions in feature_regions.values()
        for region in regions
    ]

    merged_regions = _merge_regions(
        all_regions,
        merge_gap,
    )

    candidates = []

    for event_id, (start, end) in enumerate(
        merged_regions,
        start=1,
    ):
        # Use all samples falling inside the unpadded merged interval.
        sample_mask = (
            (times >= start) &
            (times <= end)
        )

        if not np.any(sample_mask):
            continue

        strongest_score = 0.0
        strongest_feature = ""
        active_features = []

        for feature_name, scores in feature_scores.items():
            event_scores = scores[sample_mask]

            if len(event_scores) == 0:
                continue

            peak = float(np.max(event_scores))

            if peak > 0.0:
                active_features.append(feature_name)

            if peak > strongest_score:
                strongest_score = peak
                strongest_feature = feature_name

        regions_text = []

        for feature_name, regions in feature_regions.items():
            for region_start, region_end in regions:
                if (
                    region_end >= start - 1e-9
                    and region_start <= end + 1e-9
                ):
                    regions_text.append(
                        f"{feature_name}:"
                        f"{region_start:.2f}-"
                        f"{region_end:.2f}"
                    )

        candidates.append({
            "id": event_id,
            "raw_start": start,
            "raw_end": end,
            "start": max(
                0.0,
                start - DEFAULT_PADDING,
            ),
            "end": end + DEFAULT_PADDING,
            "duration": (
                end - start +
                2.0 * DEFAULT_PADDING
            ),
            "score": strongest_score,
            "features": "+".join(active_features),
            "strongest_feature": strongest_feature,
            "num_feature_regions": len(regions_text),
            "feature_regions": ";".join(regions_text),
        })

    candidate_csv_path = csv_path.with_name(
        f"{csv_path.stem}_candidate_events.csv"
    )
    autocut_path = csv_path.with_name(
        f"{csv_path.stem}_autocut.txt"
    )

    candidate_fields = [
        "id",
        "start",
        "end",
        "duration",
        "peak",
        "score",
        "features",
        "strongest_feature",
        "num_feature_regions",
        "feature_regions",
    ]

    with open(
        candidate_csv_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=candidate_fields,
        )
        writer.writeheader()

        for candidate in candidates:
            writer.writerow({
                "id": candidate["id"],
                "start": f"{candidate['start']:.2f}",
                "end": f"{candidate['end']:.2f}",
                "duration": f"{candidate['duration']:.2f}",
                "peak": f"{candidate['score']:.3f}",
                "score": f"{candidate['score']:.3f}",
                "features": candidate["features"],
                "strongest_feature": candidate["strongest_feature"],
                "num_feature_regions": candidate["num_feature_regions"],
                "feature_regions": candidate["feature_regions"],
            })

    with open(
        autocut_path,
        "w",
        encoding="utf-8",
    ) as handle:
        handle.write(f"{video_path}\n")

        for candidate in candidates:
            absolute_start = (
                source_start +
                candidate["start"]
            )
            absolute_end = (
                source_start +
                candidate["end"]
            )

            description = (
                "AUTO "
                f"score={candidate['score']:.3f} "
                f"features={candidate['features'] or 'none'}"
            )

            handle.write(
                f"{_format_autocut_time(absolute_start)} "
                f"{_format_autocut_time(absolute_end)} "
                f"5 5 {description}\n"
            )

    return (
        candidate_csv_path,
        autocut_path,
        candidates,
    )


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

def process_video(
    video_path: Path,
    output_path: Path,
    csv_path: Path,
    start: float,
    duration: float | None,
    flow_fps: float,
    flow_width: int,
    flow_height: float,
    flow_alpha: float,
    arrows: bool,
    video: bool,
    track: bool,
    merge_gap: float,
):
    interrupted = False

    width, height, source_fps, source_duration = get_video_info(
        video_path
    )

    if duration is None:
        processing_duration = max(
            0.0,
            source_duration - start,
        )
    else:
        processing_duration = max(
            0.0,
            min(
                duration,
                source_duration - start,
            ),
        )

    if processing_duration <= 0:
        raise RuntimeError(
            "Selected processing interval has zero duration."
        )

    print()
    print("Integrated Bike + Optical Flow Analysis")
    print("-----------------------------------------")
    print(f"Input:          {video_path}")
    if video:
        print(f"Output:         {output_path}")
    print(f"CSV:            {csv_path}")
    print(f"Start:          {start:.3f} s")
    print(f"Duration:       {processing_duration:.3f} s")
    print(f"Source:         {width}x{height}")
    print(f"Source FPS:     {source_fps:.3f}")
    print(f"Flow FPS:       {flow_fps:.3f}")
    print(f"Flow width:     {flow_width}")
    print(f"Flow height:    {flow_height:.1f}%")
    print(f"Flow alpha:     {flow_alpha:.2f}")
    print(f"Merge gap:      {merge_gap:.2f} s")
    print(
        f"Arrows:         {'yes' if arrows else 'no'}"
    )
    print()

    frame_bytes = width * height * 3

    decoder = start_ffmpeg_decode(
        video_path,
        start,
        processing_duration,
        width,
        height,
    )

    encoder = None
    if video:
        encoder = start_ffmpeg_encode(
            output_path,
            width,
            height,
            source_fps,
        )

    csv_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = get_csv_fieldnames()

    csv_file = open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8",
    )

    writer = csv.DictWriter(
        csv_file,
        fieldnames=fieldnames,
    )

    if writer is not None:
        writer.writeheader()
        csv_file.flush()

    # ---------------------------------------------------------------
    # Read first frame
    # ---------------------------------------------------------------

    raw = decoder.stdout.read(frame_bytes)

    if len(raw) != frame_bytes:
        decoder.kill()
        encoder.kill()
        csv_file.close()

        raise RuntimeError(
            "Could not read the first decoded video frame."
        )

    first_frame = np.frombuffer(
        raw,
        dtype=np.uint8,
    ).reshape(
        height,
        width,
        3,
    )

    tracker = None

    if track:
        tracker = initialise_tracker(
            first_frame,
        )

    # First frame is output immediately.
    first_output = first_frame.copy()

    if tracker is not None:
        first_output = draw_tracker(
            first_output,
            tracker,
        )

    if encoder is not None:
        encoder.stdin.write(
            first_output.tobytes()
        )

    if video:
        cv2.imshow(
            "Bike + Optical Flow",
            first_output,
        )

        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            raise KeyboardInterrupt

    # ---------------------------------------------------------------
    # Flow sampling state
    # ---------------------------------------------------------------

    previous_flow_frame = first_frame.copy()
    previous_flow_time = 0.0
    previous_flow_angle = None

    next_flow_time = 1.0 / flow_fps

    flow_samples = 0
    csv_rows = 0

    frame_index = 0

    processing_start_wall = cv2.getTickCount()
    last_progress_time = 0.0
    flow_field = None

    # ---------------------------------------------------------------
    # Main decode loop
    # ---------------------------------------------------------------

    try:
        while True:
            raw = decoder.stdout.read(frame_bytes)

            if len(raw) == 0:
                break

            if len(raw) != frame_bytes:
                print()
                print(
                    "Warning: incomplete frame received from FFmpeg."
                )
                break

            frame = np.frombuffer(
                raw,
                dtype=np.uint8,
            ).reshape(
                height,
                width,
                3,
            )

            frame_index += 1

            current_time = (
                frame_index /
                source_fps
            )

            # -------------------------------------------------------
            # Track reference object on EVERY source frame.
            # -------------------------------------------------------

            if tracker is not None:
                bike_dx, bike_dy, bike_speed, tracked_points = (
                    update_tracker(
                        frame,
                        tracker
                    )
                )
            else:
                bike_dx = 0.0
                bike_dy = 0.0
                bike_speed = 0.0
                tracked_points = 0

            # -------------------------------------------------------
            # Optical-flow sampling.
            #
            # We deliberately use elapsed video time rather than
            # frame-index rounding because source FPS may be 59.94.
            # -------------------------------------------------------

            flow_features = None
            
            if current_time + 1e-9 >= next_flow_time:

                flow_field, flow_features, _ = (
                    calculate_flow_features(
                        previous_flow_frame,
                        frame,
                        flow_width,
                        flow_height,
                    )
                )

                flow_samples += 1

                # Circular angular change.
                current_angle = float(
                    flow_features["flow_angle"]
                )

                if previous_flow_angle is None:
                    angle_change = 0.0
                    angle_change_abs = 0.0
                else:
                    delta = (
                        current_angle -
                        previous_flow_angle
                    )

                    angle_change = (
                        (delta + 180.0) %
                        360.0
                    ) - 180.0

                    angle_change_abs = abs(
                        angle_change
                    )

                flow_features["angle_change"] = (
                    angle_change
                )

                flow_features["angle_change_abs"] = (
                    angle_change_abs
                )

                # ---------------------------------------------------
                # WRITE CSV ROW IMMEDIATELY.
                #
                # This is deliberately inside the successful flow
                # calculation block and followed by flush().
                # ---------------------------------------------------

                row = {
                    "time": current_time,

                    "bike_x": tracker["center_x"] if tracker is not None else 0.0,
                    "bike_y": tracker["center_y"] if tracker is not None else 0.0,
                    "bike_dx": bike_dx,
                    "bike_dy": bike_dy,
                    "bike_speed": bike_speed,
                    "tracked_points": tracked_points,
                }

                row.update(flow_features)
                writer.writerow(row)

                csv_rows += 1

                # Critical: make sure the row physically reaches the
                # CSV even if processing is interrupted.
                csv_file.flush()

                previous_flow_frame = frame.copy()
                previous_flow_time = current_time
                previous_flow_angle = current_angle

                # Advance by exact flow interval.
                #
                # The while loop protects against a delayed frame
                # causing us to skip multiple sample periods.
                flow_interval = 1.0 / flow_fps

                while next_flow_time <= current_time:
                    next_flow_time += flow_interval

            # -------------------------------------------------------
            # Visual output.
            # -------------------------------------------------------

            visual = frame

            if flow_field is not None:
                visual = create_flow_overlay(
                    visual,
                    flow_field,
                    flow_width,
                    flow_height,
                    flow_alpha,
                    arrows,
                )

            if tracker is not None:
                visual = draw_tracker(
                    visual,
                    tracker,
                )

            if encoder is not None:
                encoder.stdin.write(
                    visual.tobytes()
                )

            # -------------------------------------------------------
            # Display
            # -------------------------------------------------------

            if video:
                cv2.imshow(
                    "Bike + Optical Flow",
                    visual,
                )

                key = cv2.waitKey(1) & 0xFF

                if key == 27:
                    print()
                    print("Interrupted by user.")
                    interrupted = True
                    break

            # -------------------------------------------------------
            # Progress
            # -------------------------------------------------------

            elapsed_wall = (
                cv2.getTickCount() -
                processing_start_wall
            ) / cv2.getTickFrequency()

            processed_video_time = min(
                current_time,
                processing_duration,
            )

            processing_fps = (
                frame_index /
                elapsed_wall
                if elapsed_wall > 0
                else 0.0
            )

            # Only print progress every 2 sec
            if elapsed_wall - last_progress_time >= 2.0 :
                print_progress(
                    processed_video_time,
                    elapsed_wall,
                    processing_duration,
                    processing_fps,
                    source_fps,
                    flow_samples,
                    csv_rows,
                )
                last_progress_time = elapsed_wall

            if current_time >= processing_duration:
                break

    finally:
        print()

        # -----------------------------------------------------------
        # Close decoder
        # -----------------------------------------------------------

    if interrupted:
        try:
            decoder.terminate()
        except Exception:
            pass

    try:
        decoder.wait(timeout=10)
    except subprocess.TimeoutExpired:
        decoder.kill()
        decoder.wait()

    try:
        if decoder.stdout:
            decoder.stdout.close()
    except Exception:
        pass

        # -----------------------------------------------------------
        # Finish encoder
        # -----------------------------------------------------------

        if encoder is not None:
            try:
                if interrupted:
                    encoder.terminate()
                else:
                    encoder.stdin.close()
            except Exception:
                pass

            try:
                encoder.wait(timeout=10 if interrupted else 120)
            except subprocess.TimeoutExpired:
                encoder.kill()
                encoder.wait()

        # -----------------------------------------------------------
        # Close CSV
        # -----------------------------------------------------------

        csv_file.flush()
        csv_file.close()

        if video:
            cv2.destroyAllWindows()

    if encoder is not None:
        encoder.stdin.close()
        encoder.wait()

        if encoder.returncode != 0:
            stderr = ""

            if encoder.stderr:
                try:
                    stderr = encoder.stderr.read().decode(
                        "utf-8",
                        errors="replace",
                    )
                except Exception:
                    pass

            raise RuntimeError(
                "FFmpeg encoder failed.\n"
                + stderr
            )

    if not interrupted and decoder.returncode not in (0, None):
        stderr = ""

        if decoder.stderr:
            try:
                stderr = decoder.stderr.read().decode(
                    "utf-8",
                    errors="replace",
                )
            except Exception:
                pass

        raise RuntimeError(
            "FFmpeg decoder failed.\n"
            + stderr
        )
    
    """
    # 1. Collect dictionary features for every frame in a clip
    clip_features = []
    for frame_prev, frame_curr in video_frames:
        flow, frame_dict, _ = calculate_flow_features(frame_prev, frame_curr, flow_width=112, flow_height_percent=100)
        # Convert dictionary values to a list in a deterministic order
        feature_vector = list(frame_dict.values())
        clip_features.append(feature_vector)

    # 2. Convert to a 2D NumPy array: Shape (T, F) e.g., (150, 65)
    sequence_matrix = np.array(clip_features, dtype=np.float32)

    # 3. Specify indices of columns where rate of change matters most
    # (e.g., flow_mean, flow_y, flow_x, div_abs_mean, div_normalized_mean, curl_abs_mean, flow_coherence)
    target_feature_indices = [0, 5, 6, 14, 21, 22, 12] 

    # 4. Append derivatives: Shape becomes (150, 72)
    sequence_matrix_with_deltas = append_temporal_derivatives(sequence_matrix, target_feature_indices)

    # Save directly as a lightweight NumPy array file
    np.save("clip_001_features.npy", sequence_matrix_with_deltas)
    """

    if not interrupted:
        (
            candidate_csv_path,
            autocut_path,
            candidates,
        ) = generate_candidate_outputs(
            csv_path=csv_path,
            video_path=video_path,
            source_start=start,
            merge_gap=merge_gap,
        )
    else:
        candidate_csv_path = None
        autocut_path = None
        candidates = []

    print()
    print("Completed.")
    print(f"Flow samples: {flow_samples}")
    print(f"CSV rows:     {csv_rows}")
    if video:
        print(f"Video:        {output_path}")
    print(f"CSV:          {csv_path}")
    if candidate_csv_path is not None:
        print(f"Candidates:    {candidate_csv_path}")
        print(f"AutoCut:       {autocut_path}")
        print(f"Events:        {len(candidates)}")

    # A 50-second run at 10 FPS should produce roughly 490-500
    # flow samples, depending on the exact sampling alignment.
    if csv_rows == 0:
        raise RuntimeError(
            "Processing completed but CSV contains zero data rows."
        )


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Track a bike/reference object and analyze dense "
            "optical flow in the upper portion of the video."
        )
    )

    parser.add_argument(
        "input",
        type=Path,
        help="Input video.",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("bike_flow.mp4"),
        help="Output video.",
    )

    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("bike_flow.csv"),
        help="Output CSV.",
    )

    parser.add_argument(
        "--start",
        type=parse_time,
        default=0.0,
        help=(
            "Start time. Supports seconds, MM:SS or HH:MM:SS."
        ),
    )

    parser.add_argument(
        "--duration",
        type=parse_time,
        default=None,
        help=(
            "Duration. Supports seconds, MM:SS or HH:MM:SS."
        ),
    )

    parser.add_argument(
        "--flow-fps",
        type=float,
        default=DEFAULT_FLOW_FPS,
        help="Optical-flow analysis FPS.",
    )

    parser.add_argument(
        "--flow-width",
        type=int,
        default=DEFAULT_FLOW_WIDTH,
        help="Optical-flow analysis width.",
    )

    parser.add_argument(
        "--flow-height",
        type=float,
        default=DEFAULT_FLOW_HEIGHT,
        help=(
            "Percentage of image height used for flow, "
            "measured from the top."
        ),
    )

    parser.add_argument(
        "--flow-alpha",
        type=float,
        default=DEFAULT_FLOW_ALPHA,
        help="Flow overlay alpha.",
    )

    parser.add_argument(
        "--arrows",
        action="store_true",
        help="Enable sparse optical-flow arrows. Only available with --video.",
    )

    parser.add_argument(
        "--video",
        action="store_true",
        help="Enable overlay video.",
    )

    parser.add_argument(
        "--track",
        action="store_true",
        help="Enable bike/reference tracking and ROI selection. Only available with --video.",
    )

    parser.add_argument(
        "--merge-gap",
        type=float,
        default=DEFAULT_MERGE_GAP,
        help="Maximum gap used to merge candidate regions, in seconds.",
    )

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = build_parser()
    args = parser.parse_args()

    if not args.input.exists():
        print(
            f"Error: input video does not exist:\n{args.input}",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.flow_fps <= 0:
        print(
            "Error: --flow-fps must be greater than zero.",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.flow_width < 32:
        print(
            "Error: --flow-width must be at least 32.",
            file=sys.stderr,
        )
        sys.exit(1)

    if not 1.0 <= args.flow_height <= 100.0:
        print(
            "Error: --flow-height must be between 1 and 100.",
            file=sys.stderr,
        )
        sys.exit(1)

    if not 0.0 <= args.flow_alpha <= 1.0:
        print(
            "Error: --flow-alpha must be between 0 and 1.",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.start < 0:
        print(
            "Error: --start cannot be negative.",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.duration is not None and args.duration <= 0:
        print(
            "Error: --duration must be greater than zero.",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.merge_gap < 0:
        print(
            "Error: --merge-gap cannot be negative.",
            file=sys.stderr,
        )
        sys.exit(1)

    if not args.video:
        if args.track or args.arrows:
            print(
                "Error: --track or --arrows cannot be used without --video.",
                file=sys.stderr,
            )
            sys.exit(1)

    try:
        process_video(
            video_path=args.input,
            output_path=args.output,
            csv_path=args.csv,
            start=args.start,
            duration=args.duration,
            flow_fps=args.flow_fps,
            flow_width=args.flow_width,
            flow_height=args.flow_height,
            flow_alpha=args.flow_alpha,
            arrows=args.arrows,
            video=args.video,
            track=args.track,
            merge_gap=args.merge_gap,
        )

    except KeyboardInterrupt:
        print()
        print("Interrupted.")

    except Exception as exc:
        print()
        print(
            f"ERROR: {exc}",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()