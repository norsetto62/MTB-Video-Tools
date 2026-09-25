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
    no_video: bool,
):
    """
    Ask the user to select a reference object and detect Shi-Tomasi
    features inside it.
    """

    if no_video:
        raise RuntimeError(
            "Bike/reference ROI selection requires video. "
            "Remove --no-video for the first tracking run."
        )

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
        "flow_coherence": flow_coherence,
        "flow_angle": flow_angle,

        "div_abs_mean": div_abs_mean,
        "div_abs_p90": div_abs_p90,
        "div_std": div_std,
        "div_pos_fraction": div_pos_fraction,
        "div_neg_fraction": div_neg_fraction,

        "curl_abs_mean": curl_abs_mean,
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


# ---------------------------------------------------------------------------
# Flow visualization
# ---------------------------------------------------------------------------

def create_flow_overlay(
    frame: np.ndarray,
    flow: np.ndarray,
    flow_width: int,
    flow_height_percent: float,
    alpha: float,
    draw_arrows: bool,
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

    if draw_arrows:
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
        "flow_coherence",
        "flow_angle",

        "angle_change",
        "angle_change_abs",

        "div_abs_mean",
        "div_abs_p90",
        "div_std",
        "div_pos_fraction",
        "div_neg_fraction",

        "curl_abs_mean",
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
    draw_arrows: bool,
    no_video: bool,
    no_track: bool,
):
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
    print(
        f"Arrows:         {'yes' if draw_arrows else 'no'}"
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
    if not no_video:
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

    if not no_track:
        tracker = initialise_tracker(
            first_frame,
            no_video,
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

    if not no_video:
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
            flow_field = None

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
                    draw_arrows,
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

            if not no_video:
                cv2.imshow(
                    "Bike + Optical Flow",
                    visual,
                )

                key = cv2.waitKey(1) & 0xFF

                if key == 27:
                    print()
                    print("Interrupted by user.")
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

        try:
            if decoder.stdout:
                decoder.stdout.close()
        except Exception:
            pass

        try:
            decoder.wait(timeout=10)
        except subprocess.TimeoutExpired:
            decoder.kill()
            decoder.wait()

        # -----------------------------------------------------------
        # Finish encoder
        # -----------------------------------------------------------

        if encoder is not None:
            try:
                encoder.stdin.close()
            except Exception:
                pass

            try:
                encoder.wait(timeout=120)
            except subprocess.TimeoutExpired:
                encoder.kill()
                encoder.wait()

        # -----------------------------------------------------------
        # Close CSV
        # -----------------------------------------------------------

        csv_file.flush()
        csv_file.close()

        if not no_video:
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

    if decoder.returncode not in (0, None):
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

    print()
    print("Completed.")
    print(f"Flow samples: {flow_samples}")
    print(f"CSV rows:     {csv_rows}")
    if not no_video:
        print(f"Video:        {output_path}")
    print(f"CSV:          {csv_path}")

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
        "video",
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
        "--no-arrows",
        action="store_true",
        help="Disable sparse optical-flow arrows.",
    )

    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Disable creation of the annotated output video.",
    )

    parser.add_argument(
        "--no-track",
        action="store_true",
        help="Disable bike/reference tracking and ROI selection.",
    )

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = build_parser()
    args = parser.parse_args()

    if not args.video.exists():
        print(
            f"Error: input video does not exist:\n{args.video}",
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

    try:
        process_video(
            video_path=args.video,
            output_path=args.output,
            csv_path=args.csv,
            start=args.start,
            duration=args.duration,
            flow_fps=args.flow_fps,
            flow_width=args.flow_width,
            flow_height=args.flow_height,
            flow_alpha=args.flow_alpha,
            draw_arrows=not args.no_arrows,
            no_video=args.no_video,
            no_track=args.no_track,
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