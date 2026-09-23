#!/usr/bin/env python3

import argparse
import csv
import math
import re
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

FPS = 2
WIDTH = 480

DEFAULT_INTERVALS_FILE = Path("intervals.txt")
DEFAULT_OUTPUT_DIR = Path("data") / "flow_scores"

GRID_ROWS = 3
GRID_COLS = 3


# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------

def parse_time(value: str) -> float:
    """
    Parse:
        SS
        MM:SS
        HH:MM:SS
    """
    value = value.strip()

    if not value:
        raise ValueError("Empty time value")

    if ":" not in value:
        return float(value)

    parts = value.split(":")
    if len(parts) == 2:
        minutes, seconds = parts
        return int(minutes) * 60 + float(seconds)

    if len(parts) == 3:
        hours, minutes, seconds = parts
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)

    raise ValueError(f"Invalid time format: {value}")


def format_time(seconds: float) -> str:
    seconds = max(0.0, seconds)

    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60

    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:05.2f}"

    return f"{minutes:02d}:{secs:05.2f}"


# ---------------------------------------------------------------------------
# intervals.txt parsing
# ---------------------------------------------------------------------------

def looks_like_video_path(line: str) -> bool:
    """
    Recognize a video path without requiring it to exist.

    This deliberately accepts Windows paths such as:
        D:\\Prenestini\\Mentorella\\foo.MP4
    """
    lower = line.lower()

    if lower.endswith((".mp4", ".mov", ".mkv", ".avi", ".m4v", ".mts", ".m2ts")):
        return True

    if re.match(r"^[A-Za-z]:[\\/]", line):
        return True

    if line.startswith(("./", "../", ".\\", "..\\", "/")):
        return True

    return False


def is_header(line: str) -> bool:
    """
    Ignore annotation-style headers such as:

        Start End MTB Video Remarks
    """
    tokens = line.lower().split()

    if len(tokens) >= 2:
        if tokens[0] == "start" and tokens[1] == "end":
            return True

    return False


def parse_intervals_file(path: Path):
    """
    Parse intervals.txt.

    Supported format:

        Start    End    MTB    Video    Remarks
        D:\video1.mp4
        00:00    05:00
        10:00    12:30

        D:\video2.mp4

    A video path with intervals means only those intervals are analyzed.

    A video path with no following intervals means the whole video is
    analyzed.

    A path beginning with '-' excludes that video.

    Example:

        -D:\video_to_skip.mp4

    Returns:
        {
            video_path_string: [
                (start_seconds, end_seconds),
                ...
            ]
        }

    If the list of intervals is empty, the entire video is analyzed.
    """

    if not path.exists():
        raise FileNotFoundError(f"Intervals file not found: {path}")

    result = {}

    current_video = None
    current_intervals = []

    def store_current():
        nonlocal current_video, current_intervals

        if current_video is None:
            return

        key = current_video

        if key not in result:
            result[key] = []

        result[key].extend(current_intervals)

        current_video = None
        current_intervals = []

    with path.open("r", encoding="utf-8-sig") as f:
        for line_number, raw_line in enumerate(f, start=1):
            line = raw_line.strip()

            if not line:
                continue

            if line.startswith("#"):
                continue

            if is_header(line):
                continue

            # A new video path.
            if looks_like_video_path(line):
                store_current()

                current_video = line
                current_intervals = []
                continue

            # An excluded video.
            if line.startswith("-"):
                excluded = line[1:].strip()

                if looks_like_video_path(excluded):
                    store_current()

                    # Store an explicit exclusion marker.
                    result[f"__EXCLUDE__{excluded}"] = []

                    current_video = None
                    current_intervals = []
                    continue

            # Otherwise this should be a time interval.
            parts = line.split()

            if len(parts) < 2:
                print(
                    f"Warning: ignoring line {line_number}: {line}",
                    file=sys.stderr,
                )
                continue

            if current_video is None:
                print(
                    f"Warning: interval without video on line "
                    f"{line_number}: {line}",
                    file=sys.stderr,
                )
                continue

            try:
                start = parse_time(parts[0])
                end = parse_time(parts[1])
            except ValueError as exc:
                print(
                    f"Warning: ignoring line {line_number}: {exc}",
                    file=sys.stderr,
                )
                continue

            if end <= start:
                print(
                    f"Warning: ignoring invalid interval on line "
                    f"{line_number}: {line}",
                    file=sys.stderr,
                )
                continue

            current_intervals.append((start, end))

    store_current()

    # Remove explicit exclusion markers and filter excluded paths.
    excluded_paths = set()

    for key in list(result.keys()):
        if key.startswith("__EXCLUDE__"):
            excluded_paths.add(key[len("__EXCLUDE__"):])
            del result[key]

    for excluded in excluded_paths:
        result.pop(excluded, None)

    # Merge overlapping intervals for each video.
    for video, intervals in result.items():
        if not intervals:
            continue

        intervals.sort()

        merged = [intervals[0]]

        for start, end in intervals[1:]:
            prev_start, prev_end = merged[-1]

            if start <= prev_end:
                merged[-1] = (
                    prev_start,
                    max(prev_end, end),
                )
            else:
                merged.append((start, end))

        result[video] = merged

    return result


# ---------------------------------------------------------------------------
# ffprobe
# ---------------------------------------------------------------------------

def ffprobe_video(video_path: Path):
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate",
        "-show_entries",
        "format=duration",
        "-of", "default=noprint_wrappers=1",
        str(video_path),
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=True,
    )

    values = {}

    for line in result.stdout.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key] = value.strip()

    width = int(values["width"])
    height = int(values["height"])

    fps_num, fps_den = values["r_frame_rate"].split("/")
    source_fps = float(fps_num) / float(fps_den)

    duration = float(values["duration"])

    return width, height, source_fps, duration


# ---------------------------------------------------------------------------
# FFmpeg frame reader
# ---------------------------------------------------------------------------

def start_ffmpeg(video_path: Path, start: float, duration: float, height: int):
    """
    Start FFmpeg and produce 8-bit grayscale frames at FPS and WIDTH.

    CUDA is used for HEVC decoding where supported.
    """

    scaled_height = int(round(height * WIDTH / 480))

    # Keep dimensions even.
    scaled_height -= scaled_height % 2

    if scaled_height <= 0:
        scaled_height = 2

    vf = (
        f"fps={FPS},"
        f"scale={WIDTH}:{scaled_height}:flags=fast_bilinear,"
        f"format=gray"
    )

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",

        "-hwaccel", "cuda",

        "-ss", f"{start:.3f}",
        "-i", str(video_path),

        "-t", f"{duration:.3f}",

        "-vf", vf,

        "-f", "rawvideo",
        "-pix_fmt", "gray",
        "pipe:1",
    ]

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=10**8,
    )

    return process, scaled_height


# ---------------------------------------------------------------------------
# Optical flow
# ---------------------------------------------------------------------------

def calculate_flow_features(prev_frame, current_frame):
    """
    Calculate dense Farneback optical flow.

    Returns global and 3x3 spatial features.
    """

    flow = cv2.calcOpticalFlowFarneback(
        prev_frame,
        current_frame,
        None,

        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,

        flags=0,
    )

    flow_x = flow[..., 0]
    flow_y = flow[..., 1]

    magnitude, angle = cv2.cartToPolar(
        flow_x,
        flow_y,
        angleInDegrees=False,
    )

    # ------------------------------------------------------------------
    # Global statistics
    # ------------------------------------------------------------------

    mean_x = float(np.mean(flow_x))
    mean_y = float(np.mean(flow_y))

    mean_abs_x = float(np.mean(np.abs(flow_x)))
    mean_abs_y = float(np.mean(np.abs(flow_y)))

    mean_magnitude = float(np.mean(magnitude))
    median_magnitude = float(np.median(magnitude))
    p90_magnitude = float(np.percentile(magnitude, 90))
    p95_magnitude = float(np.percentile(magnitude, 95))

    std_x = float(np.std(flow_x))
    std_y = float(np.std(flow_y))
    std_magnitude = float(np.std(magnitude))

    # Directional coherence:
    #
    # magnitude of average vector
    # ----------------------------
    # average magnitude
    #
    # Near 1 -> mostly coherent movement.
    # Near 0 -> opposing/random/local movement.
    mean_vector_magnitude = math.sqrt(
        mean_x * mean_x +
        mean_y * mean_y
    )

    if mean_magnitude > 1e-9:
        coherence = mean_vector_magnitude / mean_magnitude
    else:
        coherence = 0.0

    # Dominant direction of average vector.
    mean_angle = math.degrees(
        math.atan2(mean_y, mean_x)
    )

    # ------------------------------------------------------------------
    # Spatial 3x3 statistics
    # ------------------------------------------------------------------

    height, width = magnitude.shape

    features = {
        "flow_x": mean_x,
        "flow_y": mean_y,
        "flow_abs_x": mean_abs_x,
        "flow_abs_y": mean_abs_y,
        "flow_magnitude": mean_magnitude,
        "flow_median_magnitude": median_magnitude,
        "flow_p90_magnitude": p90_magnitude,
        "flow_p95_magnitude": p95_magnitude,
        "flow_std_x": std_x,
        "flow_std_y": std_y,
        "flow_std_magnitude": std_magnitude,
        "flow_coherence": coherence,
        "flow_angle": mean_angle,
    }

    # 3 rows x 3 columns.
    for row in range(GRID_ROWS):
        y0 = row * height // GRID_ROWS
        y1 = (row + 1) * height // GRID_ROWS

        for col in range(GRID_COLS):
            x0 = col * width // GRID_COLS
            x1 = (col + 1) * width // GRID_COLS

            region_x = flow_x[y0:y1, x0:x1]
            region_y = flow_y[y0:y1, x0:x1]
            region_mag = magnitude[y0:y1, x0:x1]

            prefix = f"g{row + 1}{col + 1}"

            features[f"{prefix}_x"] = float(np.mean(region_x))
            features[f"{prefix}_y"] = float(np.mean(region_y))
            features[f"{prefix}_abs_x"] = float(
                np.mean(np.abs(region_x))
            )
            features[f"{prefix}_abs_y"] = float(
                np.mean(np.abs(region_y))
            )
            features[f"{prefix}_mag"] = float(
                np.mean(region_mag)
            )
            features[f"{prefix}_p90"] = float(
                np.percentile(region_mag, 90)
            )

    return features


# ---------------------------------------------------------------------------
# Progress
# ---------------------------------------------------------------------------

def print_progress(
    processed_frames,
    expected_frames,
    elapsed,
    interval_start,
    interval_duration,
):
    if elapsed <= 0:
        return

    speed = processed_frames / FPS / elapsed

    processed_seconds = processed_frames / FPS

    percent = 100.0

    if expected_frames > 0:
        percent = min(
            100.0,
            processed_frames / expected_frames * 100.0,
        )

    if speed > 0:
        remaining_seconds = max(
            0.0,
            interval_duration - processed_seconds,
        )

        eta_seconds = remaining_seconds / speed
    else:
        eta_seconds = 0

    current_time = min(
        interval_duration,
        processed_seconds,
    )

    print(
        f"\r"
        f"{percent:6.2f}%  "
        f"{format_time(interval_start + current_time)} / "
        f"{format_time(interval_start + interval_duration)}  "
        f"speed {speed:.2f}x  "
        f"ETA {format_time(eta_seconds)}",
        end="",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Process one interval
# ---------------------------------------------------------------------------

def process_interval(
    video_path: Path,
    interval_start: float,
    interval_end: float,
    video_height: int,
    writer,
):
    duration = interval_end - interval_start

    process, frame_height = start_ffmpeg(
        video_path,
        interval_start,
        duration,
        video_height,
    )

    frame_size = WIDTH * frame_height

    prev_frame = None

    frame_index = 0
    flow_samples = 0

    expected_frames = max(
        1,
        int(math.ceil(duration * FPS)),
    )

    start_time = time.time()
    last_progress_time = start_time

    try:
        while True:
            raw = process.stdout.read(frame_size)

            if len(raw) < frame_size:
                break

            current_frame = np.frombuffer(
                raw,
                dtype=np.uint8,
            ).reshape(
                frame_height,
                WIDTH,
            )

            timestamp = (
                interval_start +
                frame_index / FPS
            )

            if prev_frame is not None:
                features = calculate_flow_features(
                    prev_frame,
                    current_frame,
                )

                row = {
                    "timestamp": f"{timestamp:.3f}",
                }

                row.update(
                    {
                        key: f"{value:.6f}"
                        for key, value in features.items()
                    }
                )

                writer.writerow(row)

                flow_samples += 1

            prev_frame = current_frame
            frame_index += 1

            now = time.time()

            if now - last_progress_time >= 0.5:
                print_progress(
                    frame_index,
                    expected_frames,
                    now - start_time,
                    interval_start,
                    duration,
                )
                last_progress_time = now

    finally:
        process.stdout.close()

        stderr = process.stderr.read().decode(
            "utf-8",
            errors="replace",
        )

        process.stderr.close()

        return_code = process.wait()

    print_progress(
        frame_index,
        expected_frames,
        max(time.time() - start_time, 1e-6),
        interval_start,
        duration,
    )

    print()

    if return_code != 0:
        raise RuntimeError(
            f"FFmpeg failed for interval "
            f"{format_time(interval_start)} - "
            f"{format_time(interval_end)}:\n"
            f"{stderr}"
        )

    return frame_index, flow_samples


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Analyze video using dense optical flow. "
            "Intervals are read from intervals.txt."
        )
    )

    parser.add_argument(
        "video",
        nargs="?",
        help=(
            "Optional video file. If omitted, videos are read "
            "from intervals.txt."
        ),
    )

    parser.add_argument(
        "--intervals",
        default=str(DEFAULT_INTERVALS_FILE),
        help="Intervals file (default: intervals.txt)",
    )

    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help=(
            "Output directory "
            "(default: data/flow_scores)"
        ),
    )

    args = parser.parse_args()

    intervals_file = Path(args.intervals)
    output_dir = Path(args.output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------------------
    # Determine videos / intervals
    # ------------------------------------------------------------------

    if args.video:
        video_path = Path(args.video)

        if not video_path.exists():
            raise FileNotFoundError(
                f"Video not found: {video_path}"
            )

        # Direct video argument:
        # analyze the whole video unless intervals.txt contains it.
        interval_map = {
            str(video_path): []
        }

        if intervals_file.exists():
            parsed = parse_intervals_file(
                intervals_file
            )

            # Normalize exact string path matching.
            if str(video_path) in parsed:
                interval_map = {
                    str(video_path): parsed[str(video_path)]
                }

    else:
        if not intervals_file.exists():
            raise FileNotFoundError(
                f"{intervals_file} not found. "
                "Provide a video path or create intervals.txt."
            )

        interval_map = parse_intervals_file(
            intervals_file
        )

    if not interval_map:
        print("No videos to analyze.")
        return 0

    # ------------------------------------------------------------------
    # Process videos
    # ------------------------------------------------------------------

    for video_string, intervals in interval_map.items():
        video_path = Path(video_string)

        print()
        print("=" * 70)
        print(f"Video: {video_path}")

        if not video_path.exists():
            print(
                f"ERROR: video not found: {video_path}",
                file=sys.stderr,
            )
            continue

        try:
            width, height, source_fps, total_duration = (
                ffprobe_video(video_path)
            )
        except Exception as exc:
            print(
                f"ERROR: ffprobe failed: {exc}",
                file=sys.stderr,
            )
            continue

        print(
            f"Source: {width}x{height} "
            f"{source_fps:.3f} fps  "
            f"duration {format_time(total_duration)}"
        )

        # No intervals means analyze the entire video.
        if not intervals:
            intervals = [
                (0.0, total_duration)
            ]

        # Clamp intervals to actual video duration.
        cleaned_intervals = []

        for start, end in intervals:
            start = max(0.0, start)
            end = min(total_duration, end)

            if end > start:
                cleaned_intervals.append(
                    (start, end)
                )

        intervals = cleaned_intervals

        if not intervals:
            print("No valid intervals for this video.")
            continue

        print("Intervals:")

        total_requested = 0.0

        for i, (start, end) in enumerate(intervals, 1):
            print(
                f"  {i}. "
                f"{format_time(start)} - "
                f"{format_time(end)}"
            )

            total_requested += end - start

        output_path = (
            output_dir /
            f"{video_path.stem}.csv"
        )

        fieldnames = [
            "timestamp",

            "flow_x",
            "flow_y",
            "flow_abs_x",
            "flow_abs_y",

            "flow_magnitude",
            "flow_median_magnitude",
            "flow_p90_magnitude",
            "flow_p95_magnitude",

            "flow_std_x",
            "flow_std_y",
            "flow_std_magnitude",

            "flow_coherence",
            "flow_angle",
        ]

        # Add 3x3 grid fields.
        for row in range(GRID_ROWS):
            for col in range(GRID_COLS):
                prefix = f"g{row + 1}{col + 1}"

                fieldnames.extend([
                    f"{prefix}_x",
                    f"{prefix}_y",
                    f"{prefix}_abs_x",
                    f"{prefix}_abs_y",
                    f"{prefix}_mag",
                    f"{prefix}_p90",
                ])

        print()
        print(f"Output: {output_path}")

        total_frames = 0
        total_flow_samples = 0

        overall_start = time.time()

        with output_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=fieldnames,
            )

            writer.writeheader()

            for index, (start, end) in enumerate(
                intervals,
                start=1,
            ):
                print()
                print(
                    f"Interval {index}/{len(intervals)}: "
                    f"{format_time(start)} - "
                    f"{format_time(end)}"
                )

                frames, flow_samples = process_interval(
                    video_path,
                    start,
                    end,
                    height,
                    writer,
                )

                total_frames += frames
                total_flow_samples += flow_samples

        elapsed = time.time() - overall_start

        speed = (
            total_requested / elapsed
            if elapsed > 0
            else 0
        )

        print()
        print(
            f"Completed {total_flow_samples} flow samples "
            f"from {total_frames} decoded frames "
            f"in {elapsed:.1f}s "
            f"({speed:.2f}x realtime)"
        )

        print(f"Saved: {output_path}")

    print()
    print("Done.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())