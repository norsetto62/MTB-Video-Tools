import argparse
import csv
import math
import re
import subprocess
import sys
import time
from pathlib import Path


DEFAULT_INTERVALS_FILE = Path("intervals.txt")
DEFAULT_OUTPUT_DIR = Path("data") / "motion_scores"

VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".avi", ".mkv"}


def format_time(seconds):
    seconds = max(0, float(seconds))
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)

    if h:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def parse_time(value):
    value = value.strip()

    if not value:
        return None

    # HH:MM:SS
    if re.fullmatch(r"\d+:\d{2}:\d{2}(?:\.\d+)?", value):
        h, m, s = value.split(":", 2)
        return int(h) * 3600 + int(m) * 60 + float(s)

    # MM:SS
    if re.fullmatch(r"\d+:\d{2}(?:\.\d+)?", value):
        m, s = value.split(":", 1)
        return int(m) * 60 + float(s)

    # Seconds
    try:
        return float(value)
    except ValueError:
        return None


def normalize_path(path_text, base_dir):
    path_text = path_text.strip().strip('"')

    path = Path(path_text)

    if not path.is_absolute():
        path = base_dir / path

    return path.resolve()


def paths_match(a, b):
    try:
        return a.resolve().samefile(b.resolve())
    except (FileNotFoundError, OSError):
        return str(a.resolve()).lower() == str(b.resolve()).lower()


def parse_intervals_file(intervals_path):
    """
    Parse intervals.txt.

    Supported format:

        Start    End    MTB    Video    Remarks
        D:\\video1.mp4
        00:00    01:20    4    5    description
        03:10    04:00    5    3    description

        D:\\video2.mp4
        00:00    05:00    4    4

    A line beginning with '-' excludes that video.

    A video with no following time ranges means:
        analyze the whole video.

    Returns:
        entries = [
            {
                "path": Path(...),
                "excluded": bool,
                "intervals": [(start, end), ...]
            },
            ...
        ]
    """

    if not intervals_path.exists():
        return []

    entries = []
    current = None

    with intervals_path.open("r", encoding="utf-8-sig") as f:
        for raw_line in f:
            line = raw_line.strip()

            if not line:
                continue

            if line.startswith("#"):
                continue

            # Split on whitespace.
            fields = line.split()

            # ---------------------------------------------------------
            # Ignore the optional annotation/header line.
            # Example:
            #   Start    End   MTB   Video   Remarks
            # ---------------------------------------------------------
            normalized_fields = [field.lower() for field in fields]

            if (
                len(normalized_fields) >= 4
                and normalized_fields[:4] == ["start", "end", "mtb", "video"]
            ):
                continue

            # Also accept the shorter header:
            #   Start End MTB Video
            if normalized_fields == ["start", "end", "mtb", "video"]:
                continue

            # Explicit exclusion.
            if line.startswith("-"):
                path_text = line[1:].strip()

                if path_text:
                    path = normalize_path(path_text, intervals_path.parent)

                    entries.append(
                        {
                            "path": path,
                            "excluded": True,
                            "intervals": [],
                        }
                    )

                    current = None

                continue

            # Try to interpret the first two fields as times.
            if len(fields) >= 2:
                start = parse_time(fields[0])
                end = parse_time(fields[1])

                if start is not None and end is not None:
                    if current is None:
                        print(
                            f"WARNING: interval without a preceding video:\n"
                            f"  {line}"
                        )
                        continue

                    if end <= start:
                        print(
                            f"WARNING: invalid interval (end <= start):\n"
                            f"  {line}"
                        )
                        continue

                    current["intervals"].append((start, end))
                    continue

            # Otherwise treat the line as a video path.
            path = normalize_path(line, intervals_path.parent)

            current = {
                "path": path,
                "excluded": False,
                "intervals": [],
            }

            entries.append(current)

    return entries


def get_video_duration(video_path):
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
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

    return float(result.stdout.strip())


def get_video_metadata(video_path):
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate",
        "-of",
        "csv=p=0",
        str(video_path),
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=True,
    )

    line = result.stdout.strip().splitlines()[0]
    width, height, frame_rate = line.split(",")

    width = int(width)
    height = int(height)

    num, den = frame_rate.split("/")
    fps = float(num) / float(den)

    return width, height, fps


def merge_intervals(intervals):
    if not intervals:
        return []

    intervals = sorted(intervals)

    merged = [list(intervals[0])]

    for start, end in intervals[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])

    return [(start, end) for start, end in merged]


def find_matching_entry(video_path, entries):
    for entry in entries:
        if paths_match(video_path, entry["path"]):
            return entry

    return None


def discover_videos(root):
    root = Path(root)

    if root.is_file():
        return [root]

    videos = []

    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
            videos.append(path.resolve())

    return sorted(videos)


def select_videos(input_path, intervals_path):
    entries = parse_intervals_file(intervals_path)

    # If an explicit input was supplied, use it as the starting point.
    if input_path is not None:
        input_path = Path(input_path)

        if input_path.is_file():
            candidates = [input_path.resolve()]
        elif input_path.is_dir():
            candidates = discover_videos(input_path)
        else:
            print(f"ERROR: input does not exist: {input_path}")
            sys.exit(1)

    # Otherwise derive video paths from intervals.txt.
    else:
        candidates = [
            entry["path"]
            for entry in entries
            if not entry["excluded"]
        ]

    selected = []

    for video in candidates:
        entry = find_matching_entry(video, entries)

        if entry and entry["excluded"]:
            continue

        if not video.exists():
            print(f"WARNING: video does not exist:")
            print(f"  {video}")
            continue

        selected.append(video)

    # Remove duplicates while preserving order.
    unique = []

    for video in selected:
        if not any(paths_match(video, other) for other in unique):
            unique.append(video)

    return unique, entries


def get_matching_intervals(video_path, entry, duration):
    """
    Determine which portions of a video should be analyzed.

    If there are no intervals for the video:
        analyze the whole video.

    Otherwise:
        analyze only the specified intervals.

    Intervals are clipped to the actual video duration and merged.
    """

    if entry is None or not entry["intervals"]:
        return [(0.0, duration)]

    clipped = []

    for start, end in entry["intervals"]:
        start = max(0.0, min(start, duration))
        end = max(0.0, min(end, duration))

        if end > start:
            clipped.append((start, end))

    return merge_intervals(clipped)


def run_ffmpeg_motion_analysis(
    video_path,
    intervals,
    output_csv,
    analysis_fps=2,
    analysis_width=480,
):
    """
    Analyze only the supplied intervals.

    Timestamps in the CSV remain relative to the original video.
    """

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    total_duration = sum(end - start for start, end in intervals)

    # Source metadata.
    width, height, source_fps = get_video_metadata(video_path)

    analysis_height = max(1, round(height * analysis_width / width))

    print(f"Source:       {width}x{height}")
    print(f"Source FPS:   {source_fps:.3f}")
    print(f"Intervals:")

    for start, end in intervals:
        print(
            f"  {format_time(start)} - {format_time(end)} "
            f"({end - start:.1f}s)"
        )

    print(f"Total analysis duration: {format_time(total_duration)}")
    print(f"Analysis:     {analysis_fps} FPS")
    print(
        f"Analysis size:{analysis_width}x{analysis_height}"
    )

    ffmpeg_path = "ffmpeg"

    cmd = [
        ffmpeg_path,
        "-hide_banner",
        "-loglevel",
        "error",
        "-hwaccel",
        "cuda",
        "-i",
        str(video_path),
    ]

    # We need one FFmpeg process per interval because each interval
    # has its own -ss / -t range.
    #
    # For simplicity and reliability, process intervals separately
    # and append the results to the same CSV.

    first_output = True
    total_frames = 0
    analysis_start_time = time.time()

    with output_csv.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:

        writer = csv.writer(csv_file)
        writer.writerow(["timestamp", "motion_score"])

        for interval_index, (start, end) in enumerate(intervals, 1):
            interval_duration = end - start

            print()
            print(
                f"Interval {interval_index}/{len(intervals)}: "
                f"{format_time(start)} - {format_time(end)}"
            )

            interval_cmd = [
                ffmpeg_path,
                "-hide_banner",
                "-loglevel",
                "error",
                "-hwaccel",
                "cuda",
                "-ss",
                f"{start:.3f}",
                "-i",
                str(video_path),
                "-t",
                f"{interval_duration:.3f}",
                "-vf",
                (
                    f"fps={analysis_fps},"
                    f"scale={analysis_width}:{analysis_height},"
                    "format=gray"
                ),
                "-f",
                "rawvideo",
                "-pix_fmt",
                "gray",
                "pipe:1",
            ]

            process = subprocess.Popen(
                interval_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            frame_size = analysis_width * analysis_height
            previous_frame = None

            expected_frames = max(
                1,
                int(interval_duration * analysis_fps)
            )

            interval_frames = 0
            interval_start_time = time.time()

            while True:
                raw = process.stdout.read(frame_size)

                if len(raw) < frame_size:
                    break

                interval_frames += 1
                total_frames += 1

                if previous_frame is not None:
                    # Avoid importing NumPy until needed.
                    import numpy as np

                    current = np.frombuffer(
                        raw,
                        dtype=np.uint8,
                    )

                    previous = np.frombuffer(
                        previous_frame,
                        dtype=np.uint8,
                    )

                    motion = float(
                        np.mean(
                            np.abs(
                                current.astype(np.int16)
                                - previous.astype(np.int16)
                            )
                        )
                    )

                    timestamp = (
                        start
                        + (interval_frames - 1) / analysis_fps
                    )

                    writer.writerow(
                        [
                            f"{timestamp:.3f}",
                            f"{motion:.6f}",
                        ]
                    )

                previous_frame = raw

                # Progress.
                elapsed = time.time() - analysis_start_time
                processed_duration = sum(
                    b - a
                    for a, b in intervals[: interval_index - 1]
                ) + min(
                    interval_frames / analysis_fps,
                    interval_duration,
                )

                progress = (
                    processed_duration / total_duration
                    if total_duration > 0
                    else 1.0
                )

                speed = (
                    processed_duration / elapsed
                    if elapsed > 0
                    else 0
                )

                remaining = (
                    (total_duration - processed_duration) / speed
                    if speed > 0
                    else 0
                )

                print(
                    f"\r"
                    f"{progress * 100:6.2f}%  "
                    f"{format_time(processed_duration)} / "
                    f"{format_time(total_duration)}  "
                    f"speed {speed:5.2f}x  "
                    f"ETA {format_time(remaining)}",
                    end="",
                    flush=True,
                )

            stderr = process.stderr.read().decode(
                "utf-8",
                errors="replace",
            )

            return_code = process.wait()

            print()

            if return_code != 0:
                raise RuntimeError(
                    f"FFmpeg failed for interval "
                    f"{format_time(start)}-{format_time(end)}:\n"
                    f"{stderr}"
                )

    elapsed = time.time() - analysis_start_time

    speed = (
        total_duration / elapsed
        if elapsed > 0
        else 0
    )

    print(
        f"Completed {total_frames} analysis frames "
        f"in {elapsed:.1f}s ({speed:.2f}x realtime)"
    )

    print(f"Saved: {output_csv}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze video motion using FFmpeg."
    )

    parser.add_argument(
        "input",
        nargs="?",
        help=(
            "Video file or directory. "
            "If omitted, videos are read from intervals.txt."
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
            "(default: data/motion_scores)"
        ),
    )

    parser.add_argument(
        "--fps",
        type=float,
        default=2,
        help="Analysis FPS (default: 2)",
    )

    parser.add_argument(
        "--width",
        type=int,
        default=480,
        help="Analysis width (default: 480)",
    )

    args = parser.parse_args()

    intervals_path = Path(args.intervals).resolve()
    output_dir = Path(args.output_dir).resolve()

    input_path = (
        Path(args.input).resolve()
        if args.input
        else None
    )

    videos, entries = select_videos(
        input_path,
        intervals_path,
    )

    if not videos:
        print("No videos selected.")
        return

    print(
        f"Using intervals file: {intervals_path}"
    )
    print()

    print(
        f"Selected {len(videos)} video(s):"
    )

    for video in videos:
        print(f"  {video}")

    print()

    for video_index, video_path in enumerate(
        videos,
        1,
    ):
        print(
            f"=== Video {video_index}/{len(videos)} ==="
        )
        print(video_path)

        duration = get_video_duration(video_path)

        entry = find_matching_entry(
            video_path,
            entries,
        )

        intervals = get_matching_intervals(
            video_path,
            entry,
            duration,
        )

        if not intervals:
            print("No valid intervals for this video.")
            continue

        output_csv = (
            output_dir
            / f"{video_path.stem}.csv"
        )

        run_ffmpeg_motion_analysis(
            video_path=video_path,
            intervals=intervals,
            output_csv=output_csv,
            analysis_fps=args.fps,
            analysis_width=args.width,
        )

        print()


if __name__ == "__main__":
    main()