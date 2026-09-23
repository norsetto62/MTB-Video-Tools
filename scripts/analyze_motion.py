import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np


# ============================================================
# Configuration
# ============================================================

FFMPEG = r"C:\Program Files\ffmpeg\bin\ffmpeg.exe"

FPS = 2
WIDTH = 480

DEFAULT_PERCENTILE = 85
DEFAULT_MIN_DURATION = 3.0
DEFAULT_MAX_GAP = 8.0
DEFAULT_PADDING = 1.0

VIDEO_EXTENSIONS = {
    ".mp4",
    ".mov",
    ".mkv",
    ".avi",
    ".mts",
    ".m2ts",
}


# ============================================================
# Timecode helpers
# ============================================================

def is_timecode(value):
    """Return True for MM:SS or HH:MM:SS."""
    parts = value.split(":")

    if len(parts) not in (2, 3):
        return False

    try:
        [float(x) for x in parts]
        return True
    except ValueError:
        return False


def timecode_to_seconds(value):
    parts = value.split(":")

    if len(parts) == 2:
        minutes, seconds = parts
        return int(minutes) * 60 + float(seconds)

    if len(parts) == 3:
        hours, minutes, seconds = parts
        return (
            int(hours) * 3600
            + int(minutes) * 60
            + float(seconds)
        )

    raise ValueError(f"Invalid timecode: {value}")


def seconds_to_timecode(seconds):
    seconds = max(0, float(seconds))

    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60

    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:05.2f}"

    return f"{minutes:02d}:{secs:05.2f}"


# ============================================================
# intervals.txt parser
# ============================================================

def parse_intervals_file(path):
    """
    Parse intervals.txt.

    Supported formats:

        filename.mp4

            Analyze the entire file.

        filename.mp4
        00:03 00:27 1 5 trail approach
        01:57 02:28 5 5 trail start

            Analyze only the listed intervals.

        -filename.mp4

            Explicitly skip this file.

    Filename inheritance:

        filename.mp4
        00:03 00:27 1 5 ...
        01:00 01:20 5 5 ...

    The second interval belongs to filename.mp4.

    Returns:

        {
            "filename.mp4": {
                "intervals": [
                    (start, end, mtb, film, description),
                    ...
                ],
                "skip": False
            }
        }
    """

    files = {}
    current_file = None

    with open(path, "r", encoding="utf-8-sig") as f:

        for raw_line in f:
            line = raw_line.strip()

            # Empty/comment lines
            if not line or line.startswith("#"):
                continue

            # Ignore the header
            upper = line.upper()

            if "START" in upper and "END" in upper:
                continue

            # ----------------------------------------------------
            # Try to recognize an interval line
            # ----------------------------------------------------

            parts = line.split(None, 4)

            if (
                len(parts) >= 4
                and is_timecode(parts[0])
                and is_timecode(parts[1])
            ):

                if current_file is None:
                    print(
                        f"WARNING: interval found before a filename:\n"
                        f"  {line}"
                    )
                    continue

                try:
                    start = timecode_to_seconds(parts[0])
                    end = timecode_to_seconds(parts[1])

                    mtb = int(parts[2])
                    film = int(parts[3])

                    description = (
                        parts[4].strip()
                        if len(parts) >= 5
                        else ""
                    )

                    if end <= start:
                        print(
                            f"WARNING: invalid interval "
                            f"(END <= START): {line}"
                        )
                        continue

                    files[current_file]["intervals"].append(
                        (
                            start,
                            end,
                            mtb,
                            film,
                            description,
                        )
                    )

                except ValueError:
                    print(
                        f"WARNING: invalid interval line:\n"
                        f"  {line}"
                    )

                continue

            # ----------------------------------------------------
            # Otherwise this is a filename
            # ----------------------------------------------------

            filename = line

            # Explicitly skipped file
            if filename.startswith("-"):
                filename = filename[1:].strip()

                if filename:
                    files[filename] = {
                        "intervals": [],
                        "skip": True,
                    }

                    print(
                        f"Marked to SKIP: {filename}"
                    )

                current_file = None
                continue

            # Normal file
            files[filename] = {
                "intervals": [],
                "skip": False,
            }

            current_file = filename

    return files


# ============================================================
# Video discovery
# ============================================================

def find_videos(input_path):
    path = Path(input_path)

    if path.is_file():
        return [path]

    if path.is_dir():
        videos = [
            p
            for p in path.iterdir()
            if p.is_file()
            and p.suffix.lower() in VIDEO_EXTENSIONS
        ]

        return sorted(videos)

    raise FileNotFoundError(
        f"Input path does not exist: {input_path}"
    )


# ============================================================
# Video duration
# ============================================================

def get_video_duration(video_path):
    cmd = [
        FFMPEG,
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-f",
        "null",
        "-",
    ]

    # ffmpeg doesn't always provide duration conveniently in
    # machine-readable form, so use ffprobe if available.
    ffprobe = str(Path(FFMPEG).with_name("ffprobe.exe"))

    if os.path.exists(ffprobe):
        cmd = [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
            )

            return float(result.stdout.strip())

        except Exception:
            pass

    # Fallback: use OpenCV
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    cap.release()

    if fps and fps > 0:
        return frames / fps

    return None


# ============================================================
# Motion analysis
# ============================================================

def analyze_segment(video_path, start_time, end_time, show_progress=True):
    """
    Analyze one video segment.

    Returns:
        list of (timestamp, motion_score)
    """

    duration = end_time - start_time

    if duration <= 0:
        return []

    cmd = [
        FFMPEG,
        "-hide_banner",
        "-loglevel",
        "error",

        "-ss",
        str(start_time),

        "-t",
        str(duration),

        "-hwaccel",
        "cuda",

        "-i",
        str(video_path),

        "-vf",
        f"fps={FPS},scale={WIDTH}:-1",

        "-f",
        "image2pipe",
        "-vcodec",
        "mjpeg",
        "-q:v",
        "5",

        "pipe:1",
    ]

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    scores = []

    previous = None
    frame_index = 0

    while True:

        # --------------------------------------------------------
        # Read JPEG frame from pipe
        # --------------------------------------------------------

        data = process.stdout.read(4)

        if not data:
            break

        # JPEG begins with FF D8.
        # Find the beginning of the JPEG.
        while data[:2] != b"\xff\xd8":

            more = process.stdout.read(1)

            if not more:
                break

            data += more

        if len(data) < 2:
            break

        jpeg = data

        # Read until JPEG end marker FF D9
        while True:

            chunk = process.stdout.read(4096)

            if not chunk:
                break

            jpeg += chunk

            pos = jpeg.find(b"\xff\xd9")

            if pos != -1:
                jpeg = jpeg[:pos + 2]
                break

        frame = cv2.imdecode(
            np.frombuffer(jpeg, dtype=np.uint8),
            cv2.IMREAD_GRAYSCALE,
        )

        if frame is None:
            continue

        timestamp = start_time + frame_index / FPS

        if previous is not None:

            diff = cv2.absdiff(
                frame,
                previous,
            )

            score = float(np.mean(diff))

            scores.append(
                (
                    timestamp,
                    score,
                )
            )

        previous = frame
        frame_index += 1

        # --------------------------------------------------------
        # Progress
        # --------------------------------------------------------

        if show_progress and frame_index % 100 == 0:

            elapsed = frame_index / FPS

            if duration > 0:
                pct = min(
                    100,
                    elapsed / duration * 100,
                )
            else:
                pct = 0

            print(
                f"\r    Processed {frame_index} frames "
                f"({elapsed:.1f}s, {pct:.0f}%)",
                end="",
                flush=True,
            )

    process.stdout.close()
    process.wait()

    if show_progress:
        print()

    return scores


# ============================================================
# Event detection
# ============================================================

def detect_events(
    scores,
    percentile=DEFAULT_PERCENTILE,
    min_duration=DEFAULT_MIN_DURATION,
    max_gap=DEFAULT_MAX_GAP,
    padding=DEFAULT_PADDING,
):
    """
    Convert motion scores into events.

    Returns:
        events, threshold

    Each event is:

        {
            "start": ...,
            "end": ...,
            "score": ...
        }
    """

    if not scores:
        return [], None

    values = np.array(
        [score for _, score in scores],
        dtype=float,
    )

    threshold = float(
        np.percentile(values, percentile)
    )

    active = [
        (timestamp, score)
        for timestamp, score in scores
        if score >= threshold
    ]

    if not active:
        return [], threshold

    # ------------------------------------------------------------
    # Group active samples
    # ------------------------------------------------------------

    groups = []

    group_start = active[0][0]
    group_end = active[0][0]
    group_scores = [active[0][1]]

    for timestamp, score in active[1:]:

        if timestamp - group_end <= max_gap:
            group_end = timestamp
            group_scores.append(score)

        else:
            groups.append(
                (
                    group_start,
                    group_end,
                    group_scores,
                )
            )

            group_start = timestamp
            group_end = timestamp
            group_scores = [score]

    groups.append(
        (
            group_start,
            group_end,
            group_scores,
        )
    )

    # ------------------------------------------------------------
    # Convert groups to events
    # ------------------------------------------------------------

    events = []

    for start, end, group_scores in groups:

        # Account for sampling interval.
        end += 1.0 / FPS

        duration = end - start

        if duration < min_duration:
            continue

        start = max(0, start - padding)
        end += padding

        events.append(
            {
                "start": start,
                "end": end,
                "score": max(group_scores),
            }
        )

    return events, threshold


# ============================================================
# Automatic MTB rating
# ============================================================

def motion_to_mtb(score, threshold):
    """
    Convert motion score to a rough MTB rating.

    This is deliberately only a motion-based estimate.
    Manual ratings remain authoritative.
    """

    if threshold is None or threshold <= 0:
        return 1

    ratio = score / threshold

    if ratio >= 1.40:
        return 5

    if ratio >= 1.20:
        return 4

    if ratio >= 1.05:
        return 3

    if ratio >= 0.90:
        return 2

    return 1


# ============================================================
# Merge automatic events with manual interval metadata
# ============================================================

def apply_manual_metadata(events, manual_intervals):
    """
    Associate automatically detected events with the manual
    interval that contains them.

    If an automatic event overlaps a manual interval, preserve
    the manual MTB/FILM/DESCRIPTION values.

    Returns modified events.
    """

    for event in events:

        best_match = None
        best_overlap = 0

        for (
            manual_start,
            manual_end,
            mtb,
            film,
            description,
        ) in manual_intervals:

            overlap_start = max(
                event["start"],
                manual_start,
            )

            overlap_end = min(
                event["end"],
                manual_end,
            )

            overlap = max(
                0,
                overlap_end - overlap_start,
            )

            if overlap > best_overlap:
                best_overlap = overlap

                best_match = (
                    manual_start,
                    manual_end,
                    mtb,
                    film,
                    description,
                )

        if best_match is not None:

            (
                manual_start,
                manual_end,
                mtb,
                film,
                description,
            ) = best_match

            event["mtb"] = mtb
            event["film"] = film
            event["description"] = description

        else:

            event["mtb"] = motion_to_mtb(
                event["score"],
                event.get("threshold"),
            )

            event["film"] = 1
            event["description"] = "automatic motion"

    return events


# ============================================================
# Preserve manual intervals that generated no motion event
# ============================================================

def add_unmatched_manual_intervals(
    events,
    manual_intervals,
):
    """
    Make sure every manually selected interval survives.

    This is important for things such as:

        MTB=1 FILM=5 landscape
        MTB=1 FILM=5 wildlife
        MTB=2 FILM=5 cinematic section

    even if motion detection finds little movement.
    """

    result = list(events)

    for (
        manual_start,
        manual_end,
        mtb,
        film,
        description,
    ) in manual_intervals:

        matched = False

        for event in events:

            overlap_start = max(
                manual_start,
                event["start"],
            )

            overlap_end = min(
                manual_end,
                event["end"],
            )

            overlap = max(
                0,
                overlap_end - overlap_start,
            )

            manual_duration = manual_end - manual_start

            if manual_duration > 0:
                overlap_ratio = (
                    overlap / manual_duration
                )
            else:
                overlap_ratio = 0

            # Consider it represented if there is substantial
            # overlap.
            if (
                overlap_ratio >= 0.50
                or overlap >= 3.0
            ):
                matched = True
                break

        if not matched:

            result.append(
                {
                    "start": manual_start,
                    "end": manual_end,
                    "score": 0.0,
                    "mtb": mtb,
                    "film": film,
                    "description": description,
                    "manual_only": True,
                }
            )

    return result


# ============================================================
# Analyze one video
# ============================================================

def analyze_video(
    video_path,
    args,
    manual_intervals=None,
):
    """
    Analyze an entire video.
    """

    duration = get_video_duration(video_path)

    if duration is None:
        print(
            f"ERROR: could not determine duration of "
            f"{video_path}"
        )
        return []

    print(
        f"  Duration: {duration:.2f}s"
    )

    scores = analyze_segment(
        video_path,
        0,
        duration,
    )

    events, threshold = detect_events(
        scores,
        percentile=args.percentile,
        min_duration=args.min_duration,
        max_gap=args.max_gap,
        padding=args.padding,
    )

    for event in events:
        event["threshold"] = threshold

    if manual_intervals:
        events = apply_manual_metadata(
            events,
            manual_intervals,
        )

        events = add_unmatched_manual_intervals(
            events,
            manual_intervals,
        )

    return events


# ============================================================
# Analyze selected intervals
# ============================================================

def analyze_selected_intervals(
    video_path,
    manual_intervals,
    args,
):
    """
    Analyze only the manually specified intervals.

    Each interval is treated as a region of interest.

    Manual intervals are ALWAYS preserved, even if no motion
    event is detected inside them.
    """

    all_events = []

    print(
        f"  Analyzing {len(manual_intervals)} selected intervals"
    )

    for index, (
        start,
        end,
        mtb,
        film,
        description,
    ) in enumerate(manual_intervals, 1):

        print(
            f"\n    [{index}/{len(manual_intervals)}] "
            f"{seconds_to_timecode(start)} - "
            f"{seconds_to_timecode(end)} "
            f"MTB={mtb} FILM={film}"
        )

        if description:
            print(
                f"      {description}"
            )

        scores = analyze_segment(
            video_path,
            start,
            end,
        )

        events, threshold = detect_events(
            scores,
            percentile=args.percentile,
            min_duration=args.min_duration,
            max_gap=args.max_gap,
            padding=args.padding,
        )

        for event in events:

            event["threshold"] = threshold

            # Make sure the automatic event cannot extend outside
            # the manually selected region.
            event["start"] = max(
                event["start"],
                start,
            )

            event["end"] = min(
                event["end"],
                end,
            )

        # Apply the manual rating/description.
        events = apply_manual_metadata(
            events,
            [(
                start,
                end,
                mtb,
                film,
                description,
            )],
        )

        # --------------------------------------------------------
        # IMPORTANT:
        # If motion detection produced no event, preserve the
        # complete manual interval.
        # --------------------------------------------------------

        events = add_unmatched_manual_intervals(
            events,
            [(
                start,
                end,
                mtb,
                film,
                description,
            )],
        )

        all_events.extend(events)

    # ------------------------------------------------------------
    # Sort chronologically
    # ------------------------------------------------------------

    all_events.sort(
        key=lambda e: e["start"]
    )

    return all_events


# ============================================================
# Output
# ============================================================

def write_annotations(
    output_path,
    results,
):
    """
    Write annotations_auto.txt.

    Format:

    filename.mp4
    START END MTB FILM DESCRIPTION
    ...
    """

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "START     END       MTB  FILM  DESCRIPTION\n"
        )

        for video_name, events in results.items():

            f.write(
                f"\n{video_name}\n"
            )

            for event in events:

                start = seconds_to_timecode(
                    event["start"]
                )

                end = seconds_to_timecode(
                    event["end"]
                )

                mtb = event.get("mtb", 1)
                film = event.get("film", 1)

                description = (
                    event.get("description", "")
                )

                if event.get("manual_only"):
                    if description:
                        description += " [manual]"
                    else:
                        description = "[manual]"

                f.write(
                    f"{start:<10}"
                    f"{end:<10}"
                    f"{mtb:<5}"
                    f"{film:<6}"
                    f"{description}\n"
                )


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Analyze MTB video motion and generate "
            "automatic annotations."
        )
    )

    parser.add_argument(
        "input",
        help="Video file or directory containing videos",
    )

    parser.add_argument(
        "--intervals",
        help=(
            "intervals.txt file. When supplied, ONLY files "
            "listed in this file are analyzed."
        ),
    )

    parser.add_argument(
        "--percentile",
        type=float,
        default=DEFAULT_PERCENTILE,
        help=(
            f"Motion percentile threshold "
            f"(default: {DEFAULT_PERCENTILE})"
        ),
    )

    parser.add_argument(
        "--min-duration",
        type=float,
        default=DEFAULT_MIN_DURATION,
        help=(
            f"Minimum automatic event duration in seconds "
            f"(default: {DEFAULT_MIN_DURATION})"
        ),
    )

    parser.add_argument(
        "--max-gap",
        type=float,
        default=DEFAULT_MAX_GAP,
        help=(
            f"Merge motion events separated by at most "
            f"this many seconds "
            f"(default: {DEFAULT_MAX_GAP})"
        ),
    )

    parser.add_argument(
        "--padding",
        type=float,
        default=DEFAULT_PADDING,
        help=(
            f"Padding around automatic events in seconds "
            f"(default: {DEFAULT_PADDING})"
        ),
    )

    parser.add_argument(
        "--output",
        default="annotations_auto.txt",
        help=(
            "Output annotation file "
            "(default: annotations_auto.txt)"
        ),
    )

    args = parser.parse_args()

    # ------------------------------------------------------------
    # Find videos
    # ------------------------------------------------------------

    try:
        video_files = find_videos(args.input)

    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    if not video_files:
        print(
            "No video files found."
        )
        sys.exit(1)

    print(
        f"Found {len(video_files)} video file(s)."
    )

    # ------------------------------------------------------------
    # Parse intervals file
    # ------------------------------------------------------------

    interval_data = None

    if args.intervals:

        print(
            f"\nReading intervals file: "
            f"{args.intervals}"
        )

        try:
            interval_data = parse_intervals_file(
                args.intervals
            )

        except Exception as e:
            print(
                f"ERROR reading intervals file: {e}"
            )
            sys.exit(1)

        print(
            f"Files mentioned in intervals file: "
            f"{len(interval_data)}"
        )

    # ------------------------------------------------------------
    # Analyze
    # ------------------------------------------------------------

    results = {}

    for video_path in video_files:

        filename = video_path.name

        print(
            "\n"
            + "=" * 70
        )

        print(
            f"{filename}"
        )

        # ========================================================
        # intervals.txt supplied
        # ========================================================

        if interval_data is not None:

            # ----------------------------------------------------
            # Not mentioned -> IGNORE
            # ----------------------------------------------------

            if filename not in interval_data:

                print(
                    "  SKIPPED: not listed in intervals file."
                )

                continue

            info = interval_data[filename]

            # ----------------------------------------------------
            # Explicitly disabled
            # ----------------------------------------------------

            if info["skip"]:

                print(
                    "  SKIPPED: explicitly disabled with '-'."
                )

                continue

            manual_intervals = info["intervals"]

            # ----------------------------------------------------
            # Filename with no intervals -> whole file
            # ----------------------------------------------------

            if not manual_intervals:

                print(
                    "  No intervals specified -> "
                    "analyzing entire file."
                )

                events = analyze_video(
                    video_path,
                    args,
                )

            # ----------------------------------------------------
            # Filename with intervals -> only those intervals
            # ----------------------------------------------------

            else:

                print(
                    f"  Using {len(manual_intervals)} "
                    f"selected interval(s)."
                )

                events = analyze_selected_intervals(
                    video_path,
                    manual_intervals,
                    args,
                )

        # ========================================================
        # No intervals.txt -> analyze everything
        # ========================================================

        else:

            print(
                "  No intervals file -> "
                "analyzing entire file."
            )

            events = analyze_video(
                video_path,
                args,
            )

        results[filename] = events

        # --------------------------------------------------------
        # Summary
        # --------------------------------------------------------

        print(
            f"\n  Detected/preserved events: "
            f"{len(events)}"
        )

        for i, event in enumerate(events, 1):

            print(
                f"    {i:02d} "
                f"{seconds_to_timecode(event['start'])} - "
                f"{seconds_to_timecode(event['end'])} "
                f"MTB={event.get('mtb', 1)} "
                f"FILM={event.get('film', 1)} "
                f"score={event.get('score', 0):.2f}"
            )

    # ------------------------------------------------------------
    # Write output
    # ------------------------------------------------------------

    if results:

        write_annotations(
            args.output,
            results,
        )

        print(
            "\n"
            + "=" * 70
        )

        print(
            f"Output written to: {args.output}"
        )

    else:

        print(
            "\nNo videos were analyzed."
        )


if __name__ == "__main__":
    main()