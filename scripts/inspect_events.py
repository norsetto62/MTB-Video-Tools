import argparse
import csv
import math
import subprocess
import sys
from pathlib import Path


# ============================================================
# Configuration
# ============================================================

FFMPEG = "ffmpeg"

# Number of candidate events to report by default.
DEFAULT_MAX_EVENTS = 30

# Candidate threshold.
# Events are generated from motion scores above this percentile.
DEFAULT_PERCENTILE = 90.0

# Merge peaks that are closer than this many seconds.
DEFAULT_MERGE_GAP = 8.0

# Add context around each detected event.
DEFAULT_PADDING_BEFORE = 5.0
DEFAULT_PADDING_AFTER = 5.0

# Extract one frame approximately from the middle of each event.
DEFAULT_FRAME_WIDTH = 640


# ============================================================
# Utility functions
# ============================================================

def format_time(seconds):
    """Format seconds as MM:SS or HH:MM:SS."""

    seconds = max(0, int(seconds))

    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60

    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    return f"{minutes:02d}:{secs:02d}"


def run_command(cmd):
    """Run a command and raise an error if it fails."""

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    if result.returncode != 0:
        print("\nCommand failed:")
        print(" ".join(str(x) for x in cmd))
        print(result.stderr)
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}"
        )

    return result.stdout


# ============================================================
# CSV
# ============================================================

def load_motion_csv(csv_path):
    """
    Load motion_scores CSV.

    Expected format:

        timestamp,motion_score
        0.000,0.000000
        0.500,27.761929
        ...
    """

    timestamps = []
    scores = []

    with csv_path.open(
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        reader = csv.DictReader(f)

        required = {"timestamp", "motion_score"}

        if not required.issubset(reader.fieldnames or set()):
            raise RuntimeError(
                f"CSV must contain columns: "
                f"{', '.join(sorted(required))}"
            )

        for row in reader:

            try:
                timestamp = float(row["timestamp"])
                score = float(row["motion_score"])
            except (TypeError, ValueError):
                continue

            timestamps.append(timestamp)
            scores.append(score)

    if not timestamps:
        raise RuntimeError(
            f"No valid motion data found in {csv_path}"
        )

    return timestamps, scores


# ============================================================
# Statistics
# ============================================================

def percentile(values, p):
    """Calculate percentile using NumPy."""

    import numpy as np

    return float(np.percentile(values, p))


def find_peaks(timestamps, scores, threshold):
    """
    Find local maxima above threshold.

    A point is considered a peak when its score is >= both
    neighboring points and is above the threshold.
    """

    peaks = []

    if len(scores) < 3:
        return peaks

    for i in range(1, len(scores) - 1):

        if scores[i] < threshold:
            continue

        if scores[i] < scores[i - 1]:
            continue

        if scores[i] < scores[i + 1]:
            continue

        peaks.append({
            "index": i,
            "timestamp": timestamps[i],
            "score": scores[i],
        })

    return peaks


# ============================================================
# Event grouping
# ============================================================

def build_events(
    timestamps,
    scores,
    peaks,
    merge_gap,
    padding_before,
    padding_after,
):
    """
    Convert individual peaks into larger candidate events.

    Nearby peaks are merged into the same event.
    """

    if not peaks:
        return []

    events = []

    current = {
        "start": peaks[0]["timestamp"],
        "end": peaks[0]["timestamp"],
        "peak_time": peaks[0]["timestamp"],
        "peak_score": peaks[0]["score"],
    }

    for peak in peaks[1:]:

        peak_time = peak["timestamp"]

        if peak_time - current["end"] <= merge_gap:

            current["end"] = peak_time

            if peak["score"] > current["peak_score"]:
                current["peak_time"] = peak_time
                current["peak_score"] = peak["score"]

        else:

            events.append(current)

            current = {
                "start": peak_time,
                "end": peak_time,
                "peak_time": peak_time,
                "peak_score": peak["score"],
            }

    events.append(current)

    # Expand each event with contextual padding.
    duration = timestamps[-1]

    for event in events:

        event["start"] = max(
            0,
            event["start"] - padding_before
        )

        event["end"] = min(
            duration,
            event["end"] + padding_after
        )

        event["duration"] = (
            event["end"] -
            event["start"]
        )

    return events


# ============================================================
# Event ranking
# ============================================================

def rank_events(events):
    """
    Sort events by peak motion score, highest first.
    """

    return sorted(
        events,
        key=lambda event: event["peak_score"],
        reverse=True,
    )


# ============================================================
# Frame extraction
# ============================================================

def extract_frame(
    video_path,
    timestamp,
    output_path,
    width,
):
    """
    Extract one JPEG frame from the original video.
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    cmd = [
        FFMPEG,
        "-hide_banner",
        "-loglevel", "error",

        "-ss", f"{timestamp:.3f}",

        "-hwaccel", "cuda",

        "-i", str(video_path),

        "-frames:v", "1",

        "-vf", f"scale={width}:-1",

        "-q:v", "2",

        "-y",
        str(output_path),
    ]

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    if result.returncode != 0:

        print(
            f"WARNING: could not extract frame at "
            f"{format_time(timestamp)}"
        )

        print(
            result.stderr
        )

        return False

    return True


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Inspect candidate high-motion events from "
            "a motion_scores CSV."
        )
    )

    parser.add_argument(
        "csv",
        type=Path,
        help="Motion score CSV."
    )

    parser.add_argument(
        "--video",
        type=Path,
        required=True,
        help="Original video corresponding to the CSV."
    )

    parser.add_argument(
        "--percentile",
        type=float,
        default=DEFAULT_PERCENTILE,
        help=(
            f"Motion percentile threshold "
            f"(default: {DEFAULT_PERCENTILE})."
        ),
    )

    parser.add_argument(
        "--merge-gap",
        type=float,
        default=DEFAULT_MERGE_GAP,
        help=(
            f"Merge peaks closer than this many seconds "
            f"(default: {DEFAULT_MERGE_GAP})."
        ),
    )

    parser.add_argument(
        "--before",
        type=float,
        default=DEFAULT_PADDING_BEFORE,
        help=(
            f"Seconds before event "
            f"(default: {DEFAULT_PADDING_BEFORE})."
        ),
    )

    parser.add_argument(
        "--after",
        type=float,
        default=DEFAULT_PADDING_AFTER,
        help=(
            f"Seconds after event "
            f"(default: {DEFAULT_PADDING_AFTER})."
        ),
    )

    parser.add_argument(
        "--max-events",
        type=int,
        default=DEFAULT_MAX_EVENTS,
        help=(
            f"Maximum number of events "
            f"(default: {DEFAULT_MAX_EVENTS})."
        ),
    )

    parser.add_argument(
        "--width",
        type=int,
        default=DEFAULT_FRAME_WIDTH,
        help=(
            f"Extracted frame width "
            f"(default: {DEFAULT_FRAME_WIDTH})."
        ),
    )

    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help=(
            "Output directory for extracted frames. "
            "Default: data/event_inspection/<video-name>"
        ),
    )

    args = parser.parse_args()

    csv_path = args.csv.resolve()
    video_path = args.video.resolve()

    if not csv_path.exists():
        print(
            f"ERROR: CSV does not exist:\n{csv_path}"
        )
        sys.exit(1)

    if not video_path.exists():
        print(
            f"ERROR: video does not exist:\n{video_path}"
        )
        sys.exit(1)

    if args.percentile < 0 or args.percentile > 100:
        print(
            "ERROR: percentile must be between 0 and 100."
        )
        sys.exit(1)

    print()
    print("=" * 70)
    print("Motion Event Inspector")
    print("=" * 70)

    print(f"CSV:       {csv_path}")
    print(f"Video:     {video_path}")

    timestamps, scores = load_motion_csv(
        csv_path
    )

    print(
        f"Samples:   {len(scores)}"
    )

    print(
        f"Duration:  {format_time(timestamps[-1])}"
    )

    print(
        f"Min score: {min(scores):.2f}"
    )

    print(
        f"Max score: {max(scores):.2f}"
    )

    print(
        f"Mean:      "
        f"{sum(scores) / len(scores):.2f}"
    )

    threshold = percentile(
        scores,
        args.percentile
    )

    print()
    print(
        f"Threshold: {threshold:.2f} "
        f"(percentile {args.percentile:.1f})"
    )

    peaks = find_peaks(
        timestamps,
        scores,
        threshold
    )

    print(
        f"Peaks:     {len(peaks)}"
    )

    events = build_events(
        timestamps,
        scores,
        peaks,
        args.merge_gap,
        args.before,
        args.after,
    )

    events = rank_events(events)

    if args.max_events > 0:
        events = events[:args.max_events]

    print(
        f"Events:    {len(events)}"
    )

    if not events:
        print()
        print("No candidate events found.")
        return

    # --------------------------------------------------------
    # Output directory
    # --------------------------------------------------------

    if args.output:
        output_dir = args.output.resolve()

    else:

        project_root = (
            Path(__file__).resolve().parent.parent
        )

        output_dir = (
            project_root /
            "data" /
            "event_inspection" /
            video_path.stem
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    frames_dir = output_dir / "frames"

    frames_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Print events
    # --------------------------------------------------------

    print()
    print(
        "-" * 70
    )

    print(
        f"{'#':>3}  "
        f"{'START':>8}  "
        f"{'END':>8}  "
        f"{'PEAK':>8}  "
        f"{'SCORE':>8}"
    )

    print(
        "-" * 70
    )

    for index, event in enumerate(
        events,
        start=1
    ):

        print(
            f"{index:3d}  "
            f"{format_time(event['start']):>8}  "
            f"{format_time(event['end']):>8}  "
            f"{format_time(event['peak_time']):>8}  "
            f"{event['peak_score']:8.2f}"
        )

    print(
        "-" * 70
    )

    # --------------------------------------------------------
    # Extract representative frames
    # --------------------------------------------------------

    print()
    print("Extracting representative frames...")

    extracted = []

    for index, event in enumerate(
        events,
        start=1
    ):

        timestamp = event["peak_time"]

        frame_path = (
            frames_dir /
            f"event_{index:03d}_"
            f"{int(timestamp):06d}s.jpg"
        )

        print(
            f"  {index:3d}/{len(events)}  "
            f"{format_time(timestamp)}",
            end="",
            flush=True
        )

        success = extract_frame(
            video_path,
            timestamp,
            frame_path,
            args.width,
        )

        if success:

            print("  OK")

            extracted.append(
                (
                    index,
                    event,
                    frame_path
                )
            )

        else:

            print("  FAILED")

    # --------------------------------------------------------
    # Save event list
    # --------------------------------------------------------

    events_csv = output_dir / "events.csv"

    with events_csv.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "event",
            "start",
            "end",
            "duration",
            "peak_time",
            "peak_score",
            "frame",
        ])

        for index, event, frame_path in extracted:

            writer.writerow([
                index,
                f"{event['start']:.3f}",
                f"{event['end']:.3f}",
                f"{event['duration']:.3f}",
                f"{event['peak_time']:.3f}",
                f"{event['peak_score']:.6f}",
                str(frame_path),
            ])

    print()
    print(
        f"Saved event list: {events_csv}"
    )

    print(
        f"Saved frames:     {frames_dir}"
    )


if __name__ == "__main__":
    main()