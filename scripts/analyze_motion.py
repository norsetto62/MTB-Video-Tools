import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path


# ============================================================
# Configuration
# ============================================================

ANALYSIS_FPS = 2
OUTPUT_WIDTH = 480

FFMPEG = "ffmpeg"
FFPROBE = "ffprobe"


# ============================================================
# Utility functions
# ============================================================

def run_command(cmd):
    """Run a command and return stdout as text."""
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
        raise RuntimeError(f"Command failed with exit code {result.returncode}")

    return result.stdout.strip()


def get_video_info(video_path):
    """Return width, height, fps and duration using ffprobe."""

    cmd = [
        FFPROBE,
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate,duration:format=duration",
        "-of", "default=noprint_wrappers=1",
        str(video_path),
    ]

    output = run_command(cmd)

    info = {}

    for line in output.splitlines():
        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        info[key] = value

    try:
        width = int(info["width"])
        height = int(info["height"])
    except (KeyError, ValueError):
        raise RuntimeError(f"Could not determine video dimensions: {video_path}")

    duration = None

    # Stream duration is preferable, but some DJI files don't expose it
    # there, so fall back to container duration.
    if "duration" in info:
        try:
            duration = float(info["duration"])
        except ValueError:
            pass

    if duration is None:
        # Ask ffprobe specifically for format duration.
        cmd = [
            FFPROBE,
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ]

        duration_text = run_command(cmd)

        try:
            duration = float(duration_text)
        except ValueError:
            raise RuntimeError(f"Could not determine duration: {video_path}")

    fps = None

    if "r_frame_rate" in info:
        try:
            numerator, denominator = info["r_frame_rate"].split("/")
            fps = float(numerator) / float(denominator)
        except (ValueError, ZeroDivisionError):
            pass

    return width, height, fps, duration


def format_time(seconds):
    """Format seconds as HH:MM:SS."""

    seconds = max(0, int(seconds))

    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60

    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    return f"{minutes:02d}:{secs:02d}"


def format_eta(seconds):
    if seconds is None:
        return "--:--"

    return format_time(seconds)


# ============================================================
# intervals.txt handling
# ============================================================

def load_intervals_file(path):
    """
    Returns:
        includes: set of explicitly included filenames/paths
        excludes: set of explicitly excluded filenames/paths

    If the file is missing or contains no entries, both sets are empty,
    meaning: no filtering.
    """
    includes = set()
    excludes = set()

    if not path.exists():
        return includes, excludes

    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()

            if not line or line.startswith("#"):
                continue

            if line.startswith("-"):
                line = line[1:].strip()
                if line:
                    excludes.add(line.lower())
            else:
                includes.add(line.lower())

    return includes, excludes

def path_matches(entry, video_path, input_root):
    """
    Determine whether an intervals.txt entry refers to video_path.

    Matching supports:

    - exact filename
    - relative path
    - absolute path
    """

    entry_path = Path(entry)

    # Exact absolute path
    if entry_path.is_absolute():
        try:
            return entry_path.resolve() == video_path.resolve()
        except OSError:
            return entry_path == video_path

    # Match against filename
    if entry_path.name.lower() == video_path.name.lower():
        return True

    # Match relative path from input root
    try:
        relative = video_path.resolve().relative_to(input_root.resolve())
        if str(relative).lower() == str(entry_path).lower():
            return True
    except ValueError:
        pass

    # Also allow an entry such as ./foo/bar.mp4
    normalized_entry = str(entry_path).replace("\\", "/").lstrip("./")
    normalized_video = str(video_path).replace("\\", "/").lower()

    return normalized_video.endswith("/" + normalized_entry.lower())

def select_videos(input_path, intervals_path):
    """
    Find videos and apply intervals.txt rules.

    If intervals.txt doesn't exist:
        all MP4 files are returned.

    If intervals.txt exists but is empty:
        all MP4 files are returned.

    If intervals.txt contains positive entries:
        only positively listed files are returned.

    Negative entries override positive entries.

    For a directly specified single video:
        the video is processed unless it is explicitly excluded.
    """

    input_path = input_path.resolve()

    if input_path.is_file():
        all_videos = [input_path]
        input_root = input_path.parent
    else:
        # Recursive search is useful because DJI footage is often
        # organized in subdirectories.
        all_videos = sorted(
            p for p in input_path.rglob("*")
            if p.is_file() and p.suffix.lower() == ".mp4"
        )
        input_root = input_path

    includes, excludes = load_intervals_file(intervals_path)

    # --------------------------------------------------------
    # No intervals.txt
    # --------------------------------------------------------

    if includes is None:
        return all_videos

    # --------------------------------------------------------
    # Empty intervals.txt
    #
    # No positive OR negative entries means:
    # don't filter anything.
    # --------------------------------------------------------

    if not includes and not excludes:
        return all_videos

    selected = []

    for video in all_videos:

        # Explicit exclusion always wins.
        excluded = any(
            path_matches(entry, video, input_root)
            for entry in excludes
        )

        if excluded:
            continue

        # ----------------------------------------------------
        # Direct single-file input:
        #
        # If the user explicitly supplied one video, don't
        # require it to appear in intervals.txt.
        # Only an explicit "-" entry excludes it.
        # ----------------------------------------------------

        if input_path.is_file():
            selected.append(video)
            continue

        # ----------------------------------------------------
        # Directory input:
        #
        # If there are positive entries, only those files are
        # selected.
        #
        # If there are ONLY negative entries, everything else
        # is selected.
        # ----------------------------------------------------

        if includes:
            included = any(
                path_matches(entry, video, input_root)
                for entry in includes
            )

            if not included:
                continue

        selected.append(video)

    return selected

# ============================================================
# Motion analysis
# ============================================================

def analyze_video(video_path, output_csv):
    """
    Analyze one video.

    FFmpeg performs:

        CUDA hardware decode
        ->
        sample at 2 FPS
        ->
        scale to 480 px width
        ->
        grayscale
        ->
        raw frames to Python

    Motion score = mean absolute pixel difference between
    consecutive sampled frames.
    """

    print()
    print("=" * 70)
    print(f"Video: {video_path}")
    print("=" * 70)

    width, height, source_fps, duration = get_video_info(video_path)

    output_height = round(height * OUTPUT_WIDTH / width)

    print(f"Source:       {width}x{height}")
    print(f"Source FPS:   {source_fps:.3f}" if source_fps else "Source FPS:   unknown")
    print(f"Duration:     {format_time(duration)}")
    print(f"Analysis:     {ANALYSIS_FPS} FPS")
    print(f"Analysis size:{OUTPUT_WIDTH}x{output_height}")
    print(f"Output:       {output_csv}")

    frame_size = OUTPUT_WIDTH * output_height

    # FFmpeg command.
    #
    # -hwaccel cuda:
    #     use NVIDIA hardware acceleration for decoding.
    #
    # -vf fps=2,scale=480:-1,format=gray:
    #     sample only 2 frames/sec, resize, convert to grayscale.
    #
    # -f rawvideo:
    #     send raw grayscale frames through stdout.
    #
    # -pix_fmt gray:
    #     exactly one byte per pixel.
    #
    # -an:
    #     don't decode audio.
    #
    # -sn/-dn:
    #     ignore subtitles/data streams.
    #
    cmd = [
        FFMPEG,
        "-hide_banner",
        "-loglevel", "error",

        "-hwaccel", "cuda",

        "-i", str(video_path),

        "-an",
        "-sn",
        "-dn",

        "-vf",
        f"fps={ANALYSIS_FPS},scale={OUTPUT_WIDTH}:-1,format=gray",

        "-f", "rawvideo",
        "-pix_fmt", "gray",

        "pipe:1",
    ]

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=10 ** 8,
    )

    previous_frame = None

    rows = []

    frame_number = 0

    start_time = time.time()
    last_progress_time = start_time

    try:

        while True:

            raw_frame = process.stdout.read(frame_size)

            if len(raw_frame) < frame_size:
                break

            frame_number += 1

            # NumPy is imported here so the script can give a
            # useful error if the environment is incomplete.
            import numpy as np

            current_frame = np.frombuffer(
                raw_frame,
                dtype=np.uint8
            ).reshape(
                (output_height, OUTPUT_WIDTH)
            )

            timestamp = (frame_number - 1) / ANALYSIS_FPS

            if previous_frame is None:
                motion_score = 0.0
            else:
                # Mean absolute difference.
                #
                # np.abs(current - previous) would overflow uint8,
                # so convert before subtraction.
                difference = cv2_absdiff(
                    current_frame,
                    previous_frame
                )

                motion_score = float(difference.mean())

            rows.append(
                (
                    timestamp,
                    motion_score
                )
            )

            previous_frame = current_frame

            # Progress display approximately once per second.
            now = time.time()

            if now - last_progress_time >= 1.0:

                elapsed = now - start_time

                processed_seconds = timestamp

                if processed_seconds > 0:
                    speed = processed_seconds / elapsed

                    remaining = max(
                        0,
                        duration - processed_seconds
                    )

                    eta = remaining / speed if speed > 0 else None
                else:
                    speed = 0
                    eta = None

                percent = min(
                    100.0,
                    processed_seconds / duration * 100
                ) if duration > 0 else 0

                print(
                    f"\r"
                    f"{percent:6.2f}%  "
                    f"{format_time(processed_seconds)} / "
                    f"{format_time(duration)}  "
                    f"speed {speed:5.2f}x  "
                    f"ETA {format_eta(eta)}",
                    end="",
                    flush=True
                )

                last_progress_time = now

    finally:
        process.stdout.close()

    stderr = process.stderr.read().decode(
        "utf-8",
        errors="replace"
    )

    process.stderr.close()

    return_code = process.wait()

    print()

    if return_code != 0:
        print(stderr)
        raise RuntimeError(
            f"FFmpeg failed for {video_path} "
            f"(exit code {return_code})"
        )

    # Write CSV.
    output_csv.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with output_csv.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "timestamp",
            "motion_score",
        ])

        for timestamp, score in rows:
            writer.writerow([
                f"{timestamp:.3f}",
                f"{score:.6f}",
            ])

    elapsed = time.time() - start_time

    print(
        f"Completed {frame_number} analysis frames "
        f"in {elapsed:.1f}s "
        f"({duration / elapsed:.2f}x realtime)"
    )

    print(f"Saved: {output_csv}")


def cv2_absdiff(a, b):
    """
    Small local implementation of absolute difference.

    This avoids making OpenCV mandatory for this particular
    operation and keeps the calculation explicit.
    """

    import numpy as np

    return np.abs(
        a.astype(np.int16) -
        b.astype(np.int16)
    ).astype(np.uint8)


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Analyze video motion using FFmpeg + CUDA."
    )

    parser.add_argument(
        "input",
        type=Path,
        help="Input MP4 file or directory."
    )

    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help=(
            "Output CSV. For a single input video this is the CSV path. "
            "For a directory, omit this and one CSV is created per video."
        ),
    )

    parser.add_argument(
        "--intervals",
        type=Path,
        default=None,
        help=(
            "Path to intervals.txt. "
            "If omitted, intervals.txt is searched in the current directory "
            "and then next to the input directory."
        ),
    )

    args = parser.parse_args()

    input_path = args.input.resolve()

    if not input_path.exists():
        print(f"ERROR: input does not exist:\n{input_path}")
        sys.exit(1)

    # Locate intervals.txt.
    if args.intervals:
        intervals_path = args.intervals.resolve()
    else:
        candidates = [
            Path.cwd() / "intervals.txt",
            input_path / "intervals.txt"
            if input_path.is_dir()
            else input_path.parent / "intervals.txt",
        ]

        intervals_path = next(
            (p for p in candidates if p.exists()),
            candidates[0],
        )

    videos = select_videos(
        input_path,
        intervals_path
    )

    if intervals_path.exists():
        print(f"Using intervals file: {intervals_path}")

    else:
        print("No intervals.txt found -> processing all MP4 files.")

    if not videos:
        print("No videos selected.")
        sys.exit(0)

    print()
    print(f"Selected {len(videos)} video(s):")

    for video in videos:
        print(f"  {video}")

    # --------------------------------------------------------
    # Single video
    # --------------------------------------------------------

    if len(videos) == 1:

        video = videos[0]

        if args.output:
            output_csv = args.output.resolve()

        else:
            # Default:
            # data/motion_scores/<video-name>.csv
            project_root = Path(__file__).resolve().parent.parent

            output_dir = (
                project_root /
                "data" /
                "motion_scores"
            )

            output_csv = (
                output_dir /
                f"{video.stem}.csv"
            )

        analyze_video(
            video,
            output_csv
        )

        return

    # --------------------------------------------------------
    # Multiple videos
    # --------------------------------------------------------

    if args.output:
        print(
            "ERROR: --output can only be used when analyzing "
            "a single video."
        )
        sys.exit(1)

    project_root = Path(__file__).resolve().parent.parent

    output_dir = (
        project_root /
        "data" /
        "motion_scores"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    for index, video in enumerate(videos, start=1):

        print()
        print(
            f"VIDEO {index}/{len(videos)}"
        )

        output_csv = (
            output_dir /
            f"{video.stem}.csv"
        )

        analyze_video(
            video,
            output_csv
        )


if __name__ == "__main__":
    main()