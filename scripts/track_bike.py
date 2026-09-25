r"""
Track a bike-mounted reference object using sparse optical flow.

The user selects a region of interest (ROI) in the first frame.
OpenCV Lucas-Kanade optical flow (calcOpticalFlowPyrLK) tracks
feature points inside the ROI through the video.

The script produces:
    1. An annotated MP4 showing the tracking.
    2. A CSV containing the tracked position and movement.

Examples:

    python .\scripts\track_bike.py `
        "D:\Prenestini\Mentorella\DJI_20250710113127_0012_D.MP4" `
        --start 00:05:00 `
        --duration 00:00:40

    python .\scripts\track_bike.py `
        "D:\Prenestini\Mentorella\DJI_20250710113127_0012_D.MP4" `
        --start 5:00 `
        --duration 40
"""

import argparse
import csv
import subprocess
from pathlib import Path

import cv2
import numpy as np


# ----------------------------------------------------------------------
# Defaults
# ----------------------------------------------------------------------

DEFAULT_MAX_CORNERS = 100
DEFAULT_QUALITY_LEVEL = 0.01
DEFAULT_MIN_DISTANCE = 7
DEFAULT_BLOCK_SIZE = 7

DEFAULT_WIN_SIZE = 21
DEFAULT_MAX_LEVEL = 3

DEFAULT_TRAJECTORY_LENGTH = 100

DEFAULT_OUTPUT = "track_bike.mp4"


# ----------------------------------------------------------------------
# Time parsing
# ----------------------------------------------------------------------

def parse_time(value):
    """
    Parse a time value.

    Supported:
        40          -> 40 seconds
        1:30        -> 1 minute 30 seconds
        5:00        -> 5 minutes
        1:02:35     -> 1 hour 2 minutes 35 seconds
    """

    value = str(value).strip()

    if not value:
        raise ValueError("Empty time value")

    parts = value.split(":")

    try:
        if len(parts) == 1:
            return float(parts[0])

        if len(parts) == 2:
            minutes = float(parts[0])
            seconds = float(parts[1])

            if seconds < 0 or seconds >= 60:
                raise ValueError(
                    f"Invalid seconds in time value: {value}"
                )

            return minutes * 60 + seconds

        if len(parts) == 3:
            hours = float(parts[0])
            minutes = float(parts[1])
            seconds = float(parts[2])

            if minutes < 0 or minutes >= 60:
                raise ValueError(
                    f"Invalid minutes in time value: {value}"
                )

            if seconds < 0 or seconds >= 60:
                raise ValueError(
                    f"Invalid seconds in time value: {value}"
                )

            return hours * 3600 + minutes * 60 + seconds

    except ValueError:
        raise ValueError(f"Invalid time value: {value}")

    raise ValueError(f"Invalid time format: {value}")


def format_time(seconds):
    """Format seconds as HH:MM:SS.s."""

    seconds = max(0.0, float(seconds))

    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60

    return f"{hours:02d}:{minutes:02d}:{secs:05.2f}"


# ----------------------------------------------------------------------
# Video information
# ----------------------------------------------------------------------

def get_video_info(video_path):
    """Return width, height, FPS and duration using ffprobe."""

    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate,duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
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

# ----------------------------------------------------------------------
# Feature detection
# ----------------------------------------------------------------------

def detect_features(gray, roi):
    """
    Detect Shi-Tomasi corners inside the ROI.
    """

    x, y, w, h = roi

    roi_gray = gray[y:y + h, x:x + w]

    if roi_gray.size == 0:
        return None

    corners = cv2.goodFeaturesToTrack(
        roi_gray,
        maxCorners=DEFAULT_MAX_CORNERS,
        qualityLevel=DEFAULT_QUALITY_LEVEL,
        minDistance=DEFAULT_MIN_DISTANCE,
        blockSize=DEFAULT_BLOCK_SIZE,
    )

    if corners is None:
        return None

    # Convert ROI coordinates to full-frame coordinates.
    corners[:, 0, 0] += x
    corners[:, 0, 1] += y

    return corners.astype(np.float32)


# ----------------------------------------------------------------------
# Tracking
# ----------------------------------------------------------------------

def track_features(
    previous_gray,
    current_gray,
    previous_points,
):
    """
    Track feature points using Lucas-Kanade optical flow.
    """

    if previous_points is None or len(previous_points) == 0:
        return None, None

    next_points, status, errors = cv2.calcOpticalFlowPyrLK(
        previous_gray,
        current_gray,
        previous_points,
        None,
        winSize=(DEFAULT_WIN_SIZE, DEFAULT_WIN_SIZE),
        maxLevel=DEFAULT_MAX_LEVEL,
        criteria=(
            cv2.TERM_CRITERIA_EPS |
            cv2.TERM_CRITERIA_COUNT,
            30,
            0.01,
        ),
    )

    if next_points is None or status is None:
        return None, None

    status = status.reshape(-1)

    good_previous = previous_points[status == 1]
    good_current = next_points[status == 1]

    if len(good_current) == 0:
        return None, None

    # Remove points with very large tracking errors.
    if errors is not None:
        errors = errors.reshape(-1)
        good_errors = errors[status == 1]

        valid = good_errors < 50.0

        good_previous = good_previous[valid]
        good_current = good_current[valid]

    if len(good_current) == 0:
        return None, None

    return (
        good_previous.astype(np.float32),
        good_current.astype(np.float32),
    )


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description="Track a bike-mounted reference object using sparse optical flow."
    )

    parser.add_argument(
        "input",
        help="Input video",
    )

    parser.add_argument(
        "-o",
        "--output",
        default=DEFAULT_OUTPUT,
        help="Output video path",
    )

    parser.add_argument(
        "--start",
        default="0",
        help="Start time: seconds, MM:SS or HH:MM:SS",
    )

    parser.add_argument(
        "--duration",
        default=None,
        help="Duration: seconds, MM:SS or HH:MM:SS",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    try:
        start_time = parse_time(args.start)

        if args.duration is not None:
            duration = parse_time(args.duration)
        else:
            duration = None

    except ValueError as exc:
        parser.error(str(exc))

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input video not found: {input_path}"
        )

    # ------------------------------------------------------------------
    # Video information
    # ------------------------------------------------------------------

    print("=" * 70)
    print("Bike Reference Tracking")
    print("=" * 70)
    print(f"Input:       {input_path}")
    print(f"Output:      {output_path}")
    print(f"Start:       {format_time(start_time)}")

    if duration is not None:
        print(f"Duration:    {format_time(duration)}")
    else:
        print("Duration:    until end")

    width, height, fps, video_duration = get_video_info(input_path)

    end_time = video_duration

    if duration is not None:
        end_time = min(
            start_time + duration,
            video_duration,
        )

    actual_duration = max(0.0, end_time - start_time)

    if start_time >= video_duration:
        raise ValueError(
            f"Start time {start_time:.2f}s is beyond "
            f"video duration {video_duration:.2f}s"
        )

    print()
    print(f"Source:      {width}x{height}")
    print(f"Source FPS:  {fps:.3f}")
    print(f"Video length:{format_time(video_duration)}")
    print(f"End:         {format_time(end_time)}")
    print()

    # ------------------------------------------------------------------
    # Open video
    # ------------------------------------------------------------------

    cap = cv2.VideoCapture(str(input_path))

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: {input_path}"
        )

    # Seek to requested start.
    cap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000.0)

    ret, frame = cap.read()

    if not ret or frame is None:
        cap.release()
        raise RuntimeError(
            "Could not read first frame at requested start time."
        )

    # Actual timestamp of first decoded frame.
    first_frame_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

    # ------------------------------------------------------------------
    # Select ROI
    # ------------------------------------------------------------------

    print("Select the object you want to track.")
    print("For example: the phone attached to the bike.")
    print("Press ENTER or SPACE when finished.")
    print("Press C to cancel.")
    print()

    display_frame = frame.copy()

    roi = cv2.selectROI(
        "Select bike reference object",
        display_frame,
        showCrosshair=True,
        fromCenter=False,
    )

    cv2.destroyWindow("Select bike reference object")

    x, y, w, h = [int(v) for v in roi]

    if w <= 0 or h <= 0:
        cap.release()
        raise RuntimeError("No tracking region selected.")

    # ------------------------------------------------------------------
    # Initial feature detection
    # ------------------------------------------------------------------

    previous_gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY,
    )

    previous_points = detect_features(
        previous_gray,
        (x, y, w, h),
    )

    if previous_points is None or len(previous_points) < 3:
        cap.release()
        raise RuntimeError(
            "Could not find enough feature points in the selected region."
        )

    initial_center = np.array(
        [
            x + w / 2.0,
            y + h / 2.0,
        ],
        dtype=np.float32,
    )

    current_center = initial_center.copy()

    # Trajectory of object centre.
    trajectory = [
        (
            float(current_center[0]),
            float(current_center[1]),
        )
    ]

    # ------------------------------------------------------------------
    # Output video
    # ------------------------------------------------------------------

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (width, height),
    )

    if not writer.isOpened():
        cap.release()
        raise RuntimeError(
            f"Could not open output video: {output_path}"
        )

    # ------------------------------------------------------------------
    # CSV
    # ------------------------------------------------------------------

    csv_path = output_path.with_suffix(".csv")

    csv_file = open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8",
    )

    csv_writer = csv.writer(csv_file)

    csv_writer.writerow(
        [
            "time",
            "x",
            "y",
            "dx",
            "dy",
            "speed",
            "tracked_points",
        ]
    )

    csv_writer.writerow(
        [
            first_frame_time,
            current_center[0],
            current_center[1],
            0.0,
            0.0,
            0.0,
            len(previous_points),
        ]
    )

    # ------------------------------------------------------------------
    # Processing
    # ------------------------------------------------------------------

    frame_number = 0
    trajectory_length = DEFAULT_TRAJECTORY_LENGTH

    print()
    print("Tracking started.")
    print("Press Q in the video window to stop.")
    print()

    while True:

        ret, frame = cap.read()

        if not ret or frame is None:
            break

        current_time = cap.get(
            cv2.CAP_PROP_POS_MSEC
        ) / 1000.0

        if current_time > end_time:
            break

        current_gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY,
        )

        previous_tracked, current_tracked = track_features(
            previous_gray,
            current_gray,
            previous_points,
        )

        tracked_count = 0

        dx = 0.0
        dy = 0.0

        # --------------------------------------------------------------
        # Successful tracking
        # --------------------------------------------------------------

        if (
            previous_tracked is not None
            and current_tracked is not None
            and len(current_tracked) >= 3
        ):

            tracked_count = len(current_tracked)

            displacements = (
                current_tracked -
                previous_tracked
            )

            # Median displacement is robust against individual
            # incorrectly tracked points.
            median_displacement = np.median(
                displacements,
                axis=0,
            )

            dx = float(np.asarray(median_displacement).reshape(-1)[0])
            dy = float(np.asarray(median_displacement).reshape(-1)[1])
            current_center += np.array(
                [dx, dy],
                dtype=np.float32,
            )

            # Keep tracking the successfully matched points.
            previous_points = current_tracked.reshape(
                -1, 1, 2
            ).astype(np.float32)

        # --------------------------------------------------------------
        # Tracking failure / too few points
        # --------------------------------------------------------------

        else:

            # Try to recover features inside the current estimated ROI.
            estimated_x = int(
                current_center[0] - w / 2
            )

            estimated_y = int(
                current_center[1] - h / 2
            )

            estimated_x = max(
                0,
                min(estimated_x, width - w),
            )

            estimated_y = max(
                0,
                min(estimated_y, height - h),
            )

            recovered_points = detect_features(
                current_gray,
                (
                    estimated_x,
                    estimated_y,
                    w,
                    h,
                ),
            )

            if (
                recovered_points is not None
                and len(recovered_points) >= 3
            ):
                previous_points = recovered_points

            else:
                previous_points = None

        # --------------------------------------------------------------
        # Speed
        # --------------------------------------------------------------

        speed = float(
            np.sqrt(
                dx * dx +
                dy * dy
            )
        )

        # --------------------------------------------------------------
        # Trajectory
        # --------------------------------------------------------------

        trajectory.append(
            (
                float(current_center[0]),
                float(current_center[1]),
            )
        )

        if len(trajectory) > trajectory_length:
            trajectory.pop(0)

        # --------------------------------------------------------------
        # Draw tracking information
        # --------------------------------------------------------------

        output_frame = frame.copy()

        # Draw tracked feature points.
        if (
            previous_points is not None
            and len(previous_points) > 0
        ):
            for point in previous_points:
                px, py = point.ravel()

                px = int(round(px))
                py = int(round(py))

                if (
                    0 <= px < width
                    and 0 <= py < height
                ):
                    cv2.circle(
                        output_frame,
                        (px, py),
                        3,
                        (0, 255, 0),
                        -1,
                    )

        # Estimated object bounding box.
        box_x = int(
            current_center[0] - w / 2
        )
        box_y = int(
            current_center[1] - h / 2
        )

        cv2.rectangle(
            output_frame,
            (box_x, box_y),
            (
                box_x + w,
                box_y + h,
            ),
            (0, 255, 255),
            2,
        )

        # Object centre.
        center_x = int(round(current_center[0]))
        center_y = int(round(current_center[1]))

        cv2.circle(
            output_frame,
            (center_x, center_y),
            6,
            (0, 0, 255),
            -1,
        )

        # Draw trajectory.
        if len(trajectory) >= 2:

            for i in range(1, len(trajectory)):

                p1 = (
                    int(round(trajectory[i - 1][0])),
                    int(round(trajectory[i - 1][1])),
                )

                p2 = (
                    int(round(trajectory[i][0])),
                    int(round(trajectory[i][1])),
                )

                cv2.line(
                    output_frame,
                    p1,
                    p2,
                    (255, 255, 0),
                    2,
                )

        # Information text.
        cv2.putText(
            output_frame,
            f"t = {format_time(current_time)}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            output_frame,
            f"dx = {dx:+.2f}  dy = {dy:+.2f}",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            output_frame,
            f"speed = {speed:.2f} px/frame",
            (20, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            output_frame,
            f"tracked points = {tracked_count}",
            (20, 145),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        writer.write(output_frame)

        # Show preview.
        cv2.imshow(
            "Bike reference tracking",
            output_frame,
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            print()
            print("Stopped by user.")
            break

        # --------------------------------------------------------------
        # Prepare next iteration
        # --------------------------------------------------------------

        previous_gray = current_gray

        frame_number += 1

        # If tracking has degraded substantially, re-detect features
        # around the estimated object position.
        if (
            previous_points is None
            or len(previous_points) < 10
        ):

            estimated_x = int(
                current_center[0] - w / 2
            )

            estimated_y = int(
                current_center[1] - h / 2
            )

            estimated_x = max(
                0,
                min(estimated_x, width - w),
            )

            estimated_y = max(
                0,
                min(estimated_y, height - h),
            )

            recovered_points = detect_features(
                previous_gray,
                (
                    estimated_x,
                    estimated_y,
                    w,
                    h,
                ),
            )

            if recovered_points is not None:
                previous_points = recovered_points

        # Progress every ~1 second.
        if frame_number % max(1, int(fps)) == 0:

            progress = (
                (current_time - start_time)
                / actual_duration
                * 100
                if actual_duration > 0
                else 0
            )

            print(
                f"{progress:5.1f}%  "
                f"{format_time(current_time)} / "
                f"{format_time(end_time)}  "
                f"tracked points: {tracked_count:3d}"
            )

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    cap.release()
    writer.release()
    csv_file.close()

    cv2.destroyAllWindows()

    print()
    print("=" * 70)
    print("Done")
    print("=" * 70)
    print(f"Video: {output_path}")
    print(f"CSV:   {csv_path}")
    print()


if __name__ == "__main__":
    main()