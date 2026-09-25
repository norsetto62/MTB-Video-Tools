r"""
visualize_flow.py

Visualize dense optical flow over the original video.

Architecture:
- Decode the source video ONCE.
- Output remains at the source video's native FPS and resolution.
- Optical flow is calculated only at --flow-fps.
- Optical flow is calculated only over the upper --flow-height percent
  of the image.
- The lower part of the image is left completely untouched.
- --start and --end allow processing a selected section.

Example:

    python .\scripts\visualize_flow.py `
        "D:\Prenestini\Mentorella\DJI_20250710113127_0012_D.MP4" `
        -o ".\output\flow_test_0_20_upper70.mp4" `
        --start 0 `
        --end 20 `
        --flow-fps 10 `
        --flow-height 70

For the entire video, omit --start and --end.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np


# ----------------------------------------------------------------------
# Defaults
# ----------------------------------------------------------------------

DEFAULT_FLOW_FPS = 10.0
DEFAULT_FLOW_WIDTH = 480
DEFAULT_FLOW_HEIGHT = 70.0
DEFAULT_FLOW_ALPHA = 0.55

GRID_STEP = 20
ARROW_SCALE = 2.0

FFMPEG = r"C:\Program Files\ffmpeg\bin\ffmpeg.exe"
FFPROBE = r"C:\Program Files\ffmpeg\bin\ffprobe.exe"


# ----------------------------------------------------------------------
# Video information
# ----------------------------------------------------------------------

def ffprobe_video(video_path):
    """Return width, height, FPS and duration."""

    cmd = [
        FFPROBE,
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate,duration",
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
            values[key] = value

    width = int(values["width"])
    height = int(values["height"])

    fps_num, fps_den = values["r_frame_rate"].split("/")
    fps = float(fps_num) / float(fps_den)

    duration = float(values.get("duration", 0.0))

    return width, height, fps, duration


# ----------------------------------------------------------------------
# Optical flow
# ----------------------------------------------------------------------

def calculate_flow(prev_gray, current_gray):
    """
    Calculate dense Farneback optical flow.

    Returns:
        flow       H x W x 2
        magnitude  H x W
        angle      H x W in degrees
    """

    flow = cv2.calcOpticalFlowFarneback(
        prev_gray,
        current_gray,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )

    fx = flow[..., 0]
    fy = flow[..., 1]

    magnitude, angle = cv2.cartToPolar(
        fx,
        fy,
        angleInDegrees=True,
    )

    return flow, magnitude, angle


def create_flow_overlay(
    flow,
    magnitude,
    angle,
):
    """
    Create an HSV optical-flow visualization.

    Hue:
        direction

    Brightness:
        magnitude

    Saturation:
        magnitude
    """

    # Use the 95th percentile instead of the absolute maximum.
    # This prevents a few extreme vectors from dominating the display.
    p95 = np.percentile(
        magnitude,
        95,
    )

    if p95 < 1e-6:

        normalized = np.zeros_like(
            magnitude
        )

    else:

        normalized = np.clip(
            magnitude / p95,
            0.0,
            1.0,
        )

    h, w = magnitude.shape

    hsv = np.zeros(
        (h, w, 3),
        dtype=np.uint8,
    )

    # OpenCV hue range is 0-179.
    hsv[..., 0] = (
        angle / 2.0
    ).astype(np.uint8)

    hsv[..., 1] = (
        normalized * 255
    ).astype(np.uint8)

    hsv[..., 2] = (
        normalized * 255
    ).astype(np.uint8)

    overlay = cv2.cvtColor(
        hsv,
        cv2.COLOR_HSV2BGR,
    )

    return overlay, normalized


def draw_arrows(
    image,
    flow,
    normalized_magnitude,
    step=GRID_STEP,
):
    """Draw a sparse optical-flow vector field."""

    h, w = normalized_magnitude.shape

    for y in range(
        step // 2,
        h,
        step,
    ):

        for x in range(
            step // 2,
            w,
            step,
        ):

            mag = normalized_magnitude[y, x]

            # Ignore very small vectors.
            if mag < 0.08:
                continue

            fx, fy = flow[y, x]

            end_x = int(
                x + fx * ARROW_SCALE
            )

            end_y = int(
                y + fy * ARROW_SCALE
            )

            end_x = max(
                0,
                min(w - 1, end_x),
            )

            end_y = max(
                0,
                min(h - 1, end_y),
            )

            cv2.arrowedLine(
                image,
                (x, y),
                (end_x, end_y),
                (255, 255, 255),
                1,
                cv2.LINE_AA,
                tipLength=0.25,
            )

    return image


# ----------------------------------------------------------------------
# FFmpeg encoder
# ----------------------------------------------------------------------

def start_encoder(
    output_path,
    width,
    height,
    fps,
):
    """Start FFmpeg NVENC encoder receiving BGR frames."""

    cmd = [
        FFMPEG,
        "-hide_banner",
        "-loglevel", "warning",

        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}",
        "-r", f"{fps:.6f}",
        "-i", "-",

        "-an",

        "-c:v", "h264_nvenc",
        "-preset", "p5",
        "-cq", "19",
        "-pix_fmt", "yuv420p",

        "-movflags", "+faststart",

        "-y",
        str(output_path),
    ]

    return subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
    )


# ----------------------------------------------------------------------
# Main processing
# ----------------------------------------------------------------------

def process_video(
    input_path,
    output_path,
    start_time,
    end_time,
    flow_fps,
    flow_width,
    flow_height_percent,
    flow_alpha,
    draw_vectors,
    no_display,
    no_track,
):
    print()
    print("=" * 70)
    print("Optical Flow Visualization")
    print("=" * 70)
    print(f"Input:       {input_path}")
    print(f"Output:      {output_path}")

    if start_time is None:
        print("Start:       beginning")
    else:
        print(f"Start:       {start_time:.3f} s")

    if end_time is None:
        print("End:         end of video")
    else:
        print(f"End:         {end_time:.3f} s")

    print(f"Flow FPS:    {flow_fps}")
    print(f"Flow width:  {flow_width}")
    print(
        f"Flow height: {flow_height_percent:.1f}% "
        f"(upper part only)"
    )
    print(f"Flow alpha:  {flow_alpha}")
    print(f"Arrows:      {'yes' if draw_vectors else 'no'}")
    print()

    # --------------------------------------------------------------
    # Probe source
    # --------------------------------------------------------------

    (
        source_width,
        source_height,
        source_fps,
        duration,
    ) = ffprobe_video(input_path)

    print(
        f"Source:      "
        f"{source_width}x{source_height}"
    )

    print(
        f"Source FPS:  "
        f"{source_fps:.3f}"
    )

    print(
        f"Duration:    "
        f"{duration:.1f} s"
    )

    # --------------------------------------------------------------
    # Validate interval
    # --------------------------------------------------------------

    if start_time is None:
        start_time = 0.0

    if end_time is None:
        end_time = duration

    if start_time < 0:
        raise ValueError(
            "--start cannot be negative."
        )

    if end_time <= start_time:
        raise ValueError(
            "--end must be greater than --start."
        )

    if start_time >= duration:
        raise ValueError(
            "--start is beyond the end of the video."
        )

    end_time = min(
        end_time,
        duration,
    )

    # --------------------------------------------------------------
    # Validate flow FPS
    # --------------------------------------------------------------

    if flow_fps <= 0:
        raise ValueError(
            "--flow-fps must be greater than zero."
        )

    if flow_fps > source_fps:
        print(
            f"WARNING: flow FPS ({flow_fps}) is higher "
            f"than source FPS ({source_fps:.3f})."
        )

        print(
            "Using source FPS instead."
        )

        flow_fps = source_fps

    # --------------------------------------------------------------
    # Validate flow height
    # --------------------------------------------------------------

    if not 1.0 <= flow_height_percent <= 100.0:
        raise ValueError(
            "--flow-height must be between 1 and 100."
        )

    # --------------------------------------------------------------
    # Calculate flow-analysis dimensions
    # --------------------------------------------------------------

    flow_width = min(
        flow_width,
        source_width,
    )

    flow_height_full = int(
        round(
            source_height
            * flow_height_percent
            / 100.0
        )
    )

    flow_height_full = max(
        2,
        min(
            source_height,
            flow_height_full,
        ),
    )

    # Preserve the source aspect ratio for the analyzed region.
    flow_height = int(
        round(
            flow_height_full
            * flow_width
            / source_width
        )
    )

    # Farneback prefers even dimensions.
    flow_width -= flow_width % 2
    flow_height -= flow_height % 2

    print(
        f"Flow size:   "
        f"{flow_width}x{flow_height}"
    )

    # --------------------------------------------------------------
    # Flow timing
    # --------------------------------------------------------------

    flow_interval = 1.0 / flow_fps

    next_flow_time = start_time

    # --------------------------------------------------------------
    # Source FFmpeg decoder
    # --------------------------------------------------------------

    frame_size = (
        source_width
        * source_height
        * 3
    )

    decode_cmd = [
        FFMPEG,
        "-hide_banner",
        "-loglevel", "error",

        "-hwaccel", "cuda",

        "-ss", f"{start_time:.6f}",
        "-i", str(input_path),

        "-t", f"{end_time - start_time:.6f}",

        "-f", "rawvideo",
        "-pix_fmt", "bgr24",

        "-",
    ]

    decoder = subprocess.Popen(
        decode_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=10**8,
    )

    # --------------------------------------------------------------
    # Output encoder
    # --------------------------------------------------------------

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    encoder = start_encoder(
        output_path,
        source_width,
        source_height,
        source_fps,
    )

    # --------------------------------------------------------------
    # State
    # --------------------------------------------------------------

    previous_flow_frame = None

    current_overlay = None
    current_arrows = None

    frame_index = 0
    flow_count = 0

    processing_start = time.time()

    try:

        while True:

            raw = decoder.stdout.read(
                frame_size
            )

            if len(raw) != frame_size:
                break

            frame = np.frombuffer(
                raw,
                dtype=np.uint8,
            ).reshape(
                source_height,
                source_width,
                3,
            )

            # Timestamp relative to the selected interval.
            relative_time = (
                frame_index
                / source_fps
            )

            absolute_time = (
                start_time
                + relative_time
            )

            # ------------------------------------------------------
            # Stop at requested end.
            # ------------------------------------------------------

            if absolute_time >= end_time:
                break

            # ------------------------------------------------------
            # Flow sampling
            # ------------------------------------------------------

            if (
                absolute_time + 1e-9
                >= next_flow_time
            ):

                # --------------------------------------------------
                # Crop the upper part of the frame.
                #
                # IMPORTANT:
                # The original frame is NOT cropped.
                # Only the optical-flow calculation uses this region.
                # --------------------------------------------------

                flow_source = frame[
                    :flow_height_full,
                    :,
                ]

                flow_frame = cv2.resize(
                    flow_source,
                    (
                        flow_width,
                        flow_height,
                    ),
                    interpolation=cv2.INTER_AREA,
                )

                gray = cv2.cvtColor(
                    flow_frame,
                    cv2.COLOR_BGR2GRAY,
                )

                if previous_flow_frame is not None:

                    (
                        flow,
                        magnitude,
                        angle,
                    ) = calculate_flow(
                        previous_flow_frame,
                        gray,
                    )

                    (
                        overlay_small,
                        normalized,
                    ) = create_flow_overlay(
                        flow,
                        magnitude,
                        angle,
                    )

                    # --------------------------------------------------
                    # Create a full-frame transparent overlay.
                    #
                    # Only the upper flow region is filled.
                    # The rest stays black and therefore has no effect
                    # when alpha-blended below.
                    # --------------------------------------------------

                    overlay_full = np.zeros_like(
                        frame
                    )

                    overlay_upper = cv2.resize(
                        overlay_small,
                        (
                            source_width,
                            flow_height_full,
                        ),
                        interpolation=cv2.INTER_LINEAR,
                    )

                    overlay_full[
                        :flow_height_full,
                        :,
                    ] = overlay_upper

                    current_overlay = (
                        overlay_full
                    )

                    # --------------------------------------------------
                    # Arrows
                    # --------------------------------------------------

                    if draw_vectors:

                        arrows_small = (
                            overlay_small.copy()
                        )

                        draw_arrows(
                            arrows_small,
                            flow,
                            normalized,
                        )

                        arrows_full = (
                            np.zeros_like(frame)
                        )

                        arrows_upper = cv2.resize(
                            arrows_small,
                            (
                                source_width,
                                flow_height_full,
                            ),
                            interpolation=cv2.INTER_LINEAR,
                        )

                        arrows_full[
                            :flow_height_full,
                            :,
                        ] = arrows_upper

                        current_arrows = (
                            arrows_full
                        )

                    else:

                        current_arrows = None

                    flow_count += 1

                previous_flow_frame = gray

                # Advance flow schedule until it is in the future.
                while (
                    next_flow_time
                    <= absolute_time
                ):
                    next_flow_time += (
                        flow_interval
                    )

            # ------------------------------------------------------
            # Apply flow overlay.
            # ------------------------------------------------------

            output_frame = frame.copy()

            if current_overlay is not None:

                # Because the lower part of current_overlay is black,
                # we must NOT simply blend the entire frame: doing so
                # would darken the lower region.
                #
                # Blend only the upper flow region.
                upper_original = output_frame[
                    :flow_height_full,
                    :
                ]

                upper_overlay = current_overlay[
                    :flow_height_full,
                    :
                ]

                upper_blended = cv2.addWeighted(
                    upper_original,
                    1.0 - flow_alpha,
                    upper_overlay,
                    flow_alpha,
                    0,
                )

                output_frame[
                    :flow_height_full,
                    :
                ] = upper_blended

                if current_arrows is not None:

                    upper_original = output_frame[
                        :flow_height_full,
                        :
                    ]

                    upper_arrows = current_arrows[
                        :flow_height_full,
                        :
                    ]

                    upper_blended = cv2.addWeighted(
                        upper_original,
                        0.78,
                        upper_arrows,
                        0.22,
                        0,
                    )

                    output_frame[
                        :flow_height_full,
                        :
                    ] = upper_blended

            # ------------------------------------------------------
            # Write output frame.
            # ------------------------------------------------------

            encoder.stdin.write(
                output_frame.tobytes()
            )

            frame_index += 1

            # ------------------------------------------------------
            # Progress
            # ------------------------------------------------------

            if (
                frame_index
                % int(max(1, source_fps * 2))
                == 0
            ):

                elapsed = (
                    time.time()
                    - processing_start
                )

                if elapsed > 0:

                    processed_fps = (
                        frame_index
                        / elapsed
                    )

                    speed = (
                        processed_fps
                        / source_fps
                    )

                else:

                    processed_fps = 0.0
                    speed = 0.0

                selected_duration = (
                    end_time
                    - start_time
                )

                if selected_duration > 0:

                    percent = (
                        relative_time
                        / selected_duration
                        * 100
                    )

                else:

                    percent = 100.0

                print(
                    f"\r"
                    f"{percent:6.1f}%  "
                    f"{relative_time:7.1f}s / "
                    f"{selected_duration:7.1f}s  "
                    f"{processed_fps:6.1f} fps  "
                    f"{speed:5.2f}x  "
                    f"flow samples: "
                    f"{flow_count:6d}",
                    end="",
                    flush=True,
                )

    finally:

        print()

        # ----------------------------------------------------------
        # Decoder cleanup
        # ----------------------------------------------------------

        if decoder.stdout:
            decoder.stdout.close()

        decoder.wait()

        # ----------------------------------------------------------
        # Encoder cleanup
        # ----------------------------------------------------------

        if encoder.stdin:
            encoder.stdin.close()

        encoder.wait()

    elapsed = (
        time.time()
        - processing_start
    )

    print()
    print("=" * 70)
    print("Finished")
    print("=" * 70)

    print(
        f"Frames:       "
        f"{frame_index}"
    )

    print(
        f"Flow samples: "
        f"{flow_count}"
    )

    print(
        f"Elapsed:      "
        f"{elapsed:.1f} s"
    )

    if elapsed > 0:

        processing_fps = (
            frame_index
            / elapsed
        )

        print(
            f"Processing:   "
            f"{processing_fps:.2f} fps "
            f"("
            f"{processing_fps / source_fps:.2f}x"
            f")"
        )

    if encoder.returncode != 0:

        raise RuntimeError(
            "FFmpeg encoder failed with "
            f"exit code {encoder.returncode}"
        )

    print(
        f"Output:       "
        f"{output_path}"
    )


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Visualize optical flow over "
            "the original video."
        )
    )

    parser.add_argument(
        "input",
        type=Path,
        help="Input video",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Output video",
    )

    parser.add_argument(
        "--start",
        type=float,
        default=None,
        help=(
            "Start time in seconds "
            "(default: beginning)"
        ),
    )

    parser.add_argument(
        "--end",
        type=float,
        default=None,
        help=(
            "End time in seconds "
            "(default: end of video)"
        ),
    )

    parser.add_argument(
        "--flow-fps",
        type=float,
        default=DEFAULT_FLOW_FPS,
        help=(
            "Optical-flow calculation FPS "
            f"(default: {DEFAULT_FLOW_FPS})"
        ),
    )

    parser.add_argument(
        "--flow-width",
        type=int,
        default=DEFAULT_FLOW_WIDTH,
        help=(
            "Optical-flow analysis width "
            f"(default: {DEFAULT_FLOW_WIDTH})"
        ),
    )

    parser.add_argument(
        "--flow-height",
        type=float,
        default=DEFAULT_FLOW_HEIGHT,
        help=(
            "Percentage of image height used "
            "for optical flow, measured from the top "
            f"(default: {DEFAULT_FLOW_HEIGHT})"
        ),
    )

    parser.add_argument(
        "--flow-alpha",
        type=float,
        default=DEFAULT_FLOW_ALPHA,
        help=(
            "Flow overlay transparency, 0-1 "
            f"(default: {DEFAULT_FLOW_ALPHA})"
        ),
    )

    parser.add_argument(
        "--no-arrows",
        action="store_true",
        help="Disable optical-flow direction arrows.",
    )

    args = parser.parse_args()

    # --------------------------------------------------------------
    # Basic validation
    # --------------------------------------------------------------

    if not args.input.exists():

        print(
            "ERROR: input file does not exist:",
            file=sys.stderr,
        )

        print(
            args.input,
            file=sys.stderr,
        )

        sys.exit(1)

    if args.flow_fps <= 0:

        print(
            "ERROR: --flow-fps must be greater than zero.",
            file=sys.stderr,
        )

        sys.exit(1)

    if args.flow_width <= 0:

        print(
            "ERROR: --flow-width must be greater than zero.",
            file=sys.stderr,
        )

        sys.exit(1)

    if not 1.0 <= args.flow_height <= 100.0:

        print(
            "ERROR: --flow-height must be between "
            "1 and 100.",
            file=sys.stderr,
        )

        sys.exit(1)

    if not 0.0 <= args.flow_alpha <= 1.0:

        print(
            "ERROR: --flow-alpha must be between 0 and 1.",
            file=sys.stderr,
        )

        sys.exit(1)

    try:

        process_video(
            input_path=args.input,
            output_path=args.output,
            start_time=args.start,
            end_time=args.end,
            flow_fps=args.flow_fps,
            flow_width=args.flow_width,
            flow_height_percent=args.flow_height,
            flow_alpha=args.flow_alpha,
            draw_vectors=not args.no_arrows,
            no_display=args.no_display,
            no_track=args.no_track,
        )

    except KeyboardInterrupt:

        print()
        print("Interrupted.")
        sys.exit(130)

    except Exception as exc:

        print(
            f"\nERROR: {exc}",
            file=sys.stderr,
        )

        sys.exit(1)


if __name__ == "__main__":
    main()
