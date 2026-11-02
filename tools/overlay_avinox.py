"""Render Avinox speed, gradient and one raw IMU axis over a local video.

Example:
    python tools/overlay_avinox.py ride.mp4 ride.proto --axis x --offset 0
"""
from __future__ import annotations

import argparse
import bisect
import json
import shutil
import subprocess
import time
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from autocut.dataset.avinox_proto import parse  # noqa: E402


def _probe(video: Path) -> dict[str, Any]:
    if not shutil.which("ffprobe"):
        raise RuntimeError("ffprobe not found. Install FFmpeg and ensure ffprobe is on PATH.")
    command = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,avg_frame_rate,r_frame_rate:stream_tags=creation_time",
        "-show_entries", "format=duration:format_tags=creation_time", "-of", "json", str(video),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    data = json.loads(result.stdout)
    stream = data["streams"][0]
    def rate(value: str) -> float:
        numerator, denominator = value.split("/")
        return float(numerator) / float(denominator) if float(denominator) else 0.0
    fps = rate(stream.get("avg_frame_rate", "0/0")) or rate(stream.get("r_frame_rate", "0/0"))
    return {
        "width": int(stream["width"]), "height": int(stream["height"]),
        "fps": fps, "duration": float(data.get("format", {}).get("duration", 0)),
        "creation_time": (data.get("format", {}).get("tags", {}).get("creation_time")
                          or stream.get("tags", {}).get("creation_time")),
    }


def _draw_panel(cv2: Any, frame: Any, x: int, y: int, w: int, h: int,
                label: str, value: str, align_right: bool = False) -> None:
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (12, 15, 18), -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.55, min(1.15, frame.shape[1] / 1500))
    thickness = max(1, round(scale * 2))
    cv2.putText(frame, label, (x + 14, y + 27), font, scale * 0.62,
                (190, 195, 200), 1, cv2.LINE_AA)
    text_size = cv2.getTextSize(value, font, scale, thickness)[0]
    tx = x + w - text_size[0] - 14 if align_right else x + 14
    cv2.putText(frame, value, (tx, y + h - 13), font, scale, (255, 255, 255),
                thickness, cv2.LINE_AA)


def _draw_imu(cv2: Any, frame: Any, samples: list[dict[str, Any]],
              times: list[float], current: float, axis: str, window: float) -> None:
    height, width = frame.shape[:2]
    margin = max(18, width // 60)
    panel_h = max(115, height // 4)
    x0, y0 = margin, height - panel_h - margin
    x1, y1 = width - margin, height - margin

    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (8, 12, 16), -1)
    cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)

    key = "imu" + axis.upper()
    left = current - window
    visible = [
        (t, s.get(key)) for t, s in zip(times, samples)
        if left <= t <= current and s.get(key) is not None
    ]
    title = f"IMU {axis.upper()}  |  raw"
    cv2.putText(frame, title, (x0 + 14, y0 + 25), cv2.FONT_HERSHEY_SIMPLEX,
                0.62, (225, 230, 235), 1, cv2.LINE_AA)

    graph_left, graph_right = x0 + 55, x1 - 18
    graph_top, graph_bottom = y0 + 38, y1 - 25
    cv2.line(frame, (graph_left, graph_bottom), (graph_right, graph_bottom),
             (90, 95, 100), 1, cv2.LINE_AA)
    cv2.line(frame, (graph_left, graph_top), (graph_left, graph_bottom),
             (90, 95, 100), 1, cv2.LINE_AA)

    if visible:
        vals = [float(v) for _, v in visible]
        lo, hi = min(vals), max(vals)
        span = max(hi - lo, 0.05)
        pad = span * 0.12
        lo, hi = lo - pad, hi + pad
        points = []
        for t, value in visible:
            px = graph_left + int((t - left) / window * (graph_right - graph_left))
            py = graph_bottom - int((float(value) - lo) / (hi - lo) *
                                    (graph_bottom - graph_top))
            points.append((px, py))
        for a, b in zip(points, points[1:]):
            cv2.line(frame, a, b, (70, 220, 255), 2, cv2.LINE_AA)
        for point in points:
            cv2.circle(frame, point, 2, (70, 220, 255), -1, cv2.LINE_AA)
        cv2.putText(frame, f"{hi:.2f}", (x0 + 8, graph_top + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (170, 175, 180), 1, cv2.LINE_AA)
        cv2.putText(frame, f"{lo:.2f}", (x0 + 8, graph_bottom),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (170, 175, 180), 1, cv2.LINE_AA)
    cursor_x = graph_left + int(max(0.0, min(1.0, (current - left) / window)) *
                                (graph_right - graph_left))
    cv2.line(frame, (cursor_x, graph_top), (cursor_x, graph_bottom),
             (255, 255, 255), 1, cv2.LINE_AA)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path, help="Source video")
    parser.add_argument("recording", type=Path, help="Avinox .proto recording")
    parser.add_argument("--axis", choices=("x", "y", "z"), default="x",
                        help="IMU axis to plot (default: x)")
    parser.add_argument("--offset", type=float, default=0.0,
                        help="Seconds added to video elapsed time to locate telemetry (default: 0)")
    parser.add_argument("--window", type=float, default=12.0,
                        help="Scrolling IMU history in seconds (default: 12)")
    parser.add_argument("--output", type=Path,
                        help="Output MP4 (default: <video>_avinox_<axis>.mp4)")
    parser.add_argument("--start", type=float, default=0.0,
                        help="Start encoding at this video time in seconds")
    parser.add_argument("--duration", type=float,
                        help="Optional duration to encode; useful for a short test")
    parser.add_argument("--crf", type=int, default=20,
                        help="H.264 quality: lower is higher quality (default: 20)")
    parser.add_argument("--hwaccel", choices=("auto", "cuda", "cpu"), default="auto",
                        help="Decoder: CUDA/NVDEC, CPU, or auto (default: auto)")
    parser.add_argument("--encoder", choices=("auto", "h264_nvenc", "libx264"), default="auto",
                        help="Encoder: NVENC, libx264, or auto (default: auto)")
    args = parser.parse_args()

    if not args.video.is_file():
        parser.error(f"Video not found: {args.video}")
    if not args.recording.is_file():
        parser.error(f"Avinox recording not found: {args.recording}")
    if args.window <= 0 or args.start < 0 or (args.duration is not None and args.duration <= 0):
        parser.error("--window and --duration must be positive; --start cannot be negative")

    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required; install project dependencies with: python -m pip install -e .") from exc

    info = _probe(args.video)
    creation_time = info.get("creation_time")
    if not creation_time:
        raise RuntimeError("MP4 creation_time is missing; cannot synchronize Avinox telemetry.")
    from datetime import datetime
    try:
        video_epoch = datetime.fromisoformat(creation_time.replace("Z", "+00:00")).timestamp()
    except ValueError as exc:
        raise RuntimeError(f"Invalid MP4 creation_time: {creation_time!r}") from exc

    decoded = parse(args.recording)
    metadata, samples = decoded["metadata"], decoded["samples"]
    samples = sorted(
        (s for s in samples if s.get("timestamp") is not None),
        key=lambda s: s["timestamp"],
    )
    if not samples:
        raise RuntimeError("No timestamped telemetry samples found.")
    avinox_start = float(metadata.get("start") or samples[0]["timestamp"])
    times = [float(s["timestamp"]) - avinox_start for s in samples]
    end_time = min(info["duration"], args.start + args.duration) if args.duration else info["duration"]
    if args.start >= info["duration"]:
        parser.error("--start is beyond the end of the video")
    output = args.output or args.video.with_name(
        f"{args.video.stem}_avinox_{args.axis}.mp4"
    )
    output.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")
    cap.set(cv2.CAP_PROP_POS_MSEC, args.start * 1000)
    width, height = info["width"], info["height"]
    fps = info["fps"]
    if fps <= 0:
        raise RuntimeError("Could not determine video frame rate.")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output), fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not create output video: {output}")

    frame_index = 0
    first_sample_time = times[0]
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            video_time = args.start + frame_index / fps
            if video_time >= end_time:
                break
            telemetry_time = video_epoch + video_time + args.offset - avinox_start
            nearest = bisect.bisect_left(times, telemetry_time)
            candidates = [i for i in (nearest - 1, nearest) if 0 <= i < len(samples)]
            sample_index = min(candidates, key=lambda i: abs(times[i] - telemetry_time)) if candidates else None
            sample = samples[sample_index] if sample_index is not None else {}

            speed = sample.get("speed")
            gradient = sample.get("gradient")
            speed_text = f"{float(speed):.1f} km/h" if speed is not None else "—"
            gradient_text = f"{float(gradient):.2f}" if gradient is not None else "—"
            panel_w = max(190, width // 5)
            panel_h = max(58, height // 12)
            _draw_panel(cv2, frame, 18, 18, panel_w, panel_h,
                        "SPEED", speed_text)
            _draw_panel(cv2, frame, width - panel_w - 18, 18, panel_w, panel_h,
                        "GRADIENT (raw / 100)", gradient_text, True)
            _draw_imu(cv2, frame, samples, times, telemetry_time, args.axis, args.window)
            writer.write(frame)
            frame_index += 1
            if frame_index % max(1, int(fps * 10)) == 0:
                print(f"Encoded {frame_index / fps:.0f}s / {end_time - args.start:.0f}s",
                      flush=True)
    finally:
        cap.release()
        writer.release()

    print(f"Video: {args.video}")
    print(f"MP4 creation_time: {creation_time} (Unix {video_epoch:.3f})")
    print(f"Avinox metadata start: {avinox_start:.3f} (UTC epoch)")
    print(f"Avinox start (UTC epoch): {metadata.get('start')}")
    print(f"First sample: {samples[0].get('date')}")
    print(f"Offset: {args.offset:+.3f}s (fine adjustment)")
    print(f"IMU axis: {args.axis.upper()} | window: {args.window:g}s")
    print(f"Frames: {frame_index} | output: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
