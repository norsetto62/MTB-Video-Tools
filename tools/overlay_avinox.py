"""Render Avinox speed, gradient and one raw IMU axis over a local video.

Example:
    python tools/overlay_avinox.py ride.mp4 ride.proto --axis x --offset 0
"""
from __future__ import annotations

import argparse
import bisect
import json
import queue
import shutil
import subprocess
import threading
import time
import numpy as np
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



def _video_reader_command(video: Path, start: float, duration: float,
                          cuda: bool) -> list[str]:
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error"]
    if start > 0:
        command += ["-ss", f"{start:.6f}"]
    if cuda:
        command += ["-hwaccel", "cuda"]
    command += ["-i", str(video)]
    if duration > 0:
        command += ["-t", f"{duration:.6f}"]
    if cuda:
        # Let FFmpeg negotiate the decoder's actual CUDA surface format.
        # The output pixel format requests the CPU frames OpenCV consumes.
        command += ["-pix_fmt", "bgr24"]
    command += ["-an", "-sn", "-dn", "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"]
    return command


def _video_writer_command(width: int, height: int, fps: float, output: Path,
                          encoder: str, crf: int) -> list[str]:
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}",
        "-r", f"{fps:.8f}", "-i", "pipe:0", "-an", "-c:v", encoder,
    ]
    if encoder == "h264_nvenc":
        command += ["-preset", "p4", "-rc", "constqp", "-qp", str(crf)]
    else:
        command += ["-preset", "medium", "-crf", str(crf)]
    command += ["-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output)]
    return command


def _draw_panel(cv2: Any, frame: Any, x: int, y: int, w: int, h: int,
                label: str, value: str, align_right: bool = False) -> None:
    # Blend only this panel ROI instead of copying/blending the full frame.
    roi = frame[y:y + h, x:x + w]
    import numpy as np
    background = np.empty_like(roi)
    background[:] = (12, 15, 18)
    cv2.addWeighted(roi, 0.28, background, 0.72, 0, roi)
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

    # Blend only the chart ROI, not the entire video frame.
    roi = frame[y0:y1, x0:x1]
    import numpy as np
    background = np.empty_like(roi)
    background[:] = (8, 12, 16)
    cv2.addWeighted(roi, 0.22, background, 0.78, 0, roi)

    key = "imu" + axis.upper()
    left = current - window
    # Telemetry is time-sorted: slice only the samples visible in this
    # scrolling window instead of scanning the entire recording for every frame.
    first = bisect.bisect_left(times, left)
    last = bisect.bisect_right(times, current)
    visible = [
        (t, sample.get(key))
        for t, sample in zip(times[first:last], samples[first:last])
        if sample.get(key) is not None
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
        # Fixed scale makes values comparable across frames and axes.
        lo, hi = -1.0, 1.0
        points = []
        for t, value in visible:
            px = graph_left + int((t - left) / window * (graph_right - graph_left))
            clipped = max(lo, min(hi, float(value)))
            py = graph_bottom - int((clipped - lo) / (hi - lo) *
                                    (graph_bottom - graph_top))
            point = (px, py)
            if not points or point != points[-1]:
                points.append(point)
        if len(points) > 1:
            cv2.polylines(frame, [np.asarray(points, dtype=np.int32)],
                          False, (70, 220, 255), 2, cv2.LINE_AA)
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

    width, height = info["width"], info["height"]
    fps = info["fps"]
    if fps <= 0:
        raise RuntimeError("Could not determine video frame rate.")
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg not found. Install FFmpeg and ensure it is on PATH.")

    # This utility runs on a dedicated CUDA/NVENC workstation.
    # Use the GPU directly; let FFmpeg report a real initialization/runtime error.
    use_cuda = True
    encoder = "h264_nvenc"

    decode_cmd = _video_reader_command(
        args.video, args.start, end_time - args.start, use_cuda
    )
    encode_cmd = _video_writer_command(width, height, fps, output, encoder, args.crf)
    decoder = subprocess.Popen(decode_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    encoder_process = subprocess.Popen(encode_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    # Keep a bounded frame pipeline: decoding, Python overlay rendering and
    # NVENC encoding can progress concurrently without buffering the whole ride.
    # 32 frames use about 190 MiB at 1080p or 760 MiB at 4K.
    buffer_frames = 32
    decoded_frames: queue.Queue[bytes | None] = queue.Queue(maxsize=buffer_frames)
    encoded_frames: queue.Queue[Any | None] = queue.Queue(maxsize=buffer_frames)
    worker_errors: list[tuple[str, BaseException]] = []
    decoder_stderr = bytearray()
    encoder_stderr = bytearray()
    frame_bytes = width * height * 3
    frame_index = 0
    decode_error = b""

    def drain_stderr(stream: Any, target: bytearray, stage: str) -> None:
        try:
            while chunk := stream.read(65536):
                target.extend(chunk)
                if len(target) > 1_000_000:
                    del target[:-1_000_000]
        except BaseException as exc:
            worker_errors.append((f"{stage} stderr reader", exc))

    def read_frames() -> None:
        try:
            while True:
                raw = decoder.stdout.read(frame_bytes)
                if not raw:
                    break
                if len(raw) != frame_bytes:
                    raise RuntimeError(
                        f"FFmpeg returned an incomplete frame ({len(raw)} of {frame_bytes} bytes)."
                    )
                decoded_frames.put(raw)
        except BaseException as exc:
            worker_errors.append(("decoder reader", exc))
        finally:
            decoded_frames.put(None)

    def write_frames() -> None:
        failed = False
        while True:
            raw = encoded_frames.get()
            try:
                if raw is None:
                    break
                # Keep draining if FFmpeg exits: otherwise the producer can
                # block forever on a full queue and appear to hang silently.
                if not failed:
                    try:
                        encoder_process.stdin.write(memoryview(raw))
                    except BaseException as exc:
                        worker_errors.append(("encoder writer", exc))
                        failed = True
            finally:
                encoded_frames.task_done()

    reader_thread = threading.Thread(target=read_frames, name="ffmpeg-reader", daemon=True)
    writer_thread = threading.Thread(target=write_frames, name="ffmpeg-writer", daemon=True)
    decoder_log_thread = threading.Thread(
        target=drain_stderr, args=(decoder.stderr, decoder_stderr, "decoder"),
        name="ffmpeg-decoder-log", daemon=True,
    )
    encoder_log_thread = threading.Thread(
        target=drain_stderr, args=(encoder_process.stderr, encoder_stderr, "encoder"),
        name="ffmpeg-encoder-log", daemon=True,
    )
    print("Starting CUDA decode -> OpenCV overlay -> NVENC pipeline...", flush=True)
    print("Launching FFmpeg decoder...", flush=True)
    decoder_log_thread.start()
    encoder_log_thread.start()
    reader_thread.start()
    writer_thread.start()
    last_frame_at = time.monotonic()
    try:
        while True:
            try:
                raw = decoded_frames.get(timeout=1.0)
            except queue.Empty:
                if decoder.poll() is not None and not reader_thread.is_alive():
                    break
                if time.monotonic() - last_frame_at > 60:
                    raise RuntimeError(
                        "FFmpeg decoder produced no frame for 60 seconds. "
                        f"Decoder status: {decoder.poll()}; stderr: "
                        f"{decoder_stderr.decode(errors='replace')[-4000:]}"
                    )
                continue
            if raw is None:
                break
            last_frame_at = time.monotonic()
            if frame_index == 0:
                print("First decoded frame received; processing started.", flush=True)
            frame = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3).copy()
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
            gradient_text = f"{float(gradient):.2f}%" if gradient is not None else "—"
            panel_w = max(190, width // 5)
            panel_h = max(58, height // 12)
            _draw_panel(cv2, frame, 18, 18, panel_w, panel_h, "SPEED", speed_text)
            _draw_panel(cv2, frame, width - panel_w - 18, 18, panel_w, panel_h,
                        "SLOPE", gradient_text, True)
            _draw_imu(cv2, frame, samples, times, telemetry_time, args.axis, args.window)
            # Queue the ndarray and let the writer consume its buffer directly.
            encoded_frames.put(frame)
            frame_index += 1
            if frame_index % max(1, int(fps * 10)) == 0:
                print(f"Encoded {frame_index / fps:.0f}s / {end_time - args.start:.0f}s",
                      flush=True)
    except BaseException:
        decoder.terminate()
        encoder_process.terminate()
        raise
    finally:
        # Stop workers without waiting forever on a full bounded queue.
        if reader_thread.is_alive():
            decoder.terminate()
        try:
            decoded_frames.put_nowait(None)
        except queue.Full:
            pass
        try:
            encoded_frames.put_nowait(None)
        except queue.Full:
            pass
        reader_thread.join(timeout=10)
        writer_thread.join(timeout=10)
        if decoder.stdout:
            decoder.stdout.close()
        if encoder_process.stdin:
            try:
                encoder_process.stdin.close()
            except BrokenPipeError:
                pass
        decoder.wait()
        encoder_process.wait()
        decoder_log_thread.join(timeout=5)
        encoder_log_thread.join(timeout=5)
        decode_error = bytes(decoder_stderr)
        encode_error = bytes(encoder_stderr)
        if decoder.stderr:
            decoder.stderr.close()
        if encoder_process.stderr:
            encoder_process.stderr.close()
    if worker_errors:
        stage, error = worker_errors[0]
        raise RuntimeError(f"{stage} failed: {error}") from error
    if decoder.returncode != 0:
        raise RuntimeError(f"FFmpeg decoder failed: {decode_error.decode(errors='replace')}")
    if encoder_process.returncode != 0:
        raise RuntimeError(f"FFmpeg encoder failed: {encode_error.decode(errors='replace')}")
    if frame_index == 0:
        raise RuntimeError("No video frames were decoded.")

    print(f"Video: {args.video}")
    print(f"MP4 creation_time: {creation_time} (Unix {video_epoch:.3f})")
    print(f"Avinox metadata start: {avinox_start:.3f} (UTC epoch)")
    print(f"Avinox start (UTC epoch): {metadata.get('start')}")
    print(f"First sample: {samples[0].get('date')}")
    print(f"Offset: {args.offset:+.3f}s (fine adjustment)")
    print(f"IMU axis: {args.axis.upper()} | fixed scale: -1.00 to +1.00 | window: {args.window:g}s")
    print(f"Frames: {frame_index} | output: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
