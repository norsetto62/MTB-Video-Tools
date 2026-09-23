import argparse
import subprocess
import tempfile
from pathlib import Path

import librosa
import numpy as np
import shutil

# Automatically fall back to system PATH if the explicit file isn't found
FFMPEG = shutil.which("ffmpeg") or r"C:\Program Files\ffmpeg\bin\ffmpeg.exe"
FFPROBE = shutil.which("ffprobe") or r"C:\Program Files\ffmpeg\bin\ffprobe.exe"

# Video Constants
MIN_CLIP = 3.0
FADE_OUT_DURATION = 3.0


def parse_time(value):
    if ":" in value:
        minutes, seconds = value.split(":")
        return int(minutes) * 60 + float(seconds)
    return float(value)


def probe_duration(path):
    cmd = [
        FFPROBE, "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return float(result.stdout.strip())


def analyze_audio(music_path):
    print("Analyzing audio rhythm, beats, and energy peaks...")
    y, sr = librosa.load(music_path, sr=None)
    
    _, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    measure_times = beat_times[::4]

    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    peaks = librosa.util.peak_pick(
        onset_env,
        pre_max=3, post_max=3,
        pre_avg=10, post_avg=10,
        delta=1.5, wait=10
    )
    onset_times = librosa.frames_to_time(peaks, sr=sr)

    combined_times = np.unique(np.concatenate([beat_times, onset_times]))
    combined_times.sort()

    return {
        "beat": beat_times,
        "measure": measure_times,
        "onset": onset_times if len(onset_times) > 0 else beat_times,
        "combined": combined_times,
    }


def snap_timestamp(target_time, grid_times, mode="beat"):
    if mode == "forward-beat":
        future = grid_times[grid_times >= target_time]
        return future[0] if len(future) > 0 else grid_times[-1]
    
    idx = (np.abs(grid_times - target_time)).argmin()
    return grid_times[idx]


def load_annotations(path):
    events = []
    last_video_name = None

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            # Skip empty lines or header comments
            if not line or line.startswith("#") or line.startswith("START"):
                continue

            parts = line.split(maxsplit=5)
            
            # Check if the first token is a timestamp (contains ':') or a video filename
            if ":" in parts[0] or parts[0].replace(".", "", 1).isdigit():
                # Case 1: Video name omitted -> reuse last_video_name
                if last_video_name is None:
                    raise ValueError(f"First row in annotations omits video filename: '{line}'")
                video_name = last_video_name
                start = parse_time(parts[0])
                end = parse_time(parts[1])
                mtb = int(parts[2])
                film = int(parts[3])
                description = parts[4] if len(parts) > 4 else ""
            else:
                # Case 2: Video name explicitly provided -> update last_video_name
                video_name = parts[0]
                last_video_name = video_name
                start = parse_time(parts[1])
                end = parse_time(parts[2])
                mtb = int(parts[3])
                film = int(parts[4])
                description = parts[5] if len(parts) > 5 else ""

            events.append({
                "video_name": video_name,
                "start": start,
                "end": end,
                "duration": end - start,
                "mtb": mtb,
                "film": film,
                "description": description,
            })
    return events

def select_events_beat_synced(events, target_duration, grid_times, sync_mode="beat", max_clip=10.0):
    for event in events:
        event["priority"] = event["mtb"] * 2 + event["film"]

    ranked = sorted(events, key=lambda e: (e["priority"], e["duration"]), reverse=True)
    selected = []
    current_audio_time = 0.0

    for event in ranked:
        remaining = target_duration - current_audio_time
        if remaining < MIN_CLIP:
            break

        ideal_len = min(event["duration"], max_clip) if max_clip else event["duration"]
        ideal_len = min(ideal_len, remaining)

        target_end_audio = current_audio_time + ideal_len
        snapped_end_audio = snap_timestamp(target_end_audio, grid_times, mode=sync_mode)
        clip_length = snapped_end_audio - current_audio_time

        if clip_length < MIN_CLIP or clip_length > remaining:
            continue

        event_copy = event.copy()
        event_copy["clip_length"] = clip_length
        event_copy["audio_start"] = current_audio_time
        event_copy["audio_end"] = snapped_end_audio

        selected.append(event_copy)
        current_audio_time = snapped_end_audio

    selected.sort(key=lambda e: e["start"])
    return selected, current_audio_time


def extract_clips(selected, videos_dir, temp_dir, aspect_ratio="landscape"):
    clip_files = []

    for index, event in enumerate(selected):
        video_path = videos_dir / event["video_name"]
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found in videos directory: {video_path}")

        start = event["start"]
        length = event["clip_length"]
        final_clip = temp_dir / f"clip_{index:03d}.ts"

        print(
            f"Clip {index + 1:02d} [{event['video_name']}]: {start:7.2f}s + {length:5.2f}s | "
            f"Sync: {event['audio_start']:.2f}s -> {event['audio_end']:.2f}s"
        )

        cmd_extract = [
            FFMPEG, "-hide_banner", "-loglevel", "error",
            "-ss", f"{start:.3f}",
            "-i", str(video_path),
            "-t", f"{length:.3f}",
        ]

        if aspect_ratio == "portrait":
            cmd_extract.extend(["-vf", "crop=ih*9/16:ih"])

        cmd_extract.extend([
            "-c:v", "h264_nvenc", "-preset", "p4", "-cq", "20",
            "-pix_fmt", "yuv420p", "-an",
            "-f", "mpegts",
            str(final_clip),
        ])

        subprocess.run(cmd_extract, check=True)
        clip_files.append(final_clip)

    return clip_files


def concatenate_clips(clip_files, temp_dir):
    concat_file = temp_dir / "concat.txt"
    with open(concat_file, "w", encoding="utf-8") as f:
        for clip in clip_files:
            f.write(f"file '{clip.as_posix()}'\n")

    silent_video = temp_dir / "video_only.mp4"
    cmd = [
        FFMPEG, "-hide_banner", "-loglevel", "error",
        "-f", "concat", "-safe", "0",
        "-i", str(concat_file),
        "-c", "copy",
        str(silent_video),
    ]
    subprocess.run(cmd, check=True)
    return silent_video


def add_music_with_fade(video_file, music_path, total_duration, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fade_start = max(0.0, total_duration - FADE_OUT_DURATION)
    af_filter = f"afade=t=out:st={fade_start:.3f}:d={FADE_OUT_DURATION:.3f}"

    cmd = [
        FFMPEG, "-hide_banner", "-loglevel", "error",
        "-i", str(video_file),
        "-i", str(music_path),
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "320k",
        "-af", af_filter,
        "-t", f"{total_duration:.3f}",
        "-movflags", "+faststart",
        str(output_path),
    ]
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="MTB-AutoCut Multi-Video Generator")
    
    # Required Positional Arguments
    parser.add_argument("videos_dir", type=Path, help="Directory containing input raw video files")
    parser.add_argument("music", type=Path, help="Path to input music/audio file")
    parser.add_argument("annotations", type=Path, help="Path to annotations TXT file")

    # Optional Arguments
    parser.add_argument(
        "-o", "--output-dir",
        type=Path,
        default=Path("./output"),
        help="Directory to save output file (default: ./output)"
    )
    parser.add_argument(
        "-ar", "--aspect-ratio",
        choices=["landscape", "portrait"],
        default="landscape",
        help="Target aspect ratio: landscape (16:9), portrait (9:16 center crop)"
    )
    parser.add_argument(
        "-sm", "--sync-mode",
        choices=["beat", "measure", "forward-beat", "onset", "combined"],
        default="beat",
        help="Sync strategy: 'beat' (standard beats), 'measure' (4-beat bars), 'forward-beat' (next beat), 'onset' (high-energy peaks/taiko hits), 'combined' (beats + peaks)"
    )
    parser.add_argument(
        "-mc", "--max-clip",
        type=float,
        default=10.0,
        help="Maximum clip duration in seconds (default: 10.0). Set to 0 to allow full annotated clip length."
    )
    args = parser.parse_args()

    if not args.videos_dir.is_dir():
        parser.error(f"Videos directory not found: {args.videos_dir}")
    if not args.music.is_file():
        parser.error(f"Music file not found: {args.music}")
    if not args.annotations.is_file():
        parser.error(f"Annotations file not found: {args.annotations}")

    max_clip_str = "unlimited" if args.max_clip <= 0 else f"{args.max_clip}s"
    output_filename = f"MultiClip_{args.aspect_ratio}_{args.sync_mode}_max{max_clip_str}.mp4"
    output_path = args.output_dir / output_filename

    print(f"MTB-AutoCut v1.0 (Multi-Video Support)")
    print(f"=========================================================================")
    print(f"Videos Directory: {args.videos_dir}")
    print(f"Music File:       {args.music}")
    print(f"Annotations File: {args.annotations}")
    print(f"Settings:         AR={args.aspect_ratio.upper()} | Sync={args.sync_mode} | Max Clip={max_clip_str}")
    print(f"=========================================================================\n")

    music_duration = probe_duration(args.music)
    audio_grids = analyze_audio(args.music)
    
    if args.sync_mode in ["beat", "forward-beat"]:
        grid_times = audio_grids["beat"]
    elif args.sync_mode == "measure":
        grid_times = audio_grids["measure"]
    elif args.sync_mode == "onset":
        grid_times = audio_grids["onset"]
    elif args.sync_mode == "combined":
        grid_times = audio_grids["combined"]

    events = load_annotations(args.annotations)
    target = music_duration

    max_clip_val = args.max_clip if args.max_clip > 0 else None

    selected, final_duration = select_events_beat_synced(
        events, target, grid_times, sync_mode=args.sync_mode, max_clip=max_clip_val
    )

    with tempfile.TemporaryDirectory(prefix="mtb_autocut_") as temp:
        temp_dir = Path(temp)
        
        print(f"\nProcessing clips...")
        clip_files = extract_clips(
            selected, args.videos_dir, temp_dir, aspect_ratio=args.aspect_ratio
        )
        
        print("\nConcatenating clips...")
        silent_video = concatenate_clips(clip_files, temp_dir)

        print(f"\nApplying audio fade-out and exporting final video ({final_duration:.2f}s)...")
        add_music_with_fade(silent_video, args.music, final_duration, output_path)

    print("\nDONE")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()