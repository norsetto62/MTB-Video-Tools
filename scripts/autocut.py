"""
Video-AutoCut v1.0 (Multi-Video Support)

Automatically creates video edits from:
- One or more source videos
- An annotation file with MTB / filmmaking ratings
- A music track

The annotation format supports:

Start    End   MTB   Video   Remarks
D:\\path\\video.mp4
00:00   01:19   4   2   Start of track
01:51   02:06   4   2   Tight switchback

or:

D:\\path\\video.mp4  00:00  01:19  4  2  Description

The standalone video filename applies to following intervals
until another video filename is encountered.
"""

import argparse
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path
import librosa
import numpy as np
import shutil


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

FFMPEG = shutil.which("ffmpeg") or r"C:\Program Files\ffmpeg\bin\ffmpeg.exe"
FFPROBE = shutil.which("ffprobe") or r"C:\Program Files\ffmpeg\bin\ffprobe.exe"

MIN_CLIP = 3.0
FADE_OUT_DURATION = 3.0


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def parse_time(value):
    """Convert MM:SS or seconds to seconds."""
    value = value.strip()

    if ":" in value:
        parts = value.split(":")

        if len(parts) == 2:
            minutes = float(parts[0])
            seconds = float(parts[1])
            return minutes * 60.0 + seconds

        elif len(parts) == 3:
            hours = float(parts[0])
            minutes = float(parts[1])
            seconds = float(parts[2])
            return hours * 3600.0 + minutes * 60.0 + seconds

    return float(value)


def format_time(seconds):
    """Format seconds as MM:SS."""
    seconds = max(0.0, float(seconds))

    minutes = int(seconds // 60)
    secs = int(seconds % 60)

    return f"{minutes:02d}:{secs:02d}"


def probe_duration(video_path):
    """Return media duration in seconds."""
    cmd = [
        FFPROBE,
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


# ---------------------------------------------------------------------------
# Audio analysis
# ---------------------------------------------------------------------------

AUDIO_CACHE_VERSION = 1


def music_file_hash(music_path):
    """Return a SHA-256 hash of the music file."""
    digest = hashlib.sha256()

    with Path(music_path).open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def audio_cache_path(music_path, cache_dir):
    """Return the cache filename for a music file."""
    music_path = Path(music_path)
    cache_dir = Path(cache_dir)

    path_key = hashlib.sha256(
        str(music_path.resolve()).encode("utf-8")
    ).hexdigest()[:12]

    return (
        cache_dir
        / f"{music_path.stem}_{path_key}.json"
    )


def load_cached_audio(music_path, cache_dir):
    """Load cached audio analysis if it matches the current music file."""
    cache_file = audio_cache_path(
        music_path,
        cache_dir,
    )

    if not cache_file.exists():
        return None

    try:
        source_hash = music_file_hash(music_path)

        with cache_file.open(
            "r",
            encoding="utf-8",
        ) as f:
            cached = json.load(f)

        if cached.get("cache_version") != AUDIO_CACHE_VERSION:
            return None

        if cached.get("source_hash") != source_hash:
            return None

        audio_data = {
            "duration": float(cached["duration"]),
            "tempo": float(cached["tempo"]),
            "beats": np.asarray(
                cached["beats"],
                dtype=float,
            ),
            "measures": np.asarray(
                cached["measures"],
                dtype=float,
            ),
            "onsets": np.asarray(
                cached["onsets"],
                dtype=float,
            ),
            "combined": np.asarray(
                cached["combined"],
                dtype=float,
            ),
        }

        print("Using cached audio analysis:")
        print(
            f"  Duration: {audio_data['duration']:.2f}s"
        )
        print(
            f"  Tempo: {audio_data['tempo']:.1f} BPM"
        )
        print(
            f"  Beats: {len(audio_data['beats'])}"
        )
        print(
            f"  Measures: {len(audio_data['measures'])}"
        )
        print(
            f"  Energy peaks: {len(audio_data['onsets'])}"
        )

        return audio_data

    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        json.JSONDecodeError,
    ):
        return None


def save_cached_audio(
    music_path,
    cache_dir,
    audio_data,
):
    """Save audio analysis results to the JSON cache."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    cache_file = audio_cache_path(
        music_path,
        cache_dir,
    )

    payload = {
        "cache_version": AUDIO_CACHE_VERSION,
        "source_hash": music_file_hash(music_path),
        "source": str(Path(music_path).resolve()),
        "duration": float(audio_data["duration"]),
        "tempo": float(audio_data["tempo"]),
        "beats": audio_data["beats"].tolist(),
        "measures": audio_data["measures"].tolist(),
        "onsets": audio_data["onsets"].tolist(),
        "combined": audio_data["combined"].tolist(),
    }

    with cache_file.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            payload,
            f,
            indent=2,
        )

    print(
        f"  Saved audio analysis cache: {cache_file}"
    )


def analyze_audio(
    music_path,
    cache_dir=None,
):
    """
    Analyze music and return synchronization points.

    If cache_dir is provided, reuse a matching cached analysis.

    Returns:
        dict containing:
            duration
            tempo
            beats
            onsets
            combined
    """

    if cache_dir is not None:
        cached = load_cached_audio(
            music_path,
            cache_dir,
        )

        if cached is not None:
            return cached

    print("Analyzing audio rhythm, beats, and energy peaks...")

    y, sr = librosa.load(
        str(music_path),
        sr=None,
        mono=True,
    )

    duration = len(y) / sr

    # Beat tracking
    tempo, beat_frames = librosa.beat.beat_track(
        y=y,
        sr=sr,
    )

    tempo = float(
        np.asarray(tempo).reshape(-1)[0]
    )

    beat_times = librosa.frames_to_time(
        beat_frames,
        sr=sr,
    )

    # Use every 4th beat as a rough musical measure grid
    measure_times = beat_times[::4]

    # Onset strength
    onset_strength = librosa.onset.onset_strength(
        y=y,
        sr=sr,
    )

    onset_times = librosa.frames_to_time(
        np.arange(len(onset_strength)),
        sr=sr,
    )

    # Peak picking
    peaks = librosa.util.peak_pick(
        onset_strength,
        pre_max=3,
        post_max=3,
        pre_avg=3,
        post_avg=5,
        delta=0.2,
        wait=5,
    )

    peak_times = onset_times[peaks]

    # Combine beats and energy peaks
    combined = np.concatenate(
        [
            beat_times,
            peak_times,
        ]
    )

    combined = np.unique(
        np.round(combined, 3)
    )

    combined = combined[
        (combined >= 0)
        & (combined <= duration)
    ]

    print(f"  Duration: {duration:.2f}s")
    print(f"  Tempo: {tempo:.1f} BPM")
    print(f"  Beats: {len(beat_times)}")
    print(f"  Measures: {len(measure_times)}")
    print(f"  Energy peaks: {len(peak_times)}")

    audio_data = {
        "duration": duration,
        "tempo": tempo,
        "beats": beat_times,
        "measures": measure_times,
        "onsets": peak_times,
        "combined": combined,
    }

    if cache_dir is not None:
        save_cached_audio(
            music_path,
            cache_dir,
            audio_data,
        )

    return audio_data

    """
    Analyze music and return synchronization points.

    Returns:
        dict containing:
            duration
            beats
            onsets
            combined
    """

    print("Analyzing audio rhythm, beats, and energy peaks...")

    y, sr = librosa.load(
        str(music_path),
        sr=None,
        mono=True,
    )

    duration = len(y) / sr

    # Beat tracking
    tempo, beat_frames = librosa.beat.beat_track(
        y=y,
        sr=sr,
    )

    beat_times = librosa.frames_to_time(
        beat_frames,
        sr=sr,
    )

    # Use every 4th beat as a rough musical measure grid
    measure_times = beat_times[::4]

    # Onset strength
    onset_strength = librosa.onset.onset_strength(
        y=y,
        sr=sr,
    )

    onset_times = librosa.frames_to_time(
        np.arange(len(onset_strength)),
        sr=sr,
    )

    # Peak picking
    peaks = librosa.util.peak_pick(
        onset_strength,
        pre_max=3,
        post_max=3,
        pre_avg=3,
        post_avg=5,
        delta=0.2,
        wait=5,
    )

    peak_times = onset_times[peaks]

    # Combine beats and energy peaks
    combined = np.concatenate(
        [
            beat_times,
            peak_times,
        ]
    )

    combined = np.unique(
        np.round(combined, 3)
    )

    combined = combined[
        (combined >= 0)
        & (combined <= duration)
    ]

    print(f"  Duration: {duration:.2f}s")
    print(f"  Tempo: {float(np.asarray(tempo).reshape(-1)[0]):.1f} BPM")
    print(f"  Beats: {len(beat_times)}")
    print(f"  Measures: {len(measure_times)}")
    print(f"  Energy peaks: {len(peak_times)}")

    return {
        "duration": duration,
        "beats": beat_times,
        "measures": measure_times,
        "onsets": peak_times,
        "combined": combined,
    }


def snap_timestamp(timestamp, audio_data, mode="beat"):
    """Snap a timestamp to an audio synchronization grid."""

    if mode == "forward-beat":
        grid = audio_data["beats"]
        candidates = grid[grid >= timestamp]

        if len(candidates) == 0:
            return timestamp

        return float(candidates[0])

    elif mode == "measure":
        grid = audio_data["measures"]

    elif mode == "onset":
        grid = audio_data["onsets"]

    elif mode == "combined":
        grid = audio_data["combined"]

    else:
        grid = audio_data["beats"]

    if len(grid) == 0:
        return timestamp

    index = np.argmin(
        np.abs(grid - timestamp)
    )

    return float(grid[index])


# ---------------------------------------------------------------------------
# Annotation loading
# ---------------------------------------------------------------------------

def load_annotations(annotation_file):
    """
    Load annotation file.

    Supported formats:

    Header:
        Start    End   MTB   Video   Remarks

    Standalone video filename:
        D:\\path\\video.mp4

    Followed by intervals:
        00:00   01:19   4   2   Description

    Or full rows:
        D:\\path\\video.mp4   00:00   01:19   4   2   Description
    """

    annotation_file = Path(annotation_file)

    events = []
    last_video_name = None

    with annotation_file.open(
        "r",
        encoding="utf-8-sig",
    ) as f:

        for raw_line in f:
            line = raw_line.strip()

            # Ignore empty lines
            if not line:
                continue

            # Ignore comments
            if line.startswith("#"):
                continue

            parts = line.split(
                maxsplit=5
            )

            if not parts:
                continue

            # ---------------------------------------------------------------
            # Skip column header
            # ---------------------------------------------------------------

            if parts[0].lower() == "start":
                continue

            # ---------------------------------------------------------------
            # Standalone video filename
            # ---------------------------------------------------------------

            if (
                len(parts) == 1
                and (
                    parts[0].lower().endswith(".mp4")
                    or parts[0].lower().endswith(".mov")
                    or parts[0].lower().endswith(".mkv")
                    or parts[0].lower().endswith(".avi")
                )
            ):
                last_video_name = parts[0]
                continue

            # ---------------------------------------------------------------
            # Case 1:
            # Video name omitted.
            #
            # Example:
            # 00:00  01:19  4  2  Description
            # ---------------------------------------------------------------

            if (
                ":" in parts[0]
                or parts[0].replace(".", "", 1).isdigit()
            ):

                if last_video_name is None:
                    raise ValueError(
                        f"First row in annotations omits video filename: '{line}'"
                    )

                if len(parts) < 4:
                    raise ValueError(
                        f"Invalid annotation row: '{line}'"
                    )

                video_name = last_video_name

                start = parse_time(parts[0])
                end = parse_time(parts[1])
                mtb = int(parts[2])
                film = int(parts[3])

                description = (
                    parts[4]
                    if len(parts) > 4
                    else ""
                )

            # ---------------------------------------------------------------
            # Case 2:
            # Video name explicitly provided on same row.
            #
            # Example:
            # video.mp4  00:00  01:19  4  2  Description
            # ---------------------------------------------------------------

            else:

                if len(parts) < 5:
                    raise ValueError(
                        f"Invalid annotation row: '{line}'"
                    )

                video_name = parts[0]
                last_video_name = video_name

                start = parse_time(parts[1])
                end = parse_time(parts[2])
                mtb = int(parts[3])
                film = int(parts[4])

                description = (
                    parts[5]
                    if len(parts) > 5
                    else ""
                )

            # ---------------------------------------------------------------
            # Validate
            # ---------------------------------------------------------------

            if end <= start:
                print(
                    f"WARNING: ignoring invalid interval: {line}"
                )
                continue

            duration = end - start

            if duration < MIN_CLIP:
                print(
                    f"WARNING: ignoring very short interval: {line}"
                )
                continue

            events.append(
                {
                    "video_name": video_name,
                    "start": start,
                    "end": end,
                    "duration": duration,
                    "mtb": mtb,
                    "film": film,
                    "description": description,
                }
            )

    return events


# ---------------------------------------------------------------------------
# Event selection
# ---------------------------------------------------------------------------

def select_events_beat_synced(
    events,
    audio_data,
    sync_mode="beat",
    max_clip=10.0,
):
    """
    Select annotation events until the music duration is filled.

    Priority:
        MTB rating x 2 + filmmaking rating

    At most one clip is selected from each annotation event.

    max_clip:
        Maximum clip duration.
        0 means unlimited.
    """

    music_duration = audio_data["duration"]

    ranked = []

    for event in events:

        score = (
            event["mtb"] * 2
            + event["film"]
        )

        ranked.append(
            (
                score,
                event["mtb"],
                event["film"],
                event,
            )
        )

    ranked.sort(
        key=lambda x: (
            x[0],
            x[1],
            x[2],
        ),
        reverse=True,
    )

    selected = []
    total_duration = 0.0

    for score, mtb, film, event in ranked:

        if total_duration >= music_duration:
            break

        available = event["duration"]

        if max_clip > 0:
            clip_duration = min(
                available,
                max_clip,
            )
        else:
            clip_duration = available

        clip_duration = min(
            clip_duration,
            music_duration - total_duration,
        )

        if clip_duration < MIN_CLIP:
            continue

        start = event["start"]
        end = start + clip_duration

        # Snap the end to the requested audio grid.
        snapped_end = snap_timestamp(
            end,
            audio_data,
            mode=sync_mode,
        )

        minimum_end = start + MIN_CLIP

        if snapped_end >= minimum_end:
            end = min(
                snapped_end,
                event["end"],
                start + clip_duration + 2.0,
            )

        final_duration = end - start

        if final_duration < MIN_CLIP:
            continue

        selected.append(
            {
                "video_name": event["video_name"],
                "start": start,
                "end": end,
                "duration": final_duration,
                "mtb": event["mtb"],
                "film": event["film"],
                "description": event["description"],
                "score": score,
            }
        )

        total_duration += final_duration

    return selected


# ---------------------------------------------------------------------------
# Video processing
# ---------------------------------------------------------------------------

def extract_clips(
    selected_events,
    videos_dir,
    temp_dir,
    aspect_ratio="landscape",
    verbose=False,
):
    """Extract individual video clips."""

    clip_files = []

    videos_dir = Path(videos_dir)

    for index, event in enumerate(selected_events):

        video_path = Path(
            event["video_name"]
        )

        # If annotation contains only a filename,
        # resolve it relative to videos_dir.
        if not video_path.is_absolute():
            video_path = videos_dir / video_path.name

        if not video_path.exists():
            candidate = videos_dir / video_path.name

            if candidate.exists():
                video_path = candidate
            else:
                print(
                    f"WARNING: video not found: {video_path}"
                )
                continue

        duration = event["end"] - event["start"]

        output_file = (
            Path(temp_dir)
            / f"clip_{index:04d}.ts"
        )

        print(
            f"  Clip {index + 1:02d}: "
            f"{format_time(event['start'])} - "
            f"{format_time(event['end'])} "
            f"({duration:.1f}s) "
            f"MTB={event['mtb']} "
            f"Film={event['film']} "
            f"{event['description']}"
        )

        cmd = [
            FFMPEG,
            "-y",
            "-ss",
            str(event["start"]),
            "-i",
            str(video_path),
            "-t",
            str(duration),
        ]

        # Portrait conversion if requested.
        if aspect_ratio == "portrait":
            cmd += [
                "-vf",
                "crop=ih*9/16:ih",
            ]

        cmd += [
            "-an",
            "-c:v",
            "h264_nvenc",
            "-preset",
            "p4",
            "-cq",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-f",
            "mpegts",
            str(output_file),
        ]

        subprocess.run(
            cmd,
            check=True,
            stdout=None if verbose else subprocess.DEVNULL,
            stderr=None if verbose else subprocess.DEVNULL,
        )

        clip_files.append(output_file)

    return clip_files


def concatenate_clips(
    clip_files,
    output_file,
    verbose=False,
):
    """Concatenate MPEG-TS clips without re-encoding."""

    concat_file = (
        Path(output_file).with_suffix(".txt")
    )

    with concat_file.open(
        "w",
        encoding="utf-8",
    ) as f:

        for clip in clip_files:
            f.write(
                f"file '{clip.resolve()}'\n"
            )

    cmd = [
        FFMPEG,
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_file),
        "-c",
        "copy",
        str(output_file),
    ]

    subprocess.run(
        cmd,
        check=True,
        stdout=None if verbose else subprocess.DEVNULL,
        stderr=None if verbose else subprocess.DEVNULL,
    )

    concat_file.unlink(
        missing_ok=True
    )


def add_music_with_fade(
    video_file,
    music_file,
    output_file,
    total_duration,
    verbose=False,
):
    """Add music and fade it out at the end."""

    fade_start = max(
        0.0,
        total_duration - FADE_OUT_DURATION,
    )

    audio_filter = (
        f"afade=t=out:"
        f"st={fade_start}:"
        f"d={FADE_OUT_DURATION}"
    )

    cmd = [
        FFMPEG,
        "-y",
        "-i",
        str(video_file),
        "-i",
        str(music_file),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        "320k",
        "-af",
        audio_filter,
        "-t",
        str(total_duration),
        "-movflags",
        "+faststart",
        str(output_file),
    ]

    subprocess.run(
        cmd,
        check=True,
        stdout=None if verbose else subprocess.DEVNULL,
        stderr=None if verbose else subprocess.DEVNULL,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description="Video-AutoCut v1.0"
    )

    parser.add_argument(
        "videos_dir",
        help="Directory containing source videos",
    )

    parser.add_argument(
        "music",
        help="Music file",
    )

    parser.add_argument(
        "annotations",
        help="Annotation file",
    )

    parser.add_argument(
        "-o",
        "--output-dir",
        default="./output",
        help="Output directory",
    )

    parser.add_argument(
        "-ar",
        "--aspect-ratio",
        choices=[
            "landscape",
            "portrait",
        ],
        default="landscape",
    )

    parser.add_argument(
        "-sm",
        "--sync-mode",
        choices=[
            "beat",
            "measure",
            "forward-beat",
            "onset",
            "combined",
        ],
        default="beat",
    )

    parser.add_argument(
        "-mc",
        "--max-clip",
        type=float,
        default=10.0,
        help="Maximum clip duration. 0 = unlimited.",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show FFmpeg console output.",
    )

    args = parser.parse_args()

    videos_dir = Path(
        args.videos_dir
    )

    music_file = Path(
        args.music
    )

    annotation_file = Path(
        args.annotations
    )

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print(
        "Video-AutoCut v1.0 (Multi-Video Support)"
    )
    print(
        "========================================================================="
    )
    print(
        f"Videos Directory: {videos_dir}"
    )
    print(
        f"Music File:       {music_file}"
    )
    print(
        f"Annotations File: {annotation_file}"
    )
    print(
        f"Settings:         "
        f"AR={args.aspect_ratio.upper()} | "
        f"Sync={args.sync_mode} | "
        f"Max Clip={args.max_clip:.1f}s"
    )
    print(
        "========================================================================="
    )
    print()

    # -----------------------------------------------------------------------
    # Analyze audio
    # -----------------------------------------------------------------------

    audio_cache_dir = output_dir / "audio_cache"

    audio_data = analyze_audio(
        music_file,
        cache_dir=audio_cache_dir,
    )

    # -----------------------------------------------------------------------
    # Load annotations
    # -----------------------------------------------------------------------

    events = load_annotations(
        annotation_file
    )

    print(
        f"Loaded {len(events)} annotation events."
    )

    if not events:
        raise RuntimeError(
            "No valid annotation events found."
        )

    # -----------------------------------------------------------------------
    # Select events
    # -----------------------------------------------------------------------

    selected = select_events_beat_synced(
        events,
        audio_data,
        sync_mode=args.sync_mode,
        max_clip=args.max_clip,
    )

    if not selected:
        raise RuntimeError(
            "No suitable events selected."
        )

    total_duration = sum(
        event["duration"]
        for event in selected
    )

    print()
    print(
        f"Selected {len(selected)} clips."
    )
    print(
        f"Total footage: {total_duration:.2f}s"
    )
    print(
        f"Music duration: {audio_data['duration']:.2f}s"
    )
    print()

    # -----------------------------------------------------------------------
    # Temporary working directory
    # -----------------------------------------------------------------------

    with tempfile.TemporaryDirectory(
        prefix="mtb_autocut_"
    ) as temp_dir:

        print("Extracting clips...")
        print()

        clip_files = extract_clips(
            selected,
            videos_dir,
            temp_dir,
            aspect_ratio=args.aspect_ratio,
            verbose=args.verbose,
        )

        if not clip_files:
            raise RuntimeError(
                "No clips were successfully extracted."
            )

        # -------------------------------------------------------------------
        # Concatenate
        # -------------------------------------------------------------------

        temp_video = (
            Path(temp_dir)
            / "video_concat.ts"
        )

        print()
        print("Concatenating clips...")

        concatenate_clips(
            clip_files,
            temp_video,
            verbose=args.verbose,
        )

        # Recalculate actual duration from the selected clips.
        actual_duration = sum(
            selected[i]["duration"]
            for i in range(
                min(
                    len(selected),
                    len(clip_files),
                )
            )
        )

        # Don't exceed music duration.
        actual_duration = min(
            actual_duration,
            audio_data["duration"],
        )

        # -------------------------------------------------------------------
        # Add music
        # -------------------------------------------------------------------

        output_file = (
            output_dir
            / "autocut.mp4"
        )

        print()
        print("Adding music...")

        add_music_with_fade(
            temp_video,
            music_file,
            output_file,
            actual_duration,
            verbose=args.verbose,
        )

    print()
    print(
        "========================================================================="
    )
    print("DONE")
    print(
        f"Output: {output_file}"
    )
    print(
        f"Duration: {actual_duration:.2f}s"
    )
    print(
        "========================================================================="
    )


if __name__ == "__main__":
    main()