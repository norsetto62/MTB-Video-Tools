from pathlib import Path
from .utils import convert_hms_to_s
from .config import min_clip

def load_annotations(annotation_file):
    """
    Load annotation file.

    Supported formats:

    Standalone audio filename (optional):
        D:\\path\\audio.mp3 *

    Standalone video filename:
        D:\\path\\video.mp4

    Followed by intervals:
        00:00   01:19   *

    Or full rows:
        D:\\path\\video.mp4   00:00   01:19   *
    """

    annotation_file = Path(annotation_file)

    clips = []
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
                maxsplit=4
            )

            if not parts:
                continue

            num_parts = len(parts)

            # ---------------------------------------------------------------
            # Audio filename
            # ---------------------------------------------------------------
            
            mix = False
            if (
                num_parts == 1
                and (
                    parts[0].lower().endswith(".mp3")
                    or parts[0].lower().endswith(".wav")
                    or parts[0].lower().endswith(".ogg")
                    or parts[0].lower().endswith(".flac")
                )
            ):
                audio_name = parts[0]
                audio_config = {"audio_filename}":audio_name,
                                "mix":mix,}
                continue

            if (
                num_parts == 2
                and (
                    parts[0].lower().endswith(".mp3")
                    or parts[0].lower().endswith(".wav")
                    or parts[0].lower().endswith(".ogg")
                    or parts[0].lower().endswith(".flac")
                )
            ):
                audio_name = parts[0]
                if parts[1].strip() == "*":
                    mix = True
                audio_config = {"audio_filename}":audio_name,
                                "mix":mix,}
                continue

            # ---------------------------------------------------------------
            # Standalone video filename
            # ---------------------------------------------------------------

            if (
                num_parts == 1
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
            # 00:00  01:19  *
            # ---------------------------------------------------------------

            if (
                ":" in parts[0]
                or parts[0].replace(".", "", 1).isdigit()
            ):

                if last_video_name is None:
                    raise ValueError(
                        f"First row in annotations omits video filename: '{line}'"
                    )

                if num_parts < 2:
                    raise ValueError(
                        f"Invalid annotation row: '{line}'"
                    )

                video_name = last_video_name

                start = convert_hms_to_s(parts[0])
                end = convert_hms_to_s(parts[1])

                mandatory = False
                if num_parts == 3:
                    if parts[2].strip() == "*":
                        mandatory = True

            # ---------------------------------------------------------------
            # Case 2:
            # Video name explicitly provided on same row.
            #
            # Example:
            # video.mp4  00:00  01:19  *
            # ---------------------------------------------------------------

            else:

                if len(parts) < 3:
                    raise ValueError(
                        f"Invalid annotation row: '{line}'"
                    )

                video_name = parts[0]
                last_video_name = video_name

                start = convert_hms_to_s(parts[1])
                end = convert_hms_to_s(parts[2])

                mandatory = False
                if num_parts == 4:
                    if parts[3].strip() == "*":
                        mandatory = True

            # ---------------------------------------------------------------
            # Validate
            # ---------------------------------------------------------------

            if end <= start:
                print(
                    f"WARNING: ignoring invalid interval: {line}"
                )
                continue

            duration = end - start

            if duration < min_clip:
                print(
                    f"WARNING: ignoring very short interval: {line}"
                )
                continue

            clips.append(
                {
                    "video_name": video_name,
                    "start": start,
                    "end": end,
                    "duration": duration,
                    "mandatory": mandatory,
                }
            )

    return {audio_config, clips}