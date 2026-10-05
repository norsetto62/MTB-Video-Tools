def convert_hms_to_s(value):
    """Convert MM:SS or seconds to seconds."""

    value = value.strip()

    if ":" in value:
        parts = value.split(":")

        if len(parts) == 2:
            minutes = float(parts[0])
            seconds = float(parts[1])
            if (minutes > 59.0) or (seconds > 59.0):
                    raise ValueError(
                        f"Minutes or seconds cannot be greater than 59.0"
                    )
            return minutes * 60.0 + seconds

        elif len(parts) == 3:
            hours = float(parts[0])
            minutes = float(parts[1])
            seconds = float(parts[2])
            if (minutes > 59.0) or (seconds > 59.0):
                    raise ValueError(
                        f"Minutes or seconds cannot be greater than 59.0"
                    )
            return hours * 3600.0 + minutes * 60.0 + seconds

    seconds = float(value)
    if (seconds > 59.9):
        return 0.0
    return seconds


def convert_s_to_hms(seconds):
    """Format seconds as MM:SS."""
    seconds = max(0.0, float(seconds))

    minutes = int(seconds // 60)
    secs = int(seconds % 60)

    return f"{minutes:02d}:{secs:02d}"