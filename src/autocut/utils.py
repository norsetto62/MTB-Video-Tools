"""Small domain-independent utility functions."""

import math


def convert_hms_to_s(value: str | int | float) -> float:
    """Convert MM:SS, HH:MM:SS or numeric seconds to seconds.

    Minutes and seconds must be in the range 0..59.
    Hours may be any non-negative value.
    Fractional seconds are allowed.
    """
    if isinstance(value, (int, float)):
        seconds = float(value)

        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError(f"Invalid seconds value: {value!r}")

        return seconds

    value = value.strip()

    if not value:
        raise ValueError("Empty time value")

    if ":" not in value:
        try:
            seconds = float(value)
        except ValueError as exc:
            raise ValueError(f"Invalid time value: {value!r}") from exc

        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError(f"Invalid seconds value: {value!r}")

        return seconds

    parts = value.split(":")

    if len(parts) not in (2, 3):
        raise ValueError(f"Invalid time format: {value!r}")

    try:
        numbers = [float(part) for part in parts]
    except ValueError as exc:
        raise ValueError(f"Invalid time value: {value!r}") from exc

    if any(not math.isfinite(part) or part < 0 for part in numbers):
        raise ValueError(f"Invalid time value: {value!r}")

    if len(parts) == 2:
        minutes, seconds = numbers

        if seconds >= 60:
            raise ValueError(
                f"Seconds must be less than 60: {value!r}"
            )

        return minutes * 60 + seconds
    
    hours, minutes, seconds = numbers

    if minutes >= 60:
        raise ValueError(
            f"Minutes must be less than 60: {value!r}"
        )

    if seconds >= 60:
        raise ValueError(
            f"Seconds must be less than 60: {value!r}"
        )

    return hours * 3600 + minutes * 60 + seconds


def convert_s_to_hms(seconds: float) -> str:
    """Format seconds as HH:MM:SS, or MM:SS when under one hour."""
    seconds = float(seconds)

    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError(f"Invalid seconds value: {seconds!r}")

    total_seconds = int(seconds)

    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)

    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    return f"{minutes:02d}:{secs:02d}"