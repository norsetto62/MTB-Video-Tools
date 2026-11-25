"""Tests for timestamp-based telemetry resampling in overlay_widgets."""

import sys
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from overlay_widgets import (  # noqa: E402
    interpolate_series,
    output_timestamps,
    resample_records,
)


def test_output_timestamps_uses_requested_fps_and_includes_start():
    times = output_timestamps(2.0, 3.0, 10.0)

    assert len(times) == 11
    assert times[0] == 2.0
    assert times[-1] == 3.0
    assert times[1] == 2.1


def test_resample_records_interpolates_numeric_and_keeps_nearest_discrete():
    samples = [
        {"timestamp": 0.0, "speed": 10.0, "mode": "eco"},
        {"timestamp": 1.0, "speed": 20.0, "mode": "turbo"},
    ]

    result = resample_records(
        samples,
        [0.0, 1.0],
        [0.0, 0.25, 0.5, 0.75, 1.0],
        numeric_fields=("speed",),
    )

    assert [item["speed"] for item in result] == [10.0, 12.5, 15.0, 17.5, 20.0]
    assert [item["mode"] for item in result] == [
        "eco", "eco", "eco", "turbo", "turbo"
    ]


def test_interpolate_series_uses_shortest_angular_path():
    value = interpolate_series([0.0, 1.0], [179.0, -179.0], 0.5, angular=True)

    assert abs(abs(value) - 180.0) < 1e-9


def test_interpolate_series_holds_endpoint_values():
    assert interpolate_series([1.0, 2.0], [5.0, 7.0], 0.0) == 5.0
    assert interpolate_series([1.0, 2.0], [5.0, 7.0], 3.0) == 7.0
