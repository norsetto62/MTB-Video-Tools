import math

import pytest

from autocut.utils import convert_hms_to_s, convert_s_to_hms


# ---------------------------------------------------------------------------
# convert_hms_to_s
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("00:00", 0.0),
        ("01:23", 83.0),
        ("59:59", 3599.0),
        ("60:00", 3600.0),
        ("90:15", 5415.0),
        ("120:00", 7200.0),
        ("00:00:00", 0.0),
        ("01:02:03", 3723.0),
        ("12:34:56", 45296.0),
        ("01:02:03.5", 3723.5),
        (0, 0.0),
        (83, 83.0),
        (83.5, 83.5),
        ("83", 83.0),
        ("83.5", 83.5),
    ],
)
def test_convert_hms_to_s_valid(value, expected):
    assert convert_hms_to_s(value) == pytest.approx(expected)


@pytest.mark.parametrize(
    "value",
    [
        "",
        "abc",
        "1:2:3:4",
        "72:83",
        "01:60",
        "01:72:00",
        "01:02:60",
        "-01:00",
        "-1",
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_convert_hms_to_s_invalid(value):
    with pytest.raises(ValueError):
        convert_hms_to_s(value)


def test_convert_hms_to_s_allows_hours_above_23():
    # Video durations are not clock times.
    assert convert_hms_to_s("25:00:00") == 90000.0


# ---------------------------------------------------------------------------
# convert_s_to_hms
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [
        (0, "00:00"),
        (1, "00:01"),
        (59, "00:59"),
        (60, "01:00"),
        (83, "01:23"),
        (3599, "59:59"),
        (3600, "01:00:00"),
        (3723, "01:02:03"),
        (45296, "12:34:56"),
        (90000, "25:00:00"),
    ],
)
def test_convert_s_to_hms_valid(seconds, expected):
    assert convert_s_to_hms(seconds) == expected


@pytest.mark.parametrize(
    "seconds",
    [
        -1,
        -0.5,
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_convert_s_to_hms_invalid(seconds):
    with pytest.raises(ValueError):
        convert_s_to_hms(seconds)


def test_convert_s_to_hms_discards_fractional_part():
    assert convert_s_to_hms(83.9) == "01:23"

def test_convert_hms_to_s_allows_large_minutes():
    assert convert_hms_to_s("90:15") == 5415.0
    assert convert_hms_to_s("120:00") == 7200.0


def test_convert_hms_to_s_rejects_seconds_overflow_in_two_part_format():
    with pytest.raises(ValueError):
        convert_hms_to_s("01:60")