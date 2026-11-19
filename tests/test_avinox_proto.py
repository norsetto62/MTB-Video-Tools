"""Tests for the Avinox .proto parser using synthetic binary fixtures."""

import struct

import pytest

from autocut.dataset.avinox_proto import HEADER_SIZE, MAGIC, FRAME_TYPE, parse


def _varint(value: int) -> bytes:
    value &= (1 << 64) - 1
    out = bytearray()
    while value > 0x7F:
        out.append((value & 0x7F) | 0x80)
        value >>= 7
    out.append(value)
    return bytes(out)


def _field_varint(number: int, value: int) -> bytes:
    return _varint(number << 3) + _varint(value)


def _field_fixed32(number: int, value: float) -> bytes:
    return _varint((number << 3) | 5) + struct.pack("<f", value)


def _field_fixed64(number: int, value: float) -> bytes:
    return _varint((number << 3) | 1) + struct.pack("<d", value)


def _build_proto(fields: dict[int, tuple[str, float | int]], *, ride_id=42,
                 start=1_750_000_000, end=1_750_000_000, ascent=600.0):
    payload = bytearray()
    for number, (kind, value) in fields.items():
        if kind == "varint":
            payload.extend(_field_varint(number, int(value)))
        elif kind == "fixed32":
            payload.extend(_field_fixed32(number, float(value)))
        elif kind == "fixed64":
            payload.extend(_field_fixed64(number, float(value)))
        else:
            raise AssertionError(kind)

    header = bytearray(HEADER_SIZE)
    struct.pack_into("<I", header, 0, MAGIC)
    struct.pack_into("<I", header, 6, ride_id)
    struct.pack_into("<II", header, 10, start, end)
    struct.pack_into("<ff", header, 22, ascent, 12.0)
    header[110:118] = b"AVINOX01"
    frame = struct.pack("<HH", FRAME_TYPE, len(payload)) + payload + b"\0\0"
    return bytes(header) + frame


BASE = {
    72: ("varint", 1_750_000_000),
    7: ("varint", 1850),
    8: ("varint", 2),
    9: ("varint", 7500),
    13: ("varint", 15000),
    14: ("varint", 22000),
    23: ("varint", 1250),
    31: ("fixed64", 46.1234567),
    32: ("fixed64", 11.7654321),
    39: ("varint", 51234),
    41: ("varint", 2150),
    71: ("varint", 87),
    36: ("fixed32", 0.125),
    37: ("fixed32", -0.25),
    38: ("fixed32", 0.75),
}


def test_parse_sample_scaling_gps_and_imu():
    result = parse(_build_proto(BASE), "ride.proto")
    assert result["metadata"]["rideId"] == 42
    assert result["metadata"]["ascent"] == pytest.approx(600)
    assert result["metadata"]["serial"] == "AVINOX01"
    sample = result["samples"][0]
    assert sample["speed"] == pytest.approx(18.5)
    assert sample["cadence"] == pytest.approx(75)
    assert sample["motorPower"] == pytest.approx(220)
    assert sample["distanceKm"] == pytest.approx(1.25)
    assert sample["latitude"] == pytest.approx(46.1234567)
    assert sample["longitude"] == pytest.approx(11.7654321)
    assert sample["altitude"] == pytest.approx(512.34)
    assert sample["temperature"] == pytest.approx(21.5)
    assert sample["battery"] == 87
    assert sample["imuX"] == pytest.approx(0.125)
    assert sample["imuY"] == pytest.approx(-0.25)
    assert sample["imuZ"] == pytest.approx(0.75)


def test_missing_altitude_and_temperature_are_null():
    fields = {k: v for k, v in BASE.items() if k not in (39, 41)}
    sample = parse(_build_proto(fields))["samples"][0]
    assert sample["altitude"] is None
    assert sample["temperature"] is None


def test_negative_altitude_is_preserved():
    fields = dict(BASE)
    fields[39] = ("varint", -750)
    assert parse(_build_proto(fields))["samples"][0]["altitude"] == pytest.approx(-7.5)


def test_missing_gps_is_null():
    fields = {k: v for k, v in BASE.items() if k not in (31, 32)}
    sample = parse(_build_proto(fields))["samples"][0]
    assert sample["latitude"] is None
    assert sample["longitude"] is None


def test_path_input_and_invalid_file(tmp_path):
    path = tmp_path / "ride.proto"
    path.write_bytes(_build_proto(BASE))
    assert parse(path)["metadata"]["fileName"] == "ride.proto"
    with pytest.raises(ValueError, match="not an Avinox"):
        parse(bytes(300))


def test_truncated_frame_is_rejected():
    with pytest.raises(ValueError, match="Truncated Avinox frame"):
        parse(bytes(bytearray(_build_proto(BASE)[:-3])))


def test_unknown_frame_type_is_rejected():
    data = bytearray(_build_proto(BASE))
    struct.pack_into("<H", data, HEADER_SIZE, 0x1234)
    with pytest.raises(ValueError, match="Unknown frame type"):
        parse(data)
