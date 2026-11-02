"""Parser for Avinox cloud ride .proto recordings.

The format is community reverse-engineered, not officially documented by DJI.
This module preserves the field mapping and scaling used by the reference
JavaScript parser. IMU channels are exposed as raw float32 values; their
physical units are unknown.
"""
from __future__ import annotations

from datetime import datetime, timezone
import struct
from pathlib import Path
from typing import Any

HEADER_SIZE = 251
MAGIC = 0xA5A5A5A5
FRAME_TYPE = 0x02EB


def _varint(data: memoryview, pos: int) -> tuple[int, int]:
    """Decode an unsigned protobuf varint."""
    value = 0
    shift = 0
    while True:
        if pos >= len(data):
            raise ValueError("Invalid varint")
        byte = data[pos]
        pos += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, pos
        shift += 7
        if shift >= 70:
            raise ValueError("Invalid varint")


def _signed(value: int | None) -> int | None:
    """Interpret an integer as signed 64-bit two's complement."""
    if value is None:
        return None
    value &= (1 << 64) - 1
    return value - (1 << 64) if value >= (1 << 63) else value


def _float32(bits: int | None) -> float | None:
    if bits is None:
        return None
    return struct.unpack("<f", struct.pack("<I", bits & 0xFFFFFFFF))[0]


def _float64(bits: int | None) -> float | None:
    if bits is None:
        return None
    return struct.unpack("<d", struct.pack("<Q", bits & 0xFFFFFFFFFFFFFFFF))[0]


def _message(payload: memoryview) -> dict[int, int | memoryview]:
    """Decode one protobuf message into a field-number/value mapping."""
    pos = 0
    fields: dict[int, int | memoryview] = {}
    while pos < len(payload):
        tag, pos = _varint(payload, pos)
        field_number, wire_type = tag >> 3, tag & 7
        if field_number == 0:
            raise ValueError("Invalid protobuf field number 0")
        if wire_type == 0:
            value, pos = _varint(payload, pos)
        elif wire_type == 1:
            if pos + 8 > len(payload):
                raise ValueError("Truncated protobuf fixed64 field")
            value = struct.unpack_from("<Q", payload, pos)[0]
            pos += 8
        elif wire_type == 2:
            length, pos = _varint(payload, pos)
            end = pos + length
            if end > len(payload):
                raise ValueError("Truncated protobuf length-delimited field")
            value = payload[pos:end]
            pos = end
        elif wire_type == 5:
            if pos + 4 > len(payload):
                raise ValueError("Truncated protobuf fixed32 field")
            value = struct.unpack_from("<I", payload, pos)[0]
            pos += 4
        else:
            raise ValueError(f"Unsupported protobuf wire type {wire_type}")
        fields[field_number] = value
    return fields


def _integer(value: int | memoryview | None) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int):
        raise ValueError("Expected an integer protobuf field")
    return value


def _scaled_signed(value: int | memoryview | None, divisor: float) -> float | None:
    integer = _integer(value)
    return None if integer is None else _signed(integer) / divisor


def _sample(fields: dict[int, int | memoryview]) -> dict[str, Any]:
    timestamp = _integer(fields.get(72))
    temperature = _signed(_integer(fields.get(41)))
    speed = _integer(fields.get(7))
    cadence = _integer(fields.get(9))
    motor_torque = _integer(fields.get(10))
    rider_torque = _integer(fields.get(11))
    total_torque = _integer(fields.get(12))
    rider_power = _integer(fields.get(13))
    motor_power = _integer(fields.get(14))
    total_power = _integer(fields.get(15))
    distance = _integer(fields.get(23))
    pressure = _integer(fields.get(42))
    rider_energy = _integer(fields.get(52))
    odometer = _integer(fields.get(73))

    return {
        "timestamp": timestamp,
        "date": datetime.fromtimestamp(timestamp, tz=timezone.utc) if timestamp else None,
        "speed": (speed / 100) if speed is not None and speed / 100 else 0,
        "assist": _integer(fields.get(8)),
        "cadence": (cadence / 100) if cadence is not None and cadence / 100 else 0,
        "motorTorque": (motor_torque / 100) if motor_torque is not None and motor_torque / 100 else 0,
        "riderTorque": (rider_torque / 100) if rider_torque is not None and rider_torque / 100 else 0,
        "totalTorque": (total_torque / 100) if total_torque is not None and total_torque / 100 else 0,
        "riderPower": (rider_power / 100) if rider_power is not None and rider_power / 100 else 0,
        "motorPower": (motor_power / 100) if motor_power is not None and motor_power / 100 else 0,
        "totalPower": (total_power / 100) if total_power is not None and total_power / 100 else 0,
        "gear": _integer(fields.get(22)),
        "imuX": _float32(_integer(fields.get(36))),
        "imuY": _float32(_integer(fields.get(37))),
        "imuZ": _float32(_integer(fields.get(38))),
        "distanceKm": (distance or 0) / 1000,
        "latitude": _float64(_integer(fields.get(31))),
        "longitude": _float64(_integer(fields.get(32))),
        "altitude": _scaled_signed(fields.get(39), 100),
        "gradient": _scaled_signed(fields.get(40), 100),
        "temperature": None if temperature is None or temperature == -99900 else temperature / 100,
        "pressure": (pressure / 100) if pressure is not None and pressure / 100 else None,
        "heartRate": _integer(fields.get(51)),
        "riderEnergyKj": (rider_energy or 0) / 1000,
        "battery": _integer(fields.get(71)),
        "odometerKm": (odometer or 0) / 1000,
        "event": _integer(fields.get(75)),
    }


def parse(source: bytes | bytearray | memoryview | str | Path, name: str | None = None) -> dict[str, Any]:
    """Parse an Avinox ride recording into metadata and decoded samples.

    Source may be raw bytes-like data or a filesystem path. The returned
    mapping has metadata and samples keys, matching the JS parser.
    """
    if isinstance(source, (str, Path)):
        path = Path(source)
        data = path.read_bytes()
        if name is None:
            name = path.name
    else:
        data = bytes(source)

    view = memoryview(data)
    if len(view) < HEADER_SIZE or struct.unpack_from("<I", view, 0)[0] != MAGIC:
        raise ValueError("This is not an Avinox cloud ride record.")

    start, end = struct.unpack_from("<II", view, 10)
    ride_id = struct.unpack_from("<I", view, 6)[0]
    ascent = struct.unpack_from("<f", view, 22)[0]
    descent = struct.unpack_from("<f", view, 26)[0]
    serial = bytes(view[110:128]).split(b"\0", 1)[0].decode("latin-1")

    metadata: dict[str, Any] = {
        "fileName": name or "Avinox ride",
        "rideId": ride_id,
        "start": start,
        "end": end,
        "duration": end - start,
        "ascent": ascent,
        "descent": descent,
        "serial": serial,
    }

    samples: list[dict[str, Any]] = []
    pos = HEADER_SIZE
    while pos < len(view):
        if pos + 6 > len(view):
            raise ValueError("Truncated Avinox frame")
        frame_type, length = struct.unpack_from("<HH", view, pos)
        if frame_type != FRAME_TYPE:
            raise ValueError(f"Unknown frame type at byte {pos}")
        payload_start = pos + 4
        payload_end = payload_start + length
        frame_end = payload_end + 2
        if frame_end > len(view):
            raise ValueError("Truncated Avinox frame")
        samples.append(_sample(_message(view[payload_start:payload_end])))
        pos = frame_end

    metadata["samples"] = len(samples)
    return {"metadata": metadata, "samples": samples}
