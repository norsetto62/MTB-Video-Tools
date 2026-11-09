#!/usr/bin/env python3
"""
Export DJI Osmo Action 4 telemetry embedded in MP4 files.

Reads the `djmd` timed-metadata track directly (ISO-BMFF + protobuf; stdlib only)
and writes one CSV row per telemetry sample. Raw sensor channels are retained;
processed vehicle-frame acceleration is added alongside them.

Processing defaults follow DaniilDem/dji-telemetry-export's documented approach:
  * accelerometer unit detection (g, m/s², mg)
  * automatic body-axis -> vehicle-axis mapping
  * quaternion-based gravity removal (high-pass fallback)
  * optional levelling to compensate camera pitch/roll
  * roll/pitch/yaw and GPS-derived speed/heading/distance

The DJI protobuf field layout is documented for Osmo Action 4/5/6. Validate the
processed signs and orientation against a known test clip before relying on them.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import math
import os
import re
import struct
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

MP4_EPOCH = 2_082_844_800
G_SI = 9.80665


# ------------------------------ MP4 boxes ---------------------------------

@dataclass
class Track:
    fourcc: str = ""
    handler_type: str = ""
    handler_name: str = ""
    timescale: int = 0
    offsets: list[int] = field(default_factory=list)
    sizes: list[int] = field(default_factory=list)
    times: list[float] = field(default_factory=list)

    def read_sample(self, f, i: int) -> bytes:
        f.seek(self.offsets[i])
        data = f.read(self.sizes[i])
        if len(data) != self.sizes[i]:
            raise IOError(f"Short read in metadata sample {i}")
        return data


@dataclass
class Movie:
    tracks: list[Track]
    creation_time: Optional[dt.datetime]
    duration_s: float


def _u32(b: bytes, p: int) -> int:
    return struct.unpack_from(">I", b, p)[0]


def _u64(b: bytes, p: int) -> int:
    return struct.unpack_from(">Q", b, p)[0]


def _boxes(data: bytes, start: int = 0, end: Optional[int] = None):
    """Yield (type, payload_start, payload_end) for ISO-BMFF child boxes."""
    if end is None:
        end = len(data)
    p = start
    while p + 8 <= end:
        size = _u32(data, p)
        kind = data[p + 4:p + 8]
        header = 8
        if size == 1:
            if p + 16 > end:
                break
            size = _u64(data, p + 8)
            header = 16
        elif size == 0:
            size = end - p
        if size < header or p + size > end:
            break
        yield kind, p + header, p + size
        p += size


def _find_path(data: bytes, path: tuple[bytes, ...], start=0, end=None):
    if end is None:
        end = len(data)
    if not path:
        return [(start, end)]
    found = []
    for kind, s, e in _boxes(data, start, end):
        if kind == path[0]:
            if len(path) == 1:
                found.append((s, e))
            else:
                found.extend(_find_path(data, path[1:], s, e))
    return found


def _mp4_datetime(seconds: int) -> Optional[dt.datetime]:
    if not seconds:
        return None
    try:
        return dt.datetime.fromtimestamp(seconds - MP4_EPOCH, tz=dt.timezone.utc).replace(tzinfo=None)
    except (OverflowError, OSError, ValueError):
        return None


def _parse_stbl(buf: bytes, s: int, e: int, track: Track):
    sizes: list[int] = []
    chunk_offsets: list[int] = []
    stsc: list[tuple[int, int]] = []
    stts: list[tuple[int, int]] = []

    for kind, a, b in _boxes(buf, s, e):
        if kind == b"stsd":
            # FullBox(4) + entry_count(4) + SampleEntry(size,type,...)
            if b - a >= 16:
                track.fourcc = buf[a + 12:a + 16].decode("latin1")
        elif kind == b"stsz":
            if b - a < 12:
                continue
            fixed, count = _u32(buf, a + 4), _u32(buf, a + 8)
            if fixed:
                sizes = [fixed] * count
            elif b - a >= 12 + 4 * count:
                sizes = [_u32(buf, a + 12 + 4*i) for i in range(count)]
        elif kind == b"stco":
            if b - a >= 8:
                count = _u32(buf, a + 4)
                if b - a >= 8 + 4 * count:
                    chunk_offsets = [_u32(buf, a + 8 + 4*i) for i in range(count)]
        elif kind == b"co64":
            if b - a >= 8:
                count = _u32(buf, a + 4)
                if b - a >= 8 + 8 * count:
                    chunk_offsets = [_u64(buf, a + 8 + 8*i) for i in range(count)]
        elif kind == b"stsc":
            if b - a >= 8:
                count = _u32(buf, a + 4)
                if b - a >= 8 + 12 * count:
                    stsc = [(_u32(buf, a + 8 + 12*i), _u32(buf, a + 12 + 12*i))
                            for i in range(count)]
        elif kind == b"stts":
            if b - a >= 8:
                count = _u32(buf, a + 4)
                if b - a >= 8 + 8 * count:
                    stts = [(_u32(buf, a + 8 + 8*i), _u32(buf, a + 12 + 8*i))
                            for i in range(count)]

    offsets = []
    if chunk_offsets and stsc and sizes:
        sample_i = 0
        for j, (first_chunk, samples_per_chunk) in enumerate(stsc):
            last_chunk = stsc[j + 1][0] - 1 if j + 1 < len(stsc) else len(chunk_offsets)
            for chunk_no in range(first_chunk, last_chunk + 1):
                if not 1 <= chunk_no <= len(chunk_offsets):
                    break
                pos = chunk_offsets[chunk_no - 1]
                for _ in range(samples_per_chunk):
                    if sample_i >= len(sizes):
                        break
                    offsets.append(pos)
                    pos += sizes[sample_i]
                    sample_i += 1
    sizes = sizes[:len(offsets)]

    scale = float(track.timescale or 1)
    times, tick = [], 0
    for count, delta in stts:
        for _ in range(count):
            if len(times) >= len(offsets):
                break
            times.append(tick / scale)
            tick += delta
        if len(times) >= len(offsets):
            break
    if len(times) < len(offsets):
        step = stts[-1][1] / scale if stts else 0.0
        last = times[-1] if times else 0.0
        while len(times) < len(offsets):
            last += step
            times.append(last)
    track.offsets, track.sizes, track.times = offsets, sizes, times


def read_movie(path: Path) -> Movie:
    file_size = path.stat().st_size
    with path.open("rb") as f:
        top = []
        pos = 0
        while pos + 8 <= file_size:
            f.seek(pos)
            h = f.read(16)
            if len(h) < 8:
                break
            size = struct.unpack(">I", h[:4])[0]
            kind = h[4:8]
            header = 8
            if size == 1:
                if len(h) < 16:
                    break
                size = struct.unpack(">Q", h[8:16])[0]
                header = 16
            elif size == 0:
                size = file_size - pos
            if size < header or pos + size > file_size:
                break
            top.append((kind, pos, size, header))
            pos += size

        moov = next((x for x in top if x[0] == b"moov"), None)
        if moov is None:
            raise ValueError("No moov box found in MP4")
        _, moov_pos, moov_size, moov_header = moov
        if moov_size - moov_header > 512 * 1024 * 1024:
            raise ValueError("Unreasonably large moov box")
        f.seek(moov_pos + moov_header)
        buf = f.read(moov_size - moov_header)

        creation, duration = None, 0.0
        mvhd = _find_path(buf, (b"mvhd",))
        if mvhd:
            a, b = mvhd[0]
            if b - a >= 20:
                version = buf[a]
                if version == 1 and b - a >= 32:
                    created, timescale, dur = _u64(buf, a+4), _u32(buf, a+20), _u64(buf, a+24)
                else:
                    created, timescale, dur = _u32(buf, a+4), _u32(buf, a+12), _u32(buf, a+16)
                creation = _mp4_datetime(created)
                duration = dur / timescale if timescale else 0.0

        tracks = []
        for ts, te in _find_path(buf, (b"trak",)):
            tr = Track()
            tkhd = _find_path(buf, (b"tkhd",), ts, te)
            if tkhd:
                a, b = tkhd[0]
                if b-a >= 16:
                    tr_id = _u32(buf, a+20) if buf[a] == 1 and b-a >= 24 else _u32(buf, a+12)
            mdhd = _find_path(buf, (b"mdia", b"mdhd"), ts, te)
            if mdhd:
                a, b = mdhd[0]
                if b-a >= 20:
                    tr.timescale = _u32(buf, a+20) if buf[a] == 1 and b-a >= 24 else _u32(buf, a+12)
            hdlr = _find_path(buf, (b"mdia", b"hdlr"), ts, te)
            if hdlr:
                a, b = hdlr[0]
                if b-a >= 24:
                    tr.handler_type = buf[a+8:a+12].decode("latin1")
                    name = buf[a+24:b].split(b"\0", 1)[0]
                    if name and name[0] == len(name)-1:
                        name = name[1:]
                    tr.handler_name = name.decode("utf-8", "replace").strip()
            stbl = _find_path(buf, (b"mdia", b"minf", b"stbl"), ts, te)
            if stbl:
                _parse_stbl(buf, *stbl[0], tr)
            tracks.append(tr)
    return Movie(tracks, creation, duration)


# ----------------------------- Protobuf -----------------------------------

def _protobuf_fields(data: bytes):
    """Yield (field_number, wire_type, value); supports protobuf wire types 0/1/2/5."""
    p, n = 0, len(data)

    def varint():
        nonlocal p
        value, shift = 0, 0
        while p < n:
            byte = data[p]
            p += 1
            value |= (byte & 0x7f) << shift
            if not byte & 0x80:
                return value
            shift += 7
            if shift >= 70:
                raise ValueError("Invalid protobuf varint")
        raise ValueError("Truncated protobuf varint")

    while p < n:
        key = varint()
        # DJI metadata samples may be padded with zero bytes. A zero key is
        # invalid protobuf, but an all-zero remainder is harmless padding.
        if key == 0:
            if not any(data[p:]):
                break
            preview = data[max(0, p - 8):min(n, p + 16)].hex(" ")
            raise ValueError(
                f"Invalid protobuf field number 0 at byte {p - 1}; "
                f"non-zero data follows (bytes: {preview})"
            )
        number, wire = key >> 3, key & 7
        if wire == 0:
            value = varint()
        elif wire == 1:
            if p + 8 > n:
                preview = data[max(0, p - 8):min(n, p + 16)].hex(" ")
                raise ValueError(
                    f"Truncated fixed64 at byte {p} (field {number}, "
                    f"remaining {n-p} of 8 bytes; context: {preview})"
                )
            value = data[p:p+8]; p += 8
        elif wire == 2:
            length = varint()
            if p + length > n: raise ValueError("Truncated length-delimited field")
            value = data[p:p+length]; p += length
        elif wire == 5:
            if p + 4 > n: raise ValueError("Truncated fixed32")
            value = data[p:p+4]; p += 4
        else:
            raise ValueError(f"Unsupported protobuf wire type {wire}")
        yield number, wire, value


def _values(data: bytes, field: int, wire: Optional[int] = None):
    for num, wt, value in _protobuf_fields(data):
        if num == field and (wire is None or wt == wire):
            yield value


def _sub(data: bytes, path: tuple[int, ...]) -> Optional[bytes]:
    cur = data
    for field in path:
        cur = next(_values(cur, field, 2), None)
        if cur is None:
            return None
    return cur


def _uint(data: bytes, field: int) -> Optional[int]:
    return next(_values(data, field, 0), None)


def _f32(data: bytes, field: int) -> Optional[float]:
    v = next(_values(data, field, 5), None)
    return struct.unpack("<f", v)[0] if v is not None else None


def _f64(data: bytes, field: int) -> Optional[float]:
    v = next(_values(data, field, 1), None)
    return struct.unpack("<d", v)[0] if v is not None else None


def _string(data: bytes, field: int) -> Optional[str]:
    v = next(_values(data, field, 2), None)
    return v.decode("utf-8", "replace") if v is not None else None


def _raw_varints(data: bytes):
    """Decode consecutive protobuf varints from an opaque byte payload.

    DJI stores rational values as two raw varints inside a length-delimited
    field. This payload is NOT itself a protobuf message.
    """
    values = []
    p = 0
    while p < len(data):
        value = 0
        shift = 0
        while p < len(data):
            byte = data[p]
            p += 1
            value |= (byte & 0x7f) << shift
            if not (byte & 0x80):
                values.append(value)
                break
            shift += 7
            if shift >= 70:
                raise ValueError("Invalid varint in DJI rational payload")
        else:
            raise ValueError("Truncated varint in DJI rational payload")
    return values


def _rational(data: bytes, field: int):
    # Match Rust get_rational(): the length-delimited value contains two
    # consecutive raw varints (numerator and denominator), not a submessage.
    v = next(_values(data, field, 2), None)
    if v is None:
        return None
    vals = _raw_varints(v)
    return (vals[0], vals[1]) if len(vals) >= 2 and vals[1] else None


def _signed_i64(value: int) -> int:
    return value - (1 << 64) if value >= (1 << 63) else value


def _parse_gps_time(s: Optional[str]):
    if not s:
        return None
    s = s.strip()
    if len(s) >= 19:
        s = s[:10] + " " + s[11:].replace("-", ":")
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try: return dt.datetime.strptime(s, fmt)
        except ValueError: pass
    return None


def _parse_frame(payload: bytes, index: int, track_time: float):
    frame = _sub(payload, (3,))
    if frame is None:
        return None
    header = _sub(frame, (1,))
    ts_us = _uint(header, 2) if header else None
    frame_no = (_uint(header, 1) if header else None)
    cam = _sub(frame, (2,))
    acc = quat = None
    iso = shutter_num = shutter_den = cct = None
    if cam is not None:
        a = _sub(cam, (10,))
        if a is not None:
            xyz = (_f32(a, 2), _f32(a, 3), _f32(a, 4))
            if all(x is not None for x in xyz): acc = tuple(xyz)
        q = _sub(cam, (9,))
        if q is not None:
            vals = tuple(_f32(q, i) for i in (1, 2, 3, 4))
            if all(x is not None for x in vals): quat = vals
        ex = _sub(cam, (3,))
        iso = _f32(ex, 1) if ex else None
        exp = _sub(cam, (4,))
        rat = _rational(exp, 1) if exp else None
        if rat: shutter_num, shutter_den = rat
        wb = _sub(cam, (6,))
        cct = _uint(wb, 1) if wb else None

    gps = {}
    gimbal = _sub(frame, (4,))
    if gimbal is not None:
        basic = _sub(gimbal, (2,))
        if basic is not None:
            status = _uint(basic, 3) or 0
            coords = _sub(basic, (1,))
            if status != 1 and coords is not None:
                unit = _uint(coords, 1) or 0
                lat, lon = _f64(coords, 2), _f64(coords, 3)
                if lat is not None and lon is not None:
                    if unit == 0:
                        lat, lon = math.degrees(lat), math.degrees(lon)
                    if (math.isfinite(lat) and math.isfinite(lon) and
                        not (lat == 0 and lon == 0) and -90 <= lat <= 90 and -180 <= lon <= 180):
                        alt_raw = _uint(basic, 2)
                        alt = _signed_i64(alt_raw) / 1000.0 if alt_raw is not None else None
                        velocity = _sub(gimbal, (3,))
                        vel = tuple(_f32(velocity, i) or 0.0 for i in (1,2,3)) if velocity else None
                        time_box = _sub(basic, (6,))
                        gps_time = _parse_gps_time(_string(time_box, 1)) if time_box else None
                        gps = dict(lat=lat, lon=lon, alt=alt, status=status, velocity=vel, gps_time=gps_time)
    t = (ts_us / 1e6) if ts_us is not None else track_time
    return dict(t_raw=t, timestamp_us=ts_us, frame=frame_no if frame_no is not None else index,
                acc=acc, quat=quat, iso=iso, shutter_num=shutter_num,
                shutter_den=shutter_den, cct=cct, gps=gps)


def extract_clip(path: Path):
    movie = read_movie(path)
    track = next((t for t in movie.tracks if t.fourcc == "djmd"), None)
    if track is None:
        track = next((t for t in movie.tracks if "DJI meta" in t.handler_name or "CAM meta" in t.handler_name), None)
    if track is None or not track.sizes:
        raise ValueError("No DJI metadata track (djmd / DJI meta) found")
    samples, base_ts = [], None
    clip_meta = {}
    first_gps = None
    with path.open("rb") as f:
        for i in range(len(track.sizes)):
            payload = track.read_sample(f, i)
            try:
                # Clip metadata, usually present in the first sample.
                h = _sub(payload, (1, 1))
                if h is not None:
                    for key, fld in (("proto_name",1),("proto_version",3),("serial",5),("firmware",6),("model",10)):
                        clip_meta.setdefault(key, _string(h, fld))
                video = _sub(payload, (2, 3))
                if video is not None and "width" not in clip_meta:
                    clip_meta["width"] = _uint(video, 1)
                    clip_meta["height"] = _uint(video, 2)
                    clip_meta["fps"] = _f32(video, 3)
                sample = _parse_frame(payload, i, track.times[i] if i < len(track.times) else float(i))
            except ValueError as exc:
                raise ValueError(f"DJI metadata sample {i} ({len(payload)} bytes): {exc}") from exc
            if sample is None:
                continue
            if sample["timestamp_us"] is not None:
                if base_ts is None: base_ts = sample["timestamp_us"]
                sample["t"] = (sample["timestamp_us"] - base_ts) / 1e6
            else:
                sample["t"] = sample["t_raw"]
            if sample["gps"].get("gps_time") is not None and first_gps is None:
                first_gps = (sample["t"], sample["gps"]["gps_time"])
            samples.append(sample)
    if not samples:
        raise ValueError("Metadata track contains no frame samples")
    # Absolute start time priority: GPS time, DJI filename, MP4 creation time.
    start = None
    start_source = "unknown"
    if first_gps:
        start = first_gps[1] - dt.timedelta(seconds=first_gps[0])
        start_source = "GPS time"
    else:
        m = re.match(r"DJI_(\d{14})", path.name)
        if m:
            try:
                start = dt.datetime.strptime(m.group(1), "%Y%m%d%H%M%S")
                start_source = "filename"
            except ValueError: pass
        if start is None and movie.creation_time:
            start, start_source = movie.creation_time, "MP4 creation time (UTC)"
    clip_meta.update(start_time=start, start_source=start_source,
                     metadata_track=f"{track.fourcc} ({track.handler_name or track.handler_type})",
                     sample_count=len(samples), duration_s=samples[-1]["t"])
    return samples, clip_meta


# ------------------------------ Processing --------------------------------

def median(values):
    values = sorted(values)
    if not values: return None
    return values[len(values)//2]


def detect_unit(samples):
    mags = [math.sqrt(sum(x*x for x in s["acc"])) for s in samples if s["acc"] is not None]
    m = median(mags)
    if m is None: return "unknown", 1.0, 0.0
    if 0.5 <= m <= 3.0: return "g", 1.0, m
    if 5.0 <= m <= 30.0: return "m/s2", 1.0/G_SI, m
    if 500.0 <= m <= 3000.0: return "mg", 1.0/1000.0, m
    return "unknown", 1.0, m


def q_normalize(q):
    n = math.sqrt(sum(x*x for x in q))
    return tuple(x/n for x in q) if n >= 1e-6 and math.isfinite(n) else None


def q_rotate(q, v, inverse=False):
    w, x, y, z = q
    if inverse: x, y, z = -x, -y, -z
    vx, vy, vz = v
    tx, ty, tz = 2*(y*vz-z*vy), 2*(z*vx-x*vz), 2*(x*vy-y*vx)
    return (vx+w*tx+(y*tz-z*ty),
            vy+w*ty+(z*tx-x*tz),
            vz+w*tz+(x*ty-y*tx))


def v_norm(v):
    n = math.sqrt(sum(x*x for x in v))
    return tuple(x/n for x in v) if n > 1e-9 else None


def dot(a,b): return sum(x*y for x,y in zip(a,b))
def cross(a,b): return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])


def rotate_onto(source, target, v):
    c = max(-1.0, min(1.0, dot(source,target)))
    axis = cross(source,target)
    sn = math.sqrt(dot(axis,axis))
    if sn < 1e-9:
        if c > 0: return v
        helper = (1,0,0) if abs(source[0]) < .9 else (0,1,0)
        k = v_norm(cross(source,helper)) or (0,0,1)
        kv = dot(k,v)
        return tuple(2*k[i]*kv-v[i] for i in range(3))
    k = tuple(x/sn for x in axis)
    kcross = cross(k,v)
    kv = dot(k,v)*(1-c)
    return tuple(v[i]*c+kcross[i]*sn+k[i]*kv for i in range(3))


def auto_axis_map(medians):
    order = sorted(range(3), key=lambda i: abs(medians[i]), reverse=True)
    vertical_i = order[0]
    vertical = "XYZ"[vertical_i]
    remaining = [a for a in "XYZ" if a != vertical]
    if vertical == "X":
        lateral, longitudinal = remaining[0], remaining[1]
    else:
        lateral = next(a for a in remaining if a != "X")
        longitudinal = "X"
    return dict(lateral=lateral, longitudinal=longitudinal, vertical=vertical,
                invert_lateral=True, invert_longitudinal=False,
                invert_vertical=medians[vertical_i] < 0)


def euler(q):
    q = q_normalize(q)
    if q is None: return (None,None,None)
    w,x,y,z=q
    roll=math.atan2(2*(w*x+y*z),1-2*(x*x+y*y))
    pitch=math.asin(max(-1,min(1,2*(w*y-z*x))))
    yaw=math.atan2(2*(w*z+x*y),1-2*(y*y+z*z))
    return tuple(math.degrees(a) for a in (roll,pitch,yaw))


def process(samples, gravity_mode="quat", highpass_window=1.0, level=True,
            axes="auto", invert=()):
    unit, scale, magnitude = detect_unit(samples)
    acc = [tuple(x*scale for x in s["acc"]) if s["acc"] is not None else None for s in samples]
    quats = [q_normalize(s["quat"]) if s["quat"] is not None else None for s in samples]
    present = [a for a in acc if a is not None]
    axis_medians = tuple(median([a[i] for a in present]) or 0.0 for i in range(3))
    axis_map = auto_axis_map(axis_medians) if axes == "auto" else dict(
        lateral=axes[0], longitudinal=axes[1], vertical=axes[2],
        invert_lateral=False, invert_longitudinal=False, invert_vertical=False)
    for key in invert:
        axis_map["invert_"+key] = not axis_map["invert_"+key]

    # Rotate each acceleration into world coordinates, estimate constant gravity
    # as the component-wise median, subtract there, then rotate back.
    world, valid_indices = [], []
    for i,(a,q) in enumerate(zip(acc,quats)):
        if a is not None and q is not None:
            world.append(q_rotate(q,a))
            valid_indices.append(i)
    gravity_world = None
    dynamic_quat = [None]*len(samples)
    if len(world) >= 10:
        gravity_world = tuple(median([v[k] for v in world]) or 0.0 for k in range(3))
        for i,q,w in zip(valid_indices,(quats[j] for j in valid_indices),world):
            residual = tuple(w[k]-gravity_world[k] for k in range(3))
            dynamic_quat[i] = q_rotate(q,residual,inverse=True)

    dynamic = [None]*len(samples)
    gravity_used = gravity_mode
    if gravity_mode == "none":
        dynamic = list(acc)
    elif gravity_mode == "quat" and gravity_world is not None:
        dynamic = dynamic_quat
    else:
        if gravity_mode == "quat": gravity_used = "highpass (fallback)"
        # Centred moving-average high-pass; use median sample rate as approximation.
        times = [s["t"] for s in samples]
        duration = times[-1]-times[0] if len(times)>1 else 0
        rate = (len(times)-1)/duration if duration>0 else 0
        half = max(1, round(rate*highpass_window/2)) if rate>0 else 1
        for axis in range(3):
            vals = [a[axis] if a is not None else None for a in acc]
            present_axis = [v for v in vals if v is not None]
            if not present_axis: continue
            fallback = sum(present_axis)/len(present_axis)
            prefix=[0.0]
            for v in vals: prefix.append(prefix[-1]+(fallback if v is None else v))
            for i,a in enumerate(acc):
                if a is None: continue
                lo=max(0,i-half); hi=min(len(acc),i+half+1)
                mean=(prefix[hi]-prefix[lo])/(hi-lo)
                if dynamic[i] is None: dynamic[i]=[0.0,0.0,0.0]
                dynamic[i][axis]=a[axis]-mean
        dynamic=[tuple(x) if x is not None else None for x in dynamic]

    # Level the camera frame by rotating measured gravity direction onto vehicle up.
    world_up = v_norm(gravity_world) if gravity_world is not None else None
    static_up = v_norm(axis_medians) if math.sqrt(dot(axis_medians,axis_medians)) > .5 else None
    up = [0.0,0.0,0.0]
    vi="XYZ".index(axis_map["vertical"])
    up[vi] = -1.0 if axis_map["invert_vertical"] else 1.0
    up=tuple(up)
    tilts=[]
    for i,d in enumerate(dynamic):
        if d is None: continue
        q=quats[i]
        if world_up is not None and q is not None:
            gravity_dir=v_norm(q_rotate(q,world_up,inverse=True))
        else:
            gravity_dir=static_up
        if gravity_dir is not None:
            tilts.append(math.degrees(math.acos(max(-1,min(1,dot(gravity_dir,up))))))
            if level:
                dynamic[i]=rotate_onto(gravity_dir,up,d)
    tilt=median(tilts)
    rows=[]
    for i,s in enumerate(samples):
        a=acc[i]
        d=dynamic[i]
        row=dict(sample=s, raw=a, processed=None, roll=None, pitch=None, yaw=None)
        if d is not None:
            def pick(axis, inv):
                v=d["XYZ".index(axis)]
                return -v if inv else v
            lat=pick(axis_map["lateral"],axis_map["invert_lateral"])
            lon=pick(axis_map["longitudinal"],axis_map["invert_longitudinal"])
            vert=pick(axis_map["vertical"],axis_map["invert_vertical"])
            row["processed"]=(lat,lon,vert,math.hypot(lat,lon))
        if s["quat"] is not None:
            row["roll"],row["pitch"],row["yaw"]=euler(s["quat"])
        rows.append(row)
    return rows, dict(unit=unit, unit_median=magnitude, axis_medians=axis_medians,
                      axes=axis_map, gravity_used=gravity_used,
                      gravity_world=gravity_world, tilt_deg=tilt, levelled=bool(level and tilt is not None))


# ---------------------------------- CSV ------------------------------------

def export_csv(path: Path, samples, rows, info):
    gps_fixes=[r["sample"]["gps"] for r in rows if r["sample"]["gps"]]
    distance=0.0
    previous=None
    fields=[
        "Frame","Elapsed time (s)","Timestamp",
        "Accelerometer X raw (g)","Accelerometer Y raw (g)","Accelerometer Z raw (g)",
        "Quaternion W","Quaternion X","Quaternion Y","Quaternion Z",
        "Roll (deg)","Pitch (deg)","Yaw (deg)",
        "Lateral acceleration (g)","Longitudinal acceleration (g)","Vertical acceleration (g)","Combined acceleration (g)",
        "ISO","Shutter numerator","Shutter denominator","Shutter (s)","Colour temperature (K)",
        "Latitude","Longitude","Altitude (m)","GPS speed (m/s)","GPS heading (deg)","GPS distance (m)","GPS time"
    ]
    start=info.get("start_time")
    with path.open("w",newline="",encoding="utf-8-sig") as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            s=r["sample"]; gps=s["gps"]; q=s["quat"]; p=r["processed"]
            t=s["t"]
            timestamp=(start+dt.timedelta(seconds=t)).isoformat(sep=" ") if start else ""
            out={"Frame":s["frame"],"Elapsed time (s)":f"{t:.6f}","Timestamp":timestamp,
                 "Roll (deg)":r["roll"],"Pitch (deg)":r["pitch"],"Yaw (deg)":r["yaw"],
                 "ISO":s["iso"],"Shutter numerator":s["shutter_num"],"Shutter denominator":s["shutter_den"],
                 "Shutter (s)":(s["shutter_num"]/s["shutter_den"] if s["shutter_num"] and s["shutter_den"] else None),
                 "Colour temperature (K)":s["cct"]}
            if r["raw"] is not None:
                for name,val in zip(("Accelerometer X raw (g)","Accelerometer Y raw (g)","Accelerometer Z raw (g)"),r["raw"]):
                    out[name]=val
            if q is not None:
                for name,val in zip(("Quaternion W","Quaternion X","Quaternion Y","Quaternion Z"),q): out[name]=val
            if p is not None:
                for name,val in zip(("Lateral acceleration (g)","Longitudinal acceleration (g)","Vertical acceleration (g)","Combined acceleration (g)"),p): out[name]=val
            if gps:
                speed=heading=None
                if previous is not None:
                    pt,plat,plon=previous
                    d=6371008.8*2*math.asin(min(1,math.sqrt(
                        math.sin(math.radians(gps["lat"]-plat)/2)**2+
                        math.cos(math.radians(plat))*math.cos(math.radians(gps["lat"]))*
                        math.sin(math.radians(gps["lon"]-plon)/2)**2)))
                    distance+=d
                    if gps["velocity"] is None and t>pt: speed=d/(t-pt)
                    if d>.05 and gps["velocity"] is None:
                        y=math.sin(math.radians(gps["lon"]-plon))*math.cos(math.radians(gps["lat"]))
                        x=math.cos(math.radians(plat))*math.sin(math.radians(gps["lat"]))-math.sin(math.radians(plat))*math.cos(math.radians(gps["lat"]))*math.cos(math.radians(gps["lon"]-plon))
                        heading=(math.degrees(math.atan2(y,x))+360)%360
                if gps["velocity"] is not None:
                    vx,vy,_=gps["velocity"]
                    speed=math.hypot(vx,vy)
                    if vx or vy: heading=(math.degrees(math.atan2(vx,vy))+360)%360
                previous=(t,gps["lat"],gps["lon"])
                out.update({"Latitude":gps["lat"],"Longitude":gps["lon"],"Altitude (m)":gps["alt"],
                            "GPS speed (m/s)":speed,"GPS heading (deg)":heading,
                            "GPS distance (m)":distance,
                            "GPS time":gps["gps_time"].isoformat(sep=" ") if gps["gps_time"] else ""})
            writer.writerow(out)


def main():
    ap=argparse.ArgumentParser(description="Extract DJI Osmo Action 4 telemetry from MP4 to CSV")
    ap.add_argument("video",type=Path,help="Input DJI MP4 file")
    ap.add_argument("-o","--output",type=Path,help="Output CSV (default: same folder/name.csv)")
    ap.add_argument("--gravity",choices=("quat","highpass","none"),default="quat",
                    help="Gravity removal method (default: quat)")
    ap.add_argument("--highpass-window",type=float,default=1.0,help="High-pass window in seconds")
    ap.add_argument("--axes",default="auto",help="auto or three letters: lateral,longitudinal,vertical (e.g. yxz)")
    ap.add_argument("--invert",default="",help="Comma-separated axes to invert: lateral,longitudinal,vertical")
    ap.add_argument("--no-level",action="store_true",help="Do not compensate camera roll/pitch relative to vehicle")
    args=ap.parse_args()
    if not args.video.is_file():
        ap.error(f"File not found: {args.video}")
    if args.axes != "auto" and (len(args.axes)!=3 or set(args.axes.upper())!={"X","Y","Z"}):
        ap.error("--axes must be 'auto' or a permutation such as 'yxz'")
    axes=args.axes.upper()
    invert=[x.strip().lower() for x in args.invert.split(",") if x.strip()]
    if any(x not in ("lateral","longitudinal","vertical") for x in invert):
        ap.error("--invert accepts lateral,longitudinal,vertical")
    try:
        samples,info=extract_clip(args.video)
        rows,proc=process(samples,args.gravity,args.highpass_window,not args.no_level,axes,invert)
        output=args.output or args.video.with_suffix(".csv")
        export_csv(output,samples,rows,info)
    except Exception as exc:
        print(f"ERROR: {exc}",file=sys.stderr)
        return 1
    print(f"Camera: {info.get('model') or 'unknown'}")
    print(f"Samples: {len(samples)} | duration: {info['duration_s']:.2f} s")
    print(f"Accelerometer unit: {proc['unit']} (median magnitude {proc['unit_median']:.3f})")
    print(f"Axis mapping: lateral={proc['axes']['lateral']}, longitudinal={proc['axes']['longitudinal']}, vertical={proc['axes']['vertical']}")
    print(f"Gravity removal: {proc['gravity_used']} | levelled: {proc['levelled']}")
    if proc["tilt_deg"] is not None: print(f"Estimated camera tilt: {proc['tilt_deg']:.1f} deg")
    print(f"GPS fixes: {sum(bool(s['gps']) for s in samples)}")
    print(f"CSV written: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
