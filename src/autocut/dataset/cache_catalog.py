"""SQLite inventory for on-disk motion-feature cache files.

The catalog is a rebuildable index: NumPy arrays remain in their existing
NPZ files, and indexing never opens or decodes source videos.
"""
from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)
CATALOG_FILENAME = "feature_cache.sqlite3"
CATALOG_SCHEMA_VERSION = 1


def catalog_path(cache_dir: str | Path) -> Path:
    """Return the SQLite catalog path for a feature-cache directory."""
    return Path(cache_dir) / CATALOG_FILENAME


def _connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute(
        """CREATE TABLE IF NOT EXISTS cache_entries (
            cache_path TEXT PRIMARY KEY,
            cache_version INTEGER,
            array_format_version INTEGER,
            feature_schema_version INTEGER,
            feature_names_json TEXT,
            source_path TEXT,
            source_fingerprint_mode TEXT,
            source_size_bytes INTEGER,
            source_mtime_ns INTEGER,
            source_sha256 TEXT,
            clip_start REAL,
            clip_end REAL,
            sample_fps REAL,
            processing_width INTEGER,
            roi_json TEXT,
            flow_config_json TEXT,
            rows INTEGER,
            first_timestamp REAL,
            last_timestamp REAL,
            file_size_bytes INTEGER,
            indexed_mtime_ns INTEGER,
            valid INTEGER NOT NULL DEFAULT 0,
            issue TEXT
        )"""
    )
    connection.execute(
        """CREATE TABLE IF NOT EXISTS catalog_meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )"""
    )
    connection.execute(
        "INSERT OR REPLACE INTO catalog_meta(key, value) VALUES (?, ?)",
        ("schema_version", str(CATALOG_SCHEMA_VERSION)),
    )
    connection.commit()
    return connection


def _read_entry(path: Path) -> dict[str, Any]:
    """Read NPZ metadata and basic array coverage without touching the source video."""
    stat = path.stat()
    entry: dict[str, Any] = {
        "cache_path": str(path.resolve()),
        "file_size_bytes": stat.st_size,
        "indexed_mtime_ns": stat.st_mtime_ns,
        "valid": 0,
        "issue": None,
    }
    try:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != {"metadata", "features", "timestamps"}:
                raise ValueError("unexpected NPZ members")
            raw = archive["metadata"]
            if raw.shape != ():
                raise ValueError("metadata is not a scalar")
            metadata = json.loads(str(raw.item()))
            if not isinstance(metadata, dict):
                raise ValueError("metadata is not an object")
            clip = metadata.get("clip", {})
            features = archive["features"]
            timestamps = archive["timestamps"]
            entry.update({
                "cache_version": metadata.get("cache_version"),
                "array_format_version": metadata.get("array_format_version"),
                "feature_schema_version": metadata.get("feature_schema_version"),
                "feature_names_json": json.dumps(metadata.get("feature_names"), separators=(",", ":")),
                "source_path": metadata.get("source_path"),
                "source_fingerprint_mode": metadata.get("source_fingerprint_mode"),
                "source_size_bytes": metadata.get("source_size_bytes"),
                "source_mtime_ns": metadata.get("source_mtime_ns"),
                "source_sha256": metadata.get("source_sha256"),
                "clip_start": clip.get("start"),
                "clip_end": clip.get("end"),
                "sample_fps": metadata.get("sample_fps"),
                "processing_width": metadata.get("processing_width"),
                "roi_json": json.dumps(metadata.get("roi"), sort_keys=True, separators=(",", ":")),
                "flow_config_json": json.dumps(metadata.get("flow_config"), sort_keys=True, separators=(",", ":")),
                "rows": int(features.shape[0]) if features.ndim else 0,
                "first_timestamp": float(timestamps[0]) if timestamps.ndim == 1 and len(timestamps) else None,
                "last_timestamp": float(timestamps[-1]) if timestamps.ndim == 1 and len(timestamps) else None,
            })
            if (features.ndim != 2 or timestamps.ndim != 1
                    or features.shape[0] != timestamps.shape[0]
                    or features.shape[0] < 2):
                raise ValueError("invalid feature/timestamp array shapes")
            if not np.isfinite(features).all() or not np.isfinite(timestamps).all():
                raise ValueError("non-finite array values")
            if not np.all(np.diff(timestamps) > 0):
                raise ValueError("timestamps are not strictly increasing")
            entry["valid"] = 1
    except (OSError, ValueError, TypeError, KeyError, EOFError, json.JSONDecodeError) as exc:
        entry["issue"] = str(exc)
    return entry


_COLUMNS = (
    "cache_path", "cache_version", "array_format_version", "feature_schema_version",
    "feature_names_json", "source_path", "source_fingerprint_mode", "source_size_bytes",
    "source_mtime_ns", "source_sha256", "clip_start", "clip_end", "sample_fps",
    "processing_width", "roi_json", "flow_config_json", "rows", "first_timestamp",
    "last_timestamp", "file_size_bytes", "indexed_mtime_ns", "valid", "issue",
)


def index_cache_file(cache_dir: str | Path, cache_file: str | Path) -> bool:
    """Index one existing NPZ. Returns True when its basic structure is valid."""
    root = Path(cache_dir).resolve()
    path = Path(cache_file).resolve()
    if path.suffix.lower() != ".npz" or not path.is_file():
        return False
    entry = _read_entry(path)
    placeholders = ", ".join("?" for _ in _COLUMNS)
    updates = ", ".join(f"{column}=excluded.{column}" for column in _COLUMNS if column != "cache_path")
    with _connect(catalog_path(root)) as connection:
        connection.execute(
            f"INSERT INTO cache_entries ({', '.join(_COLUMNS)}) VALUES ({placeholders}) "
            f"ON CONFLICT(cache_path) DO UPDATE SET {updates}",
            tuple(entry.get(column) for column in _COLUMNS),
        )
    if not entry["valid"]:
        logger.warning("Invalid feature cache file indexed: %s (%s)", path, entry["issue"])
    return bool(entry["valid"])


def index_existing_cache(cache_dir: str | Path) -> dict[str, int]:
    """Inventory existing NPZ files without extracting or hashing source videos."""
    root = Path(cache_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    counts = {"found": 0, "valid": 0, "invalid": 0}
    for path in root.rglob("*.npz"):
        if path.resolve() == catalog_path(root).resolve():
            continue
        counts["found"] += 1
        if index_cache_file(root, path):
            counts["valid"] += 1
        else:
            counts["invalid"] += 1
    logger.info(
        "Feature-cache catalog inventory: %d found, %d valid, %d invalid",
        counts["found"], counts["valid"], counts["invalid"],
    )
    return counts


def list_cache_entries(cache_dir: str | Path) -> list[dict[str, Any]]:
    """Return catalog entries for diagnostics and tests."""
    path = catalog_path(cache_dir)
    if not path.is_file():
        return []
    with _connect(path) as connection:
        rows = connection.execute(
            "SELECT * FROM cache_entries ORDER BY source_path, clip_start, cache_path"
        ).fetchall()
    return [dict(row) for row in rows]

def check_cache_compatibility(
    metadata: dict[str, Any],
    *,
    video_path: str | Path,
    clip_start: float,
    clip_end: float,
    sample_fps: float,
    processing_width: int,
    flow_config: dict[str, Any],
    roi: dict[str, float],
    feature_names: list[str],
    cache_version: int,
    array_format_version: int,
    feature_schema_version: int,
    source_sha256: str | None = None,
    timestamp_start: float | None = None,
    timestamp_end: float | None = None,
    timestamps: np.ndarray | None = None,
    tolerance: float = 1e-6,
) -> dict[str, Any]:
    """Evaluate reuse against an explicit extraction contract.

    This function only checks metadata and supplied timestamps; it never changes
    cache files or extracts source video. Interval reuse requires the stored
    clip to contain the requested interval and, when timestamps are supplied,
    the requested sampling lattice to align with the stored sequence.
    """
    reasons: list[str] = []
    source = Path(video_path).expanduser().resolve()
    try:
        stat = source.stat()
    except OSError as exc:
        return {"compatible": False, "reasons": [f"source_unavailable: {exc}"]}

    expected = {
        "cache_version": cache_version,
        "array_format_version": array_format_version,
        "feature_schema_version": feature_schema_version,
        "feature_names": feature_names,
        "source_path": str(source),
        "source_size_bytes": stat.st_size,
        "source_mtime_ns": stat.st_mtime_ns,
        "sample_fps": float(sample_fps),
        "processing_width": int(processing_width),
        "roi": roi,
        "flow_config": flow_config,
    }
    for field, value in expected.items():
        actual = metadata.get(field)
        if field in {"sample_fps"}:
            equal = isinstance(actual, (int, float)) and abs(float(actual) - float(value)) <= tolerance
        else:
            equal = actual == value
        if not equal:
            reasons.append(f"{field}_mismatch")

    mode = metadata.get("source_fingerprint_mode", "metadata")
    if mode == "sha256":
        if not source_sha256:
            reasons.append("source_sha256_not_verified")
        elif metadata.get("source_sha256") != source_sha256:
            reasons.append("source_sha256_mismatch")
    elif mode != "metadata":
        reasons.append("unknown_source_fingerprint_mode")

    stored_clip = metadata.get("clip")
    if not isinstance(stored_clip, dict):
        reasons.append("clip_metadata_missing")
    else:
        try:
            stored_start = float(stored_clip["start"])
            stored_end = float(stored_clip["end"])
            if stored_start > clip_start + tolerance or stored_end < clip_end - tolerance:
                reasons.append("requested_interval_not_covered")
        except (KeyError, TypeError, ValueError):
            reasons.append("clip_metadata_invalid")

    if timestamps is not None:
        values = np.asarray(timestamps)
        if values.ndim != 1 or values.size < 2 or not np.isfinite(values).all():
            reasons.append("timestamps_invalid")
        else:
            step = 1.0 / float(sample_fps)
            if not np.all(np.diff(values) > 0):
                reasons.append("timestamps_not_increasing")
            elif not np.allclose(np.diff(values), step, rtol=0.0, atol=tolerance):
                reasons.append("sampling_grid_mismatch")
            if timestamp_start is not None and abs(float(values[0]) - timestamp_start) > tolerance:
                reasons.append("timestamp_start_mismatch")
            if timestamp_end is not None and abs(float(values[-1]) - timestamp_end) > tolerance:
                reasons.append("timestamp_end_mismatch")

    return {"compatible": not reasons, "reasons": reasons}

