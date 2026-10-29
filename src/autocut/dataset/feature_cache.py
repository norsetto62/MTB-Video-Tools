"""Versioned, integrity-checked cache for extracted motion features.

Only raw feature sequences are cached. Windowing and model scaling remain
separate so training folds can fit their own scaler without data leakage.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np

from ..data_models import Clip, FeatureSequence
from ..motion.features import FEATURE_NAMES
from ..motion.flow import DEFAULT_FARNEBACK, FarnebackConfig


logger = logging.getLogger(__name__)

FEATURE_CACHE_VERSION = 3
FEATURE_SCHEMA_VERSION = 1
_ARRAY_FORMAT_VERSION = 1


def source_file_hash(path: str | Path) -> str:
    """Return a SHA-256 digest of the complete source-file contents."""
    source = Path(path).expanduser()
    digest = hashlib.sha256()
    total_bytes = source.stat().st_size
    bytes_read = 0
    next_report = 256 * 1024 * 1024
    with source.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
            bytes_read += len(chunk)
            if bytes_read >= next_report and bytes_read < total_bytes:
                logger.info(
                    "Hashing video for feature-cache validation: %s (%.1f/%.1f GiB, %.0f%%)",
                    source, bytes_read / (1024 ** 3), total_bytes / (1024 ** 3),
                    100.0 * bytes_read / max(total_bytes, 1),
                )
                next_report = ((bytes_read // (256 * 1024 * 1024)) + 1) * (256 * 1024 * 1024)
    if total_bytes >= 256 * 1024 * 1024:
        logger.info(
            "Video hash complete for feature cache: %s (%.1f GiB)",
            source, total_bytes / (1024 ** 3),
        )
    return digest.hexdigest()


def _config_payload(config: FarnebackConfig) -> dict[str, Any]:
    """Convert the optical-flow config into stable, JSON-safe values."""
    values = {
        name: getattr(config, name)
        for name in ("pyr_scale", "levels", "winsize", "iterations",
                     "poly_n", "poly_sigma", "flags")
    }
    # JSON round-trip also rejects non-serializable values.
    return json.loads(json.dumps(values, sort_keys=True, allow_nan=False))


def _identity(
    video_path: str | Path,
    clip: Clip,
    *,
    sample_fps: float,
    processing_width: int,
    flow_config: FarnebackConfig,
    roi_left: float = 0.15,
    roi_right: float = 0.85,
    roi_bottom: float = 0.50,
    roi_top: float = 0.10,
    source_fingerprint_mode: str = "metadata",
) -> tuple[dict[str, Any], str]:
    source = Path(video_path).expanduser().resolve()
    stat = source.stat()
    if source_fingerprint_mode not in {"metadata", "sha256"}:
        raise ValueError("source_fingerprint_mode must be 'metadata' or 'sha256'.")
    metadata = {
        "cache_version": FEATURE_CACHE_VERSION,
        "array_format_version": _ARRAY_FORMAT_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_names": list(FEATURE_NAMES),
        "source_path": str(source),
        "source_fingerprint_mode": source_fingerprint_mode,
        "source_size_bytes": stat.st_size,
        "source_mtime_ns": stat.st_mtime_ns,
        "clip": {"start": float(clip.start), "end": float(clip.end)},
        "sample_fps": float(sample_fps),
        "processing_width": int(processing_width),
        "roi": {"left": float(roi_left), "right": float(roi_right), "bottom": float(roi_bottom), "top": float(roi_top)},
        "flow_config": _config_payload(flow_config),
    }
    if source_fingerprint_mode == "sha256":
        metadata["source_sha256"] = source_file_hash(source)
    encoded = json.dumps(metadata, sort_keys=True, separators=(",", ":"), allow_nan=False)
    key = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    return metadata, key


def feature_cache_path(
    video_path: str | Path,
    cache_dir: str | Path,
    clip: Clip,
    *,
    sample_fps: float = 2.0,
    processing_width: int = 480,
    flow_config: FarnebackConfig = DEFAULT_FARNEBACK,
    roi_left: float = 0.15,
    roi_right: float = 0.85,
    roi_bottom: float = 0.50,
    roi_top: float = 0.10,
    source_fingerprint_mode: str = "metadata",
) -> Path:
    """Return the deterministic cache path; metadata fingerprinting is the default."""
    _, key = _identity(
        video_path, clip, sample_fps=sample_fps,
        processing_width=processing_width, flow_config=flow_config,
        roi_left=roi_left, roi_right=roi_right, roi_bottom=roi_bottom, roi_top=roi_top,
        source_fingerprint_mode=source_fingerprint_mode,
    )
    return Path(cache_dir) / f"{key}.npz"


def _array_digest(features: np.ndarray, timestamps: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in (features, timestamps):
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(json.dumps(contiguous.shape).encode("ascii"))
        digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def save_feature_sequence(
    sequence: FeatureSequence,
    video_path: str | Path,
    cache_dir: str | Path,
    clip: Clip,
    *,
    sample_fps: float = 2.0,
    processing_width: int = 480,
    flow_config: FarnebackConfig = DEFAULT_FARNEBACK,
    roi_left: float = 0.15,
    roi_right: float = 0.85,
    roi_bottom: float = 0.50,
    roi_top: float = 0.10,
    source_fingerprint_mode: str = "metadata",
) -> Path:
    """Write a feature sequence and metadata via an atomic file replacement."""
    if not isinstance(sequence, FeatureSequence):
        raise TypeError("sequence must be a FeatureSequence instance.")
    metadata, identity_key = _identity(
        video_path, clip, sample_fps=sample_fps,
        processing_width=processing_width, flow_config=flow_config,
        roi_left=roi_left, roi_right=roi_right, roi_bottom=roi_bottom, roi_top=roi_top,
        source_fingerprint_mode=source_fingerprint_mode,
    )
    features = np.asarray(sequence.features)
    timestamps = np.asarray(sequence.timestamps)
    _validate_arrays(features, timestamps, clip=clip)
    metadata["features_sha256"] = _array_digest(features, timestamps)

    # Keep the path based on extraction identity, not output contents.
    target = Path(cache_dir) / f"{identity_key}.npz"
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            np.savez_compressed(
                stream,
                metadata=np.asarray(json.dumps(metadata, sort_keys=True, separators=(",", ":"), allow_nan=False)),
                features=features,
                timestamps=timestamps,
            )
            stream.flush()
            os.fsync(stream.fileno())
        Path(temp_name).replace(target)
    except Exception:
        try:
            Path(temp_name).unlink(missing_ok=True)
        except OSError:
            pass
        raise

    # Keep the catalog synchronized with every successful cache write. Catalog
    # failures must not invalidate an atomically-written NPZ; lookup can rebuild
    # the entry on the next read.
    try:
        from .cache_catalog import index_cache_file

        index_cache_file(target.parent, target)
    except Exception as exc:  # catalog is a rebuildable optimization
        logger.warning("Could not index feature cache %s: %s", target, exc)
    return target


def _validate_arrays(
    features: np.ndarray,
    timestamps: np.ndarray,
    *,
    clip: Clip | None = None,
) -> None:
    """Reject malformed feature arrays before writing or constructing a model input."""
    if features.ndim != 2 or features.shape[1] != len(FEATURE_NAMES) or features.shape[0] < 2:
        raise ValueError(
            f"features must have shape (N, {len(FEATURE_NAMES)}) with N >= 2; "
            f"got {features.shape}."
        )
    if features.dtype != np.float32:
        raise ValueError(f"features must have dtype float32; got {features.dtype}.")
    if timestamps.ndim != 1 or timestamps.shape[0] != features.shape[0]:
        raise ValueError("timestamps must be 1D and match the feature sample count.")
    if timestamps.dtype != np.float64:
        raise ValueError(f"timestamps must have dtype float64; got {timestamps.dtype}.")
    if not np.isfinite(features).all() or not np.isfinite(timestamps).all():
        raise ValueError("cached features and timestamps must be finite.")
    if not np.all(np.diff(timestamps) > 0):
        raise ValueError("cached timestamps must be strictly increasing.")
    if clip is not None and (
        timestamps[0] <= clip.start or timestamps[-1] > clip.end + 1e-6
    ):
        raise ValueError("cached timestamps must lie within the clip interval.")


def load_feature_sequence(
    video_path: str | Path,
    cache_dir: str | Path,
    clip: Clip,
    *,
    sample_fps: float = 2.0,
    processing_width: int = 480,
    flow_config: FarnebackConfig = DEFAULT_FARNEBACK,
    roi_left: float = 0.15,
    roi_right: float = 0.85,
    roi_bottom: float = 0.50,
    roi_top: float = 0.10,
    source_fingerprint_mode: str = "metadata",
) -> FeatureSequence | None:
    """Load a compatible cache entry; missing, stale, or corrupt entries are misses."""
    try:
        expected, key = _identity(
            video_path, clip, sample_fps=sample_fps,
            processing_width=processing_width, flow_config=flow_config,
            roi_left=roi_left, roi_right=roi_right, roi_bottom=roi_bottom, roi_top=roi_top,
            source_fingerprint_mode=source_fingerprint_mode,
        )
        target = Path(cache_dir) / f"{key}.npz"
        from .cache_catalog import lookup_cache_file, remove_cache_entry

        catalog_valid = lookup_cache_file(cache_dir, target)
        if not target.is_file():
            if catalog_valid is False:
                remove_cache_entry(cache_dir, target)
            return None
        # The catalog is only a rebuildable index, never the source of truth.
        # A stale "invalid" row must not veto an existing NPZ; validate the
        # NPZ directly and repair the row if the file proves valid.
        if catalog_valid is False:
            logger.warning(
                "Cache catalog marked an existing NPZ invalid; validating file directly: %s",
                target,
            )
        with np.load(target, allow_pickle=False) as archive:
            if set(archive.files) != {"metadata", "features", "timestamps"}:
                logger.warning("Feature cache miss: unexpected NPZ members in %s", target)
                return None
            raw_metadata = archive["metadata"]
            if raw_metadata.shape != ():
                logger.warning("Feature cache miss: malformed metadata array in %s", target)
                return None
            metadata = json.loads(str(raw_metadata.item()))
            features = archive["features"]
            timestamps = archive["timestamps"]
        if not isinstance(metadata, dict):
            logger.warning("Feature cache miss: metadata is not an object in %s", target)
            return None
        digest = metadata.pop("features_sha256", None)
        if metadata != expected:
            logger.warning(
                "Feature cache miss: extraction identity mismatch in %s; expected key=%s",
                target, key,
            )
            return None
        if not isinstance(digest, str):
            logger.warning("Feature cache miss: missing content digest in %s", target)
            return None
        _validate_arrays(features, timestamps, clip=clip)
        if _array_digest(features, timestamps) != digest:
            logger.warning("Feature cache miss: feature-array digest mismatch in %s", target)
            return None
        if catalog_valid is False:
            try:
                from .cache_catalog import index_cache_file
                index_cache_file(cache_dir, target)
            except Exception as exc:
                logger.warning("Could not repair cache catalog entry for %s: %s", target, exc)
        return FeatureSequence(
            features=np.array(features, dtype=np.float32, copy=True),
            timestamps=np.array(timestamps, dtype=np.float64, copy=True),
        )
    except (OSError, ValueError, TypeError, KeyError, EOFError, json.JSONDecodeError) as exc:
        logger.warning("Feature cache miss: could not validate cache entry for %s: %s", video_path, exc)
        return None
