"""Versioned, integrity-checked cache for extracted motion features.

Only raw feature sequences are cached. Windowing and model scaling remain
separate so training folds can fit their own scaler without data leakage.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np

from ..data_models import Clip, FeatureSequence
from ..motion.features import FEATURE_NAMES
from ..motion.flow import DEFAULT_FARNEBACK, FarnebackConfig


FEATURE_CACHE_VERSION = 1
FEATURE_SCHEMA_VERSION = 1
_ARRAY_FORMAT_VERSION = 1


def source_file_hash(path: str | Path) -> str:
    """Return a SHA-256 digest of the complete source-file contents."""
    digest = hashlib.sha256()
    with Path(path).expanduser().open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
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
) -> tuple[dict[str, Any], str]:
    source = Path(video_path).expanduser().resolve()
    fingerprint = source_file_hash(source)
    metadata = {
        "cache_version": FEATURE_CACHE_VERSION,
        "array_format_version": _ARRAY_FORMAT_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_names": list(FEATURE_NAMES),
        "source_path": str(source),
        "source_sha256": fingerprint,
        "clip": {"start": float(clip.start), "end": float(clip.end)},
        "sample_fps": float(sample_fps),
        "processing_width": int(processing_width),
        "flow_config": _config_payload(flow_config),
    }
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
) -> Path:
    """Return the deterministic cache path for a source and extraction contract."""
    _, key = _identity(
        video_path, clip, sample_fps=sample_fps,
        processing_width=processing_width, flow_config=flow_config,
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
) -> Path:
    """Write a feature sequence and metadata via an atomic file replacement."""
    if not isinstance(sequence, FeatureSequence):
        raise TypeError("sequence must be a FeatureSequence instance.")
    metadata, identity_key = _identity(
        video_path, clip, sample_fps=sample_fps,
        processing_width=processing_width, flow_config=flow_config,
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
) -> FeatureSequence | None:
    """Load a compatible cache entry; missing, stale, or corrupt entries are misses."""
    try:
        expected, key = _identity(
            video_path, clip, sample_fps=sample_fps,
            processing_width=processing_width, flow_config=flow_config,
        )
        target = Path(cache_dir) / f"{key}.npz"
        with np.load(target, allow_pickle=False) as archive:
            if set(archive.files) != {"metadata", "features", "timestamps"}:
                return None
            raw_metadata = archive["metadata"]
            if raw_metadata.shape != ():
                return None
            metadata = json.loads(str(raw_metadata.item()))
            features = archive["features"]
            timestamps = archive["timestamps"]
        if not isinstance(metadata, dict):
            return None
        digest = metadata.pop("features_sha256", None)
        if metadata != expected or not isinstance(digest, str):
            return None
        _validate_arrays(features, timestamps, clip=clip)
        if _array_digest(features, timestamps) != digest:
            return None
        return FeatureSequence(
            features=np.array(features, dtype=np.float32, copy=True),
            timestamps=np.array(timestamps, dtype=np.float64, copy=True),
        )
    except (OSError, ValueError, TypeError, KeyError, EOFError, json.JSONDecodeError):
        return None
