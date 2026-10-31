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


def _read_valid_cache(
    path: Path,
    *,
    expected_common: dict[str, Any],
    requested_clip: Clip,
    sample_fps: float,
    source_sha256: str | None,
) -> FeatureSequence | None:
    """Read, integrity-check and temporally slice one compatible NPZ."""
    try:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != {"metadata", "features", "timestamps"}:
                return None
            raw = archive["metadata"]
            if raw.shape != ():
                return None
            metadata = json.loads(str(raw.item()))
            features = np.array(archive["features"], copy=True)
            timestamps = np.array(archive["timestamps"], copy=True)
        if not isinstance(metadata, dict):
            return None
        digest = metadata.pop("features_sha256", None)
        if not isinstance(digest, str) or _array_digest(features, timestamps) != digest:
            return None
        if any(metadata.get(key) != value for key, value in expected_common.items()):
            return None
        mode = metadata.get("source_fingerprint_mode")
        if mode == "sha256":
            if source_sha256 is None or metadata.get("source_sha256") != source_sha256:
                return None
        elif mode != "metadata":
            return None
        _validate_arrays(features, timestamps)
        stored_clip = metadata.get("clip")
        if not isinstance(stored_clip, dict):
            return None
        stored_start = float(stored_clip["start"])
        stored_end = float(stored_clip["end"])
        tolerance = 1e-6
        if stored_start > requested_clip.start + tolerance or stored_end < requested_clip.end - tolerance:
            return None

        # VideoReader's feature timestamps are start + n/fps, n >= 1.
        # Reuse is safe only when the requested interval starts on the
        # cached sampling lattice; otherwise a fresh decode has a different
        # sampling phase and the cached rows are not equivalent.
        step = 1.0 / sample_fps
        phase = (requested_clip.start - stored_start) / step
        if abs(phase - round(phase)) > tolerance:
            return None
        mask = (
            (timestamps > requested_clip.start + tolerance)
            & (timestamps < requested_clip.end - tolerance)
        )
        selected_features = features[mask]
        selected_timestamps = timestamps[mask]
        if selected_features.shape[0] < 2:
            return None
        expected_first = requested_clip.start + step
        if abs(float(selected_timestamps[0]) - expected_first) > tolerance:
            return None
        if not np.allclose(np.diff(selected_timestamps), step, rtol=0.0, atol=tolerance):
            return None
        return FeatureSequence(
            features=np.asarray(selected_features, dtype=np.float32),
            timestamps=np.asarray(selected_timestamps, dtype=np.float64),
        )
    except (OSError, ValueError, TypeError, KeyError, EOFError, json.JSONDecodeError):
        return None


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
    """Load an exact or temporally covering compatible cache entry.

    Exact identity is checked first. If absent, catalog candidates for the
    same source video are validated and sliced to the requested interval.
    """
    try:
        expected, key = _identity(
            video_path, clip, sample_fps=sample_fps,
            processing_width=processing_width, flow_config=flow_config,
            roi_left=roi_left, roi_right=roi_right, roi_bottom=roi_bottom, roi_top=roi_top,
            source_fingerprint_mode=source_fingerprint_mode,
        )
        common = {name: value for name, value in expected.items() if name != "clip"}
        source = Path(video_path).expanduser().resolve()
        stat = source.stat()
        common["source_path"] = str(source)
        common["source_size_bytes"] = stat.st_size
        common["source_mtime_ns"] = stat.st_mtime_ns
        exact_path = Path(cache_dir) / f"{key}.npz"

        from .cache_catalog import find_cache_candidates, index_cache_file, lookup_cache_file

        candidates: list[Path] = []
        if exact_path.is_file():
            lookup_cache_file(cache_dir, exact_path)
            candidates.append(exact_path)
        for candidate in find_cache_candidates(cache_dir, source_path=source):
            if candidate not in candidates:
                candidates.append(candidate)

        # A SHA-256 source fingerprint is computed only if at least one
        # candidate actually uses that mode.
        source_sha256 = None
        for candidate in candidates:
            try:
                with np.load(candidate, allow_pickle=False) as archive:
                    raw = archive["metadata"]
                    if raw.shape == ():
                        metadata = json.loads(str(raw.item()))
                        if metadata.get("source_fingerprint_mode") == "sha256":
                            source_sha256 = source_file_hash(source)
                            break
            except (OSError, ValueError, KeyError, EOFError, json.JSONDecodeError):
                continue

        for candidate in candidates:
            result = _read_valid_cache(
                candidate,
                expected_common=common,
                requested_clip=clip,
                sample_fps=sample_fps,
                source_sha256=source_sha256,
            )
            if result is not None:
                if candidate != exact_path:
                    logger.info(
                        "Reusing covering feature cache %s for interval [%.3f, %.3f]",
                        candidate, clip.start, clip.end,
                    )
                try:
                    index_cache_file(cache_dir, candidate)
                except Exception as exc:
                    logger.warning("Could not refresh cache catalog for %s: %s", candidate, exc)
                return result
        return None
    except (OSError, ValueError, TypeError) as exc:
        logger.warning("Feature cache miss: could not validate cache entry for %s: %s", video_path, exc)
        return None
