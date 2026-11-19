"""Canonical cache directory resolution for pipeline consumers.

The configured paths.cache_dir is the cache root (default: data/cache).
Feature sequences live in its features/ child and audio artifacts in audio/.
"""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any


DEFAULT_CACHE_ROOT = Path("data/cache")


def _get(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(key, default)
    try:
        return getattr(value, key)
    except AttributeError:
        return default


def cache_root(settings: Mapping[str, Any] | Any) -> Path:
    """Resolve the configured cache root from a full settings object or paths section."""
    paths = _get(settings, "paths", settings)
    value = _get(paths, "cache_dir", DEFAULT_CACHE_ROOT)
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ValueError("paths.cache_dir must be a non-empty path.")
    return Path(value).expanduser()


def feature_cache_dir(settings: Mapping[str, Any] | Any) -> Path:
    """Return the single canonical directory for motion-feature NPZ files."""
    return cache_root(settings) / "features"


def audio_cache_dir(settings: Mapping[str, Any] | Any) -> Path:
    """Return the single canonical directory for audio cache artifacts."""
    return cache_root(settings) / "audio"
