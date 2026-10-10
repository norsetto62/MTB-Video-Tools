"""Versioned, content-validated cache for music analysis results."""

from __future__ import annotations

from pathlib import Path
import hashlib
import json
import os
import tempfile
from typing import Any


AUDIO_CACHE_VERSION = 1


def music_file_hash(music_path: str | Path) -> str:
    """Return the SHA-256 digest of the file contents, not its path or mtime."""
    path = Path(music_path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audio_cache_path(music_path: str | Path, cache_dir: str | Path) -> Path:
    """Return a stable cache path for the normalized source path."""
    source = str(Path(music_path).expanduser().resolve())
    key = hashlib.sha256(source.encode("utf-8")).hexdigest()
    return Path(cache_dir) / f"{key}.json"


def _payload(analysis: Any) -> dict[str, Any]:
    return {
        "cache_version": AUDIO_CACHE_VERSION,
        "source_path": str(analysis.source_path),
        "source_hash": analysis.source_hash,
        "duration": analysis.duration,
        "tempo_bpm": analysis.tempo_bpm,
        "beats": list(analysis.beats),
        "measures": list(analysis.measures),
        "onsets": list(analysis.onsets),
        "combined": list(analysis.combined),
    }


def save_cached_audio(analysis: Any, cache_dir: str | Path) -> Path:
    """Persist a validated analysis atomically, leaving no partial final file."""
    from .analysis import AudioAnalysis

    if not isinstance(analysis, AudioAnalysis):
        raise TypeError("analysis must be an AudioAnalysis instance.")
    target = audio_cache_path(analysis.source_path, cache_dir)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = _payload(analysis)
    fd, temp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
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


def load_cached_audio(music_path: str | Path, cache_dir: str | Path):
    """Load a cache entry; invalid, stale, or incompatible entries are misses."""
    from .analysis import AudioAnalysis

    source = Path(music_path).expanduser()
    target = audio_cache_path(source, cache_dir)
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("cache_version") != AUDIO_CACHE_VERSION:
            return None
        digest = music_file_hash(source)
        if payload.get("source_hash") != digest:
            return None
        if Path(payload.get("source_path", "")).resolve() != source.resolve():
            return None
        return AudioAnalysis(
            source_path=Path(payload["source_path"]),
            source_hash=payload["source_hash"],
            duration=payload["duration"],
            tempo_bpm=payload["tempo_bpm"],
            beats=tuple(payload["beats"]),
            measures=tuple(payload["measures"]),
            onsets=tuple(payload["onsets"]),
            combined=tuple(payload["combined"]),
        )
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return None
