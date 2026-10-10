"""Contract tests for the versioned music-analysis cache."""
import json
from pathlib import Path

import pytest

from autocut.audio.analysis import AudioAnalysis
from autocut.audio.cache import (
    AUDIO_CACHE_VERSION,
    audio_cache_path,
    load_cached_audio,
    music_file_hash,
    save_cached_audio,
)


def make_analysis(path: Path, digest: str | None = None):
    return AudioAnalysis(
        source_path=Path(path),
        source_hash=digest or music_file_hash(path),
        duration=8.0,
        tempo_bpm=120.0,
        beats=(0.5, 1.0, 1.5),
        measures=(0.5,),
        onsets=(0.75, 1.25),
        combined=(0.5, 0.75, 1.0, 1.25, 1.5),
    )


def test_music_hash_tracks_content_not_only_path(tmp_path):
    music = tmp_path / "music.mp3"
    music.write_bytes(b"first")
    first = music_file_hash(music)
    music.write_bytes(b"second")
    assert music_file_hash(music) != first


def test_cache_path_is_stable_for_same_source_and_cache_dir(tmp_path):
    music = tmp_path / "music.mp3"
    music.write_bytes(b"music")
    cache_dir = tmp_path / "data" / "audio_cache"
    assert audio_cache_path(music, cache_dir) == audio_cache_path(music, cache_dir)
    assert audio_cache_path(music, cache_dir).parent == cache_dir


def test_round_trip_cache_hit(tmp_path):
    music = tmp_path / "music.mp3"
    music.write_bytes(b"audio bytes")
    cache_dir = tmp_path / "cache"
    expected = make_analysis(music)
    save_cached_audio(expected, cache_dir)
    actual = load_cached_audio(music, cache_dir)
    assert actual == expected


def test_changed_source_content_invalidates_cache(tmp_path):
    music = tmp_path / "music.mp3"
    music.write_bytes(b"before")
    cache_dir = tmp_path / "cache"
    save_cached_audio(make_analysis(music), cache_dir)
    music.write_bytes(b"after")
    assert load_cached_audio(music, cache_dir) is None


def test_incompatible_cache_version_is_a_miss(tmp_path):
    music = tmp_path / "music.mp3"
    music.write_bytes(b"audio")
    cache_dir = tmp_path / "cache"
    save_cached_audio(make_analysis(music), cache_dir)
    cache_path = audio_cache_path(music, cache_dir)
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    payload["cache_version"] = AUDIO_CACHE_VERSION + 1
    cache_path.write_text(json.dumps(payload), encoding="utf-8")
    assert load_cached_audio(music, cache_dir) is None


@pytest.mark.parametrize("payload_text", ["{", "[]", "null", '{"duration": "long"}'])
def test_malformed_cache_is_a_miss(tmp_path, payload_text):
    music = tmp_path / "music.mp3"
    music.write_bytes(b"audio")
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    audio_cache_path(music, cache_dir).write_text(payload_text, encoding="utf-8")
    assert load_cached_audio(music, cache_dir) is None


def test_cache_rejects_out_of_range_or_unsorted_timestamps(tmp_path):
    music = tmp_path / "music.mp3"
    music.write_bytes(b"audio")
    cache_dir = tmp_path / "cache"
    save_cached_audio(make_analysis(music), cache_dir)
    cache_path = audio_cache_path(music, cache_dir)
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    payload["beats"] = [1.0, 0.5]
    cache_path.write_text(json.dumps(payload), encoding="utf-8")
    assert load_cached_audio(music, cache_dir) is None


def test_cache_save_uses_atomic_replace(tmp_path, monkeypatch):
    music = tmp_path / "music.mp3"
    music.write_bytes(b"audio")
    cache_dir = tmp_path / "cache"
    expected = make_analysis(music)
    real_replace = Path.replace
    replaced = []

    def track_replace(self, target):
        replaced.append((self, Path(target)))
        return real_replace(self, target)

    monkeypatch.setattr(Path, "replace", track_replace)
    save_cached_audio(expected, cache_dir)
    assert replaced, "cache save should atomically replace the final cache file"
    assert replaced[-1][1] == audio_cache_path(music, cache_dir)
    assert load_cached_audio(music, cache_dir) == expected


def test_cache_write_failure_does_not_require_discarding_analysis(tmp_path):
    # Cache persistence is deliberately a separate operation from successful
    # analysis. The caller can log a warning and still return the analysis.
    music = tmp_path / "music.mp3"
    music.write_bytes(b"audio")
    expected = make_analysis(music)
    cache_dir = tmp_path / "not-a-directory"
    cache_dir.write_text("block mkdir", encoding="utf-8")
    with pytest.raises(OSError):
        save_cached_audio(expected, cache_dir)
