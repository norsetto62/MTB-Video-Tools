"""Contract tests for the versioned motion-feature cache."""

import json
from pathlib import Path

import numpy as np
import pytest

from autocut.data_models import Clip, FeatureSequence
from autocut.dataset.feature_cache import (
    FEATURE_CACHE_VERSION,
    feature_cache_path,
    source_file_hash,
    load_feature_sequence,
    save_feature_sequence,
)
from autocut.motion.features import FEATURE_NAMES
from autocut.motion.flow import FarnebackConfig


def _fixture(tmp_path: Path):
    video = tmp_path / "ride.mp4"
    video.write_bytes(b"video bytes")
    clip = Clip(video_name="ride", start=1.0, end=3.0)
    sequence = FeatureSequence(
        features=np.arange(3 * len(FEATURE_NAMES), dtype=np.float32).reshape(3, -1),
        timestamps=np.asarray([1.5, 2.0, 2.5], dtype=np.float64),
    )
    cache_dir = tmp_path / "feature-cache"
    return video, clip, sequence, cache_dir


def test_feature_cache_round_trip_and_deterministic_path(tmp_path):
    video, clip, expected, cache_dir = _fixture(tmp_path)
    first_path = feature_cache_path(video, cache_dir, clip)
    assert first_path == feature_cache_path(video, cache_dir, clip)
    save_feature_sequence(expected, video, cache_dir, clip)
    assert first_path.is_file()
    actual = load_feature_sequence(video, cache_dir, clip)
    assert actual is not None
    np.testing.assert_array_equal(actual.features, expected.features)
    np.testing.assert_array_equal(actual.timestamps, expected.timestamps)



def test_metadata_fingerprint_is_default_and_does_not_hash_source(tmp_path, monkeypatch):
    video, clip, sequence, cache_dir = _fixture(tmp_path)

    def fail_if_hashing(_path):
        raise AssertionError("metadata mode must not hash the full video")

    monkeypatch.setattr("autocut.dataset.feature_cache.source_file_hash", fail_if_hashing)
    path = save_feature_sequence(sequence, video, cache_dir, clip)
    assert path.is_file()
    assert load_feature_sequence(video, cache_dir, clip) is not None


def test_sha256_fingerprint_remains_available_as_opt_in(tmp_path, monkeypatch):
    video, clip, sequence, cache_dir = _fixture(tmp_path)
    calls = []
    real_hash = source_file_hash

    def track_hash(path):
        calls.append(Path(path))
        return real_hash(path)

    monkeypatch.setattr("autocut.dataset.feature_cache.source_file_hash", track_hash)
    save_feature_sequence(sequence, video, cache_dir, clip, source_fingerprint_mode="sha256")
    assert calls
    assert load_feature_sequence(video, cache_dir, clip, source_fingerprint_mode="sha256") is not None


def test_metadata_fingerprint_changes_when_file_stat_changes(tmp_path):
    video, clip, sequence, cache_dir = _fixture(tmp_path)
    save_feature_sequence(sequence, video, cache_dir, clip)
    video.write_bytes(b"video bytes with a different size")
    assert load_feature_sequence(video, cache_dir, clip) is None

def test_feature_cache_misses_when_source_content_changes(tmp_path):
    video, clip, sequence, cache_dir = _fixture(tmp_path)
    save_feature_sequence(sequence, video, cache_dir, clip)
    video.write_bytes(b"changed video bytes")
    assert load_feature_sequence(video, cache_dir, clip) is None


def test_feature_cache_key_includes_clip_and_extraction_settings(tmp_path):
    video, clip, sequence, cache_dir = _fixture(tmp_path)
    save_feature_sequence(sequence, video, cache_dir, clip)
    assert load_feature_sequence(
        video, cache_dir, Clip(video_name="ride", start=0.0, end=3.0)
    ) is None
    assert load_feature_sequence(
        video, cache_dir, clip, sample_fps=4.0
    ) is None
    assert load_feature_sequence(
        video, cache_dir, clip, processing_width=640
    ) is None
    changed_flow = FarnebackConfig(winsize=19)
    assert load_feature_sequence(video, cache_dir, clip, flow_config=changed_flow) is None



def test_feature_cache_key_includes_roi_settings(tmp_path):
    video, clip, sequence, cache_dir = _fixture(tmp_path)
    original = feature_cache_path(video, cache_dir, clip)
    changed = feature_cache_path(video, cache_dir, clip, roi_left=0.20)
    assert original != changed
    save_feature_sequence(sequence, video, cache_dir, clip)
    assert load_feature_sequence(video, cache_dir, clip, roi_left=0.20) is None


def test_feature_cache_key_includes_sampling_rate_but_not_window_geometry(tmp_path):
    video, clip, sequence, cache_dir = _fixture(tmp_path)
    original = feature_cache_path(video, cache_dir, clip, sample_fps=2.0)
    assert original != feature_cache_path(video, cache_dir, clip, sample_fps=4.0)
    # Window duration and stride are intentionally downstream of the raw
    # feature cache; build_windows applies both on every invocation.
    assert original == feature_cache_path(video, cache_dir, clip, sample_fps=2.0)


def test_feature_cache_rejects_incompatible_version(tmp_path):
    video, clip, sequence, cache_dir = _fixture(tmp_path)
    path = save_feature_sequence(sequence, video, cache_dir, clip)
    with np.load(path, allow_pickle=False) as archive:
        payload = {name: archive[name] for name in archive.files}
    metadata = json.loads(str(payload["metadata"].item()))
    metadata["cache_version"] = FEATURE_CACHE_VERSION + 1
    payload["metadata"] = np.asarray(json.dumps(metadata))
    np.savez_compressed(path, **payload)
    assert load_feature_sequence(video, cache_dir, clip) is None


def test_feature_cache_rejects_corrupt_arrays_even_if_metadata_is_intact(tmp_path):
    video, clip, sequence, cache_dir = _fixture(tmp_path)
    path = save_feature_sequence(sequence, video, cache_dir, clip)
    with np.load(path, allow_pickle=False) as archive:
        payload = {name: archive[name] for name in archive.files}
    payload["features"] = payload["features"].copy()
    payload["features"][0, 0] += 1.0
    np.savez_compressed(path, **payload)
    assert load_feature_sequence(video, cache_dir, clip) is None


@pytest.mark.parametrize(
    ("features", "timestamps"),
    [
        (np.zeros((3, 53), dtype=np.float32), np.asarray([1.0, 2.0, 3.0], dtype=np.float64)),
        (np.full((3, 54), np.nan, dtype=np.float32), np.asarray([1.0, 2.0, 3.0], dtype=np.float64)),
        (np.zeros((3, 54), dtype=np.float32), np.asarray([1.0, 1.0, 3.0], dtype=np.float64)),
    ],
)
def test_save_rejects_invalid_arrays(tmp_path, features, timestamps):
    video, clip, _, cache_dir = _fixture(tmp_path)
    with pytest.raises(ValueError):
        save_feature_sequence(
            FeatureSequence(features=features, timestamps=timestamps),
            video, cache_dir, clip,
        )


def test_feature_cache_uses_atomic_replace(tmp_path, monkeypatch):
    video, clip, expected, cache_dir = _fixture(tmp_path)
    real_replace = Path.replace
    replaced = []

    def track_replace(self, target):
        replaced.append(Path(target))
        return real_replace(self, target)

    monkeypatch.setattr(Path, "replace", track_replace)
    path = save_feature_sequence(expected, video, cache_dir, clip)
    assert replaced == [path]
    assert load_feature_sequence(video, cache_dir, clip) is not None


def test_feature_cache_rejects_timestamps_outside_clip(tmp_path):
    video, clip, _, cache_dir = _fixture(tmp_path)
    outside = FeatureSequence(
        features=np.zeros((3, 54), dtype=np.float32),
        timestamps=np.asarray([0.5, 1.0, 3.5], dtype=np.float64),
    )
    with pytest.raises(ValueError, match="within the clip interval"):
        save_feature_sequence(outside, video, cache_dir, clip)



def test_feature_cache_reuses_covering_interval_and_slices_without_rewriting_npz(tmp_path):
    video = tmp_path / "ride.mp4"
    video.write_bytes(b"video bytes")
    cache_dir = tmp_path / "feature-cache"
    cached_clip = Clip(video_name="ride", start=0.0, end=6.0)
    cached = FeatureSequence(
        features=np.arange(11 * len(FEATURE_NAMES), dtype=np.float32).reshape(11, -1),
        timestamps=np.arange(0.5, 6.0, 0.5, dtype=np.float64),
    )
    cache_path = save_feature_sequence(cached, video, cache_dir, cached_clip)
    original_bytes = cache_path.read_bytes()

    requested = Clip(video_name="ride", start=2.0, end=4.0)
    actual = load_feature_sequence(video, cache_dir, requested)

    assert actual is not None
    np.testing.assert_array_equal(actual.timestamps, np.asarray([2.5, 3.0, 3.5]))
    np.testing.assert_array_equal(actual.features, cached.features[4:7])
    assert cache_path.read_bytes() == original_bytes


def test_feature_cache_rejects_covering_interval_with_different_sampling_phase(tmp_path):
    video = tmp_path / "ride.mp4"
    video.write_bytes(b"video bytes")
    cache_dir = tmp_path / "feature-cache"
    cached_clip = Clip(video_name="ride", start=0.0, end=6.0)
    cached = FeatureSequence(
        features=np.ones((11, len(FEATURE_NAMES)), dtype=np.float32),
        timestamps=np.arange(0.5, 6.0, 0.5, dtype=np.float64),
    )
    save_feature_sequence(cached, video, cache_dir, cached_clip)

    # The requested start is halfway between samples on the cached lattice.
    requested = Clip(video_name="ride", start=0.25, end=2.25)
    assert load_feature_sequence(video, cache_dir, requested) is None


def test_feature_cache_does_not_reuse_interval_without_full_temporal_coverage(tmp_path):
    video = tmp_path / "ride.mp4"
    video.write_bytes(b"video bytes")
    cache_dir = tmp_path / "feature-cache"
    cached_clip = Clip(video_name="ride", start=1.0, end=3.0)
    cached = FeatureSequence(
        features=np.ones((3, len(FEATURE_NAMES)), dtype=np.float32),
        timestamps=np.asarray([1.5, 2.0, 2.5], dtype=np.float64),
    )
    save_feature_sequence(cached, video, cache_dir, cached_clip)

    requested = Clip(video_name="ride", start=1.0, end=4.0)
    assert load_feature_sequence(video, cache_dir, requested) is None


def test_feature_cache_indexes_and_reuses_existing_unindexed_npz(tmp_path):
    video = tmp_path / "ride.mp4"
    video.write_bytes(b"video bytes")
    cache_dir = tmp_path / "feature-cache"
    cached_clip = Clip(video_name="ride", start=0.0, end=6.0)
    cached = FeatureSequence(
        features=np.ones((11, len(FEATURE_NAMES)), dtype=np.float32),
        timestamps=np.arange(0.5, 6.0, 0.5, dtype=np.float64),
    )
    cache_path = save_feature_sequence(cached, video, cache_dir, cached_clip)
    original_bytes = cache_path.read_bytes()

    # Simulate an NPZ copied into the canonical directory without a catalog row.
    from autocut.dataset.cache_catalog import catalog_path
    catalog_path(cache_dir).unlink(missing_ok=True)
    requested = Clip(video_name="ride", start=2.0, end=4.0)
    actual = load_feature_sequence(video, cache_dir, requested)

    assert actual is not None
    assert cache_path.read_bytes() == original_bytes
