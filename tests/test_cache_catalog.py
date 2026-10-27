"""Lifecycle tests for the SQLite feature-cache catalog."""

import numpy as np

from autocut.data_models import Clip, FeatureSequence
from autocut.dataset.cache_catalog import (
    catalog_path,
    index_existing_cache,
    list_cache_entries,
    lookup_cache_file,
)
from autocut.dataset.feature_cache import save_feature_sequence
from autocut.motion.features import FEATURE_NAMES


def _fixture(tmp_path):
    video = tmp_path / "ride.mp4"
    video.write_bytes(b"source video bytes")
    cache_dir = tmp_path / "features"
    clip = Clip(video_name="ride", start=1.0, end=3.0)
    sequence = FeatureSequence(
        features=np.ones((3, len(FEATURE_NAMES)), dtype=np.float32),
        timestamps=np.asarray([1.5, 2.0, 2.5], dtype=np.float64),
    )
    return video, cache_dir, clip, sequence


def test_save_feature_sequence_indexes_written_npz(tmp_path):
    video, cache_dir, clip, sequence = _fixture(tmp_path)

    cache_file = save_feature_sequence(sequence, video, cache_dir, clip)

    assert catalog_path(cache_dir).is_file()
    entries = list_cache_entries(cache_dir)
    assert len(entries) == 1
    assert entries[0]["cache_path"] == str(cache_file.resolve())
    assert entries[0]["valid"] == 1
    assert lookup_cache_file(cache_dir, cache_file) is True


def test_lookup_indexes_cache_file_created_outside_save_api(tmp_path):
    video, cache_dir, clip, sequence = _fixture(tmp_path)
    cache_file = save_feature_sequence(sequence, video, cache_dir, clip)

    # Simulate a legacy/external NPZ that exists before the catalog entry.
    from autocut.dataset.cache_catalog import remove_cache_entry

    remove_cache_entry(cache_dir, cache_file)
    assert list_cache_entries(cache_dir) == []

    assert lookup_cache_file(cache_dir, cache_file) is True
    assert len(list_cache_entries(cache_dir)) == 1


def test_lookup_reindexes_file_when_stat_changes(tmp_path):
    video, cache_dir, clip, sequence = _fixture(tmp_path)
    cache_file = save_feature_sequence(sequence, video, cache_dir, clip)
    assert lookup_cache_file(cache_dir, cache_file) is True

    # Replace with a malformed NPZ and force a distinct file stat.
    cache_file.write_bytes(b"not an npz cache")
    assert lookup_cache_file(cache_dir, cache_file) is False
    entry = list_cache_entries(cache_dir)[0]
    assert entry["valid"] == 0



def test_full_rebuild_removes_catalog_rows_for_deleted_npz(tmp_path):
    video, cache_dir, clip, sequence = _fixture(tmp_path)
    first = save_feature_sequence(sequence, video, cache_dir, clip)
    second_clip = Clip(video_name="ride", start=3.0, end=5.0)
    second = save_feature_sequence(sequence, video, cache_dir, second_clip)
    assert len(list_cache_entries(cache_dir)) == 2

    second.unlink()
    counts = index_existing_cache(cache_dir)

    assert counts == {"found": 1, "valid": 1, "invalid": 0}
    entries = list_cache_entries(cache_dir)
    assert [entry["cache_path"] for entry in entries] == [str(first.resolve())]


def test_full_rebuild_reconstructs_catalog_from_npz_files(tmp_path):
    video, cache_dir, clip, sequence = _fixture(tmp_path)
    cache_file = save_feature_sequence(sequence, video, cache_dir, clip)
    catalog = catalog_path(cache_dir)

    # Remove the index database; the NPZ must be sufficient to reconstruct it.
    catalog.unlink()
    for suffix in ("-wal", "-shm"):
        sidecar = catalog.with_name(catalog.name + suffix)
        if sidecar.exists():
            sidecar.unlink()

    counts = index_existing_cache(cache_dir)

    assert counts == {"found": 1, "valid": 1, "invalid": 0}
    entries = list_cache_entries(cache_dir)
    assert len(entries) == 1
    assert entries[0]["cache_path"] == str(cache_file.resolve())
    assert entries[0]["valid"] == 1
