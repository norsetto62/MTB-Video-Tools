"""Tests for explicit, manifest-driven feature-cache cleanup."""
import json

import numpy as np
import pytest

from autocut.data_models import Clip, FeatureSequence
from autocut.dataset.cache_catalog import catalog_path, list_cache_entries
from autocut.dataset.cache_cleanup import cleanup_feature_cache
from autocut.dataset.feature_cache import save_feature_sequence
from autocut.motion.features import FEATURE_NAMES


def _write_cache(tmp_path, start=1.0):
    video = tmp_path / "ride.mp4"
    if not video.exists():
        video.write_bytes(b"source video bytes")
    cache_dir = tmp_path / "features"
    clip = Clip(video_name="ride", start=start, end=start + 2.0)
    sequence = FeatureSequence(
        features=np.full((3, len(FEATURE_NAMES)), start, dtype=np.float32),
        timestamps=np.asarray([start + 0.5, start + 1.0, start + 1.5], dtype=np.float64),
    )
    return save_feature_sequence(sequence, video, cache_dir, clip)


def _manifest(path, cache_dir, files):
    manifest = path / "keep.json"
    manifest.write_text(
        json.dumps([file.relative_to(cache_dir).as_posix() for file in files]),
        encoding="utf-8",
    )
    return manifest


def test_cleanup_dry_run_reports_candidates_without_mutating_files_or_catalog(tmp_path):
    keep_file = _write_cache(tmp_path, 1.0)
    obsolete_file = _write_cache(tmp_path, 4.0)
    cache_dir = keep_file.parent
    keep_bytes = keep_file.read_bytes()
    obsolete_bytes = obsolete_file.read_bytes()
    entries_before = list_cache_entries(cache_dir)
    manifest = _manifest(tmp_path, cache_dir, [keep_file])

    report = cleanup_feature_cache(cache_dir, manifest)

    assert report["dry_run"] is True
    assert report["npz_found"] == 2
    assert report["npz_kept"] == 1
    assert report["npz_obsolete"] == 1
    assert report["delete_candidates"] == [str(obsolete_file.resolve())]
    assert keep_file.read_bytes() == keep_bytes
    assert obsolete_file.read_bytes() == obsolete_bytes
    assert list_cache_entries(cache_dir) == entries_before


def test_cleanup_apply_deletes_only_excluded_npz_and_reconciles_catalog(tmp_path):
    keep_file = _write_cache(tmp_path, 1.0)
    obsolete_file = _write_cache(tmp_path, 4.0)
    cache_dir = keep_file.parent
    keep_bytes = keep_file.read_bytes()
    manifest = _manifest(tmp_path, cache_dir, [keep_file])

    report = cleanup_feature_cache(cache_dir, manifest, apply=True)

    assert report["dry_run"] is False
    assert report["deleted"] == [str(obsolete_file.resolve())]
    assert keep_file.is_file()
    assert keep_file.read_bytes() == keep_bytes
    assert not obsolete_file.exists()
    entries = list_cache_entries(cache_dir)
    assert [entry["cache_path"] for entry in entries] == [str(keep_file.resolve())]
    assert entries[0]["valid"] == 1
    assert report["catalog_entries_after"] == 1
    assert catalog_path(cache_dir).is_file()


def test_cleanup_rejects_missing_or_invalid_keep_manifest_entry_without_deletion(tmp_path):
    keep_file = _write_cache(tmp_path, 1.0)
    obsolete_file = _write_cache(tmp_path, 4.0)
    cache_dir = keep_file.parent
    manifest = _manifest(tmp_path, cache_dir, [keep_file, cache_dir / "missing.npz"])

    with pytest.raises(ValueError, match="does not exist"):
        cleanup_feature_cache(cache_dir, manifest, apply=True)

    assert keep_file.exists()
    assert obsolete_file.exists()


def test_cleanup_rejects_manifest_path_outside_cache(tmp_path):
    keep_file = _write_cache(tmp_path, 1.0)
    cache_dir = keep_file.parent
    outside = tmp_path / "outside.npz"
    outside.write_bytes(b"not a cache")
    manifest = tmp_path / "keep.json"
    manifest.write_text(json.dumps([str(outside)]), encoding="utf-8")

    with pytest.raises(ValueError, match="escapes cache directory"):
        cleanup_feature_cache(cache_dir, manifest, apply=True)

    assert outside.exists()
    assert keep_file.exists()
