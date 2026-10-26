"""Tests for the non-destructive SQLite feature-cache inventory and contract."""

import json
from pathlib import Path

import numpy as np

from autocut.data_models import Clip, FeatureSequence
from autocut.dataset.cache_catalog import (
    check_cache_compatibility,
    index_existing_cache,
    list_cache_entries,
)
from autocut.dataset.feature_cache import (
    FEATURE_CACHE_VERSION,
    _ARRAY_FORMAT_VERSION,
    FEATURE_SCHEMA_VERSION,
    _config_payload,
    save_feature_sequence,
)
from autocut.motion.features import FEATURE_NAMES
from autocut.motion.flow import DEFAULT_FARNEBACK


def _sequence():
    return FeatureSequence(
        features=np.zeros((3, len(FEATURE_NAMES)), dtype=np.float32),
        timestamps=np.asarray([1.5, 2.0, 2.5], dtype=np.float64),
    )


def test_inventory_indexes_valid_cache_without_modifying_npz(tmp_path):
    video = tmp_path / "ride.mp4"
    video.write_bytes(b"source video")
    cache_dir = tmp_path / "cache"
    clip = Clip(video_name="ride", start=1.0, end=3.0)
    cache_file = save_feature_sequence(_sequence(), video, cache_dir, clip)
    before = cache_file.read_bytes()

    counts = index_existing_cache(cache_dir)

    assert counts == {"found": 1, "valid": 1, "invalid": 0}
    assert cache_file.read_bytes() == before
    entries = list_cache_entries(cache_dir)
    assert len(entries) == 1
    entry = entries[0]
    assert entry["valid"] == 1
    assert entry["source_path"] == str(video.resolve())
    assert entry["clip_start"] == 1.0
    assert entry["clip_end"] == 3.0
    assert entry["rows"] == 3
    assert entry["first_timestamp"] == 1.5
    assert entry["last_timestamp"] == 2.5


def test_inventory_is_idempotent_and_records_malformed_npz(tmp_path):
    video = tmp_path / "ride.mp4"
    video.write_bytes(b"source video")
    cache_dir = tmp_path / "cache"
    save_feature_sequence(
        _sequence(), video, cache_dir,
        Clip(video_name="ride", start=1.0, end=3.0),
    )
    malformed = cache_dir / "broken.npz"
    malformed.write_bytes(b"not an npz")
    before = malformed.read_bytes()

    assert index_existing_cache(cache_dir) == {
        "found": 2, "valid": 1, "invalid": 1
    }
    assert index_existing_cache(cache_dir) == {
        "found": 2, "valid": 1, "invalid": 1
    }
    entries = list_cache_entries(cache_dir)
    assert len(entries) == 2
    broken = next(item for item in entries if item["cache_path"] == str(malformed.resolve()))
    assert broken["valid"] == 0
    assert broken["issue"]
    assert malformed.read_bytes() == before


def _contract(video, clip_start=1.0, clip_end=3.0):
    stat = video.stat()
    metadata = {
        "cache_version": FEATURE_CACHE_VERSION,
        "array_format_version": _ARRAY_FORMAT_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_names": list(FEATURE_NAMES),
        "source_path": str(video.resolve()),
        "source_fingerprint_mode": "metadata",
        "source_size_bytes": stat.st_size,
        "source_mtime_ns": stat.st_mtime_ns,
        "clip": {"start": clip_start, "end": clip_end},
        "sample_fps": 2.0,
        "processing_width": 480,
        "roi": {"left": 0.15, "right": 0.85, "bottom": 0.5, "top": 0.1},
        "flow_config": _config_payload(DEFAULT_FARNEBACK),
    }
    return metadata


def _check(video, metadata, **overrides):
    args = {
        "video_path": video,
        "clip_start": 1.5,
        "clip_end": 2.5,
        "sample_fps": 2.0,
        "processing_width": 480,
        "flow_config": _config_payload(DEFAULT_FARNEBACK),
        "roi": {"left": 0.15, "right": 0.85, "bottom": 0.5, "top": 0.1},
        "feature_names": list(FEATURE_NAMES),
        "cache_version": FEATURE_CACHE_VERSION,
        "array_format_version": _ARRAY_FORMAT_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "timestamps": np.asarray([1.5, 2.0, 2.5], dtype=np.float64),
    }
    args.update(overrides)
    return check_cache_compatibility(metadata, **args)


def test_compatibility_accepts_covered_interval_on_matching_contract(tmp_path):
    video = tmp_path / "ride.mp4"
    video.write_bytes(b"source video")
    result = _check(video, _contract(video))
    assert result == {"compatible": True, "reasons": []}


def test_compatibility_rejects_uncovered_interval_and_sampling_grid(tmp_path):
    video = tmp_path / "ride.mp4"
    video.write_bytes(b"source video")
    metadata = _contract(video, clip_start=2.0, clip_end=3.0)

    uncovered = _check(video, metadata)
    assert not uncovered["compatible"]
    assert "requested_interval_not_covered" in uncovered["reasons"]

    misaligned = _check(
        video, _contract(video),
        timestamps=np.asarray([1.5, 2.1, 2.5], dtype=np.float64),
    )
    assert not misaligned["compatible"]
    assert "sampling_grid_mismatch" in misaligned["reasons"]


def test_compatibility_reports_each_changed_extraction_setting(tmp_path):
    video = tmp_path / "ride.mp4"
    video.write_bytes(b"source video")
    result = _check(
        video, _contract(video),
        sample_fps=4.0,
        processing_width=640,
        roi={"left": 0.2, "right": 0.85, "bottom": 0.5, "top": 0.1},
        flow_config={"winsize": 99},
        feature_schema_version=FEATURE_SCHEMA_VERSION + 1,
    )
    assert not result["compatible"]
    assert {
        "sample_fps_mismatch",
        "processing_width_mismatch",
        "roi_mismatch",
        "flow_config_mismatch",
        "feature_schema_version_mismatch",
    }.issubset(set(result["reasons"]))


def test_compatibility_requires_sha256_when_cache_uses_sha256_mode(tmp_path):
    video = tmp_path / "ride.mp4"
    video.write_bytes(b"source video")
    metadata = _contract(video)
    metadata["source_fingerprint_mode"] = "sha256"
    metadata["source_sha256"] = "saved-hash"

    result = _check(video, metadata)
    assert not result["compatible"]
    assert "source_sha256_not_verified" in result["reasons"]

    result = _check(video, metadata, source_sha256="different-hash")
    assert "source_sha256_mismatch" in result["reasons"]
