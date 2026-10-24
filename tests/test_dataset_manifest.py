"""Tests for dataset manifest parsing and validation."""

from pathlib import Path

import pytest

from autocut.dataset.manifest import (
    DatasetRecord,
    get_records_for_group,
    get_unique_groups,
    load_manifest,
)


def _write_manifest(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    return path


def test_load_manifest_valid_and_group_helpers(tmp_path: Path) -> None:
    video_a = tmp_path / "vid1.mp4"
    annotation_a = tmp_path / "ann1.txt"
    video_b = tmp_path / "vid2.mp4"
    annotation_b = tmp_path / "ann2.txt"
    for file_path in (video_a, annotation_a, video_b, annotation_b):
        file_path.touch()

    manifest = _write_manifest(
        tmp_path / "manifest.yaml",
        """
videos:
  - id: video_alpha
    group_id: group_a
    video: vid1.mp4
    annotation: ann1.txt
  - id: video_beta
    group_id: group_b
    video: vid2.mp4
    annotation: ann2.txt
""",
    )

    records = load_manifest(manifest, verify_files=True)
    assert len(records) == 2
    assert isinstance(records, tuple)
    assert isinstance(records[0], DatasetRecord)
    assert records[0].id == "video_alpha"
    assert records[0].video_name == "video_alpha"
    assert records[0].group_id == "group_a"
    assert records[0].video_path == video_a.resolve()
    assert records[0].annotation_path == annotation_a.resolve()
    assert records[0].manifest_path == manifest.resolve()
    assert get_unique_groups(records) == ("group_a", "group_b")
    assert get_records_for_group(records, "group_a") == (records[0],)
    assert get_records_for_group(records, "missing") == ()


def test_relative_paths_are_based_on_manifest_not_working_directory(tmp_path: Path, monkeypatch) -> None:
    manifest_dir = tmp_path / "dataset"
    other_dir = tmp_path / "elsewhere"
    manifest_dir.mkdir()
    other_dir.mkdir()
    (manifest_dir / "video.mp4").touch()
    (manifest_dir / "annotation.txt").touch()
    (other_dir / "video.mp4").touch()
    (other_dir / "annotation.txt").touch()

    manifest = _write_manifest(
        manifest_dir / "manifest.yaml",
        """
videos:
  - id: clip
    group_id: group
    video: video.mp4
    annotation: annotation.txt
""",
    )
    monkeypatch.chdir(other_dir)

    record = load_manifest(manifest, verify_files=True)[0]
    assert record.video_path == (manifest_dir / "video.mp4").resolve()
    assert record.annotation_path == (manifest_dir / "annotation.txt").resolve()


def test_absolute_paths_are_preserved(tmp_path: Path) -> None:
    video = tmp_path / "video.mp4"
    annotation = tmp_path / "annotation.txt"
    video.touch()
    annotation.touch()
    manifest = _write_manifest(
        tmp_path / "manifest.yaml",
        f"""
videos:
  - id: clip
    group_id: group
    video: "{video.as_posix()}"
    annotation: "{annotation.as_posix()}"
""",
    )

    record = load_manifest(manifest, verify_files=True)[0]
    assert record.video_path == video.resolve()
    assert record.annotation_path == annotation.resolve()


def test_load_manifest_duplicate_id_is_case_insensitive(tmp_path: Path) -> None:
    manifest = _write_manifest(
        tmp_path / "manifest.yaml",
        """
videos:
  - id: Clip_1
    group_id: g1
    video: v.mp4
    annotation: a.txt
  - id: clip_1
    group_id: g2
    video: v2.mp4
    annotation: a2.txt
""",
    )
    with pytest.raises(ValueError, match="Duplicate video ID"):
        load_manifest(manifest)


def test_load_manifest_missing_fields(tmp_path: Path) -> None:
    manifest = _write_manifest(
        tmp_path / "manifest.yaml",
        """
videos:
  - id: clip_1
    group_id: g1
    video: v.mp4
""",
    )
    with pytest.raises(ValueError, match="missing required field"):
        load_manifest(manifest)


@pytest.mark.parametrize("bad_group", [None, "", "   ", 7, True])
def test_load_manifest_rejects_invalid_group_id(tmp_path: Path, bad_group) -> None:
    import yaml

    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {"videos": [{
                "id": "clip",
                "group_id": bad_group,
                "video": "v.mp4",
                "annotation": "a.txt",
            }]}
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="group_id"):
        load_manifest(manifest)


def test_load_manifest_invalid_yaml_and_root(tmp_path: Path) -> None:
    malformed = _write_manifest(tmp_path / "malformed.yaml", "videos: [")
    with pytest.raises(ValueError, match="Invalid YAML"):
        load_manifest(malformed)

    invalid_root = _write_manifest(tmp_path / "root.yaml", "42")
    with pytest.raises(ValueError, match="root"):
        load_manifest(invalid_root)


def test_load_manifest_rejects_empty_entries_and_non_mapping_entry(tmp_path: Path) -> None:
    empty = _write_manifest(tmp_path / "empty.yaml", "videos: []")
    with pytest.raises(ValueError, match="no entries"):
        load_manifest(empty)

    non_mapping = _write_manifest(tmp_path / "entry.yaml", "videos:\n  - clip")
    with pytest.raises(ValueError, match="must be a mapping"):
        load_manifest(non_mapping)


def test_verify_files_false_allows_missing_sources_but_true_rejects_them(tmp_path: Path) -> None:
    manifest = _write_manifest(
        tmp_path / "manifest.yaml",
        """
videos:
  - id: clip
    group_id: group
    video: missing.mp4
    annotation: missing.txt
""",
    )
    record = load_manifest(manifest)[0]
    assert record.video_path == (tmp_path / "missing.mp4").resolve()
    with pytest.raises(FileNotFoundError, match="Video file not found"):
        load_manifest(manifest, verify_files=True)


def test_load_manifest_requires_manifest_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Dataset manifest not found"):
        load_manifest(tmp_path / "missing.yaml")



def test_load_manifest_allows_annotation_free_video(tmp_path: Path) -> None:
    video = tmp_path / "unlabeled.mp4"
    video.touch()
    manifest = _write_manifest(
        tmp_path / "manifest.yaml",
        """
videos:
  - id: unlabeled
    group_id: train_extra
    video: unlabeled.mp4
""",
    )

    record = load_manifest(manifest, verify_files=True)[0]
    assert record.video_path == video.resolve()
    assert record.annotation_path is None


def test_load_manifest_allows_explicit_null_annotation(tmp_path: Path) -> None:
    video = tmp_path / "unlabeled.mp4"
    video.touch()
    manifest = _write_manifest(
        tmp_path / "manifest.yaml",
        """
videos:
  - id: unlabeled
    group_id: train_extra
    video: unlabeled.mp4
    annotation: null
""",
    )

    record = load_manifest(manifest, verify_files=True)[0]
    assert record.annotation_path is None
