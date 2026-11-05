from pathlib import Path

import yaml

from autocut.data_models import Annotation, VideoInfo
from autocut.dataset import audit as audit_module
from autocut.dataset.audit import audit_manifest, audit_production_annotations
from autocut.config_loader import load_config


def _run_config(tmp_path: Path):
    path = tmp_path / "autocut.yaml"
    path.write_text(
        "selection:\n  min_clip: 3.0\n",
        encoding="utf-8",
    )
    return load_config(path)


def test_manifest_audit_reports_video_annotations_and_window_estimates(tmp_path, monkeypatch):
    video = tmp_path / "ride.mp4"
    video.touch()
    annotation = tmp_path / "ride.txt"
    annotation.write_text("placeholder", encoding="utf-8")
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        yaml.safe_dump({
            "videos": [{
                "id": "ride",
                "group_id": "session-a",
                "video": str(video),
                "annotation": str(annotation),
            }]
        }),
        encoding="utf-8",
    )
    split_config = tmp_path / "train.yaml"
    split_config.write_text(
        "data:\n  train_groups: [session-a]\n  validation_groups: [session-b]\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        audit_module, "get_video_info",
        lambda _: VideoInfo(width=640, height=360, fps=30.0, duration=10.0),
    )
    monkeypatch.setattr(
        audit_module, "load_training_annotations",
        lambda _: (video, [
            Annotation(start=0.0, end=5.0, score=1, remarks=""),
            Annotation(start=5.0, end=10.0, score=3, remarks=""),
        ]),
    )

    report = audit_manifest(
        manifest, _run_config(tmp_path), training_config=split_config
    )

    assert report["status"] == "error"  # session-b is not in this manifest
    assert report["summary"]["manifest_entries"] == 1
    assert report["summary"]["annotation_intervals"] == 2
    assert report["summary"]["annotation_seconds_by_target"]["1"] == 5.0
    assert report["summary"]["estimated_total_windows"] == 4
    assert report["videos"][0]["window_estimate"]["pure_windows"] == 2
    assert report["videos"][0]["window_estimate"]["conflicting_windows"] == 2
    assert report["summary"]["estimated_pure_windows"] == 2
    assert report["group_split"]["status"] == "invalid"


def test_manifest_audit_warns_when_split_is_not_configured(tmp_path, monkeypatch):
    video = tmp_path / "ride.mp4"
    video.touch()
    annotation = tmp_path / "ride.txt"
    annotation.touch()
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        yaml.safe_dump({"videos": [{
            "id": "ride", "group_id": "session-a",
            "video": str(video), "annotation": str(annotation),
        }]}),
        encoding="utf-8",
    )
    split_config = tmp_path / "train.yaml"
    split_config.write_text("data:\n  train_groups: []\n  validation_groups: []\n")

    monkeypatch.setattr(
        audit_module, "get_video_info",
        lambda _: VideoInfo(width=640, height=360, fps=30.0, duration=10.0),
    )
    monkeypatch.setattr(
        audit_module, "load_training_annotations",
        lambda _: (video, [Annotation(0.0, 10.0, 2, "")]),
    )

    report = audit_manifest(
        manifest, _run_config(tmp_path), training_config=split_config
    )
    assert report["status"] == "warning"
    assert report["group_split"]["status"] == "not_configured"
    assert any(issue["code"] == "split_not_configured" for issue in report["warnings"])


def test_manifest_audit_flags_duplicate_ids_and_out_of_bounds_annotations(tmp_path, monkeypatch):
    video = tmp_path / "ride.mp4"
    video.touch()
    annotation = tmp_path / "ride.txt"
    annotation.touch()
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        yaml.safe_dump({"videos": [
            {"id": "Ride", "group_id": "a", "video": str(video), "annotation": str(annotation)},
            {"id": "ride", "group_id": "a", "video": str(video), "annotation": str(annotation)},
        ]}),
        encoding="utf-8",
    )
    split_config = tmp_path / "train.yaml"
    split_config.write_text("data:\n  train_groups: [a]\n  validation_groups: [b]\n")

    monkeypatch.setattr(
        audit_module, "get_video_info",
        lambda _: VideoInfo(width=640, height=360, fps=30.0, duration=10.0),
    )
    monkeypatch.setattr(
        audit_module, "load_training_annotations",
        lambda _: (video, [Annotation(8.0, 12.0, 2, "")]),
    )

    report = audit_manifest(
        manifest, _run_config(tmp_path), training_config=split_config
    )
    assert report["status"] == "error"
    assert any(issue["code"] == "video_invalid" and "outside" in issue["message"]
               for issue in report["errors"])
    assert any("Duplicate video id" in issue["message"] for issue in report["errors"])


def test_production_audit_flags_clips_beyond_video_duration(tmp_path, monkeypatch):
    video = tmp_path / "ride.mp4"
    video.touch()
    annotations = tmp_path / "edit.txt"
    annotations.write_text(f"{video}\n00:00 00:08 *\n", encoding="utf-8")

    monkeypatch.setattr(
        audit_module, "get_video_info",
        lambda _: VideoInfo(width=640, height=360, fps=30.0, duration=5.0),
    )

    report = audit_production_annotations(annotations, _run_config(tmp_path))
    assert report["status"] == "error"
    assert report["clips"][0]["status"] == "error"
    assert report["errors"][0]["code"] == "clip_out_of_bounds"



def test_annotation_bounds_use_rounded_video_duration(tmp_path, monkeypatch):
    video = tmp_path / "rounding.mp4"
    video.touch()
    annotation = tmp_path / "rounding.txt"
    annotation.touch()
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        yaml.safe_dump({"videos": [{
            "id": "rounding", "group_id": "session",
            "video": str(video), "annotation": str(annotation),
        }]}),
        encoding="utf-8",
    )
    split_config = tmp_path / "train.yaml"
    split_config.write_text(
        "data:\n  train_groups: [session]\n  validation_groups: [other]\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        audit_module, "get_video_info",
        lambda _: VideoInfo(width=640, height=360, fps=30.0, duration=59.99),
    )
    monkeypatch.setattr(
        audit_module, "load_training_annotations",
        lambda _: (video, [Annotation(55.0, 60.0, 2, "")]),
    )

    report = audit_manifest(
        manifest, _run_config(tmp_path), training_config=split_config
    )
    assert not any("outside" in issue["message"] for issue in report["errors"])
    assert report["videos"][0]["video_info"]["duration_seconds"] == 59.99
    assert report["videos"][0]["annotation_boundary_duration_seconds"] == 60.0


def test_annotation_bounds_still_reject_more_than_rounded_second(tmp_path, monkeypatch):
    video = tmp_path / "rounding.mp4"
    video.touch()
    annotation = tmp_path / "rounding.txt"
    annotation.touch()
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        yaml.safe_dump({"videos": [{
            "id": "rounding", "group_id": "session",
            "video": str(video), "annotation": str(annotation),
        }]}),
        encoding="utf-8",
    )
    split_config = tmp_path / "train.yaml"
    split_config.write_text(
        "data:\n  train_groups: [session]\n  validation_groups: [other]\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        audit_module, "get_video_info",
        lambda _: VideoInfo(width=640, height=360, fps=30.0, duration=59.49),
    )
    monkeypatch.setattr(
        audit_module, "load_training_annotations",
        lambda _: (video, [Annotation(55.0, 60.0, 2, "")]),
    )

    report = audit_manifest(
        manifest, _run_config(tmp_path), training_config=split_config
    )
    assert any("rounded video duration" in issue["message"] for issue in report["errors"])



def test_manifest_audit_preserves_legacy_cwd_relative_path_resolution(
    tmp_path: Path, monkeypatch
):
    manifest_dir = tmp_path / "manifest_dir"
    cwd_dir = tmp_path / "working_dir"
    manifest_dir.mkdir()
    cwd_dir.mkdir()

    manifest_video = manifest_dir / "ride.mp4"
    manifest_annotation = manifest_dir / "ride.txt"
    cwd_video = cwd_dir / "ride.mp4"
    cwd_annotation = cwd_dir / "ride.txt"
    for path in (manifest_video, manifest_annotation, cwd_video, cwd_annotation):
        path.touch()

    manifest = manifest_dir / "manifest.yaml"
    manifest.write_text(
        yaml.safe_dump({"videos": [{
            "id": "ride",
            "group_id": "session-a",
            "video": "ride.mp4",
            "annotation": "ride.txt",
        }]}),
        encoding="utf-8",
    )
    split_config = tmp_path / "train.yaml"
    split_config.write_text(
        "data:\n  train_groups: []\n  validation_groups: []\n",
        encoding="utf-8",
    )

    monkeypatch.chdir(cwd_dir)
    monkeypatch.setattr(
        audit_module,
        "get_video_info",
        lambda _: VideoInfo(width=640, height=360, fps=30.0, duration=10.0),
    )
    monkeypatch.setattr(
        audit_module,
        "load_training_annotations",
        lambda _: (cwd_video, [Annotation(0.0, 10.0, 2, "")]),
    )

    report = audit_manifest(
        manifest, _run_config(tmp_path), training_config=split_config
    )

    assert report["videos"][0]["video"] == str(cwd_video.resolve())
    assert report["videos"][0]["annotation"] == str(cwd_annotation.resolve())
