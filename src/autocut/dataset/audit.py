"""Source-data audit for AutoCut.

This first audit layer validates manifests, annotation files, source videos,
interval bounds, and configured group splits. Feature-cache compatibility is
reported as not checked until the cache artifact contract is implemented.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

import yaml

from ..annotations import load_annotations, load_training_annotations
from ..config_loader import RunConfig
from ..video.probe import get_video_info


def _resolve_path(value: object, *, relative_to: Path) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("path must be a non-empty string")
    path = Path(value.strip()).expanduser()
    if path.is_absolute():
        return path
    # Scanner manifests conventionally store paths relative to the project
    # root. Prefer the current working directory, then the manifest directory.
    cwd_candidate = path.resolve()
    if cwd_candidate.exists():
        return cwd_candidate
    return (relative_to / path).resolve()


def _add_issue(report: dict[str, Any], severity: str, code: str, message: str) -> None:
    report[severity].append({"code": code, "message": message})


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8-sig") as stream:
            document = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML manifest {path}: {exc}") from exc

    if not isinstance(document, Mapping):
        raise ValueError(f"Manifest root must be a mapping: {path}")
    videos = document.get("videos")
    if not isinstance(videos, list):
        raise ValueError(f"Manifest must contain a 'videos' list: {path}")
    if not videos:
        raise ValueError(f"Manifest contains no videos: {path}")
    for index, item in enumerate(videos, start=1):
        if not isinstance(item, Mapping):
            raise ValueError(f"Manifest videos[{index - 1}] must be a mapping")
    return [dict(item) for item in videos]


def _read_split_config(path: Path, report: dict[str, Any], manifest_groups: set[str]) -> None:
    if not path.is_file():
        _add_issue(
            report, "warnings", "split_config_missing",
            f"Group split was not checked because training config was not found: {path}",
        )
        report["group_split"] = {"status": "not_checked", "config": str(path)}
        return

    try:
        with path.open("r", encoding="utf-8-sig") as stream:
            raw = yaml.safe_load(stream)
        data = raw.get("data", {}) if isinstance(raw, Mapping) else {}
        train_groups = data.get("train_groups", [])
        validation_groups = data.get("validation_groups", [])
        if not isinstance(train_groups, list) or not isinstance(validation_groups, list):
            raise ValueError("data.train_groups and data.validation_groups must be lists")
        if not train_groups or not validation_groups:
            _add_issue(
                report, "warnings", "split_not_configured",
                "Training/validation group split is incomplete; set non-empty "
                "data.train_groups and data.validation_groups in the training YAML.",
            )
            report["group_split"] = {
                "status": "not_configured",
                "train_groups": train_groups,
                "validation_groups": validation_groups,
                "unassigned_groups": sorted(manifest_groups),
            }
            return
        if any(not isinstance(group, str) or not group.strip()
               for group in train_groups + validation_groups):
            raise ValueError("group IDs must be non-empty strings")
        train = [group.strip() for group in train_groups]
        validation = [group.strip() for group in validation_groups]
        overlap = sorted(set(train) & set(validation))
        unknown = sorted((set(train) | set(validation)) - manifest_groups)
        unassigned = sorted(manifest_groups - (set(train) | set(validation)))
        if overlap:
            _add_issue(
                report, "errors", "split_overlap",
                "Groups occur in both training and validation: " + ", ".join(overlap),
            )
        if unknown:
            _add_issue(
                report, "errors", "split_unknown_group",
                "Configured groups are absent from the manifest: " + ", ".join(unknown),
            )
        if unassigned:
            _add_issue(
                report, "warnings", "split_unassigned_group",
                "Manifest groups are not assigned to either split: " + ", ".join(unassigned),
            )
        report["group_split"] = {
            "status": "valid" if not overlap and not unknown else "invalid",
            "train_groups": train,
            "validation_groups": validation,
            "unassigned_groups": unassigned,
            "unknown_groups": unknown,
            "overlap_groups": overlap,
        }
    except (OSError, AttributeError, TypeError, ValueError, yaml.YAMLError) as exc:
        _add_issue(report, "errors", "split_config_invalid", f"Cannot validate group split in {path}: {exc}")
        report["group_split"] = {"status": "invalid", "config": str(path)}


def _annotation_boundary_duration(duration: float) -> float:
    """Round a probed duration to the whole-second precision of annotations.

    Annotation files are authored with whole-second boundaries, while
    container durations often include fractional seconds that are not useful
    for deciding whether such a boundary is valid. Keep the raw duration for
    reporting and window calculations; use this rounded value only for the
    annotation-boundary check.
    """
    return float(math.floor(duration + 0.5))


def _estimated_windows(duration: float, window: float, stride: float) -> int:
    if duration + 1e-9 < window:
        return 0
    return max(0, int(math.floor((duration - window + 1e-9) / stride)) + 1)


def audit_manifest(
    source: str | Path,
    settings: RunConfig,
    *,
    training_config: str | Path = "config/train.yaml",
) -> dict[str, Any]:
    """Audit all source videos and training annotations listed in a YAML manifest."""
    manifest_path = Path(source).expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Dataset manifest not found: {manifest_path}")

    report: dict[str, Any] = {
        "source": str(manifest_path),
        "kind": "dataset_manifest",
        "status": "ok",
        "summary": {},
        "videos": [],
        "errors": [],
        "warnings": [],
        "feature_cache": {
            "status": "not_checked",
            "reason": "Feature-cache compatibility is not yet implemented in this audit layer.",
            "cache_dir": str(settings.cache_dir),
        },
        "window_estimates": {
            "window_duration_seconds": settings.common.window_duration,
            "stride_seconds": settings.common.window_stride,
            "total_windows": 0,
            "annotated_windows_estimate": 0,
        },
    }

    items = _load_manifest(manifest_path)
    seen_ids: set[str] = set()
    manifest_groups: set[str] = set()
    total_duration = 0.0
    total_annotations = 0
    total_windows = 0
    total_labeled_windows = 0
    target_seconds = {str(i): 0.0 for i in range(4)}

    for index, item in enumerate(items, start=1):
        video_id = item.get("id")
        group_id = item.get("group_id")
        video_label = str(video_id) if isinstance(video_id, str) and video_id.strip() else f"item #{index}"
        entry: dict[str, Any] = {
            "id": video_label,
            "group_id": group_id if isinstance(group_id, str) else None,
            "status": "ok",
            "errors": [],
            "warnings": [],
        }

        if not isinstance(video_id, str) or not video_id.strip():
            entry["errors"].append("Missing or empty id.")
        else:
            normalized_id = video_id.strip().casefold()
            if normalized_id in seen_ids:
                entry["errors"].append(f"Duplicate video id (case-insensitive): {video_id!r}.")
            seen_ids.add(normalized_id)

        if not isinstance(group_id, str) or not group_id.strip():
            entry["errors"].append("Missing or empty group_id.")
        else:
            manifest_groups.add(group_id.strip())

        try:
            video_path = _resolve_path(item.get("video"), relative_to=manifest_path.parent)
            entry["video"] = str(video_path)
            if not video_path.is_file():
                raise FileNotFoundError(f"Video file not found: {video_path}")
            info = get_video_info(video_path)
            entry["video_info"] = {
                "width": info.width,
                "height": info.height,
                "fps": info.fps,
                "duration_seconds": info.duration,
            }
            total_duration += info.duration
            duration = info.duration
        except (OSError, RuntimeError, ValueError) as exc:
            entry["errors"].append(str(exc))
            duration = None
            video_path = None

        annotations: list[Any] = []
        annotation_path: Path | None = None
        try:
            annotation_path = _resolve_path(item.get("annotation"), relative_to=manifest_path.parent)
            entry["annotation"] = str(annotation_path)
            parsed_video, annotations = load_training_annotations(annotation_path)
            if video_path is not None and parsed_video.resolve() != video_path.resolve():
                entry["warnings"].append(
                    "Video path in annotation file differs from the manifest video path: "
                    f"{parsed_video} != {video_path}"
                )
            entry["annotation_count"] = len(annotations)
            entry["annotated_seconds"] = round(sum(a.end - a.start for a in annotations), 3)
            entry["target_seconds"] = {
                str(target): round(sum(a.end - a.start for a in annotations if a.score == target), 3)
                for target in range(4)
            }
            total_annotations += len(annotations)
            for target in range(4):
                target_seconds[str(target)] += sum(
                    a.end - a.start for a in annotations if a.score == target
                )
            if duration is not None:
                boundary_duration = _annotation_boundary_duration(duration)
                entry["annotation_boundary_duration_seconds"] = boundary_duration
                out_of_bounds = [
                    a for a in annotations
                    if a.start < -1e-6 or a.end > boundary_duration + 1e-6
                ]
                if out_of_bounds:
                    entry["errors"].append(
                        f"{len(out_of_bounds)} annotation interval(s) fall outside "
                        f"the rounded video duration (0–{boundary_duration:.0f}s; "
                        f"raw probe duration {duration:.6f}s)."
                    )
                windows = _estimated_windows(
                    duration, settings.common.window_duration, settings.common.window_stride
                )
                labeled = 0
                for window_index in range(windows):
                    start = window_index * settings.common.window_stride
                    end = start + settings.common.window_duration
                    if any(min(end, a.end) > max(start, a.start) for a in annotations):
                        labeled += 1
                entry["window_estimate"] = {
                    "total": windows,
                    "overlapping_annotations": labeled,
                }
                total_windows += windows
                total_labeled_windows += labeled
        except (OSError, RuntimeError, ValueError, TypeError) as exc:
            entry["errors"].append(f"Annotation validation failed: {exc}")

        if entry["errors"]:
            entry["status"] = "error"
            for message in entry["errors"]:
                _add_issue(report, "errors", "video_invalid", f"{video_label}: {message}")
        for message in entry["warnings"]:
            _add_issue(report, "warnings", "video_warning", f"{video_label}: {message}")
        report["videos"].append(entry)

    _read_split_config(Path(training_config).expanduser(), report, manifest_groups)
    report["summary"] = {
        "manifest_entries": len(items),
        "unique_ids": len(seen_ids),
        "groups": sorted(manifest_groups),
        "videos_with_errors": sum(bool(v["errors"]) for v in report["videos"]),
        "total_video_duration_seconds": round(total_duration, 3),
        "annotation_intervals": total_annotations,
        "annotation_seconds_by_target": {
            key: round(value, 3) for key, value in target_seconds.items()
        },
        "estimated_total_windows": total_windows,
        "estimated_windows_overlapping_annotations": total_labeled_windows,
    }
    if report["errors"]:
        report["status"] = "error"
    elif report["warnings"]:
        report["status"] = "warning"
    return report


def audit_production_annotations(source: str | Path, settings: RunConfig) -> dict[str, Any]:
    """Audit a production annotation TXT and probe its referenced media."""
    path = Path(source).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Annotation file not found: {path}")

    report: dict[str, Any] = {
        "source": str(path),
        "kind": "production_annotations",
        "status": "ok",
        "clips": [],
        "errors": [],
        "warnings": [],
    }
    clips, audio = load_annotations(path, settings.common)
    durations: dict[str, float] = {}
    for clip in clips:
        video_path = Path(clip.video_name)
        entry = {
            "video": str(video_path),
            "start": clip.start,
            "end": clip.end,
            "duration": clip.duration,
            "mandatory": clip.mandatory,
            "status": "ok",
        }
        try:
            info = get_video_info(video_path)
            durations[str(video_path)] = info.duration
            boundary_duration = _annotation_boundary_duration(info.duration)
            entry["annotation_boundary_duration_seconds"] = boundary_duration
            if clip.end > boundary_duration + 1e-6:
                entry["status"] = "error"
                message = (
                    f"Clip {clip.start:.3f}-{clip.end:.3f}s exceeds "
                    f"rounded video duration {boundary_duration:.0f}s "
                    f"(raw probe duration {info.duration:.6f}s): {video_path}"
                )
                entry["error"] = message
                _add_issue(report, "errors", "clip_out_of_bounds", message)
            entry["video_duration_seconds"] = info.duration
        except (OSError, RuntimeError, ValueError) as exc:
            entry["status"] = "error"
            entry["error"] = str(exc)
            _add_issue(report, "errors", "video_invalid", str(exc))
        report["clips"].append(entry)

    report["audio"] = None if audio is None else {"path": audio.path, "mix": audio.mix}
    report["summary"] = {
        "clips": len(clips),
        "videos": len(durations),
        "mandatory_clips": sum(clip.mandatory for clip in clips),
        "total_requested_duration_seconds": round(sum(clip.duration for clip in clips), 3),
    }
    if report["errors"]:
        report["status"] = "error"
    return report


def audit_source(
    source: str | Path,
    settings: RunConfig,
    *,
    training_config: str | Path = "config/train.yaml",
) -> dict[str, Any]:
    """Select manifest or production-annotation auditing by file extension."""
    path = Path(source).expanduser()
    if path.suffix.lower() in {".yaml", ".yml"}:
        return audit_manifest(path, settings, training_config=training_config)
    return audit_production_annotations(path, settings)


def print_audit_report(report: Mapping[str, Any]) -> None:
    """Print a stable, human-readable JSON report."""
    print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
