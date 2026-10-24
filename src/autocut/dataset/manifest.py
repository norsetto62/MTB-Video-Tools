"""Load and validate YAML dataset manifests for AutoCut.

Manifest paths are resolved relative to the manifest file's directory, making
resolution independent of the process working directory. Media probing and
annotation parsing are deliberately left to higher-level validation layers.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DatasetRecord:
    """A validated manifest entry for one source video and its annotations."""

    id: str
    group_id: str
    video_path: Path
    annotation_path: Path | None
    manifest_path: Path

    @property
    def video_name(self) -> str:
        """Alias for the unique manifest entry ID."""
        return self.id


def _resolve_manifest_path(
    value: object,
    *,
    field: str,
    entry_id: str,
    base: Path,
    prefer_cwd: bool = False,
) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"Manifest entry {entry_id!r} has invalid {field!r} path; "
            "expected a non-empty string."
        )

    path = Path(value.strip()).expanduser()
    if not path.is_absolute():
        if prefer_cwd:
            cwd_candidate = path.resolve()
            if cwd_candidate.exists():
                return cwd_candidate
        path = base / path
    try:
        return path.resolve()
    except (OSError, RuntimeError) as exc:
        raise ValueError(
            f"Cannot resolve {field!r} path for manifest entry {entry_id!r}: {value!r}"
        ) from exc


def _read_entries(raw: Any, manifest_path: Path) -> list[Any]:
    if isinstance(raw, list):
        entries = raw
    elif isinstance(raw, Mapping):
        # "videos" is the established AutoCut manifest schema. The aliases
        # allow simple standalone manifests without changing that schema.
        key = next(
            (name for name in ("videos", "entries", "records", "datasets") if name in raw),
            None,
        )
        if key is None:
            raise ValueError(
                f"Manifest {manifest_path} must be a list or contain a 'videos' list."
            )
        entries = raw[key]
        if not isinstance(entries, list):
            raise ValueError(f"Manifest field {key!r} must be a list: {manifest_path}")
    else:
        raise ValueError(
            f"Manifest root in {manifest_path} must be a list or mapping."
        )

    if not entries:
        raise ValueError(f"Dataset manifest contains no entries: {manifest_path}")
    return entries


def load_manifest(
    path: str | Path,
    *,
    verify_files: bool = False,
    prefer_cwd: bool = False,
) -> tuple[DatasetRecord, ...]:
    """Parse a YAML manifest into validated records in manifest order.

    Relative video and annotation paths are resolved against the manifest
    directory by default. `prefer_cwd=True` enables the legacy audit
    convention: use a matching path from the current working directory first,
    falling back to the manifest directory.

    Args:
        path: YAML manifest path.
        verify_files: If true, require each referenced video and annotation
            path to be an existing regular file. Does not probe video media or
            parse annotations.
        prefer_cwd: Preserve legacy current-working-directory-first resolution.
            New dataset consumers should leave this false.

    Raises:
        FileNotFoundError: If the manifest or a referenced file (when
            verify_files is true) is missing.
        ValueError: If YAML or manifest structure/fields are invalid.
    """
    manifest_path = Path(path).expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Dataset manifest not found: {manifest_path}")

    try:
        with manifest_path.open("r", encoding="utf-8-sig") as stream:
            raw = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in manifest {manifest_path}: {exc}") from exc

    if raw is None:
        raise ValueError(f"Dataset manifest is empty: {manifest_path}")

    entries = _read_entries(raw, manifest_path)
    records: list[DatasetRecord] = []
    seen_ids: set[str] = set()
    required_fields = ("id", "group_id", "video")

    for index, item in enumerate(entries):
        if not isinstance(item, Mapping):
            raise ValueError(
                f"Manifest entry at index {index} must be a mapping/dictionary."
            )

        missing = [field for field in required_fields if field not in item]
        if missing:
            raise ValueError(
                f"Manifest entry at index {index} is missing required field(s): "
                + ", ".join(missing)
            )

        entry_id = item["id"]
        if not isinstance(entry_id, str) or not entry_id.strip():
            raise ValueError(
                f"Manifest entry at index {index} has invalid non-empty string 'id'."
            )
        entry_id = entry_id.strip()

        # Match audit's existing duplicate-ID policy and avoid IDs that collide
        # on case-insensitive filesystems such as the default Windows setup.
        normalized_id = entry_id.casefold()
        if normalized_id in seen_ids:
            raise ValueError(
                f"Duplicate video ID found in manifest (case-insensitive): {entry_id!r}."
            )
        seen_ids.add(normalized_id)

        group_id = item["group_id"]
        if not isinstance(group_id, str) or not group_id.strip():
            raise ValueError(
                f"Manifest entry {entry_id!r} has invalid non-empty string 'group_id'."
            )
        group_id = group_id.strip()

        video_path = _resolve_manifest_path(
            item["video"],
            field="video",
            entry_id=entry_id,
            base=manifest_path.parent,
            prefer_cwd=prefer_cwd,
        )
        annotation_value = item.get("annotation")
        annotation_path = None
        if annotation_value is not None:
            annotation_path = _resolve_manifest_path(
                annotation_value,
                field="annotation",
                entry_id=entry_id,
                base=manifest_path.parent,
                prefer_cwd=prefer_cwd,
            )

        if verify_files:
            if not video_path.is_file():
                raise FileNotFoundError(
                    f"Video file not found for entry {entry_id!r}: {video_path}"
                )
            if annotation_path is not None and not annotation_path.is_file():
                raise FileNotFoundError(
                    f"Annotation file not found for entry {entry_id!r}: {annotation_path}"
                )

        records.append(
            DatasetRecord(
                id=entry_id,
                group_id=group_id,
                video_path=video_path,
                annotation_path=annotation_path,
                manifest_path=manifest_path,
            )
        )

    return tuple(records)


def get_unique_groups(records: Sequence[DatasetRecord]) -> tuple[str, ...]:
    """Return sorted unique group IDs for deterministic fold generation."""
    return tuple(sorted({record.group_id for record in records}))


def get_records_for_group(
    records: Sequence[DatasetRecord], group_id: str
) -> tuple[DatasetRecord, ...]:
    """Return records in group_id, preserving input order."""
    return tuple(record for record in records if record.group_id == group_id)
