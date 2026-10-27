"""Explicit, manifest-driven cleanup for the motion-feature cache.

NPZ files are not disposable merely because they are unindexed. Cleanup
requires a keep manifest created from a completed regeneration run. Dry-run
is the default; deletion requires --apply.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Iterable

from .cache_catalog import _read_entry, index_existing_cache, list_cache_entries

logger = logging.getLogger(__name__)


def _load_keep_manifest(manifest_path: str | Path, cache_dir: Path) -> set[Path]:
    """Load a JSON list of NPZ paths, absolute or relative to cache_dir."""
    payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not all(isinstance(item, str) for item in payload):
        raise ValueError("Keep manifest must be a JSON array of NPZ path strings.")

    root = cache_dir.resolve()
    keep: set[Path] = set()
    for item in payload:
        candidate = Path(item)
        if not candidate.is_absolute():
            candidate = root / candidate
        candidate = candidate.resolve()
        if candidate.suffix.lower() != ".npz":
            raise ValueError(f"Keep-manifest entry is not an NPZ: {item}")
        if not candidate.is_relative_to(root):
            raise ValueError(f"Keep-manifest entry escapes cache directory: {item}")
        if candidate in keep:
            raise ValueError(f"Duplicate keep-manifest entry: {item}")
        if not candidate.is_file():
            raise ValueError(f"Keep-manifest cache file does not exist: {candidate}")
        entry = _read_entry(candidate)
        if not entry["valid"]:
            raise ValueError(f"Keep-manifest cache file is invalid: {candidate}: {entry['issue']}")
        keep.add(candidate)
    return keep


def cleanup_feature_cache(
    cache_dir: str | Path,
    keep_manifest: str | Path,
    *,
    apply: bool = False,
) -> dict[str, object]:
    """Plan or apply cleanup, preserving exactly the valid NPZs in a manifest.

    No files are deleted unless apply=True. Before deletion, every manifest
    entry is checked for existence and structural validity. The catalog is
    rebuilt after an applied cleanup so stale rows are removed and survivors
    are indexed. NPZ files are never rewritten.
    """
    root = Path(cache_dir).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Feature cache directory does not exist: {root}")

    keep = _load_keep_manifest(keep_manifest, root)
    existing = sorted(path.resolve() for path in root.rglob("*.npz"))
    delete = [path for path in existing if path not in keep]
    report: dict[str, object] = {
        "cache_dir": str(root),
        "manifest_entries": len(keep),
        "npz_found": len(existing),
        "npz_kept": len(keep),
        "npz_obsolete": len(delete),
        "deleted": [],
        "dry_run": not apply,
    }

    if not apply:
        report["delete_candidates"] = [str(path) for path in delete]
        return report

    # Delete only files excluded by the verified allowlist. Do not touch
    # SQLite/WAL/SHM directly; reconcile the catalog through its API.
    deleted: list[str] = []
    for path in delete:
        path.unlink()
        deleted.append(str(path))
    counts = index_existing_cache(root)
    entries = list_cache_entries(root)
    indexed_paths = {Path(entry["cache_path"]).resolve() for entry in entries}
    if indexed_paths != keep:
        raise RuntimeError(
            "Post-cleanup catalog does not match the keep manifest: "
            f"{len(indexed_paths)} indexed, {len(keep)} expected."
        )
    report.update({
        "deleted": deleted,
        "dry_run": False,
        "catalog_rebuild": counts,
        "catalog_entries_after": len(entries),
    })
    return report


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Remove obsolete feature-cache NPZs using a completed-run keep manifest."
    )
    parser.add_argument("--cache-dir", type=Path, required=True, help="Canonical feature-cache directory.")
    parser.add_argument(
        "--keep-manifest", type=Path, required=True,
        help="JSON array of NPZ filenames relative to cache-dir or absolute paths to preserve.",
    )
    parser.add_argument(
        "--apply", action="store_true",
        help="Actually delete NPZs excluded from the manifest. Without this flag, only report candidates.",
    )
    parser.add_argument("--verbose", action="store_true", help="Print the cleanup report as JSON.")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    try:
        report = cleanup_feature_cache(args.cache_dir, args.keep_manifest, apply=args.apply)
    except (OSError, ValueError, json.JSONDecodeError, RuntimeError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
