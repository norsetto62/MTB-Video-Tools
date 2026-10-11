import re, subprocess, yaml, shutil
from pathlib import Path, PureWindowsPath

root = Path("data/annotations")
items = []

def seconds(s):
    p = [int(x) for x in s.split(":")]
    return p[0]*3600+p[1]*60+p[2] if len(p)==3 else p[0]*60+p[1]

def validate_items(items):
    errors = []
    seen_ids = set()

    ffprobe = shutil.which("ffprobe")

    for index, item in enumerate(items):
        item_label = item.get("id") or f"item #{index + 1}"

        # 1. Duplicate or empty IDs
        video_id = item.get("id")
        if not isinstance(video_id, str) or not video_id.strip():
            errors.append(f"{item_label}: missing or empty id")
        else:
            normalized_id = video_id.strip().casefold()
            if normalized_id in seen_ids:
                errors.append(f"{item_label}: duplicate id {video_id!r}")
            else:
                seen_ids.add(normalized_id)

        # 2. Missing or unprobeable video
        video = item.get("video")
        if not isinstance(video, str) or not video.strip():
            errors.append(f"{item_label}: missing or empty video path")
            continue

        video_path = Path(video)
        if not video_path.is_file():
            errors.append(f"{item_label}: video file not found: {video}")
            continue

        if ffprobe:
            try:
                result = subprocess.run(
                    [
                        ffprobe, "-v", "error",
                        "-show_entries", "format=duration",
                        "-of", "default=noprint_wrappers=1:nokey=1",
                        str(video_path),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=15,
                    check=False,
                )
                if result.returncode != 0:
                    errors.append(
                        f"{item_label}: ffprobe failed for {video}: "
                        f"{result.stderr.strip()}"
                    )
                else:
                    try:
                        duration = float(result.stdout.strip())
                        if duration <= 0:
                            errors.append(
                                f"{item_label}: invalid video duration: {duration}"
                            )
                    except ValueError:
                        errors.append(
                            f"{item_label}: ffprobe returned invalid duration"
                        )
            except (OSError, subprocess.TimeoutExpired) as exc:
                errors.append(f"{item_label}: could not probe video: {exc}")
        else:
            errors.append(
                f"{item_label}: ffprobe not found; video duration not verified"
            )

        # 4. Empty or missing group ID
        group_id = item.get("group_id")
        if not isinstance(group_id, str) or not group_id.strip():
            errors.append(f"{item_label}: missing or empty group_id")

    return errors


for f in sorted(root.rglob("*")):
    if not f.is_file() or f.name.lower() == "readme.md" or f.suffix.lower() not in {".txt", ".ann"}:
        continue
    lines = f.read_text(encoding="utf-8-sig", errors="replace").splitlines()
    video = next((x.strip() for x in lines if x.strip() and not x.lstrip().startswith("#") and not x.lstrip().lower().startswith("start")), "")
    if not video:
        continue
    intervals = []
    for line in lines[1:]:
        m = re.match(r"^\s*(\d{1,2}:\d{2}(?::\d{2})?)\s+(\d{1,2}:\d{2}(?::\d{2})?)\s+([0-3])(?:\s|$)", line)
        if m:
            a, b, target = seconds(m[1]), seconds(m[2]), int(m[3])
            if b > a:
                intervals.append((a, b, target))
    if not intervals:
        print(f"WARNING: no target intervals parsed: {f}")
        continue
    duration = max(b for a,b,t in intervals)
    try:
        p = subprocess.run(["ffprobe","-v","error","-show_entries","format=duration","-of","default=noprint_wrappers=1:nokey=1",video], capture_output=True, text=True, timeout=10)
        if p.returncode == 0 and float(p.stdout.strip()) > 0:
            duration = float(p.stdout.strip())
    except (OSError, ValueError, subprocess.TimeoutExpired):
        pass
    totals = {str(t): sum(b-a for a,b,k in intervals if k == t) for t in range(4)}
    covered = sum(b-a for a,b,t in intervals)
    vidpath = Path(video.replace("\\", "/"))
    stem = f.stem.lower().replace(" ", "_")
    group_id = PureWindowsPath(video).parent.name.lower().replace(" ", "_")
    items.append({
        "id": stem, "group_id": group_id, "video": video,
        "annotation": f.as_posix(), "duration_seconds": round(duration, 2),
        "annotated_seconds": round(covered, 2),
        "annotation_coverage_pct": round(100*min(covered,duration)/duration, 2) if duration else 0,
        "target_seconds": {k: round(v, 2) for k,v in totals.items()},
        "target_pct_of_video": {k: round(100*v/duration, 2) if duration else 0 for k,v in totals.items()},
    })

errors = validate_items(items)

if errors:
    print("Manifest validation failed:")
    for error in errors:
        print(f"  - {error}")
    raise SystemExit(1)

# Continue to write the YAML here.
items.sort(key=lambda x: (x["id"], x["group_id"], x["video"], x["annotation"]))
out = Path("data/dataset_manifest.yaml")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(yaml.safe_dump({"videos": items}, sort_keys=False, allow_unicode=True), encoding="utf-8")
print(out.read_text(encoding="utf-8"))
print(f"# Scanned {len(items)} annotation files; saved {out}", flush=True)
