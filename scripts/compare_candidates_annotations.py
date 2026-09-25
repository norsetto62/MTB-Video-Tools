import csv
import re
from pathlib import Path


CANDIDATES = Path(
    r"output\candidate_events\mentorella_flow_candidate_events.csv"
)

ANNOTATIONS = Path(
    r"data\annotations\Mentorella.txt"
)

MERGE_GAP = 5.0


def timestamp_to_seconds(ts):
    minutes, seconds = map(int, ts.split(":"))
    return minutes * 60 + seconds


def seconds_to_timestamp(seconds):
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"


# ------------------------------------------------------------
# Read candidate events
# ------------------------------------------------------------

with open(CANDIDATES, encoding="utf-8", newline="") as f:
    events = list(csv.DictReader(f))

events.sort(key=lambda e: float(e["start"]))


# ------------------------------------------------------------
# Merge nearby candidate events
# ------------------------------------------------------------

merged = []

for event in events:

    start = float(event["start"])
    end = float(event["end"])

    if not merged:
        merged.append({
            "start": start,
            "end": end,
            "count": 1,
        })
        continue

    previous = merged[-1]

    gap = start - previous["end"]

    if gap <= MERGE_GAP:
        previous["end"] = max(previous["end"], end)
        previous["count"] += 1
    else:
        merged.append({
            "start": start,
            "end": end,
            "count": 1,
        })


# ------------------------------------------------------------
# Read annotations
# ------------------------------------------------------------

annotations = []

for line in ANNOTATIONS.read_text(encoding="utf-8").splitlines():

    m = re.match(
        r"^\s*(\d{2}:\d{2})\s+"
        r"(\d{2}:\d{2})\s+"
        r"(\d+)\s+"
        r"(\d+)\s+(.*)$",
        line,
    )

    if not m:
        continue

    start, end, mtb, video, remarks = m.groups()

    annotations.append({
        "start": timestamp_to_seconds(start),
        "end": timestamp_to_seconds(end),
        "mtb": int(mtb),
        "video": int(video),
        "remarks": remarks.strip(),
    })


# ------------------------------------------------------------
# Compare merged sequences with annotations
# ------------------------------------------------------------

print()
print("=" * 90)
print(f"Original candidate events: {len(events)}")
print(f"Merge gap:                 {MERGE_GAP:.1f} s")
print(f"Merged candidate events:   {len(merged)}")
print(f"Annotations:               {len(annotations)}")
print("=" * 90)

mtb4_hit = 0

for i, ann in enumerate(annotations, 1):

    ann_start = ann["start"]
    ann_end = ann["end"]
    ann_duration = ann_end - ann_start

    overlapping = []

    for event in merged:

        overlap_start = max(event["start"], ann_start)
        overlap_end = min(event["end"], ann_end)

        if overlap_start < overlap_end:
            overlapping.append(event)

    # --------------------------------------------------------
    # Calculate union of candidate coverage
    # --------------------------------------------------------

    ranges = []

    for event in overlapping:
        ranges.append((
            max(event["start"], ann_start),
            min(event["end"], ann_end),
        ))

    ranges.sort()

    merged_ranges = []

    for start, end in ranges:

        if not merged_ranges or start > merged_ranges[-1][1]:
            merged_ranges.append([start, end])
        else:
            merged_ranges[-1][1] = max(
                merged_ranges[-1][1],
                end
            )

    covered_seconds = sum(
        end - start
        for start, end in merged_ranges
    )

    coverage = (
        100 * covered_seconds / ann_duration
        if ann_duration > 0
        else 0
    )

    hit = bool(overlapping)

    if ann["mtb"] == 4 and hit:
        mtb4_hit += 1

    print(
        f"{i:2d}  "
        f"{seconds_to_timestamp(ann_start)}–"
        f"{seconds_to_timestamp(ann_end)}  "
        f"MTB {ann['mtb']}  "
        f"Sequences: {len(overlapping):2d}  "
        f"Coverage: {coverage:5.1f}%  "
        f"Hit: {'YES' if hit else 'NO'}"
    )

    print(f"    {ann['remarks']}")


# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------

inside = 0

for event in merged:

    if any(
        event["start"] < ann["end"]
        and event["end"] > ann["start"]
        for ann in annotations
    ):
        inside += 1

outside = len(merged) - inside

print()
print("=" * 90)
print("SUMMARY")
print("=" * 90)

print(f"Merged sequences:             {len(merged)}")
print(f"Inside annotations:           {inside}")
print(f"Outside annotations:          {outside}")
print(f"Percentage inside:             {100 * inside / len(merged):.1f}%")
print()
print(f"MTB 4 intervals hit:           {mtb4_hit}/6")
print(f"MTB 4 intervals missed:        {6 - mtb4_hit}/6")
print()

# ------------------------------------------------------------
# Print merged sequences OUTSIDE all annotations
# ------------------------------------------------------------

print()
print("=" * 90)
print("SEQUENCES OUTSIDE ALL ANNOTATIONS")
print("=" * 90)

outside_sequences = []

for i, event in enumerate(merged, 1):

    overlapping = any(
        event["start"] < ann["end"]
        and event["end"] > ann["start"]
        for ann in annotations
    )

    if not overlapping:
        outside_sequences.append(event)

for i, event in enumerate(outside_sequences, 1):

    start = event["start"]
    end = event["end"]
    duration = end - start

    print(
        f"{i:2d}  "
        f"{seconds_to_timestamp(start)}–"
        f"{seconds_to_timestamp(end)}  "
        f"Duration: {duration:5.1f}s  "
        f"Original events merged: {event['count']:2d}"
    )

print()
print(f"Outside sequences: {len(outside_sequences)}")