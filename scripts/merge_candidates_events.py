import csv
from pathlib import Path

CANDIDATES = Path(
    r"output\candidate_events\mentorella_0004_flow_candidate_events.csv"
)

with open(CANDIDATES, encoding="utf-8", newline="") as f:
    events = list(csv.DictReader(f))

events.sort(key=lambda e: float(e["start"]))


def merge_events(events, gap):
    merged = []

    for e in events:
        start = float(e["start"])
        end = float(e["end"])

        if not merged:
            merged.append([start, end, 1])
            continue

        previous = merged[-1]

        if start - previous[1] <= gap:
            # Extend previous event
            previous[1] = max(previous[1], end)
            previous[2] += 1
        else:
            merged.append([start, end, 1])

    return merged


print()
print(f"Original candidate events: {len(events)}")
print()

for gap in [1.5, 2, 3, 5, 7, 10]:
    merged = merge_events(events, gap)

    total_duration = sum(end - start for start, end, _ in merged)
    longest = max(end - start for start, end, _ in merged)

    print(
        f"Gap {gap:4.1f}s -> "
        f"{len(merged):3d} events | "
        f"total duration {total_duration:6.1f}s | "
        f"longest {longest:5.1f}s"
    )

    print("Merged events:")
    for row in merged:
        print(row)