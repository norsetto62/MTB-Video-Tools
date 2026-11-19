"""Export and visualize telemetry from an Avinox .proto ride recording.

Usage:
    python tools/parse_avinox.py path/to/ride.proto
    python tools/parse_avinox.py path/to/ride.proto --output-dir data/avinox --show
"""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from autocut.dataset.avinox_proto import parse  # noqa: E402


def _json_default(value: Any) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def export_csv(samples: list[dict[str, Any]], destination: Path) -> Path:
    """Write all decoded channels, one telemetry sample per row."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    columns = list(dict.fromkeys(key for sample in samples for key in sample))
    with destination.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for sample in samples:
            row = dict(sample)
            if isinstance(row.get("date"), datetime):
                # Keep the original UTC timestamp and add a local-time companion.
                row["date_local"] = row["date"].astimezone().isoformat()
                row["date"] = row["date"].isoformat()
            writer.writerow(row)
    return destination


def export_metadata(metadata: dict[str, Any], destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    return destination


def plot_ride(samples: list[dict[str, Any]], metadata: dict[str, Any],
              destination: Path, *, show: bool = False) -> Path:
    """Save separate time-series panels and a GPS track when coordinates exist."""
    try:
        import matplotlib
        if not show:
            matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection
        from matplotlib.colors import Normalize
    except ImportError as exc:
        raise ImportError(
            "Visualization requires matplotlib. Install it with: "
            "python -m pip install -e '.[plot]'"
        ) from exc

    if not samples:
        raise ValueError("The recording contains no telemetry samples.")

    origin = samples[0].get("timestamp") or 0
    times = [
        (sample.get("timestamp") - origin) if sample.get("timestamp") is not None else i
        for i, sample in enumerate(samples)
    ]

    def values(key: str) -> list[float]:
        result = []
        for sample in samples:
            value = sample.get(key)
            result.append(float("nan") if value is None else float(value))
        return result

    gps = [
        (i, float(sample["longitude"]), float(sample["latitude"]))
        for i, sample in enumerate(samples)
        if sample.get("longitude") is not None
        and sample.get("latitude") is not None
        and -180 <= float(sample["longitude"]) <= 180
        and -90 <= float(sample["latitude"]) <= 90
    ]
    has_gps = len(gps) >= 2

    # GPS gets its own map-like panel; otherwise use four telemetry panels.
    panel_count = 5 if has_gps else 4
    fig, axes = plt.subplots(panel_count, 1, figsize=(16, 3.0 * panel_count),
                             sharex=False)
    start = datetime.fromtimestamp(metadata["start"], timezone.utc).astimezone() if metadata.get("start") else None
    end = datetime.fromtimestamp(metadata["end"], timezone.utc).astimezone() if metadata.get("end") else None
    time_range = (
        f"Start: {start:%Y-%m-%d %H:%M:%S %Z}  |  "
        f"End: {end:%Y-%m-%d %H:%M:%S %Z}"
        if start and end else "Start/end time unavailable"
    )
    fig.suptitle(
        f"{metadata.get('fileName', 'Avinox ride')} — "
        f"{len(samples)} samples, {metadata.get('duration', 0)} s\\n{time_range}",
        fontsize=13, fontweight="bold",
    )

    ax = axes[0]
    ax.plot(times, values("speed"), label="Speed")
    ax.set_ylabel("Speed (km/h)")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.25)

    ax = axes[1]
    ax.plot(times, values("riderPower"), label="Rider power")
    ax.plot(times, values("motorPower"), label="Motor power")
    ax.plot(times, values("totalPower"), label="Total power", alpha=0.75)
    ax.set_ylabel("Power (W)")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.25)

    ax = axes[2]
    ax.plot(times, values("gradient"), label="Recorded gradient")
    ax.set_ylabel("Gradient (raw / 100)")
    ax.set_title("Recorded gradient — physical unit not yet verified", loc="left")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.25)

    ax = axes[3]
    ax.plot(times, values("imuX"), label="IMU X")
    ax.plot(times, values("imuY"), label="IMU Y")
    ax.plot(times, values("imuZ"), label="IMU Z")
    ax.set_ylabel("IMU raw float32")
    ax.set_title("IMU axes — raw values, independent scale", loc="left")
    ax.legend(loc="upper right", ncol=3)
    ax.grid(alpha=0.25)
    ax.set_xlabel("Elapsed time (s)")

    if has_gps:
        ax = axes[4]
        lons = [p[1] for p in gps]
        lats = [p[2] for p in gps]
        points = list(zip(lons, lats))
        if len(points) >= 2:
            segments = [[points[i], points[i + 1]] for i in range(len(points) - 1)]
            elapsed = [times[p[0]] for p in gps]
            collection = LineCollection(
                segments, cmap="viridis",
                norm=Normalize(min(elapsed), max(elapsed) or 1),
                linewidth=2.0,
            )
            collection.set_array((__import__("numpy").asarray(elapsed[:-1]) +
                                  __import__("numpy").asarray(elapsed[1:])) / 2)
            ax.add_collection(collection)
            fig.colorbar(collection, ax=ax, label="Elapsed time (s)")
        ax.scatter(lons[0], lats[0], marker="o", s=55, label="Start", zorder=3)
        ax.scatter(lons[-1], lats[-1], marker="X", s=65, label="Finish", zorder=3)
        ax.set_xlim(min(lons), max(lons)) if min(lons) != max(lons) else ax.set_xlim(lons[0] - 0.0001, lons[0] + 0.0001)
        ax.set_ylim(min(lats), max(lats)) if min(lats) != max(lats) else ax.set_ylim(lats[0] - 0.0001, lats[0] + 0.0001)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title(f"GPS track — {len(gps)} valid fixes", loc="left")
        ax.legend(loc="best")
        ax.grid(alpha=0.25)

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=160, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", type=Path, help="Avinox .proto recording")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/avinox"),
        help="Directory for CSV, metadata JSON and plot (default: data/avinox)",
    )
    parser.add_argument("--show", action="store_true", help="Open the plot window after saving it")
    args = parser.parse_args()

    if not args.recording.is_file():
        parser.error(f"Recording does not exist: {args.recording}")

    result = parse(args.recording)
    metadata, samples = result["metadata"], result["samples"]
    stem = args.recording.stem
    csv_path = export_csv(samples, args.output_dir / f"{stem}.csv")
    json_path = export_metadata(metadata, args.output_dir / f"{stem}_metadata.json")
    plot_path = plot_ride(samples, metadata, args.output_dir / f"{stem}_overview.png",
                          show=args.show)

    gps_count = sum(s.get("latitude") is not None and s.get("longitude") is not None
                    for s in samples)
    start = datetime.fromtimestamp(metadata["start"], timezone.utc).astimezone() if metadata.get("start") else None
    end = datetime.fromtimestamp(metadata["end"], timezone.utc).astimezone() if metadata.get("end") else None
    print(f"Ride ID: {metadata.get('rideId')}")
    print(f"Start (local): {start.isoformat() if start else 'n/a'}")
    print(f"End   (local): {end.isoformat() if end else 'n/a'}")
    print(f"Samples: {len(samples)}")
    print(f"GPS fixes: {gps_count}/{len(samples)}")
    print(f"Duration: {metadata.get('duration')} s")
    print(f"CSV: {csv_path}")
    print(f"Metadata: {json_path}")
    print(f"Plot: {plot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
