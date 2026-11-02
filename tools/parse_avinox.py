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

# Allow running this file directly from a source checkout without requiring
# an editable install.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from autocut.dataset.avinox_proto import parse  # noqa: E402


def _json_default(value: Any) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def export_csv(samples: list[dict[str, Any]], destination: Path) -> Path:
    """Write one row per telemetry sample, preserving all decoded channels."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    columns = list(dict.fromkeys(key for sample in samples for key in sample))
    with destination.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for sample in samples:
            row = dict(sample)
            if isinstance(row.get("date"), datetime):
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
    """Create a diagnostic overview of the main ride channels and GPS track."""
    try:
        import matplotlib
        if not show:
            matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
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

    def series(key: str) -> list[float]:
        values = []
        for sample in samples:
            value = sample.get(key)
            values.append(float("nan") if value is None else float(value))
        return values

    fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True)
    fig.suptitle(
        f"{metadata.get('fileName', 'Avinox ride')} — "
        f"{len(samples)} samples, {metadata.get('duration', 0)} s",
        fontsize=14, fontweight="bold",
    )

    axes[0].plot(times, series("speed"), label="Speed")
    axes[0].set_ylabel("Speed (km/h)")
    axes[0].legend(loc="upper right")
    axes[0].grid(alpha=0.25)

    axes[1].plot(times, series("riderPower"), label="Rider power")
    axes[1].plot(times, series("motorPower"), label="Motor power")
    axes[1].plot(times, series("totalPower"), label="Total power", alpha=0.75)
    axes[1].set_ylabel("Power (W)")
    axes[1].legend(loc="upper right")
    axes[1].grid(alpha=0.25)

    axes[2].plot(times, series("imuX"), label="IMU X")
    axes[2].plot(times, series("imuY"), label="IMU Y")
    axes[2].plot(times, series("imuZ"), label="IMU Z")
    axes[2].plot(times, series("gradient"), label="Recorded gradient", alpha=0.8)
    axes[2].set_ylabel("Raw IMU / gradient")
    axes[2].legend(loc="upper right", ncol=4)
    axes[2].grid(alpha=0.25)

    latitudes, longitudes = series("latitude"), series("longitude")
    gps_valid = [
        i for i, (lat, lon) in enumerate(zip(latitudes, longitudes))
        if lat == lat and lon == lon and -90 <= lat <= 90 and -180 <= lon <= 180
    ]
    if gps_valid:
        axes[3].plot(
            [longitudes[i] for i in gps_valid],
            [latitudes[i] for i in gps_valid],
            linewidth=1.2,
        )
        axes[3].scatter(longitudes[gps_valid[0]], latitudes[gps_valid[0]],
                        marker="o", label="Start")
        axes[3].scatter(longitudes[gps_valid[-1]], latitudes[gps_valid[-1]],
                        marker="x", label="End")
        axes[3].legend(loc="best")
        axes[3].set_ylabel("Latitude")
        axes[3].set_xlabel("Longitude")
        axes[3].set_aspect("equal", adjustable="datalim")
    else:
        axes[3].text(0.5, 0.5, "No valid GPS samples",
                     ha="center", va="center", transform=axes[3].transAxes)
        axes[3].set_axis_off()

    if gps_valid:
        axes[3].grid(alpha=0.25)
    else:
        axes[2].set_xlabel("Elapsed time (s)")
    if gps_valid:
        for axis in axes[:3]:
            axis.set_xlabel("Elapsed time (s)")

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
    metadata = result["metadata"]
    samples = result["samples"]
    stem = args.recording.stem
    csv_path = export_csv(samples, args.output_dir / f"{stem}.csv")
    json_path = export_metadata(metadata, args.output_dir / f"{stem}_metadata.json")
    plot_path = plot_ride(samples, metadata, args.output_dir / f"{stem}_overview.png",
                          show=args.show)

    print(f"Ride ID: {metadata.get('rideId')}")
    print(f"Samples: {len(samples)}")
    print(f"Duration: {metadata.get('duration')} s")
    print(f"CSV: {csv_path}")
    print(f"Metadata: {json_path}")
    print(f"Plot: {plot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
