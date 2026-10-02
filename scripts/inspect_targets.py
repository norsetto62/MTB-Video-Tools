import argparse
import sys
from pathlib import Path
import numpy as np


def inspect_datasets(data_dir):
    data_path = Path(data_dir)
    npz_files = sorted(list(data_path.glob("*.npz")))

    if not npz_files:
        print(f"No .npz files found in: {data_path}")
        return

    print("=" * 90)
    print("TARGET SCORE (y) DISTRIBUTION ANALYSIS")
    print("=" * 90)
    print(
        f"{'Dataset Video':<24} | {'Samples':<8} | {'Min':<8} | {'Max':<8} | "
        f"{'Mean':<8} | {'Std':<8} | {'IQR (25-75%)':<14}"
    )
    print("-" * 90)

    all_y = []

    for file_path in npz_files:
        data = np.load(file_path, allow_pickle=True)
        if "y" not in data:
            print(f"{file_path.stem:<24} | Missing 'y' target array")
            continue

        y = data["y"]
        all_y.append(y)

        y_min = np.min(y)
        y_max = np.max(y)
        y_mean = np.mean(y)
        y_std = np.std(y)
        p25, p75 = np.percentile(y, [25, 75])

        print(
            f"{file_path.stem:<24} | {len(y):<8} | {y_min:<8.4f} | {y_max:<8.4f} | "
            f"{y_mean:<8.4f} | {y_std:<8.4f} | [{p25:.2f}, {p75:.2f}]"
        )

    print("-" * 90)
    if all_y:
        concat_y = np.concatenate(all_y)
        g_min = np.min(concat_y)
        g_max = np.max(concat_y)
        g_mean = np.mean(concat_y)
        g_std = np.std(concat_y)
        gp25, gp75 = np.percentile(concat_y, [25, 75])

        print(
            f"{'GLOBAL COMBINED':<24} | {len(concat_y):<8} | {g_min:<8.4f} | {g_max:<8.4f} | "
            f"{g_mean:<8.4f} | {g_std:<8.4f} | [{gp25:.2f}, {gp75:.2f}]"
        )
    print("=" * 90)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect target score distributions across .npz datasets.")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=r"C:\VideoTools\MTB-Video-Tools\output\datasets",
        help="Path to directory containing dataset .npz files",
    )
    args = parser.parse_args()

    try:
        inspect_datasets(args.data_dir)
    except KeyboardInterrupt:
        print("\n[!] Execution interrupted by user.")
        sys.exit(0)