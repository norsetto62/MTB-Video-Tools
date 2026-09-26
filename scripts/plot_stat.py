# plot_stat.py

import sys
import pandas as pd
import matplotlib.pyplot as plt


def main():
    if len(sys.argv) < 3:
        print("Usage:")
        print("  python plot_stat.py <csv_file> <statistic>")
        print()
        print("Example:")
        print("  python plot_stat.py motion_scores.csv flow_coherence")
        sys.exit(1)

    csv_file = sys.argv[1]
    statistic = sys.argv[2]

    # Read CSV
    df = pd.read_csv(csv_file)

    # Check columns
    if "time" not in df.columns:
        print("ERROR: CSV does not contain a 'time' column.")
        sys.exit(1)

    if statistic not in df.columns:
        print(f"ERROR: Statistic '{statistic}' not found.")
        print("\nAvailable columns:")
        for col in df.columns:
            print(f"  {col}")
        sys.exit(1)

    # Remove rows where the requested statistic is missing
    data = df[["time", statistic]].dropna()
    sma = data.rolling(5, min_periods = 1 ).mean()

    # Plot
    plt.figure(figsize=(14, 6))
    plt.plot(data["time"], data[statistic])
    plt.plot(data["time"], sma[statistic])

    plt.xlabel("Time (seconds)")
    plt.ylabel(statistic)
    plt.title(f"{statistic} over time")

    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save next to the input CSV
    output_file = csv_file.rsplit(".", 1)[0] + f"_{statistic}.png"
    plt.savefig(output_file, dpi=150)
    print(f"Plot saved to: {output_file}")

if __name__ == "__main__":
    main()