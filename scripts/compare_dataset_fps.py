from pathlib import Path
import argparse
import numpy as np


def load_dataset(path):
    d = np.load(path, allow_pickle=True)

    X = d["X"]
    meta = d["metadata"].item()

    return len(X), meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir-2fps", default="output/datasets_2fps")
    parser.add_argument("--dir-10fps", default="output/datasets_10fps")
    args = parser.parse_args()

    d2 = {
        p.stem.replace("_dataset", ""): p
        for p in Path(args.dir_2fps).glob("*_dataset.npz")
    }

    d10 = {
        p.stem.replace("_dataset", ""): p
        for p in Path(args.dir_10fps).glob("*_dataset.npz")
    }

    common = sorted(set(d2) & set(d10))

    passed = failed = 0

    print()
    print("2-FPS vs 10-FPS WINDOW PARITY")
    print("=" * 72)

    for name in common:
        n2, m2 = load_dataset(d2[name])
        n10, m10 = load_dataset(d10[name])

        ok = (
            n2 == n10
            and m2["window"] == m10["window"]
            and m2["stride"] == m10["stride"]
            and m2["flow_start"] == m10["flow_start"]
        )

        if ok:
            passed += 1
            status = "PASS"
        else:
            failed += 1
            status = "FAIL"

        # Number of windows is the actual parity check.
        # The 2-FPS dataset should have 8 samples/window,
        # the 10-FPS dataset should have 40.
        print(
            f"{name:<12} {status:<5} "
            f"windows {n2} vs {n10}"
        )

    print("-" * 72)
    print(f"TOTAL: {passed} PASS / {failed} FAIL")
    print()
    print("RESULT:", "PASS" if failed == 0 else "FAIL")


if __name__ == "__main__":
    main()