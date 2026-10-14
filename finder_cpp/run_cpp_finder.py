#!/usr/bin/env python3
"""Run finder_cpp on a CSV and write cluster labels.

Usage:
  python run_cpp_finder.py input.csv
  python run_cpp_finder.py input.csv -x Region -y Fresh
  python run_cpp_finder.py input.csv -x x -y y -o out_cpp_clusters.csv
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Same defaults as pip FINDER / finder_cpp CLI
THRESHOLD = 10
POINTS_PER_DIMENSION = 15
ALGORITHM = "DbscanLoop"
MIN_THRESHOLD = 5
MAX_THRESHOLD = 21
DECAY = 0.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run C++ FINDER (finder_cpp) and save x,y,cluster CSV.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python run_cpp_finder.py example/COS7_Tubulin_AF647_nstorm.csv\n"
            "  python run_cpp_finder.py example/Wholesale-customers-data.csv "
            "-x Region -y Fresh\n"
        ),
    )
    parser.add_argument(
        "input_csv",
        type=Path,
        help="input CSV path",
    )
    parser.add_argument(
        "-x",
        "--x-col",
        default="x",
        metavar="NAME",
        help="column name for X coordinates (default: x)",
    )
    parser.add_argument(
        "-y",
        "--y-col",
        default="y",
        metavar="NAME",
        help="column name for Y coordinates (default: y)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        metavar="PATH",
        help="output CSV path (default: <input>_cpp_clusters.csv)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_csv: Path = args.input_csv
    if not input_csv.is_file():
        print(f"error: input CSV not found: {input_csv}")
        return 1

    try:
        import finder_cpp
    except ImportError as e:
        print(f"Failed to import finder_cpp: {e}")
        print("Build first: cd finder_cpp && pip install -e . --no-build-isolation")
        return 1

    output_csv = args.output
    if output_csv is None:
        output_csv = input_csv.with_name(f"{input_csv.stem}_cpp_clusters.csv")

    print("=" * 60)
    print("C++ FINDER (finder_cpp)")
    print("=" * 60)
    print(f"Input:  {input_csv}")
    print(f"Output: {output_csv}")
    print(f"Columns: x={args.x_col!r}, y={args.y_col!r}")
    print(
        f"Config: threshold={THRESHOLD}, ppd={POINTS_PER_DIMENSION}, "
        f"algo={ALGORITHM}, minmax_threshold=[{MIN_THRESHOLD}, {MAX_THRESHOLD}], "
        f"decay={DECAY}"
    )

    df = pd.read_csv(input_csv)
    for col in (args.x_col, args.y_col):
        if col not in df.columns:
            print(f"error: column {col!r} not in CSV. Available: {list(df.columns)}")
            return 1

    coords = df[[args.x_col, args.y_col]].to_numpy(dtype=np.float32)
    print(f"Points: {len(coords)}")

    t0 = time.time()
    result = finder_cpp.run_finder_2d(
        coords,
        threshold=THRESHOLD,
        points_per_dimension=POINTS_PER_DIMENSION,
        algorithm=ALGORITHM,
        min_threshold=MIN_THRESHOLD,
        max_threshold=MAX_THRESHOLD,
        decay=DECAY,
    )
    elapsed = time.time() - t0

    if isinstance(result, dict):
        labels = np.asarray(result["labels"])
        selected = {"sigma": result["sigma"], "threshold": result["threshold"]}
    else:
        labels = np.asarray(result)
        selected = None

    unique = set(labels.tolist())
    n_clusters = len(unique) - (1 if -1 in unique else 0)
    n_noise = int(np.sum(labels == -1))

    print(f"\nDone in {elapsed:.2f} s")
    if selected is not None:
        print(f"Selected parameters: {selected}")
    print(f"Clusters: {n_clusters}")
    print(f"Noise points: {n_noise}")

    out = pd.DataFrame({"x": coords[:, 0], "y": coords[:, 1], "cluster": labels})
    out.to_csv(output_csv, index=False)
    print(f"Wrote {output_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
