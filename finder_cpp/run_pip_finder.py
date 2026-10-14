#!/usr/bin/env python3
"""Run pip FINDER (finder_smlm) on a CSV and write cluster labels.

Usage:
  python run_pip_finder.py input.csv
  python run_pip_finder.py input.csv -x Region -y Fresh
  python run_pip_finder.py input.csv -x x -y y -o out_pip_clusters.csv
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Same defaults as finder_cpp / verzep finder_smlm
THRESHOLD = 10
POINTS_PER_DIMENSION = 15
ALGO = "DbscanLoop"
MINMAX_THRESHOLD = [5, 21]
ONE_TWO_D = "twoD"
SIMILARITY_SCORE_COMPUTATION = "threshold"
LOG_THRESHOLDS = False
LOG_SIGMAS = True
ADAPTIVE_SIGMA_BOUNDARIES = False
DECAY = 0.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run pip FINDER (finder_smlm) and save x,y,cluster CSV.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python run_pip_finder.py example/COS7_Tubulin_AF647_nstorm.csv\n"
            "  python run_pip_finder.py example/Wholesale-customers-data.csv "
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
        help="output CSV path (default: <input>_pip_clusters.csv)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_csv: Path = args.input_csv
    if not input_csv.is_file():
        print(f"error: input CSV not found: {input_csv}")
        return 1

    try:
        from finder import Finder
    except ImportError as e:
        print(f"Failed to import finder (finder_smlm): {e}")
        print("Install with: pip install finder_smlm")
        return 1

    output_csv = args.output
    if output_csv is None:
        output_csv = input_csv.with_name(f"{input_csv.stem}_pip_clusters.csv")

    print("=" * 60)
    print("pip FINDER (finder_smlm)")
    print("=" * 60)
    print(f"Input:  {input_csv}")
    print(f"Output: {output_csv}")
    print(f"Columns: x={args.x_col!r}, y={args.y_col!r}")
    print(
        f"Config: threshold={THRESHOLD}, ppd={POINTS_PER_DIMENSION}, "
        f"algo={ALGO}, minmax_threshold={MINMAX_THRESHOLD}, "
        f"similarity={SIMILARITY_SCORE_COMPUTATION}, decay={DECAY}"
    )

    df = pd.read_csv(input_csv)
    for col in (args.x_col, args.y_col):
        if col not in df.columns:
            print(f"error: column {col!r} not in CSV. Available: {list(df.columns)}")
            return 1

    xc = df[[args.x_col, args.y_col]].to_numpy(dtype=np.float64)
    print(f"Points: {len(xc)}")

    finder = Finder(
        threshold=THRESHOLD,
        points_per_dimension=POINTS_PER_DIMENSION,
        algo=ALGO,
        minmax_threshold=MINMAX_THRESHOLD,
        one_two_d=ONE_TWO_D,
        similarity_score_computation=SIMILARITY_SCORE_COMPUTATION,
        log_thresholds=LOG_THRESHOLDS,
        log_sigmas=LOG_SIGMAS,
        adaptive_sigma_boundaries=ADAPTIVE_SIGMA_BOUNDARIES,
        decay=DECAY,
        n_dim=2,
    )

    t0 = time.time()
    labels = np.asarray(finder.fit(xc))
    elapsed = time.time() - t0

    unique = set(labels.tolist())
    n_clusters = len(unique) - (1 if -1 in unique else 0)
    n_noise = int(np.sum(labels == -1))

    print(f"\nDone in {elapsed:.2f} s")
    print(f"Selected parameters: {finder.selected_parameters}")
    print(f"Clusters: {n_clusters}")
    print(f"Noise points: {n_noise}")

    out = pd.DataFrame({"x": xc[:, 0], "y": xc[:, 1], "cluster": labels})
    out.to_csv(output_csv, index=False)
    print(f"Wrote {output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
