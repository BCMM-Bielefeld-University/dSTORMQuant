#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

THRESHOLD = 10
POINTS_PER_DIMENSION = 15
ALGORITHM = "DbscanLoop"
MIN_THRESHOLD = 5
MAX_THRESHOLD = 21
DECAY = 0.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run C++ FINDER on CSV files and save x,y,cluster CSVs."
    )
    parser.add_argument(
        "input",
        type=Path,
        nargs="?",
        default=Path("examples"),
        help="input CSV file or folder containing CSVs (default: examples)",
    )
    parser.add_argument("-x", "--x-col", default="x", metavar="NAME",
                        help="column name for X coordinates (default: x)")
    parser.add_argument("-y", "--y-col", default="y", metavar="NAME",
                        help="column name for Y coordinates (default: y)")
    parser.add_argument("-o", "--output-dir", type=Path, default=Path("results_cpp"),
                        metavar="DIR", help="output folder (default: results_cpp)")
    return parser.parse_args()


def run_one(finder_cpp, input_csv: Path, output_csv: Path, x_col: str, y_col: str) -> bool:
    df = pd.read_csv(input_csv)
    for col in (x_col, y_col):
        if col not in df.columns:
            print(f"  skip: column {col!r} not in CSV. Available: {list(df.columns)}")
            return False

    coords = df[[x_col, y_col]].to_numpy(dtype=np.float32)
    print(f"  Points: {len(coords)}")

    t0 = time.perf_counter()
    result = finder_cpp.run_finder_2d(
        coords,
        threshold=THRESHOLD,
        points_per_dimension=POINTS_PER_DIMENSION,
        algorithm=ALGORITHM,
        min_threshold=MIN_THRESHOLD,
        max_threshold=MAX_THRESHOLD,
        decay=DECAY,
    )
    elapsed = time.perf_counter() - t0

    if isinstance(result, dict):
        labels = np.asarray(result["labels"])
        print(f"  Selected: sigma={result['sigma']}, threshold={result['threshold']}")
    else:
        labels = np.asarray(result)

    unique = set(labels.tolist())
    n_clusters = len(unique) - (1 if -1 in unique else 0)
    n_noise = int(np.sum(labels == -1))
    print(f"  Done in {elapsed:.2f} s | clusters: {n_clusters} | noise: {n_noise}")

    pd.DataFrame({"x": coords[:, 0], "y": coords[:, 1], "cluster": labels}).to_csv(
        output_csv, index=False
    )
    print(f"  Wrote {output_csv}")
    return True


def main() -> int:
    args = parse_args()

    try:
        import finder_cpp
    except ImportError as e:
        print(f"Failed to import finder_cpp: {e}")
        print("Build first: cd finder_cpp && pip install -e . --no-build-isolation")
        return 1

    if args.input.is_file():
        csv_files = [args.input]
    elif args.input.is_dir():
        csv_files = sorted(args.input.glob("*.csv"))
    else:
        print(f"error: input not found: {args.input}")
        return 1

    if not csv_files:
        print(f"error: no CSV files found in {args.input}")
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("C++ FINDER (finder_cpp)")
    print("=" * 60)
    print(f"Input:  {args.input} ({len(csv_files)} file(s))")
    print(f"Output: {args.output_dir}")
    print(f"Columns: x={args.x_col!r}, y={args.y_col!r}")

    ok = 0
    t_start = time.perf_counter()  # end-to-end timer: read + FINDER + save, all files
    for i, csv_path in enumerate(csv_files, 1):
        print(f"\n[{i}/{len(csv_files)}] {csv_path.name}")
        out_path = args.output_dir / f"{csv_path.stem}_cpp_clusters.csv"
        try:
            if run_one(finder_cpp, csv_path, out_path, args.x_col, args.y_col):
                ok += 1
        except Exception as e:  # keep going if one file fails
            print(f"  error: {e}")
    total_elapsed = time.perf_counter() - t_start

    time_txt = args.output_dir / "total_time.txt"
    time_txt.write_text(
        f"C++ FINDER end-to-end time\n"
        f"Total time: {total_elapsed:.3f} s\n"
        f"Files processed: {ok}/{len(csv_files)}\n"
    )

    print(f"\nFinished: {ok}/{len(csv_files)} succeeded")
    print(f"Total end-to-end time: {total_elapsed:.2f} s (saved to {time_txt})")
    return 0 if ok == len(csv_files) else 1


if __name__ == "__main__":
    sys.exit(main())