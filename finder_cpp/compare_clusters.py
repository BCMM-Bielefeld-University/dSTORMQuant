#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    fowlkes_mallows_score,
    normalized_mutual_info_score,
)


def _n_clusters(labels: np.ndarray) -> int:
    unique = set(labels.tolist())
    return len(unique) - (1 if -1 in unique else 0)


def compare_labels(labels_a: np.ndarray, labels_b: np.ndarray) -> dict[str, float | int]:
    if len(labels_a) != len(labels_b):
        raise ValueError(
            f"Length mismatch: {len(labels_a)} vs {len(labels_b)} "
            "(CSVs must list the same points in the same order)"
        )

    noise_a = labels_a == -1
    noise_b = labels_b == -1

    return {
        "n_points": int(len(labels_a)),
        "n_clusters_a": _n_clusters(labels_a),
        "n_clusters_b": _n_clusters(labels_b),
        "n_noise_a": int(np.sum(noise_a)),
        "n_noise_b": int(np.sum(noise_b)),
        "adjusted_rand_index": float(adjusted_rand_score(labels_a, labels_b)),
        "adjusted_mutual_info": float(adjusted_mutual_info_score(labels_a, labels_b)),
        "normalized_mutual_info": float(
            normalized_mutual_info_score(labels_a, labels_b)
        ),
        "fowlkes_mallows": float(fowlkes_mallows_score(labels_a, labels_b)),
    }


def load_cluster_csv(path: Path) -> tuple[pd.DataFrame, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"CSV not found: {path}")
    df = pd.read_csv(path)
    if "cluster" not in df.columns:
        raise ValueError(f"{path} missing 'cluster' column")
    return df, df["cluster"].to_numpy()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare two clustering result CSVs (x, y, cluster).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python compare_clusters.py pip_clusters.csv cpp_clusters.csv\n"
            "  python compare_clusters.py --a pip_clusters.csv --b cpp_clusters.csv\n"
        ),
    )
    parser.add_argument(
        "csv_a",
        nargs="?",
        type=Path,
        help="first clustering CSV",
    )
    parser.add_argument(
        "csv_b",
        nargs="?",
        type=Path,
        help="second clustering CSV",
    )
    parser.add_argument(
        "-a",
        "--csv-a",
        dest="opt_a",
        type=Path,
        metavar="PATH",
        help="first clustering CSV (alternative to positional)",
    )
    parser.add_argument(
        "-b",
        "--csv-b",
        dest="opt_b",
        type=Path,
        metavar="PATH",
        help="second clustering CSV (alternative to positional)",
    )
    args = parser.parse_args(argv)

    path_a = args.opt_a or args.csv_a
    path_b = args.opt_b or args.csv_b
    if path_a is None or path_b is None:
        parser.error("two CSV paths are required (positional or -a/-b)")

    args.csv_a = path_a
    args.csv_b = path_b
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        df_a, labels_a = load_cluster_csv(args.csv_a)
        df_b, labels_b = load_cluster_csv(args.csv_b)

        if {"x", "y"}.issubset(df_a.columns) and {"x", "y"}.issubset(df_b.columns):
            if not (
                np.allclose(df_a["x"].to_numpy(), df_b["x"].to_numpy())
                and np.allclose(df_a["y"].to_numpy(), df_b["y"].to_numpy())
            ):
                raise ValueError(
                    "x/y coordinates differ between CSVs — "
                    "point order or data must match for a fair comparison"
                )

        metrics = compare_labels(labels_a, labels_b)
    except (OSError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    print("=" * 60)
    print("Clustering comparison")
    print("=" * 60)
    print(f"A: {args.csv_a}")
    print(f"B: {args.csv_b}")
    print(f"Points: {metrics['n_points']}")
    print(f"Clusters: A={metrics['n_clusters_a']}  B={metrics['n_clusters_b']}")
    print(f"Noise:    A={metrics['n_noise_a']}  B={metrics['n_noise_b']}")
    print()
    print("Partition agreement:")
    print(f"  Adjusted Rand Index (ARI):     {metrics['adjusted_rand_index']:.6f}")
    print(f"  Adjusted Mutual Info (AMI):    {metrics['adjusted_mutual_info']:.6f}")
    print(f"  Normalized Mutual Info (NMI):  {metrics['normalized_mutual_info']:.6f}")
    print(f"  Fowlkes–Mallows Index:         {metrics['fowlkes_mallows']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
