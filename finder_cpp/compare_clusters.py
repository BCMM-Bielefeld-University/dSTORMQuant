#!/usr/bin/env python3
"""
Compare two FINDER clustering CSVs (same points, columns x,y,cluster).

Scientific metrics for partition agreement (label IDs need not match):
  - Adjusted Rand Index (ARI)  — chance-corrected pairwise agreement [0–1; 1 = identical]
  - Adjusted Mutual Information (AMI)
  - Normalized Mutual Information (NMI)
  - Fowlkes–Mallows Index (FMI)
  - Noise agreement            — fraction of points both label as noise (-1)
  - Cluster counts             — descriptive, not a similarity score

ARI / AMI are the usual primary numbers in papers when comparing two clusterings
without ground truth (or when treating one run as reference).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    fowlkes_mallows_score,
    normalized_mutual_info_score,
)

EXAMPLE_DIR = Path(__file__).resolve().parent / "example"
DEFAULT_A = EXAMPLE_DIR / "COS7_Tubulin_AF647_nstorm_pip_clusters.csv"
DEFAULT_B = EXAMPLE_DIR / "COS7_Tubulin_AF647_nstorm_cpp_clusters.csv"


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
    both_noise = float(np.mean(noise_a & noise_b))
    noise_xor = float(np.mean(noise_a ^ noise_b))

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
        "both_noise_fraction": both_noise,
        "noise_disagreement_fraction": noise_xor,
        "exact_label_match_fraction": float(np.mean(labels_a == labels_b)),
    }


def load_cluster_csv(path: Path) -> tuple[pd.DataFrame, np.ndarray]:
    df = pd.read_csv(path)
    if "cluster" not in df.columns:
        raise ValueError(f"{path} missing 'cluster' column")
    return df, df["cluster"].to_numpy()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "csv_a",
        nargs="?",
        type=Path,
        default=DEFAULT_A,
        help=f"first clustering CSV (default: {DEFAULT_A.name})",
    )
    parser.add_argument(
        "csv_b",
        nargs="?",
        type=Path,
        default=DEFAULT_B,
        help=f"second clustering CSV (default: {DEFAULT_B.name})",
    )
    args = parser.parse_args()

    df_a, labels_a = load_cluster_csv(args.csv_a)
    df_b, labels_b = load_cluster_csv(args.csv_b)

    # Guard against reordered rows
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

    print("=" * 60)
    print("Clustering comparison")
    print("=" * 60)
    print(f"A: {args.csv_a}")
    print(f"B: {args.csv_b}")
    print(f"Points: {metrics['n_points']}")
    print(
        f"Clusters: A={metrics['n_clusters_a']}  B={metrics['n_clusters_b']}"
    )
    print(f"Noise:    A={metrics['n_noise_a']}  B={metrics['n_noise_b']}")
    print()
    print("Partition agreement (label IDs may differ; higher = more similar):")
    print(f"  Adjusted Rand Index (ARI):     {metrics['adjusted_rand_index']:.6f}")
    print(f"  Adjusted Mutual Info (AMI):    {metrics['adjusted_mutual_info']:.6f}")
    print(f"  Normalized Mutual Info (NMI):  {metrics['normalized_mutual_info']:.6f}")
    print(f"  Fowlkes–Mallows Index:         {metrics['fowlkes_mallows']:.6f}")
    print()
    print("Noise / raw labels:")
    print(f"  Both noise (fraction):         {metrics['both_noise_fraction']:.6f}")
    print(f"  Noise disagreement (fraction): {metrics['noise_disagreement_fraction']:.6f}")
    print(
        f"  Exact label ID match:          {metrics['exact_label_match_fraction']:.6f}"
        "  (low is OK if IDs are permuted)"
    )
    print()
    print("Primary report metric: ARI (≈1 means nearly identical partitions).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
