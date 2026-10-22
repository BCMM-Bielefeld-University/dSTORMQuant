#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.path import Path as MplPath
from scipy.spatial import ConvexHull, QhullError
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    fowlkes_mallows_score,
    normalized_mutual_info_score,
)

MIN_POINTS_PER_CLUSTER = 3


def _n_clusters(labels: np.ndarray) -> int:
    unique = set(labels.tolist())
    return len(unique) - (1 if -1 in unique else 0)


def cluster_mask(coords: np.ndarray, labels: np.ndarray,
                 bounds: tuple[float, float, float, float], dim: int) -> np.ndarray:
    """Rasterise the convex hull of every cluster into one flattened boolean mask."""
    # -----------------------------------------------------------------------
    # Derived from IoU_calc() in ARI_IoU_source.R
    #   Author:  DJ-Nieves
    #   Source:  https://github.com/DJ-Nieves/ARI-and-IoU-cluster-analysis-evaluation
    #   License: GNU General Public License v3.0 (GPL-3.0)
    # Modified: ported from R to Python (scipy ConvexHull + matplotlib Path in
    # place of chull/owin/as.mask), noise label is -1 instead of 0.
    # Modified 2026-10-15. Remains under GPL-3.0, ABSOLUTELY NO WARRANTY.
    # See https://www.gnu.org/licenses/gpl-3.0.html
    # -----------------------------------------------------------------------
    xmin, xmax, ymin, ymax = bounds
    xs = xmin + (np.arange(dim) + 0.5) * (xmax - xmin) / dim  # pixel centres
    ys = ymin + (np.arange(dim) + 0.5) * (ymax - ymin) / dim
    mask = np.zeros((dim, dim), dtype=bool)  # [y, x]

    for lab in np.unique(labels):
        if lab == -1:  # noise
            continue
        pts = coords[labels == lab]
        if len(pts) < MIN_POINTS_PER_CLUSTER:
            continue
        try:
            hull = ConvexHull(pts)  # R: chull(cluster)
        except QhullError:  # collinear / degenerate
            continue
        verts = pts[hull.vertices]

        ix = np.where((xs >= verts[:, 0].min()) & (xs <= verts[:, 0].max()))[0]
        iy = np.where((ys >= verts[:, 1].min()) & (ys <= verts[:, 1].max()))[0]
        if len(ix) == 0 or len(iy) == 0:
            continue

        # R: owin(poly=...) + as.mask(..., dimyx=mask_dim)
        gx, gy = np.meshgrid(xs[ix], ys[iy])
        inside = MplPath(verts).contains_points(
            np.column_stack([gx.ravel(), gy.ravel()])
        ).reshape(gx.shape)
        mask[np.ix_(iy, ix)] |= inside
    return mask


def iou_score(coords: np.ndarray, labels_a: np.ndarray, labels_b: np.ndarray,
              dim: int) -> float:
    """IoU of the two flattened cluster masks (overlap / union)."""
    # -----------------------------------------------------------------------
    # Derived from IoU_calc() in ARI_IoU_source.R
    #   Author:  DJ-Nieves
    #   Source:  https://github.com/DJ-Nieves/ARI-and-IoU-cluster-analysis-evaluation
    #   License: GNU General Public License v3.0 (GPL-3.0)
    # Modified: ported from R to Python; compares two label arrays instead of
    # a ground-truth set against several results. Modified 2026-10-15.
    # Remains under GPL-3.0, ABSOLUTELY NO WARRANTY.
    # -----------------------------------------------------------------------
    bounds = (coords[:, 0].min(), coords[:, 0].max(),
              coords[:, 1].min(), coords[:, 1].max())
    mask_a = cluster_mask(coords, labels_a, bounds, dim)  # R: flat_GT_mask
    mask_b = cluster_mask(coords, labels_b, bounds, dim)  # R: flat_Res_mask
    union = np.count_nonzero(mask_a | mask_b)             # R: flat_union_mask
    if union == 0:
        return float("nan")
    return float(np.count_nonzero(mask_a & mask_b) / union)  # R: overlap/union


def compare_labels(labels_a: np.ndarray, labels_b: np.ndarray) -> dict[str, float | int]:
    if len(labels_a) != len(labels_b):
        raise ValueError(
            f"Length mismatch: {len(labels_a)} vs {len(labels_b)} "
            "(CSVs must list the same points in the same order)"
        )
    return {
        "n_points": int(len(labels_a)),
        "n_clusters_a": _n_clusters(labels_a),
        "n_clusters_b": _n_clusters(labels_b),
        "n_noise_a": int(np.sum(labels_a == -1)),
        "n_noise_b": int(np.sum(labels_b == -1)),
        "adjusted_rand_index": float(adjusted_rand_score(labels_a, labels_b)),
        "adjusted_mutual_info": float(adjusted_mutual_info_score(labels_a, labels_b)),
        "normalized_mutual_info": float(normalized_mutual_info_score(labels_a, labels_b)),
        "fowlkes_mallows": float(fowlkes_mallows_score(labels_a, labels_b)),
    }


def load_cluster_csv(path: Path) -> tuple[pd.DataFrame, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"CSV not found: {path}")
    df = pd.read_csv(path)
    if "cluster" not in df.columns:
        raise ValueError(f"{path} missing 'cluster' column")
    return df, df["cluster"].to_numpy()


def compare_pair(path_a: Path, path_b: Path, mask_dim: int) -> dict[str, float | int]:
    """Load two cluster CSVs and return all metrics including IoU."""
    df_a, labels_a = load_cluster_csv(path_a)
    df_b, labels_b = load_cluster_csv(path_b)

    has_xy = {"x", "y"}.issubset(df_a.columns) and {"x", "y"}.issubset(df_b.columns)
    if has_xy:
        if len(df_a) != len(df_b):
            raise ValueError(f"Length mismatch: {len(df_a)} vs {len(df_b)}")
        if not (
            np.allclose(df_a["x"].to_numpy(), df_b["x"].to_numpy())
            and np.allclose(df_a["y"].to_numpy(), df_b["y"].to_numpy())
        ):
            raise ValueError(
                "x/y coordinates differ between CSVs — "
                "point order or data must match for a fair comparison"
            )

    metrics = compare_labels(labels_a, labels_b)
    metrics["iou"] = (
        iou_score(df_a[["x", "y"]].to_numpy(dtype=np.float64),
                  labels_a, labels_b, mask_dim)
        if has_xy else float("nan")
    )
    return metrics


def print_metrics(metrics: dict, path_a: Path, path_b: Path, mask_dim: int) -> None:
    print("=" * 60)
    print("Clustering comparison (A = base, B = compared)")
    print("=" * 60)
    print(f"A: {path_a}")
    print(f"B: {path_b}")
    print(f"Points: {metrics['n_points']}")
    print(f"Clusters: A={metrics['n_clusters_a']}  B={metrics['n_clusters_b']}")
    print(f"Noise:    A={metrics['n_noise_a']}  B={metrics['n_noise_b']}")
    print()
    print("Partition agreement:")
    print(f"  Adjusted Rand Index (ARI):     {metrics['adjusted_rand_index']:.6f}")
    print(f"  Adjusted Mutual Info (AMI):    {metrics['adjusted_mutual_info']:.6f}")
    print(f"  Normalized Mutual Info (NMI):  {metrics['normalized_mutual_info']:.6f}")
    print(f"  Fowlkes–Mallows Index:         {metrics['fowlkes_mallows']:.6f}")
    print()
    print("Spatial overlap of cluster polygons:")
    print(f"  IoU (convex hulls, {mask_dim}x{mask_dim} mask): {metrics['iou']:.6f}")


def _stem_key(path: Path) -> str:
    """foo_pip_clusters.csv / foo_cpp_clusters.csv -> foo"""
    stem = path.stem
    for suffix in ("_pip_clusters", "_cpp_clusters", "_clusters"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def compare_folders(dir_a: Path, dir_b: Path, mask_dim: int, out_csv: Path) -> int:
    files_a = {_stem_key(p): p for p in sorted(dir_a.glob("*.csv"))}
    files_b = {_stem_key(p): p for p in sorted(dir_b.glob("*.csv"))}

    common = sorted(files_a.keys() & files_b.keys())
    only_a = sorted(files_a.keys() - files_b.keys())
    only_b = sorted(files_b.keys() - files_a.keys())

    if not common:
        print(f"error: no matching CSV names between {dir_a} and {dir_b}", file=sys.stderr)
        return 1
    for k in only_a:
        print(f"warning: only in A ({dir_a}): {files_a[k].name}")
    for k in only_b:
        print(f"warning: only in B ({dir_b}): {files_b[k].name}")

    print("=" * 60)
    print(f"Comparing {len(common)} pair(s): A={dir_a} (base)  B={dir_b}")
    print("=" * 60)

    rows = []
    for i, key in enumerate(common, 1):
        print(f"[{i}/{len(common)}] {key}")
        try:
            m = compare_pair(files_a[key], files_b[key], mask_dim)
        except Exception as e:  # keep going if one pair fails
            print(f"  error: {e}")
            continue
        print(f"  clusters A/B: {m['n_clusters_a']}/{m['n_clusters_b']} | "
              f"ARI={m['adjusted_rand_index']:.4f} | IoU={m['iou']:.4f}")
        rows.append({"name": key, "file_a": files_a[key].name,
                     "file_b": files_b[key].name, **m})

    if not rows:
        print("error: no pair could be compared", file=sys.stderr)
        return 1

    summary = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_csv, index=False)

    print()
    print(f"Compared {len(rows)}/{len(common)} pair(s)")
    print(f"Mean ARI: {summary['adjusted_rand_index'].mean():.4f}  "
          f"Mean IoU: {summary['iou'].mean():.4f}")
    print(f"Wrote {out_csv}")
    return 0 if len(rows) == len(common) else 1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare clustering result CSVs (x, y, cluster). A is the base. "
                    "Inputs can be two CSV files or two folders.",
    )
    parser.add_argument("csv_a", nargs="?", type=Path, help="base CSV or folder (pip)")
    parser.add_argument("csv_b", nargs="?", type=Path, help="compared CSV or folder (cpp)")
    parser.add_argument("-a", "--csv-a", dest="opt_a", type=Path, metavar="PATH")
    parser.add_argument("-b", "--csv-b", dest="opt_b", type=Path, metavar="PATH")
    parser.add_argument(
        "-o", "--output", type=Path, default=Path("comparison_summary.csv"), metavar="PATH",
        help="summary CSV for folder mode (default: comparison_summary.csv)",
    )
    parser.add_argument(
        "--mask-dim", type=int, default=2400, metavar="N",
        help="IoU raster resolution, N x N pixels over the data bounding box "
             "(default: 2400, i.e. ROI 2000 + 400 as in the R script)",
    )
    args = parser.parse_args(argv)

    path_a = args.opt_a or args.csv_a
    path_b = args.opt_b or args.csv_b
    if path_a is None or path_b is None:
        parser.error("two paths are required (positional or -a/-b)")
    args.csv_a, args.csv_b = path_a, path_b
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.csv_a.is_dir() and args.csv_b.is_dir():
        return compare_folders(args.csv_a, args.csv_b, args.mask_dim, args.output)
    if args.csv_a.is_dir() != args.csv_b.is_dir():
        print("error: give two folders or two files, not one of each", file=sys.stderr)
        return 1

    try:
        metrics = compare_pair(args.csv_a, args.csv_b, args.mask_dim)
    except (OSError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print_metrics(metrics, args.csv_a, args.csv_b, args.mask_dim)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())