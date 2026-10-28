#!/usr/bin/env python3
"""Convert AIM paper Dryad .mat files to CSV.

Coordinates in the paper datasets are in camera pixels with pixel size 100 nm.
"""

from __future__ import annotations

import argparse
import numpy as np
import pandas as pd

from pathlib import Path

PIXEL_SIZE_NM = 100.0
SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_DIR = SCRIPT_DIR / "dataset"

# Dryad mats under scripts/dataset/
MAT_DATASETS = (
    "CTCF_MCF10A_DRB_6h.mat",
    "Origami_PAINT.mat",
    "simulationSMLM.mat",
)


def locs_to_csv(
    frames: np.ndarray,
    x_px: np.ndarray,
    y_px: np.ndarray,
    out_csv: Path,
    *,
    pixel_nm: float = PIXEL_SIZE_NM,
) -> Path:
    df = pd.DataFrame(
        {
            "frameIndex": frames.astype(np.int64),
            "x (nm)": x_px.astype(float) * pixel_nm,
            "y (nm)": y_px.astype(float) * pixel_nm,
            "channelIndex": np.zeros(len(frames), dtype=np.int64),
        }
    )
    ok = (
        np.isfinite(df["x (nm)"])
        & np.isfinite(df["y (nm)"])
        & (df["x (nm)"] > 0)
        & (df["y (nm)"] > 0)
    )
    n_drop = int((~ok).sum())
    if n_drop:
        print(f"  dropped {n_drop:,} non-finite/non-positive rows")
        df = df.loc[ok].reset_index(drop=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(
        f"  wrote {out_csv}  n={len(df):,}  "
        f"frames={int(df['frameIndex'].min())}-{int(df['frameIndex'].max())}  "
        f"pixel={pixel_nm} nm"
    )
    return out_csv


def load_mat_fyx(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load F,X,Y from MATLAB v5 or v7.3 (HDF5) .mat."""
    try:
        from scipy.io import loadmat

        mat = loadmat(path, squeeze_me=True, struct_as_record=False)
        keys = [k for k in mat if not k.startswith("__")]
        if all(k in mat for k in ("F", "X", "Y")):
            return (
                np.asarray(mat["F"]).ravel(),
                np.asarray(mat["X"]).ravel(),
                np.asarray(mat["Y"]).ravel(),
            )
        raise KeyError(f"Expected F,X,Y in {keys}")
    except NotImplementedError:
        import h5py

        with h5py.File(path, "r") as h5:

            def _read(name: str) -> np.ndarray:
                if name not in h5:
                    raise KeyError(name)
                return np.asarray(h5[name]).ravel()

            return _read("F"), _read("X"), _read("Y")


def export_simulation_gt(mat_path: Path, csv_dir: Path) -> Path | None:
    """If present, export driftXT/driftYT from simulationSMLM.mat."""
    try:
        from scipy.io import loadmat

        mat = loadmat(mat_path, squeeze_me=True, variable_names=["driftXT", "driftYT"])
    except Exception:
        return None
    if "driftXT" not in mat or "driftYT" not in mat:
        return None
    dx = np.asarray(mat["driftXT"]).ravel()
    dy = np.asarray(mat["driftYT"]).ravel()
    out = csv_dir / "simulationSMLM_groundtruth_drift_px.csv"
    pd.DataFrame(
        {"frame": np.arange(1, len(dx) + 1), "dx_px": dx, "dy_px": dy}
    ).to_csv(out, index=False)
    print(f"  ground-truth drift -> {out}")
    return out


def convert_mat_files(mat_dir: Path, csv_dir: Path) -> list[Path]:
    written: list[Path] = []
    for name in MAT_DATASETS:
        path = mat_dir / name
        if not path.exists() or path.stat().st_size < 1000:
            print(f"  skip missing/invalid: {path}")
            continue
        print(f"Converting {path.name} ...")
        f, x, y = load_mat_fyx(path)
        out = csv_dir / f"{path.stem}.csv"
        locs_to_csv(f, x, y, out)
        written.append(out)
        if name == "simulationSMLM.mat":
            export_simulation_gt(path, csv_dir)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--from-mat",
        action="store_true",
        help="convert Dryad .mat files from scripts/dataset/ (default)",
    )
    args = parser.parse_args()
    if not args.from_mat:
        args.from_mat = True

    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Converting Dryad .mat files from {DATASET_DIR}/ (if present)...")
    convert_mat_files(DATASET_DIR, DATASET_DIR)


if __name__ == "__main__":
    main()
