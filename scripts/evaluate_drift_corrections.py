#!/usr/bin/env python3
"""Compare the results of all drift corrections methods with ground-truth.
"""

from __future__ import annotations

import json
import numpy as np
import pandas as pd

from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_DIR = SCRIPT_DIR / "dataset"
OUT_ROOT = SCRIPT_DIR / "output"
PIXEL_NM = 100.0
STEM = "simulationSMLM"
GT_PATH = DATASET_DIR / "simulationSMLM_groundtruth_drift_px.csv"

# backend label -> default out folder from the three aim_drift_*.py scripts
BACKENDS = (
    ("dSTORMQuant", "aim_drift_dSTORMQuant"),
    ("picasso_via_dSTORMQuant", "aim_drift_picasso_via_dSTORMQuant"),
    ("picasso", "aim_drift_picasso"),
)


def load_gt(path: Path) -> np.ndarray:
    df = pd.read_csv(path)
    return df[["dx_px", "dy_px"]].to_numpy(dtype=float)


def load_est(out_folder: str) -> np.ndarray:
    path = OUT_ROOT / out_folder / STEM / "drift_corrected" / "input_aimdrift.txt"
    if not path.exists():
        raise FileNotFoundError(path)
    arr = np.loadtxt(path)
    return arr[:, :2]


def residual_metrics(est: np.ndarray, gt: np.ndarray) -> dict:
    """Compare after aligning both curves to frame 0 (subtract drift[0])."""
    n = min(len(est), len(gt))
    e = est[:n] - est[0]
    g = gt[:n] - gt[0]
    dx = (e[:, 0] - g[:, 0]) * PIXEL_NM
    dy = (e[:, 1] - g[:, 1]) * PIXEL_NM
    r = np.sqrt(dx**2 + dy**2)
    return {
        "n_frames": int(n),
        "residual_std_dx_nm": float(np.std(dx)),
        "residual_std_dy_nm": float(np.std(dy)),
        "residual_std_dr_nm": float(np.std(r)),
        "rmse_dx_nm": float(np.sqrt(np.mean(dx**2))),
        "rmse_dy_nm": float(np.sqrt(np.mean(dy**2))),
        "rmse_dr_nm": float(np.sqrt(np.mean(r**2))),
        "max_abs_dr_nm": float(np.max(r)),
        "median_abs_dr_nm": float(np.median(r)),
    }


def main() -> None:
    gt = load_gt(GT_PATH)
    rows = []
    for backend, out_folder in BACKENDS:
        est = load_est(out_folder)
        m = residual_metrics(est, gt)
        rows.append({"dataset": STEM, "backend": backend, "align": "frame0", **m})
        print(
            f"{backend:28s}  RMSE_r={m['rmse_dr_nm']:.4f} nm  "
            f"std_dx={m['residual_std_dx_nm']:.4f}  std_dy={m['residual_std_dy_nm']:.4f}  "
            f"max|dr|={m['max_abs_dr_nm']:.4f} nm"
        )

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    csv_out = OUT_ROOT / "aim_drift_residuals_vs_gt.csv"
    json_out = OUT_ROOT / "aim_drift_residuals_vs_gt.json"
    pd.DataFrame(rows).to_csv(csv_out, index=False)
    json_out.write_text(
        json.dumps(
            {
                "pixel_nm": PIXEL_NM,
                "stem": STEM,
                "groundtruth": str(GT_PATH),
                "drift_roots": {
                    label: str(OUT_ROOT / folder) for label, folder in BACKENDS
                },
                "note": (
                    "Residuals after aligning estimated and GT drifts to frame0 "
                    "(subtract drift[0]). Units nm (pixel_size=100)."
                ),
                "rows": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nSaved {csv_out}")
    print(f"Saved {json_out}")


if __name__ == "__main__":
    main()
