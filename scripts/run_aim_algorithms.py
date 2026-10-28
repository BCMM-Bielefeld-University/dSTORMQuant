#!/usr/bin/env python3
"""Run the three AIM drift scripts on evaluation CSVs and save their outputs.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_DIR = SCRIPT_DIR / "dataset"
OUT_ROOT = SCRIPT_DIR / "output"

BACKENDS = (
    ("dSTORMQuant", "aim_drift_dSTORMQuant.py", "aim_drift_dSTORMQuant"),
    (
        "picasso_via_dSTORMQuant",
        "aim_drift_picasso_via_dSTORMQuant.py",
        "aim_drift_picasso_via_dSTORMQuant",
    ),
    ("picasso", "aim_drift_picasso.py", "aim_drift_picasso"),
)

DATASETS = (
    "simulationSMLM.csv",
    "CTCF_MCF10A_DRB_6h.csv",
    "Origami_PAINT.csv",
)

COMMON_ARGS = (
    "--pixel",
    "100",
    "--segmentation",
    "50",
    "--intersect-d-nm",
    "20",
    "--roi-r-nm",
    "60",
    "--no-napari",
)


def run_one(backend: str, script_name: str, out_folder: str, csv_path: Path) -> float:
    out_dir = OUT_ROOT / out_folder
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(SCRIPT_DIR / script_name),
        str(csv_path),
        "--out-dir",
        str(out_dir),
        *COMMON_ARGS,
    ]
    print(f"\n=== RUN {backend} <- {csv_path.name} ===", flush=True)
    print(" ", " ".join(cmd), flush=True)
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=str(SCRIPT_DIR.parent))
    elapsed = time.perf_counter() - t0
    if proc.returncode != 0:
        raise RuntimeError(f"{backend} failed on {csv_path.name} (exit {proc.returncode})")
    print(f"  OK in {elapsed / 60:.1f} min -> {out_dir / csv_path.stem}", flush=True)
    return elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DATASETS),
        help="CSV names under scripts/dataset/",
    )
    args = parser.parse_args()

    run_times: dict[str, dict[str, float]] = {}
    for name in args.datasets:
        csv_path = DATASET_DIR / name
        if not csv_path.exists():
            raise FileNotFoundError(csv_path)
        stem = csv_path.stem
        run_times[stem] = {}
        for backend, script_name, out_folder in BACKENDS:
            run_times[stem][backend] = run_one(backend, script_name, out_folder, csv_path)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    manifest = OUT_ROOT / "aim_backend_runs.json"
    payload = {
        "dataset_dir": str(DATASET_DIR),
        "out_root": str(OUT_ROOT),
        "params": {
            "pixel_nm": 100,
            "segmentation": 50,
            "intersect_d_nm": 20,
            "roi_r_nm": 60,
        },
        "backends": {
            label: str(OUT_ROOT / folder) for label, _, folder in BACKENDS
        },
        "run_times_sec": run_times,
    }
    manifest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved run manifest -> {manifest}")
    print("All done.")


if __name__ == "__main__":
    main()
