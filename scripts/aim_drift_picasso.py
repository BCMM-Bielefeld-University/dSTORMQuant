#!/usr/bin/env python3
"""Run full upstream Picasso AIM drift correction on localization CSV files.
"""

from __future__ import annotations

import argparse
import json
import numpy as np
import pandas as pd

from numpy import savetxt
from picasso import aim, io
from pathlib import Path
from typing import Any

from dSTORMQuant.utils.data_handling import read_localization_csv
from dSTORMQuant.visualization.visualization import (
    plot_drift_data,
    save_napari_points_screenshot,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_DIR = SCRIPT_DIR / "dataset"
DEFAULT_OUT_DIR = SCRIPT_DIR / "output" / "aim_drift_picasso"

PIXEL_SIZE_NM = 117.0
SEGMENTATION = 100
INTERSECT_D_NM = 20.0
ROI_R_NM = 60.0
_CHANNEL_COLORS = ("cyan", "magenta", "yellow", "lime", "orange", "white")


def _resolve_input(path: Path) -> Path:
    """Resolve CSV path; bare names are looked up under scripts/dataset/."""
    if path.exists():
        return path.resolve()
    for cand in (DATASET_DIR / path, DATASET_DIR / path.name):
        if cand.exists():
            return cand.resolve()
    raise SystemExit(f"Input not found: {path} (also checked under {DATASET_DIR})")


def _channels_from_df(df: pd.DataFrame) -> dict[str, tuple[str, str, int]]:
    if "channelIndex" not in df.columns:
        df = df.copy()
        df["channelIndex"] = 0
    channels: dict[str, tuple[str, str, int]] = {}
    for i, ch in enumerate(sorted(int(c) for c in df["channelIndex"].unique())):
        color = _CHANNEL_COLORS[i % len(_CHANNEL_COLORS)]
        key = f"Ch{ch}"
        channels[key] = (color, key, ch)
    return channels


def _save_napari(
    df: pd.DataFrame,
    out_dir: Path,
    filename: str,
    *,
    first: bool,
    use_napari: bool,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    save_napari_points_screenshot(
        df,
        _channels_from_df(df),
        str(out_dir),
        filename,
        first=first,
        scale=1,
        use_napari=use_napari,
    )
    if use_napari:
        print(f"  napari -> {out_dir / filename}")


def _short_stem(csv_path: Path, max_len: int = 40) -> str:
    stem = csv_path.stem
    return stem[:max_len].rstrip("_") if len(stem) > max_len else stem


def _write_run_info(out_dir: Path, payload: dict[str, Any]) -> None:
    path = out_dir / "run_info.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"  run info -> {path}")


def _pick(
    df: pd.DataFrame, aliases: tuple[str, ...], *, required: bool = True
) -> str | None:
    for name in aliases:
        if name in df.columns:
            return name
    if required:
        raise SystemExit(
            "Missing column. Tried: "
            + ", ".join(repr(a) for a in aliases)
            + f"\nAvailable: {list(df.columns)}"
        )
    return None


def csv_to_locs(
    df: pd.DataFrame, pixel_nm: float, source: str
) -> tuple[pd.DataFrame, list[dict]]:
    """Build Picasso locs DataFrame + info (camera pixels)."""
    frame_col = _pick(df, ("frameIndex", "frame", "frames"))
    x_col = _pick(df, ("x (nm)", "x [nm]", "x"))
    y_col = _pick(df, ("y (nm)", "y [nm]", "y"))
    photons_col = _pick(
        df,
        ("intensity (photons)", "intensity (photon)", "intensity [photon]", "photons"),
        required=False,
    )
    sx_col = _pick(df, ("sigmaX (nm)", "sigma_x [nm]", "sx"), required=False)
    sy_col = _pick(df, ("sigmaY (nm)", "sigma_y [nm]", "sy"), required=False)
    sigma_col = _pick(df, ("sigma [nm]", "sigma (nm)", "sigma"), required=False)
    bg_col = _pick(
        df,
        ("background (photons/nm^2)", "background [photon]", "background", "bg"),
        required=False,
    )
    lp_col = _pick(
        df,
        ("localization precision (nm)", "uncertainty [nm]", "uncertainty_xy [nm]", "lp"),
        required=False,
    )
    ch_col = _pick(df, ("channelIndex", "channel"), required=False)

    frames = df[frame_col].astype(np.int64)
    frames = frames - int(frames.min())
    x = df[x_col].astype(float) / pixel_nm
    y = df[y_col].astype(float) / pixel_nm

    out = pd.DataFrame(
        {
            "frame": frames.astype(np.uint32),
            "x": x.astype(np.float32),
            "y": y.astype(np.float32),
        }
    )
    if photons_col is not None:
        out["photons"] = df[photons_col].astype(np.float32)
    if sx_col is not None and sy_col is not None:
        out["sx"] = (df[sx_col].astype(float) / pixel_nm).astype(np.float32)
        out["sy"] = (df[sy_col].astype(float) / pixel_nm).astype(np.float32)
    elif sigma_col is not None:
        s = (df[sigma_col].astype(float) / pixel_nm).astype(np.float32)
        out["sx"] = s
        out["sy"] = s
    if bg_col is not None:
        out["bg"] = df[bg_col].astype(np.float32)
    if lp_col is not None:
        out["lp"] = df[lp_col].astype(np.float32)
    if ch_col is not None:
        out["channelIndex"] = df[ch_col].fillna(0).astype(np.uint32)
    else:
        out["channelIndex"] = np.zeros(len(out), dtype=np.uint32)

    ok = np.isfinite(out["x"]) & np.isfinite(out["y"]) & (out["x"] > 0) & (out["y"] > 0)
    n_drop = int((~ok).sum())
    if n_drop:
        print(f"  dropped {n_drop:,} non-finite / non-positive x,y rows")
        out = out.loc[ok].reset_index(drop=True)

    out = out.sort_values(["channelIndex", "frame"], kind="mergesort").reset_index(
        drop=True
    )

    info = [
        {
            "Generated by": "aim_drift_picasso.py (upstream Picasso AIM)",
            "Frames": int(out["frame"].max()) + 1,
            "Height": int(np.ceil(float(out["y"].max()))),
            "Width": int(np.ceil(float(out["x"].max()))),
            "Pixelsize": float(pixel_nm),
            "PixelSize": float(pixel_nm),
            "Source": source,
        }
    ]
    return out, info


def run_one(
    csv_path: Path,
    out_dir: Path,
    *,
    pixel_nm: float,
    segmentation: int,
    intersect_d_px: float,
    roi_r_px: float,
    use_napari: bool,
) -> None:
    print(f"\n=== {csv_path.name} ===")
    df = read_localization_csv(csv_path)
    locs, info = csv_to_locs(df, pixel_nm, str(csv_path))
    print(f"  locs: {len(locs):,}")

    stem = _short_stem(csv_path)
    file_out = out_dir / stem
    file_out.mkdir(parents=True, exist_ok=True)
    drift_dir = file_out / "drift_corrected"
    drift_dir.mkdir(parents=True, exist_ok=True)

    _write_run_info(
        file_out,
        {
            "mode": "upstream Picasso AIM (picasso.aim)",
            "input": str(csv_path.resolve()),
            "pixel_size_nm": pixel_nm,
            "segmentation": segmentation,
            "intersect_d_px": intersect_d_px,
            "roi_r_px": roi_r_px,
            "backend": "picasso.aim.aim",
        },
    )

    _save_napari(
        df,
        file_out,
        "napari_view_before_drift_correction.png",
        first=True,
        use_napari=use_napari,
    )

    hdf5_in = file_out / "input_locs.hdf5"
    io.save_locs(str(hdf5_in), locs, info)
    print(f"  wrote {hdf5_in.name}")

    print(
        f"  Picasso aim.aim: segmentation={segmentation}, "
        f"intersect_d={intersect_d_px:.4f} px, roi_r={roi_r_px:.4f} px"
    )
    undrifted, new_info, drift = aim.aim(
        locs,
        info,
        segmentation=segmentation,
        intersect_d=intersect_d_px,
        roi_r=roi_r_px,
        progress="console",
    )

    hdf5_out = file_out / "after_drift_correction.hdf5"
    io.save_locs(str(hdf5_out), undrifted, new_info)
    print(f"  wrote {hdf5_out.name}")

    drift_csv = file_out / "aim_drift.csv"
    if isinstance(drift, pd.DataFrame):
        drift_df = drift
    else:
        drift_df = pd.DataFrame(drift)
    drift_df.to_csv(drift_csv, index=False)
    print(f"  wrote {drift_csv.name}")

    if "x" in drift_df.columns and "y" in drift_df.columns:
        drift_xy = drift_df[["x", "y"]].to_numpy(dtype=float)
    elif "dx" in drift_df.columns and "dy" in drift_df.columns:
        drift_xy = drift_df[["dx", "dy"]].to_numpy(dtype=float)
    else:
        raise SystemExit(
            f"Unrecognized Picasso drift columns: {drift_df.columns.tolist()}"
        )

    drift_txt = drift_dir / "input_aimdrift.txt"
    savetxt(drift_txt, drift_xy, header="dx\tdy", newline="\r\n")
    plot_drift_data(str(drift_txt), file_out, "input", pixel_nm)
    print(f"  wrote {drift_txt}")
    print("  wrote drift_correction_plot_input.png/.svg")

    # Full nm CSV + Napari are optional; skip on large datasets when --no-napari.
    if use_napari:
        export = undrifted.copy()
        for col in ("x", "y", "sx", "sy"):
            if col in export.columns:
                export[col] = export[col].astype(float) * pixel_nm
        csv_out = file_out / "after_drift_correction.csv"
        export.to_csv(csv_out, index=False)
        csv_out_dc = drift_dir / "after_drift_correction.csv"
        export.to_csv(csv_out_dc, index=False)
        print(f"  wrote {csv_out.name} and {csv_out_dc} ({len(export):,} rows, nm)")
        _save_napari(
            export,
            file_out,
            "napari_view_after_drift_correction.png",
            first=False,
            use_napari=True,
        )
    else:
        print("  skipped after_drift_correction.csv / Napari (--no-napari)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="CSV name(s) under scripts/dataset/ or absolute/relative CSV path(s)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"output root (default: {DEFAULT_OUT_DIR})",
    )
    parser.add_argument("--pixel", type=float, default=PIXEL_SIZE_NM)
    parser.add_argument("--segmentation", type=int, default=SEGMENTATION)
    parser.add_argument("--intersect-d-nm", type=float, default=INTERSECT_D_NM)
    parser.add_argument("--roi-r-nm", type=float, default=ROI_R_NM)
    parser.add_argument(
        "--no-napari",
        action="store_true",
        help="skip Napari before/after screenshots",
    )
    args = parser.parse_args()

    pixel_nm = args.pixel
    intersect_d_px = args.intersect_d_nm / pixel_nm
    roi_r_px = args.roi_r_nm / pixel_nm

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(
        "AIM parameters (upstream Picasso AIM):\n"
        f"  pixel_size={pixel_nm} nm\n"
        f"  segmentation={args.segmentation}\n"
        f"  intersect_d_nm={args.intersect_d_nm} ({intersect_d_px:.4f} px)\n"
        f"  roi_r_nm={args.roi_r_nm} ({roi_r_px:.4f} px)\n"
        f"  napari screenshots: {not args.no_napari}\n"
        f"  dataset_dir={DATASET_DIR}\n"
        f"  out_dir={out_dir}"
    )

    for p in args.inputs:
        resolved = _resolve_input(p)
        if resolved.suffix.lower() != ".csv":
            raise SystemExit(f"Expected CSV, got: {resolved}")
        run_one(
            resolved,
            out_dir,
            pixel_nm=pixel_nm,
            segmentation=args.segmentation,
            intersect_d_px=intersect_d_px,
            roi_r_px=roi_r_px,
            use_napari=not args.no_napari,
        )

    print("\nAll done.")


if __name__ == "__main__":
    main()
