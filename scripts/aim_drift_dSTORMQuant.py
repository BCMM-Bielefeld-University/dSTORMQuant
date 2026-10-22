#!/usr/bin/env python3

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd

from dSTORMQuant.processing.drift_correction.drift import apply_aim_drift
from dSTORMQuant.visualization.visualization import save_napari_points_screenshot

# Manual AIM defaults
PIXEL_SIZE_NM = 117.0
SEGMENTATION = 100
INTERSECT_D_NM = 20.0
ROI_R_NM = 60.0
SANITY_CHECKS = {
    "filter_boundary": True,
    "filter_parameters": True,
    "remove_invalid_values": True,
}
DRIFT_VALIDATION = {
    "use": False,
    "max_segment_drift_nm": 35.0,
    "n_previous_segments": 10,
}
_CHANNEL_COLORS = ("cyan", "magenta", "yellow", "lime", "orange", "white")


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


def run_one(
    csv_path: Path,
    out_dir: Path,
    *,
    pixel_size: float,
    segmentation: int,
    intersect_d_nm: float,
    roi_r_nm: float,
    use_napari: bool,
) -> str:
    intersect_d = intersect_d_nm / pixel_size
    roi_r = roi_r_nm / pixel_size

    print(
        f"AIM (dSTORMQuant current): pixel={pixel_size} nm, "
        f"segmentation={segmentation}, "
        f"intersect_d={intersect_d_nm} nm ({intersect_d:.4f} px), "
        f"roi_r={roi_r_nm} nm ({roi_r:.4f} px)"
    )

    stem = csv_path.stem
    short = stem[:40].rstrip("_") if len(stem) > 40 else stem
    file_out = out_dir / short
    file_out.mkdir(parents=True, exist_ok=True)
    (file_out / "drift_corrected").mkdir(parents=True, exist_ok=True)

    local_csv = file_out / "input.csv"
    if csv_path.resolve() != local_csv.resolve():
        shutil.copy2(csv_path, local_csv)

    df_before = pd.read_csv(local_csv)
    _save_napari(
        df_before,
        file_out,
        "napari_view_before_drift_correction.png",
        first=True,
        use_napari=use_napari,
    )

    corrected_csv, _drift = apply_aim_drift(
        str(local_csv),
        local_csv.name,
        file_out,
        pixel_size,
        segmentation,
        intersect_d,
        roi_r,
        sanity_config=dict(SANITY_CHECKS),
        drift_validation_config=dict(DRIFT_VALIDATION),
    )
    print(f"  drift-corrected CSV -> {corrected_csv}")

    df_after = pd.read_csv(corrected_csv)
    _save_napari(
        df_after,
        file_out / "drift_corrected",
        "napari_view_after_drift_correction.png",
        first=False,
        use_napari=use_napari,
    )
    return corrected_csv


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="one or more single-channel localization CSV files",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/output/aim_drift_dSTORMQuant"),
        help="output root (default: data/output/aim_drift_dSTORMQuant)",
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

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    inputs = [p.resolve() for p in args.inputs]
    for p in inputs:
        if not p.exists():
            raise SystemExit(f"Input not found: {p}")
        if p.suffix.lower() != ".csv":
            raise SystemExit(f"Expected CSV, got: {p}")

    print(
        f"AIM parameters (dSTORMQuant current, unmodified AIM):\n"
        f"  pixel_size={args.pixel} nm\n"
        f"  segmentation={args.segmentation}\n"
        f"  intersect_d_nm={args.intersect_d_nm}\n"
        f"  roi_r_nm={args.roi_r_nm}\n"
        f"  FFT scale: (2*roi_r)/box  [as in AIM.py]\n"
        f"  napari screenshots: {not args.no_napari}"
    )

    for csv_path in inputs:
        print(f"\n=== {csv_path.name} ===")
        run_one(
            csv_path,
            out_dir,
            pixel_size=args.pixel,
            segmentation=args.segmentation,
            intersect_d_nm=args.intersect_d_nm,
            roi_r_nm=args.roi_r_nm,
            use_napari=not args.no_napari,
        )

    print("\nAll done.")


if __name__ == "__main__":
    main()
