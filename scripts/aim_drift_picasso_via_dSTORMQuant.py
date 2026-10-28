#!/usr/bin/env python3
"""Run Picasso AIM settings through dSTORMQuant ``apply_aim_drift``.

Uses the adapted AIM in this repository with Picasso AIM counterparts:

- ``fft_peak_scale: intersect_d``
- ``exclude_self_round2: true``
- ``drift_reference: mean``
"""

from __future__ import annotations

import pandas as pd
import argparse
import json
import shutil

from pathlib import Path
from typing import Any

from dSTORMQuant.processing.drift_correction.drift import apply_aim_drift
from dSTORMQuant.visualization.visualization import save_napari_points_screenshot

SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_DIR = SCRIPT_DIR / "dataset"
DEFAULT_OUT_DIR = SCRIPT_DIR / "output" / "aim_drift_picasso_via_dSTORMQuant"

PIXEL_SIZE_NM = 117.0
SEGMENTATION = 100
INTERSECT_D_NM = 20.0
ROI_R_NM = 60.0


def _resolve_input(path: Path) -> Path:
    """Resolve CSV path; bare names are looked up under scripts/dataset/."""
    if path.exists():
        return path.resolve()
    for cand in (DATASET_DIR / path, DATASET_DIR / path.name):
        if cand.exists():
            return cand.resolve()
    raise SystemExit(f"Input not found: {path} (also checked under {DATASET_DIR})")


def _link_or_copy(src: Path, dst: Path) -> None:
    """Prefer a hard link for large CSVs; fall back to copy."""
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        dst.hardlink_to(src)
    except OSError:
        shutil.copy2(src, dst)


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
# Picasso AIM counterparts exposed by drift_correction.aim_options
AIM_OPTIONS = {
    "fft_peak_scale": "intersect_d",
    "exclude_self_round2": True,
    "drift_reference": "mean",
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


def _short_stem(csv_path: Path, max_len: int = 40) -> str:
    stem = csv_path.stem
    return stem[:max_len].rstrip("_") if len(stem) > max_len else stem


def _write_run_info(out_dir: Path, payload: dict[str, Any]) -> None:
    path = out_dir / "run_info.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"  run info -> {path}")


def _list_drift_outputs(drift_corrected_dir: Path, base: str) -> list[str]:
    names = [
        f"{base}_after_drift_correction.csv",
        f"{base}_aimdrift.txt",
        f"{base}_drift_segments.csv",
    ]
    found: list[str] = []
    for name in names:
        p = drift_corrected_dir / name
        if p.exists():
            found.append(str(p))
    parent = drift_corrected_dir.parent
    for pattern in (
        f"drift_correction_plot_{base}.png",
        f"drift_correction_plot_{base}.svg",
    ):
        for p in parent.rglob(pattern):
            found.append(str(p))
    return found


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
    aim_options = dict(AIM_OPTIONS)

    print(
        f"AIM (Picasso settings using dSTORMQuant): pixel={pixel_size} nm, "
        f"segmentation={segmentation}, "
        f"intersect_d={intersect_d_nm} nm ({intersect_d:.4f} px), "
        f"roi_r={roi_r_nm} nm ({roi_r:.4f} px), "
        f"aim_options={aim_options}"
    )

    stem = _short_stem(csv_path)
    file_out = out_dir / stem
    file_out.mkdir(parents=True, exist_ok=True)
    drift_dir = file_out / "drift_corrected"
    drift_dir.mkdir(parents=True, exist_ok=True)

    local_csv = file_out / "input.csv"
    if csv_path.resolve() != local_csv.resolve():
        _link_or_copy(csv_path, local_csv)

    _write_run_info(
        file_out,
        {
            "mode": "Picasso AIM settings using dSTORMQuant apply_aim_drift",
            "input": str(csv_path.resolve()),
            "pixel_size_nm": pixel_size,
            "segmentation": segmentation,
            "intersect_d_nm": intersect_d_nm,
            "roi_r_nm": roi_r_nm,
            "aim_options": aim_options,
            "drift_validation": DRIFT_VALIDATION,
            "sanity_checks": SANITY_CHECKS,
            "backend": "dSTORMQuant.processing.drift_correction.drift.apply_aim_drift",
        },
    )

    if use_napari:
        df_before = pd.read_csv(local_csv)
        _save_napari(
            df_before,
            file_out,
            "napari_view_before_drift_correction.png",
            first=True,
            use_napari=True,
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
        aim_options=aim_options,
        # Large Dryad tables: skip multi‑GB corrected CSV when Napari is off.
        export_corrected_csv=use_napari,
    )
    print(f"  drift-corrected output -> {corrected_csv}")

    for path in _list_drift_outputs(drift_dir, local_csv.stem):
        print(f"  output -> {path}")

    # Avoid reloading multi‑GB corrected CSVs when Napari is disabled.
    if use_napari:
        df_after = pd.read_csv(corrected_csv)
        _save_napari(
            df_after,
            drift_dir,
            "napari_view_after_drift_correction.png",
            first=False,
            use_napari=True,
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

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_paths: list[Path] = []
    for p in args.inputs:
        resolved = _resolve_input(p)
        if resolved.suffix.lower() != ".csv":
            raise SystemExit(f"Expected CSV, got: {resolved}")
        csv_paths.append(resolved)

    print(
        "AIM parameters (Picasso settings via dSTORMQuant):\n"
        f"  pixel_size={args.pixel} nm\n"
        f"  segmentation={args.segmentation}\n"
        f"  intersect_d_nm={args.intersect_d_nm}\n"
        f"  roi_r_nm={args.roi_r_nm}\n"
        f"  aim_options={AIM_OPTIONS}\n"
        f"  napari screenshots: {not args.no_napari}\n"
        f"  dataset_dir={DATASET_DIR}\n"
        f"  out_dir={out_dir}"
    )

    for csv_path in csv_paths:
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
