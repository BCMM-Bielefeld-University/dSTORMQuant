#!/usr/bin/env python3
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy.special import spence

# ---------------------------------------------------------------------------
# Equations (Mortensen et al., Nat. Methods 7, 377-381, 2010)
# Same forms as GDSC SMLM Gaussian2DPeakResultHelper.getVarianceX / getMLVarianceX.
#
# Symbols
#   s   : PSF standard deviation (nm)                 -> sigma
#   a   : pixel size at the sample (nm)               -> a
#   N   : detected photons in the spot                -> n
#   b^2 : background variance, photons per pixel      -> b2
#         (b2 = bg * a^2 if bg is given in photons/nm^2)
#
# Pixelation-corrected PSF variance:
#   s_a^2 = s^2 + a^2 / 12
#
# Dimensionless background parameter (paper's tau):
#   tau = 2*pi * s_a^2 * b^2 / (N * a^2)          (code uses k = 1/tau)
#
# Eq. 5 (MLE, Cramer-Rao-limit variance):
#   <(dx)^2> = (s_a^2 / N) * 1 / (1 + I)
#   I = INT_0^1 ln(t) / (1 + k*t) dt = Li2(-k) / k,   k = N*a^2 / (2*pi*s_a^2*b^2)
#   with Li2(-k) = spence(1 + k)  (scipy's spence(z) = Li2(1 - z))
#   Limit b -> 0:  <(dx)^2> -> s_a^2 / N
#
# Eq. 6 (least-squares fit):
#   <(dx)^2> = (s_a^2 / N) * (16/9 + 8*pi*s_a^2*b^2 / (N*a^2))
#
# Reported precision (1D, per axis):
#   precision = sqrt(F * <(dx)^2>)
#   F = 1 for sCMOS/CCD, F = 2 for EMCCD (excess noise factor, --emccd)
# ---------------------------------------------------------------------------

# Accepted header aliases (first match in the CSV wins).
BG_ALIASES = (
    "background (photons/nm^2)",
    "background [photon]",
    "background [photons]",
    "background",
    "bg",  # dSTORMQuant internal
)
SIGMA_ALIASES = ("sigma [nm]", "sigma (nm)", "sigma")
SIGMA_X_ALIASES = ("sigmaX (nm)", "sigma_x [nm]", "sigma_x", "sx")
SIGMA_Y_ALIASES = ("sigmaY (nm)", "sigma_y [nm]", "sigma_y", "sy")
INTENSITY_ALIASES = (
    "intensity (photons)",
    "intensity (photon)",
    "intensity [photon]",
    "photons",
)
# Optional column to compare against (exported uncertainty / precision).
REF_ALIASES = (
    "localization precision (nm)",
    "uncertainty [nm]",
    "uncertainty_xy [nm]",
    "uncertainty",
    "lp",  # dSTORMQuant internal
)
CHANNEL_ALIASES = ("channelIndex", "channel")
CHANNEL_NAME_ALIASES = ("channelName",)


def var_mle(sigma: np.ndarray, a: np.ndarray, n: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """Mortensen Eq. 5 variance (nm^2), vectorised.

    <(dx)^2> = (s_a^2 / N) * 1 / (1 + I)
    s_a^2    = sigma^2 + a^2 / 12
    k        = N * a^2 / (2*pi * s_a^2 * b^2)        (= 1 / tau)
    I        = INT_0^1 ln(t) / (1 + k*t) dt = Li2(-k) / k
    """
    # s_a^2 = sigma^2 + a^2/12
    sa2 = sigma**2 + a**2 / 12.0
    # k = N a^2 / (2 pi s_a^2 b^2)
    k = n * a**2 / (2.0 * np.pi * sa2 * np.maximum(b2, 1e-300))
    # I = Li2(-k)/k, and Li2(-k) = spence(1 + k); I = 0 when b^2 = 0
    integral = np.where(b2 > 0, spence(1.0 + k) / k, 0.0)
    # <(dx)^2> = (s_a^2 / N) / (1 + I)
    return (sa2 / n) / (1.0 + integral)


def var_lsq(sigma: np.ndarray, a: np.ndarray, n: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """Mortensen Eq. 6 variance (nm^2), vectorised.

    <(dx)^2> = (s_a^2 / N) * (16/9 + 8*pi * s_a^2 * b^2 / (N * a^2))
    s_a^2    = sigma^2 + a^2 / 12
    """
    sa2 = sigma**2 + a**2 / 12.0
    return (sa2 / n) * (16.0 / 9.0 + 8.0 * np.pi * sa2 * b2 / (n * a**2))


def parse_pixel_map(s: str) -> dict[int, float]:
    """'0=117,1=116.5' -> {0: 117.0, 1: 116.5}"""
    out: dict[int, float] = {}
    for item in s.split(","):
        key, value = item.split("=")
        out[int(key.strip())] = float(value)
    return out


def pick_column(df: pd.DataFrame, aliases: tuple[str, ...], *, required: bool = True) -> str | None:
    for name in aliases:
        if name in df.columns:
            return name
    if required:
        raise SystemExit(
            "Missing required column. Tried: "
            + ", ".join(repr(a) for a in aliases)
            + f"\nAvailable: {list(df.columns)}"
        )
    return None


def resolve_sigma(df: pd.DataFrame, sigma_mode: str) -> pd.Series:
    """Build one sigma series from a single sigma column or sigmaX/sigmaY."""
    single = pick_column(df, SIGMA_ALIASES, required=False)
    if single is not None:
        return df[single].astype(float)

    sx_name = pick_column(df, SIGMA_X_ALIASES)
    sy_name = pick_column(df, SIGMA_Y_ALIASES)
    assert sx_name is not None and sy_name is not None
    sx, sy = df[sx_name].astype(float), df[sy_name].astype(float)
    return {
        "mean": (sx + sy) / 2.0,
        "geo": np.sqrt(sx * sy),
        "x": sx,
        "y": sy,
    }[sigma_mode]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("csv", help="input localization CSV")
    parser.add_argument(
        "--pixel",
        type=float,
        default=117.0,
        help="pixel size at sample in nm (default: 117)",
    )
    parser.add_argument(
        "--pixel-map",
        help="per-channel pixel sizes, e.g. '0=117,1=116.5' (overrides --pixel)",
    )
    parser.add_argument(
        "--bg-unit",
        choices=["per_nm2", "per_pixel"],
        default="per_nm2",
        help="units of the background column (default: per_nm2)",
    )
    parser.add_argument(
        "--sigma-mode",
        choices=["mean", "geo", "x", "y"],
        default="mean",
        help="how to combine sigmaX/sigmaY when both exist (ignored for a single sigma column)",
    )
    parser.add_argument(
        "--emccd",
        action="store_true",
        help="apply EMCCD factor 2 on variance (default: off)",
    )
    parser.add_argument("--out", help="output CSV path")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    bg_col = pick_column(df, BG_ALIASES)
    n_col = pick_column(df, INTENSITY_ALIASES)
    ref_col = pick_column(df, REF_ALIASES, required=False)
    ch_col = pick_column(df, CHANNEL_ALIASES, required=False)
    ch_name_col = pick_column(df, CHANNEL_NAME_ALIASES, required=False)

    pmap = parse_pixel_map(args.pixel_map) if args.pixel_map else {}
    if ch_col is not None:
        a = df[ch_col].map(lambda c: pmap.get(int(c), args.pixel)).astype(float)
    else:
        a = pd.Series(args.pixel, index=df.index, dtype=float)

    sigma = resolve_sigma(df, args.sigma_mode)
    n_photons = df[n_col].astype(float)
    bg = df[bg_col].astype(float)
    if args.bg_unit == "per_nm2":
        # photons/nm^2 -> photons/pixel:  b^2 = bg * a^2
        b2 = bg * a**2
    else:
        b2 = bg  # already photons/pixel

    ok = (sigma > 0) & (n_photons > 0) & (b2 >= 0) & np.isfinite(sigma + n_photons + b2)
    # precision = sqrt(F * <(dx)^2>), F = 2 for EMCCD excess noise, else 1
    emccd_factor = 2.0 if args.emccd else 1.0

    df["pixel_size_nm"] = a
    df["sigma_used_nm"] = sigma
    df["bg_photons_per_pixel"] = b2
    # Eq. 5 (MLE)
    df["prec_mle_nm"] = np.where(
        ok,
        np.sqrt(emccd_factor * var_mle(sigma.to_numpy(), a.to_numpy(), n_photons.to_numpy(), b2.to_numpy())),
        np.nan,
    )
    # Eq. 6 (least squares)
    df["prec_lsq_nm"] = np.where(
        ok,
        np.sqrt(emccd_factor * var_lsq(sigma.to_numpy(), a.to_numpy(), n_photons.to_numpy(), b2.to_numpy())),
        np.nan,
    )

    groups: list[tuple[object, pd.DataFrame]]
    if ch_col is not None:
        groups = list(df.groupby(ch_col, sort=True))
    else:
        groups = [("all", df)]

    print(
        f"Rows: {len(df)}, valid: {int(ok.sum())}, "
        f"EMCCD factor: {emccd_factor:g}, bg-unit: {args.bg_unit}"
    )
    print(f"Columns used: bg={bg_col!r}, intensity={n_col!r}, ref={ref_col!r}")

    for ch, group in groups:
        name = ""
        if ch_name_col is not None and ch_name_col in group.columns:
            raw_name = group[ch_name_col].iloc[0]
            if pd.notna(raw_name):
                name = str(raw_name)
        mask = ok.loc[group.index]
        gv = group.loc[mask]
        if gv.empty:
            print(f"\n=== Channel {ch} {name} === no valid rows")
            continue

        print(
            f"\n=== Channel {ch} {name} === rows: {len(group)}, valid: {len(gv)}, "
            f"pixel = {gv['pixel_size_nm'].iloc[0]:g} nm"
        )
        print(
            f"  a > sigma in "
            f"{(gv['pixel_size_nm'] > gv['sigma_used_nm']).mean() * 100:.1f}% of rows "
            f"(Mortensen assumes a <= sigma)"
        )
        print(
            f"  median sigma = {gv['sigma_used_nm'].median():.1f} nm, "
            f"N = {gv[n_col].median():.0f} photons, "
            f"bg = {gv['bg_photons_per_pixel'].median():.3g} photons/pixel"
        )
        ref_txt = ""
        if ref_col is not None:
            ref_txt = f"{ref_col} = {gv[ref_col].median():.2f} | "
        print(
            f"  median precision (nm): {ref_txt}"
            f"Eq.5 MLE = {gv['prec_mle_nm'].median():.2f} | "
            f"Eq.6 LSQ = {gv['prec_lsq_nm'].median():.2f}"
        )
        if ref_col is not None:
            for computed in ("prec_mle_nm", "prec_lsq_nm"):
                ratio = gv[computed] / gv[ref_col]
                corr = gv[[computed, ref_col]].corr().iloc[0, 1]
                print(
                    f"  {computed}: median ratio to {ref_col} = {ratio.median():.3f}, "
                    f"correlation = {corr:.4f}"
                )

    out = args.out or args.csv.rsplit(".", 1)[0] + "_recomputed.csv"
    df.to_csv(out, index=False)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()