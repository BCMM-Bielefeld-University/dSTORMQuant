"""Mortensen et al. localization-precision recalculation.

Equations from Mortensen et al., Nat. Methods 7, 377-381 (2010), matching
GDSC SMLM Gaussian2DPeakResultHelper.getVarianceX / getMLVarianceX.

Symbols
  s   : PSF standard deviation (nm)
  a   : pixel size at the sample (nm)
  N   : detected photons in the spot
  b^2 : background variance, photons per pixel
        (b2 = bg * a^2 if bg is given in photons/nm^2)

Pixelation-corrected PSF variance:
  s_a^2 = s^2 + a^2 / 12

Eq. 5 (MLE):
  <(dx)^2> = (s_a^2 / N) * 1 / (1 + I)
  I = Li2(-k) / k,  k = N*a^2 / (2*pi*s_a^2*b^2)

Eq. 6 (least-squares):
  <(dx)^2> = (s_a^2 / N) * (16/9 + 8*pi*s_a^2*b^2 / (N*a^2))

Reported precision (1D, per axis):
  precision = sqrt(F * <(dx)^2>)
  F = 1 for sCMOS/CCD, F = 2 for EMCCD
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy.special import spence

from dSTORMQuant.utils.logger import get_logger

logger = get_logger()

SigmaMode = Literal["mean", "geo", "x", "y"]
BgUnit = Literal["per_nm2", "per_pixel"]
LPMethod = Literal["mle", "lsq"]


def var_mle(
    sigma: np.ndarray, a: np.ndarray, n: np.ndarray, b2: np.ndarray
) -> np.ndarray:
    """Mortensen Eq. 5 variance (nm^2), vectorised."""
    sa2 = sigma**2 + a**2 / 12.0
    k = n * a**2 / (2.0 * np.pi * sa2 * np.maximum(b2, 1e-300))
    integral = np.where(b2 > 0, spence(1.0 + k) / k, 0.0)
    return (sa2 / n) / (1.0 + integral)


def var_lsq(
    sigma: np.ndarray, a: np.ndarray, n: np.ndarray, b2: np.ndarray
) -> np.ndarray:
    """Mortensen Eq. 6 variance (nm^2), vectorised."""
    sa2 = sigma**2 + a**2 / 12.0
    return (sa2 / n) * (16.0 / 9.0 + 8.0 * np.pi * sa2 * b2 / (n * a**2))


def resolve_sigma(
    df: pd.DataFrame,
    *,
    sigma_col: str | None,
    sigma_x_col: str | None,
    sigma_y_col: str | None,
    sigma_mode: SigmaMode,
) -> pd.Series:
    """Build one sigma series from CSV columns present in ``df``.

    Preference:
      1. Both ``sigma_x`` and ``sigma_y`` if present (combined via ``sigma_mode``)
      2. Else single ``sigma`` if configured and present
      3. Else raise with the headers that were tried
    """
    have_xy = (
        sigma_x_col is not None
        and sigma_y_col is not None
        and sigma_x_col in df.columns
        and sigma_y_col in df.columns
    )
    if have_xy:
        sx = df[sigma_x_col].astype(float)
        sy = df[sigma_y_col].astype(float)
        return {
            "mean": (sx + sy) / 2.0,
            "geo": np.sqrt(sx * sy),
            "x": sx,
            "y": sy,
        }[sigma_mode]

    if sigma_col is not None and sigma_col in df.columns:
        return df[sigma_col].astype(float)

    tried = []
    if sigma_x_col or sigma_y_col:
        tried.append(f"sigma_x/sigma_y={sigma_x_col!r}/{sigma_y_col!r}")
    if sigma_col:
        tried.append(f"sigma={sigma_col!r}")
    raise ValueError(
        "LP calculation: need both sigma_x and sigma_y columns, or a single "
        "sigma column. Tried: "
        + (", ".join(tried) if tried else "(no sigma columns configured)")
        + f". Available: {list(df.columns)}"
    )


def compute_localization_precision(
    sigma: np.ndarray,
    a: np.ndarray,
    n: np.ndarray,
    b2: np.ndarray,
    *,
    method: LPMethod = "mle",
    emccd: bool = False,
) -> np.ndarray:
    """Return 1D localization precision (nm) for valid rows; NaN elsewhere."""
    ok = (sigma > 0) & (n > 0) & (b2 >= 0) & np.isfinite(sigma + n + b2)
    emccd_factor = 2.0 if emccd else 1.0
    var_fn = var_mle if method == "mle" else var_lsq
    precision = np.full(len(sigma), np.nan, dtype=float)
    if np.any(ok):
        precision[ok] = np.sqrt(
            emccd_factor * var_fn(sigma[ok], a[ok], n[ok], b2[ok])
        )
    return precision


def lp_input_columns(config: dict[str, Any]) -> list[str]:
    """Exact config-mapped input headers needed for LP (excludes output_column)."""
    lp_cfg = config.get("lp_calculation") or {}
    if not lp_cfg.get("use", False):
        return []
    columns = lp_cfg.get("columns") or {}
    names: list[str] = []
    for key in ("intensity", "background", "sigma", "sigma_x", "sigma_y"):
        value = columns.get(key)
        if value and value not in names:
            names.append(value)
    return names


def lp_passthrough_columns(config: dict[str, Any]) -> list[str]:
    """CSV columns to carry through AIM when LP recalculation is enabled.

    Keeps the exact config-mapped headers (plus output_column) so later stages
    can recompute with strict column mapping after drift renames.
    """
    lp_cfg = config.get("lp_calculation") or {}
    if not lp_cfg.get("use", False):
        return []
    names = [lp_cfg.get("output_column", "LP_calculated")]
    for name in lp_input_columns(config):
        if name not in names:
            names.append(name)
    return names


def apply_lp_calculation(
    df: pd.DataFrame,
    config: dict[str, Any],
    *,
    stage: str | None = None,
) -> pd.DataFrame:
    """Recompute Mortensen LP into ``output_column``; leave any existing LP intact.

    Uses ``config['lp_calculation']`` and ``config['drift_correction']['pixel_size']``.
    When ``lp_calculation.use`` is false, returns ``df`` unchanged.

    Each localization uses only its own intensity, background, and sigma (plus
    the shared pixel size). Column headers must match ``lp_calculation.columns``
    exactly (no aliases).

    Raises:
        ValueError: If enabled but required mapped columns are missing.
    """
    lp_cfg = config.get("lp_calculation") or {}
    if not lp_cfg.get("use", False):
        return df

    columns = lp_cfg.get("columns") or {}
    intensity_col = columns.get("intensity", "intensity (photons)")
    background_col = columns.get("background", "background (photons/nm^2)")
    sigma_col = columns.get("sigma")
    sigma_x_col = columns.get("sigma_x", "sigmaX (nm)")
    sigma_y_col = columns.get("sigma_y", "sigmaY (nm)")
    output_col = lp_cfg.get("output_column", "LP_calculated")
    method: LPMethod = lp_cfg.get("method", "mle")
    emccd = bool(lp_cfg.get("emccd", False))
    bg_unit: BgUnit = lp_cfg.get("bg_unit", "per_nm2")
    sigma_mode: SigmaMode = lp_cfg.get("sigma_mode", "mean")
    pixel_size = float(config["drift_correction"]["pixel_size"])
    stage_txt = f" [{stage}]" if stage else ""

    for col, label in (
        (intensity_col, "intensity"),
        (background_col, "background"),
    ):
        if col not in df.columns:
            raise ValueError(
                f"LP calculation{stage_txt}: {label} column {col!r} not found. "
                f"Available: {list(df.columns)}"
            )

    sigma = resolve_sigma(
        df,
        sigma_col=sigma_col if sigma_col else None,
        sigma_x_col=sigma_x_col if sigma_x_col else None,
        sigma_y_col=sigma_y_col if sigma_y_col else None,
        sigma_mode=sigma_mode,
    )
    n_photons = df[intensity_col].astype(float)
    bg = df[background_col].astype(float)
    a = pd.Series(pixel_size, index=df.index, dtype=float)
    if bg_unit == "per_nm2":
        b2 = bg * a**2
    else:
        b2 = bg

    precision = compute_localization_precision(
        sigma.to_numpy(),
        a.to_numpy(),
        n_photons.to_numpy(),
        b2.to_numpy(),
        method=method,
        emccd=emccd,
    )

    out = df.copy()
    out[output_col] = precision
    n_valid = int(np.isfinite(precision).sum())
    logger.info(
        "LP recalculation%s (%s, emccd=%s): wrote %s for %d/%d localizations "
        "(pixel_size=%.4g nm; intensity=%r, background=%r)",
        stage_txt,
        method,
        emccd,
        output_col,
        n_valid,
        len(out),
        pixel_size,
        intensity_col,
        background_col,
    )
    return out
