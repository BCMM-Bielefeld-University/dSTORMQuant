from __future__ import annotations

from typing import Any

import pandas as pd

from dSTORMQuant.utils.logger import get_logger

logger = get_logger()


def apply_filters(
    df: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[dict[str, pd.DataFrame], dict[str, bool]]:
    """Apply sigma, intensity, localization-precision, and p-value filters in sequence.

    Filter thresholds are read from ``config['filtering']``. Stages are skipped when
    required measurement columns are absent.

    Args:
        df: Localization dataframe before filtering.
        config: Full pipeline configuration containing a ``filtering`` section.

    Returns:
        Tuple of:

        - ``results``: Snapshot dataframe after each nominal stage key
          (``after_sigma_filter``, ``after_photons_count_filter``, etc.).
        - ``applied``: Boolean flags indicating whether each stage modified data.
    """
    filtering = config["filtering"]
    results: dict[str, pd.DataFrame] = {}
    applied: dict[str, bool] = {
        "after_sigma_filter": False,
        "after_photons_count_filter": False,
        "after_localization_precision_filter": False,
        "after_pvalue_filter": False,
    }

    current = df
    # --- Sigma ---
    sigma_cfg = filtering["sigma"]
    if {"sx", "sy"}.issubset(current.columns):
        sigma_min = float(sigma_cfg["min_value"])
        sigma_max = float(sigma_cfg["max_value"])
        filtered = current[
            (current["sx"] >= sigma_min)
            & (current["sx"] <= sigma_max)
            & (current["sy"] >= sigma_min)
            & (current["sy"] <= sigma_max)
        ].copy()
        n0, n1 = len(current), len(filtered)
        logger.info(
            f"Sigma filter [{sigma_min}, {sigma_max}] nm: {n1}/{n0} localizations retained"
        )
        current = filtered
        applied["after_sigma_filter"] = True
    else:
        logger.warning(
            "Sigma filter requires columns sx/sy, which are missing; skipping sigma filter."
        )

    results["after_sigma_filter"] = current.copy()

    # --- Intensity ---
    int_cfg = filtering["intensity"]
    if "photons" in current.columns:
        intensity_min = float(int_cfg["min_value"])
        filtered = current[current["photons"] >= intensity_min].copy()
        n0, n1 = len(current), len(filtered)
        logger.info(
            f"Intensity filter (min_value={intensity_min:.1f} photons): "
            f"{n1}/{n0} localizations retained"
        )
        current = filtered
        applied["after_photons_count_filter"] = True
    else:
        logger.warning(
            "Photon count filter requires column photons, which is missing; skipping photon filter."
        )

    results["after_photons_count_filter"] = current.copy()

    # --- Localization precision ---
    lp_cfg = filtering["localization_precision"]
    if "lp" in current.columns:
        lp_thr = float(lp_cfg["threshold_value"])
        filtered = current[current["lp"] < lp_thr].copy()
        n0, n1 = len(current), len(filtered)
        logger.info(
            f"Localization precision filter (threshold={lp_thr} nm): "
            f"{n1}/{n0} localizations retained"
        )
        current = filtered
        applied["after_localization_precision_filter"] = True
    else:
        logger.warning(
            "Localization precision filter requires column lp, which is missing; "
            "skipping localization precision filter."
        )

    results["after_localization_precision_filter"] = current.copy()

    # --- P-value ---
    pv_cfg = filtering["p_value"]
    if "pvalue" in current.columns:
        pv_thr = float(pv_cfg["threshold_value"])
        filtered = current[current["pvalue"] < pv_thr].copy()
        n0, n1 = len(current), len(filtered)
        logger.info(
            f"P-value filter (threshold={pv_thr}): {n1}/{n0} localizations retained"
        )
        current = filtered
        applied["after_pvalue_filter"] = True
    else:
        logger.warning(
            "P-value filter requires column pvalue, which is missing; skipping p-value filter."
        )

    results["after_pvalue_filter"] = current.copy()

    return results, applied
