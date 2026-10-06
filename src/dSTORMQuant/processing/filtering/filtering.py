from __future__ import annotations

import json

from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import pandas as pd

from dSTORMQuant.utils.logger import get_logger

logger = get_logger()


class FilterSkipReason(str, Enum):
    DISABLED = "disabled_in_config"
    MISSING_COLUMNS = "missing_columns"


@dataclass(frozen=True)
class FilterStepReport:
    step_key: str
    display_name: str
    enabled: bool
    applied: bool
    skip_reason: FilterSkipReason | None
    required_columns: tuple[str, ...] = ()

    @property
    def skipped_missing_columns(self) -> bool:
        """Enabled in config but not applied because quality columns were absent."""
        return self.enabled and self.skip_reason == FilterSkipReason.MISSING_COLUMNS


def _filter_step_enabled(step_cfg: dict[str, Any]) -> bool:
    """Return whether a filter step is enabled (defaults to True if omitted)."""
    return bool(step_cfg.get("use", True))


def apply_filters(
    df: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[dict[str, pd.DataFrame], dict[str, bool], list[FilterStepReport]]:
    """Apply sigma, intensity, localization-precision, and p-value filters in sequence.

    Filter thresholds are read from ``config['filtering']``. Each stage is skipped when
    ``use: false`` or when required measurement columns are absent.

    Args:
        df: Localization dataframe before filtering.
        config: Full pipeline configuration containing a ``filtering`` section.

    Returns:
        Tuple of:

        - ``results``: Snapshot dataframe after each nominal stage key.
        - ``applied``: Boolean flags indicating whether each stage modified data.
        - ``reports``: Per-step status for logging and ``filtering_report.json``.
    """
    filtering = config["filtering"]
    results: dict[str, pd.DataFrame] = {}
    applied: dict[str, bool] = {
        "after_sigma_filter": False,
        "after_photons_count_filter": False,
        "after_localization_precision_filter": False,
        "after_pvalue_filter": False,
    }
    reports: list[FilterStepReport] = []

    current = df

    def _finish_step(
        step_key: str,
        display_name: str,
        enabled: bool,
        skip_reason: FilterSkipReason | None,
        required: tuple[str, ...],
        *,
        did_apply: bool,
    ) -> None:
        applied[step_key] = did_apply
        reports.append(
            FilterStepReport(
                step_key=step_key,
                display_name=display_name,
                enabled=enabled,
                applied=did_apply,
                skip_reason=skip_reason,
                required_columns=required,
            )
        )

    # --- Sigma ---
    sigma_cfg = filtering["sigma"]
    sigma_enabled = _filter_step_enabled(sigma_cfg)
    if not sigma_enabled:
        logger.info("Sigma filter disabled in config (use: false); skipping.")
        _finish_step(
            "after_sigma_filter",
            "sigma",
            False,
            FilterSkipReason.DISABLED,
            ("sx", "sy"),
            did_apply=False,
        )
    elif {"sx", "sy"}.issubset(current.columns):
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
        _finish_step(
            "after_sigma_filter",
            "sigma",
            True,
            None,
            ("sx", "sy"),
            did_apply=True,
        )
    else:
        logger.warning(
            "Sigma filter requires columns sx/sy, which are missing; skipping sigma filter."
        )
        _finish_step(
            "after_sigma_filter",
            "sigma",
            True,
            FilterSkipReason.MISSING_COLUMNS,
            ("sx", "sy"),
            did_apply=False,
        )

    results["after_sigma_filter"] = current.copy()

    # --- Intensity ---
    int_cfg = filtering["intensity"]
    int_enabled = _filter_step_enabled(int_cfg)
    if not int_enabled:
        logger.info("Intensity filter disabled in config (use: false); skipping.")
        _finish_step(
            "after_photons_count_filter",
            "intensity (photons)",
            False,
            FilterSkipReason.DISABLED,
            ("photons",),
            did_apply=False,
        )
    elif "photons" in current.columns:
        intensity_min = float(int_cfg["min_value"])
        filtered = current[current["photons"] >= intensity_min].copy()
        n0, n1 = len(current), len(filtered)
        logger.info(
            f"Intensity filter (min_value={intensity_min:.1f} photons): "
            f"{n1}/{n0} localizations retained"
        )
        current = filtered
        _finish_step(
            "after_photons_count_filter",
            "intensity (photons)",
            True,
            None,
            ("photons",),
            did_apply=True,
        )
    else:
        logger.warning(
            "Photon count filter requires column photons, which is missing; skipping photon filter."
        )
        _finish_step(
            "after_photons_count_filter",
            "intensity (photons)",
            True,
            FilterSkipReason.MISSING_COLUMNS,
            ("photons",),
            did_apply=False,
        )

    results["after_photons_count_filter"] = current.copy()

    # --- Localization precision ---
    lp_cfg = filtering["localization_precision"]
    lp_enabled = _filter_step_enabled(lp_cfg)
    if not lp_enabled:
        logger.info(
            "Localization precision filter disabled in config (use: false); skipping."
        )
        _finish_step(
            "after_localization_precision_filter",
            "localization precision",
            False,
            FilterSkipReason.DISABLED,
            ("lp",),
            did_apply=False,
        )
    elif "lp" in current.columns:
        lp_thr = float(lp_cfg["threshold_value"])
        filtered = current[current["lp"] < lp_thr].copy()
        n0, n1 = len(current), len(filtered)
        logger.info(
            f"Localization precision filter (threshold={lp_thr} nm): "
            f"{n1}/{n0} localizations retained"
        )
        current = filtered
        _finish_step(
            "after_localization_precision_filter",
            "localization precision",
            True,
            None,
            ("lp",),
            did_apply=True,
        )
    else:
        logger.warning(
            "Localization precision filter requires column lp, which is missing; "
            "skipping localization precision filter."
        )
        _finish_step(
            "after_localization_precision_filter",
            "localization precision",
            True,
            FilterSkipReason.MISSING_COLUMNS,
            ("lp",),
            did_apply=False,
        )

    results["after_localization_precision_filter"] = current.copy()

    # --- P-value ---
    pv_cfg = filtering["p_value"]
    pv_enabled = _filter_step_enabled(pv_cfg)
    if not pv_enabled:
        logger.info("P-value filter disabled in config (use: false); skipping.")
        _finish_step(
            "after_pvalue_filter",
            "p-value",
            False,
            FilterSkipReason.DISABLED,
            ("pvalue",),
            did_apply=False,
        )
    elif "pvalue" in current.columns:
        pv_thr = float(pv_cfg["threshold_value"])
        filtered = current[current["pvalue"] < pv_thr].copy()
        n0, n1 = len(current), len(filtered)
        logger.info(
            f"P-value filter (threshold={pv_thr}): {n1}/{n0} localizations retained"
        )
        current = filtered
        _finish_step(
            "after_pvalue_filter",
            "p-value",
            True,
            None,
            ("pvalue",),
            did_apply=True,
        )
    else:
        logger.warning(
            "P-value filter requires column pvalue, which is missing; skipping p-value filter."
        )
        _finish_step(
            "after_pvalue_filter",
            "p-value",
            True,
            FilterSkipReason.MISSING_COLUMNS,
            ("pvalue",),
            did_apply=False,
        )

    results["after_pvalue_filter"] = current.copy()

    log_filtering_summary(reports, filtering)
    return results, applied, reports


def log_filtering_summary(
    reports: list[FilterStepReport],
    filtering_cfg: dict[str, Any],
) -> None:
    """Emit a prominent log block summarizing which filters ran."""
    n_enabled = sum(1 for r in reports if r.enabled)
    n_applied = sum(1 for r in reports if r.applied)
    lines = [
        "Filtering summary "
        f"({n_applied}/{n_enabled} enabled steps actually applied to data):"
    ]
    for r in reports:
        if not r.enabled:
            status = "disabled (use: false)"
        elif r.applied:
            status = "applied"
        elif r.skip_reason == FilterSkipReason.MISSING_COLUMNS:
            status = f"NOT APPLIED — missing columns {list(r.required_columns)}"
        else:
            status = "not applied"
        lines.append(f"  - {r.display_name}: {status}")

    any_missing_column_skips = any(r.skipped_missing_columns for r in reports)
    msg = "\n".join(lines)
    if any_missing_column_skips:
        logger.warning(
            "%s\n"
            "Some enabled filters were skipped (missing quality columns after "
            "drift correction). Details in filtering_report.json under test_data/.",
            msg,
        )
    else:
        logger.info(msg)


def write_filtering_report_json(
    reports: list[FilterStepReport],
    path: str | Path,
    *,
    input_file: str,
) -> None:
    """Write per-step filtering status for downstream users and QC."""
    payload = {
        "input_file": input_file,
        "steps": [
            {
                **{k: v for k, v in asdict(r).items() if k != "skip_reason"},
                "skip_reason": r.skip_reason.value if r.skip_reason else None,
                "skipped_missing_columns": r.skipped_missing_columns,
            }
            for r in reports
        ],
        "enabled_steps": sum(1 for r in reports if r.enabled),
        "applied_steps": sum(1 for r in reports if r.applied),
        "skipped_missing_columns": any(r.skipped_missing_columns for r in reports),
    }
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    logger.info(f"Wrote filtering report: {out}")
