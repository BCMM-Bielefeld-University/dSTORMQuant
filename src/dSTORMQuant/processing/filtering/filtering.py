from __future__ import annotations

import json

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from dSTORMQuant.utils.logger import get_logger

logger = get_logger()

# Internal post-drift field -> exact CSV header(s) users must provide.
_CSV_HEADERS_FOR_INTERNAL: dict[str, tuple[str, ...]] = {
    "sx": ("sigmaX (nm)",),
    "sy": ("sigmaY (nm)",),
    "photons": ("intensity (photons)",),
    "lp": ("localization precision (nm)",),
    "pvalue": ("p-value", "pvalue"),
}


def _expected_csv_headers(internal_columns: tuple[str, ...]) -> tuple[str, ...]:
    """Map internal filter columns to documented input CSV header names."""
    headers: list[str] = []
    for col in internal_columns:
        headers.extend(_CSV_HEADERS_FOR_INTERNAL.get(col, (col,)))
    return tuple(headers)


def _warn_filter_skipped_for_column_names(
    step_label: str,
    expected_csv_headers: tuple[str, ...],
) -> None:
    """Log why a filter was skipped: CSV headers must match the documented schema."""
    named = ", ".join(f"'{h}'" for h in expected_csv_headers)
    logger.warning(
        "%s skipped: required quality column(s) not found after drift correction. "
        "Reason: input CSV header names must match the documented dSTORMQuant schema "
        "exactly (expected: %s). Rename your export columns to these names and re-run. "
        "See filtering_report.json in test_data/.",
        step_label,
        named,
    )


@dataclass(frozen=True)
class FilterStepReport:
    step_name: str
    enabled: bool
    applied: bool
    skipped_missing_columns: bool
    expected_csv_columns: tuple[str, ...] = ()


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
        enabled: bool,
        required: tuple[str, ...],
        *,
        did_apply: bool,
        skipped_missing_columns: bool = False,
    ) -> None:
        applied[step_key] = did_apply
        step_name = step_key.removeprefix("after_")
        expected_csv = _expected_csv_headers(required)
        reports.append(
            FilterStepReport(
                step_name=step_name,
                enabled=enabled,
                applied=did_apply,
                skipped_missing_columns=skipped_missing_columns,
                expected_csv_columns=expected_csv,
            )
        )
        if skipped_missing_columns:
            _warn_filter_skipped_for_column_names(step_name, expected_csv)

    # --- Sigma ---
    sigma_cfg = filtering["sigma"]
    sigma_enabled = _filter_step_enabled(sigma_cfg)
    if not sigma_enabled:
        logger.info("Sigma filter disabled in config (use: false); skipping.")
        _finish_step(
            "after_sigma_filter",
            False,
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
            True,
            ("sx", "sy"),
            did_apply=True,
        )
    else:
        _finish_step(
            "after_sigma_filter",
            True,
            ("sx", "sy"),
            did_apply=False,
            skipped_missing_columns=True,
        )

    results["after_sigma_filter"] = current.copy()

    # --- Intensity ---
    int_cfg = filtering["intensity"]
    int_enabled = _filter_step_enabled(int_cfg)
    if not int_enabled:
        logger.info("Intensity filter disabled in config (use: false); skipping.")
        _finish_step(
            "after_photons_count_filter",
            False,
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
            True,
            ("photons",),
            did_apply=True,
        )
    else:
        _finish_step(
            "after_photons_count_filter",
            True,
            ("photons",),
            did_apply=False,
            skipped_missing_columns=True,
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
            False,
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
            True,
            ("lp",),
            did_apply=True,
        )
    else:
        _finish_step(
            "after_localization_precision_filter",
            True,
            ("lp",),
            did_apply=False,
            skipped_missing_columns=True,
        )

    results["after_localization_precision_filter"] = current.copy()

    # --- P-value ---
    pv_cfg = filtering["p_value"]
    pv_enabled = _filter_step_enabled(pv_cfg)
    if not pv_enabled:
        logger.info("P-value filter disabled in config (use: false); skipping.")
        _finish_step(
            "after_pvalue_filter",
            False,
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
            True,
            ("pvalue",),
            did_apply=True,
        )
    else:
        _finish_step(
            "after_pvalue_filter",
            True,
            ("pvalue",),
            did_apply=False,
            skipped_missing_columns=True,
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
        elif r.skipped_missing_columns:
            expected = ", ".join(f"'{h}'" for h in r.expected_csv_columns)
            status = (
                "NOT APPLIED — CSV headers must match documented names "
                f"(expected: {expected})"
            )
        else:
            status = "not applied"
        lines.append(f"  - {r.step_name}: {status}")

    any_missing_column_skips = any(r.skipped_missing_columns for r in reports)
    msg = "\n".join(lines)
    if any_missing_column_skips:
        logger.warning(
            "%s\n"
            "Reason: one or more enabled filters were skipped because the input "
            "CSV did not use the documented quality column names "
            "(e.g. 'sigmaX (nm)', 'sigmaY (nm)', 'intensity (photons)', "
            "'localization precision (nm)', 'p-value'). Rename columns to match "
            "exactly, then re-run. Details in filtering_report.json under test_data/.",
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
        "steps": [asdict(r) for r in reports],
        "number_of_enabled_steps": sum(1 for r in reports if r.enabled),
        "number_of_applied_steps": sum(1 for r in reports if r.applied),
        "skipped_missing_columns": any(r.skipped_missing_columns for r in reports),
    }
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    logger.info(f"Wrote filtering report: {out}")
