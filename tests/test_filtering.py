"""Tests for localization filtering."""

from __future__ import annotations

import pandas as pd

from dSTORMQuant.processing.filtering.filtering import apply_filters


def test_apply_filters_removes_out_of_range_rows(
    synthetic_localizations: pd.DataFrame, filtering_config: dict
) -> None:
    results, applied = apply_filters(synthetic_localizations, filtering_config)

    assert applied["after_sigma_filter"] is True
    assert applied["after_photons_count_filter"] is True
    assert applied["after_localization_precision_filter"] is True
    assert applied["after_pvalue_filter"] is True

    final = results["after_pvalue_filter"]
    # Rows 0 and 1 pass all thresholds; rows 2 and 3 fail at least one stage.
    assert len(final) == 2
    assert set(final.index.tolist()) == {0, 1}


def test_apply_filters_skips_missing_measurement_columns(
    filtering_config: dict,
) -> None:
    df = pd.DataFrame(
        {
            "x": [1.0, 2.0],
            "y": [3.0, 4.0],
            "frame": [1, 2],
        }
    )
    results, applied = apply_filters(df, filtering_config)

    assert applied["after_sigma_filter"] is False
    assert applied["after_photons_count_filter"] is False
    assert applied["after_localization_precision_filter"] is False
    assert applied["after_pvalue_filter"] is False
    assert len(results["after_pvalue_filter"]) == 2


def test_apply_filters_skips_disabled_steps(
    synthetic_localizations: pd.DataFrame,
) -> None:
    config = {
        "filtering": {
            "sigma": {"use": False, "min_value": 50.0, "max_value": 250.0},
            "intensity": {"min_value": 300.0},
            "localization_precision": {"threshold_value": 20.0},
            "p_value": {"use": False, "threshold_value": 0.1},
        }
    }
    results, applied = apply_filters(synthetic_localizations, config)

    assert applied["after_sigma_filter"] is False
    assert applied["after_pvalue_filter"] is False
    assert applied["after_photons_count_filter"] is True
    assert applied["after_localization_precision_filter"] is True
    # With sigma and p-value disabled, rows 0–2 pass intensity + lp; row 3 fails lp.
    assert len(results["after_pvalue_filter"]) == 3
