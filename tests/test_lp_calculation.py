"""Tests for Mortensen LP recalculation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from dSTORMQuant.core.config.loader import load_config
from dSTORMQuant.processing.lp_calculation import (
    apply_lp_calculation,
    compute_localization_precision,
    var_mle,
)


def test_var_mle_zero_background_limit() -> None:
    """When b^2 -> 0, variance -> s_a^2 / N."""
    sigma = np.array([100.0])
    a = np.array([117.0])
    n = np.array([1000.0])
    b2 = np.array([0.0])
    sa2 = sigma**2 + a**2 / 12.0
    np.testing.assert_allclose(var_mle(sigma, a, n, b2), sa2 / n, rtol=1e-10)


def test_apply_lp_calculation_disabled_is_noop() -> None:
    df = pd.DataFrame(
        {
            "intensity (photons)": [1000.0],
            "background (photons/nm^2)": [0.001],
            "sigmaX (nm)": [120.0],
            "sigmaY (nm)": [110.0],
        }
    )
    config = {
        "lp_calculation": {"use": False},
        "drift_correction": {"pixel_size": 117},
    }
    out = apply_lp_calculation(df, config)
    assert "LP_calculated" not in out.columns
    assert out is df or out.equals(df)


def test_apply_lp_calculation_requires_exact_config_headers() -> None:
    """Internal AIM names alone are not enough; config headers must be present."""
    df = pd.DataFrame(
        {
            "photons": [1000.0],
            "bg": [0.001],
            "sx": [120.0],
            "sy": [110.0],
        }
    )
    config = {
        "lp_calculation": {
            "use": True,
            "method": "mle",
            "emccd": False,
            "bg_unit": "per_nm2",
            "sigma_mode": "mean",
            "output_column": "LP_calculated",
            "columns": {
                "intensity": "intensity (photons)",
                "background": "background (photons/nm^2)",
                "sigma": None,
                "sigma_x": "sigmaX (nm)",
                "sigma_y": "sigmaY (nm)",
            },
        },
        "drift_correction": {"pixel_size": 117},
    }
    with pytest.raises(ValueError, match="intensity"):
        apply_lp_calculation(df, config, stage="after_drift_correction")


def test_apply_lp_calculation_writes_column_keeps_original_lp() -> None:
    df = pd.DataFrame(
        {
            "intensity (photons)": [1000.0, 2000.0],
            "background (photons/nm^2)": [0.001, 0.002],
            "sigmaX (nm)": [120.0, 130.0],
            "sigmaY (nm)": [110.0, 125.0],
            "localization precision (nm)": [8.0, 7.0],
        }
    )
    config = {
        "lp_calculation": {
            "use": True,
            "method": "mle",
            "emccd": False,
            "bg_unit": "per_nm2",
            "sigma_mode": "mean",
            "output_column": "LP_calculated",
            "columns": {
                "intensity": "intensity (photons)",
                "background": "background (photons/nm^2)",
                "sigma": None,
                "sigma_x": "sigmaX (nm)",
                "sigma_y": "sigmaY (nm)",
            },
        },
        "drift_correction": {"pixel_size": 117},
    }
    out = apply_lp_calculation(df, config)
    assert "LP_calculated" in out.columns
    assert out["localization precision (nm)"].tolist() == [8.0, 7.0]
    assert np.all(np.isfinite(out["LP_calculated"]))
    assert np.all(out["LP_calculated"] > 0)


def _lp_config(**column_overrides: object) -> dict:
    columns = {
        "intensity": "intensity (photons)",
        "background": "background (photons/nm^2)",
        "sigma": "sigma [nm]",
        "sigma_x": "sigmaX (nm)",
        "sigma_y": "sigmaY (nm)",
    }
    columns.update(column_overrides)
    return {
        "lp_calculation": {
            "use": True,
            "method": "lsq",
            "emccd": False,
            "bg_unit": "per_nm2",
            "sigma_mode": "mean",
            "output_column": "LP_calculated",
            "columns": columns,
        },
        "drift_correction": {"pixel_size": 117},
    }


def test_apply_lp_calculation_single_sigma_column() -> None:
    df = pd.DataFrame(
        {
            "intensity (photons)": [1500.0],
            "background (photons/nm^2)": [0.001],
            "sigma [nm]": [115.0],
        }
    )
    out = apply_lp_calculation(df, _lp_config())
    assert np.isfinite(out["LP_calculated"].iloc[0])


def test_apply_lp_calculation_prefers_sigma_xy_when_both_exist() -> None:
    """If sigmaX/sigmaY and sigma are all present, use X/Y (mean mode)."""
    df = pd.DataFrame(
        {
            "intensity (photons)": [1500.0],
            "background (photons/nm^2)": [0.001],
            "sigma [nm]": [200.0],  # deliberately different
            "sigmaX (nm)": [100.0],
            "sigmaY (nm)": [100.0],
        }
    )
    from dSTORMQuant.processing.lp_calculation.lp_calc import resolve_sigma

    sigma = resolve_sigma(
        df,
        sigma_col="sigma [nm]",
        sigma_x_col="sigmaX (nm)",
        sigma_y_col="sigmaY (nm)",
        sigma_mode="mean",
    )
    assert float(sigma.iloc[0]) == 100.0


def test_apply_lp_calculation_falls_back_to_sigma_when_xy_missing() -> None:
    df = pd.DataFrame(
        {
            "intensity (photons)": [1500.0],
            "background (photons/nm^2)": [0.001],
            "sigma [nm]": [115.0],
            # sigmaX / sigmaY absent
        }
    )
    out = apply_lp_calculation(df, _lp_config())
    assert np.isfinite(out["LP_calculated"].iloc[0])


def test_apply_lp_calculation_missing_column_raises() -> None:
    df = pd.DataFrame({"x (nm)": [1.0]})
    config = {
        "lp_calculation": {
            "use": True,
            "columns": {
                "intensity": "intensity (photons)",
                "background": "background (photons/nm^2)",
                "sigma": None,
                "sigma_x": "sigmaX (nm)",
                "sigma_y": "sigmaY (nm)",
            },
        },
        "drift_correction": {"pixel_size": 117},
    }
    with pytest.raises(ValueError, match="intensity"):
        apply_lp_calculation(df, config)


def test_default_config_includes_lp_calculation(default_config_path: Path) -> None:
    config = load_config(default_config_path)
    assert "lp_calculation" in config
    assert "use" in config["lp_calculation"]
    assert config["lp_calculation"]["output_column"] == "LP_calculated"


def test_lp_calculation_omitted_uses_defaults(default_config_path: Path) -> None:
    with open(default_config_path) as f:
        data = yaml.safe_load(f)
    data.pop("lp_calculation", None)
    from dSTORMQuant.core.config.models import SMLMConfig

    cfg = SMLMConfig(**data)
    assert cfg.lp_calculation.use is False
    assert cfg.lp_calculation.output_column == "LP_calculated"


def test_compute_localization_precision_matches_manual() -> None:
    sigma = np.array([100.0])
    a = np.array([117.0])
    n = np.array([1000.0])
    b2 = np.array([10.0])
    prec = compute_localization_precision(sigma, a, n, b2, method="mle", emccd=False)
    expected = np.sqrt(var_mle(sigma, a, n, b2))
    np.testing.assert_allclose(prec, expected)
