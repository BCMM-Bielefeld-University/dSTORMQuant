"""Shared fixtures for dSTORMQuant tests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "config.yaml"


@pytest.fixture
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture
def default_config_path() -> Path:
    return DEFAULT_CONFIG_PATH


@pytest.fixture
def filtering_config() -> dict:
    """Minimal filtering section matching apply_filters() expectations."""
    return {
        "filtering": {
            "sigma": {"min_value": 50.0, "max_value": 250.0},
            "intensity": {"min_value": 300.0},
            "localization_precision": {"threshold_value": 20.0},
            "p_value": {"threshold_value": 0.1},
        }
    }


@pytest.fixture
def synthetic_localizations() -> pd.DataFrame:
    """Tiny synthetic localization table for unit tests."""
    return pd.DataFrame(
        {
            "x": [100.0, 105.0, 500.0, 1000.0],
            "y": [200.0, 202.0, 500.0, 1000.0],
            "frame": [1, 2, 10, 20],
            "sx": [80.0, 90.0, 40.0, 300.0],
            "sy": [85.0, 95.0, 45.0, 310.0],
            "photons": [500.0, 400.0, 100.0, 600.0],
            "lp": [10.0, 15.0, 25.0, 5.0],
            "pvalue": [0.01, 0.05, 0.2, 0.02],
        }
    )


@pytest.fixture
def invalid_config_dict(default_config_path: Path) -> dict:
    """Valid YAML structure with one intentionally broken field."""
    with open(default_config_path) as f:
        data = yaml.safe_load(f)
    data["channels"]["Ch1"]["color"] = "not-a-hex"
    return data
