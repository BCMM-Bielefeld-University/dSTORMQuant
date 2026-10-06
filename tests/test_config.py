"""Tests for configuration loading and validation."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from pydantic import ValidationError

from dSTORMQuant.core.config.loader import get_config_params, load_config
from dSTORMQuant.core.config.models import SMLMConfig


def test_load_default_config_yaml(default_config_path: Path) -> None:
    """Default shipped config.yaml must load and validate."""
    config = load_config(default_config_path)

    assert "data" in config
    assert "filtering" in config
    assert "clustering" in config
    assert config["data"]["input"]["file_format"] == "csv"
    assert config["drift_correction"]["pixel_size"] == 117


def test_load_config_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist.yaml"
    with pytest.raises(FileNotFoundError):
        load_config(missing)


def test_smlm_config_rejects_invalid_channel_color(invalid_config_dict: dict) -> None:
    with pytest.raises(ValidationError):
        SMLMConfig(**invalid_config_dict)


def test_smlm_config_rejects_incomplete_metadata_keys(default_config_path: Path) -> None:
    with open(default_config_path) as f:
        data = yaml.safe_load(f)
    data["data"]["input"]["required_metadata_columns"].pop("file_name")

    with pytest.raises(ValidationError):
        SMLMConfig(**data)


def test_get_config_params_extracts_section(default_config_path: Path) -> None:
    config = load_config(default_config_path)
    drift = get_config_params(config, "drift_correction")

    assert drift["pixel_size"] == 117
    assert "sanity_checks" in drift
