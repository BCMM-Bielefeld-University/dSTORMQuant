"""Tests for localization CSV loading and axial-coordinate handling."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from dSTORMQuant.utils.data_handling import (
    find_axial_columns,
    load_data,
    warn_if_axial_coordinates_present,
)


def test_find_axial_columns_detects_common_aliases() -> None:
    cols = ["x (nm)", "y (nm)", "z (nm)", "frameIndex"]
    assert find_axial_columns(cols) == ["z (nm)"]
    assert find_axial_columns(["X", "Y", "Z"]) == ["Z"]


def test_warn_if_axial_coordinates_present_logs_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level("WARNING"):
        found = warn_if_axial_coordinates_present(
            ["x (nm)", "y (nm)", "z (nm)"], context="unit test"
        )
    assert found == ["z (nm)"]
    assert "2D analysis only" in caplog.text
    assert "z (nm)" in caplog.text


def test_load_data_warns_and_drops_z_column(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    csv_path = tmp_path / "locs.csv"
    pd.DataFrame(
        {
            "x (nm)": [1.0, 2.0],
            "y (nm)": [3.0, 4.0],
            "z (nm)": [10.0, 20.0],
            "channelIndex": [0, 0],
            "frameIndex": [1, 2],
            "channelName": ["Ch1", "Ch1"],
        }
    ).to_csv(csv_path, index=False)

    with caplog.at_level("WARNING"):
        df = load_data(csv_path)

    assert "z (nm)" not in df.columns
    assert "channelName" not in df.columns
    assert list(df.columns) == ["x (nm)", "y (nm)", "channelIndex", "frameIndex"]
    assert "2D analysis only" in caplog.text


def test_load_data_no_warning_without_z(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    csv_path = tmp_path / "locs_2d.csv"
    pd.DataFrame(
        {
            "x (nm)": [1.0],
            "y (nm)": [2.0],
            "channelIndex": [0],
            "frameIndex": [1],
        }
    ).to_csv(csv_path, index=False)

    with caplog.at_level("WARNING"):
        df = load_data(csv_path)

    assert "z (nm)" not in df.columns
    assert "2D analysis only" not in caplog.text
    assert len(df) == 1
