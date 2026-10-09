"""Tests for localization CSV loading and axial-coordinate handling."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from dSTORMQuant.core.pipeline import load_and_validate_data
from dSTORMQuant.utils.data_handling import (
    AxialDataNotSupportedError,
    axial_column_has_numeric_values,
    find_axial_columns,
    load_data,
    warn_if_axial_coordinates_present,
)

_REQUIRED = ["x (nm)", "y (nm)", "channelIndex", "frameIndex"]


def test_find_axial_columns_detects_common_aliases() -> None:
    cols = ["x (nm)", "y (nm)", "z (nm)", "frameIndex"]
    assert find_axial_columns(cols) == ["z (nm)"]
    assert find_axial_columns(["X", "Y", "Z"]) == ["Z"]


def test_axial_column_has_numeric_values() -> None:
    assert axial_column_has_numeric_values(pd.Series([1.0, 2.0])) is True
    assert axial_column_has_numeric_values(pd.Series([float("nan"), None])) is False
    assert axial_column_has_numeric_values(pd.Series(["", None, "nan"])) is False
    assert axial_column_has_numeric_values(pd.Series([None, 0.0])) is False
    assert axial_column_has_numeric_values(pd.Series([0.0, 0.0, 0.0])) is False
    assert axial_column_has_numeric_values(pd.Series([0.0, 1.0])) is True


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


def test_load_data_aborts_when_z_has_nonzero_values(tmp_path: Path) -> None:
    csv_path = tmp_path / "locs_3d.csv"
    pd.DataFrame(
        {
            "x (nm)": [1.0, 2.0],
            "y (nm)": [3.0, 4.0],
            "z (nm)": [10.0, 20.0],
            "channelIndex": [0, 0],
            "frameIndex": [1, 2],
        }
    ).to_csv(csv_path, index=False)

    with pytest.raises(AxialDataNotSupportedError, match="non-zero numeric values"):
        load_data(csv_path)


def test_load_data_drops_empty_z_column(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    csv_path = tmp_path / "locs_empty_z.csv"
    pd.DataFrame(
        {
            "x (nm)": [1.0, 2.0],
            "y (nm)": [3.0, 4.0],
            "z (nm)": [float("nan"), float("nan")],
            "channelIndex": [0, 0],
            "frameIndex": [1, 2],
            "channelName": ["Ch1", "Ch1"],
        }
    ).to_csv(csv_path, index=False)

    with caplog.at_level("WARNING"):
        df = load_data(csv_path)

    assert "z (nm)" not in df.columns
    assert "channelName" not in df.columns
    assert "empty or all-zero" in caplog.text


def test_load_data_drops_all_zero_z_column(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    csv_path = tmp_path / "locs_zero_z.csv"
    pd.DataFrame(
        {
            "x (nm)": [1.0, 2.0],
            "y (nm)": [3.0, 4.0],
            "z (nm)": [0.0, 0.0],
            "channelIndex": [0, 0],
            "frameIndex": [1, 2],
        }
    ).to_csv(csv_path, index=False)

    with caplog.at_level("WARNING"):
        df = load_data(csv_path)

    assert "z (nm)" not in df.columns
    assert "empty or all-zero" in caplog.text
    assert len(df) == 2


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


def test_load_and_validate_aborts_when_z_has_nonzero_values(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    csv_path = tmp_path / "with_z.csv"
    pd.DataFrame(
        {
            "x (nm)": [1.0],
            "y (nm)": [2.0],
            "z (nm)": [3.0],
            "channelIndex": [0],
            "frameIndex": [1],
        }
    ).to_csv(csv_path, index=False)

    with caplog.at_level("ERROR"):
        df, reason = load_and_validate_data(str(csv_path), _REQUIRED)

    assert df is None
    assert reason is not None
    assert "non-zero numeric values" in reason
    assert "2D analysis only" in caplog.text


def test_load_and_validate_drops_empty_z(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    csv_path = tmp_path / "empty_z.csv"
    pd.DataFrame(
        {
            "x (nm)": [1.0],
            "y (nm)": [2.0],
            "z (nm)": [float("nan")],
            "channelIndex": [0],
            "frameIndex": [1],
            "localization precision (nm)": [5.0],
        }
    ).to_csv(csv_path, index=False)

    with caplog.at_level("WARNING"):
        df, reason = load_and_validate_data(str(csv_path), _REQUIRED)

    assert reason is None
    assert df is not None
    assert "z (nm)" not in df.columns
    assert "localization precision (nm)" in df.columns
    assert "empty or all-zero" in caplog.text


def test_load_and_validate_drops_all_zero_z(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    csv_path = tmp_path / "zero_z.csv"
    pd.DataFrame(
        {
            "x (nm)": [1.0],
            "y (nm)": [2.0],
            "z (nm)": [0.0],
            "channelIndex": [0],
            "frameIndex": [1],
        }
    ).to_csv(csv_path, index=False)

    with caplog.at_level("WARNING"):
        df, reason = load_and_validate_data(str(csv_path), _REQUIRED)

    assert reason is None
    assert df is not None
    assert "z (nm)" not in df.columns
    assert "empty or all-zero" in caplog.text


def test_load_and_validate_succeeds_without_z(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "no_z.csv"
    pd.DataFrame(
        {
            "x (nm)": [1.0],
            "y (nm)": [2.0],
            "channelIndex": [0],
            "frameIndex": [1],
        }
    ).to_csv(csv_path, index=False)

    df, reason = load_and_validate_data(str(csv_path), _REQUIRED)
    assert reason is None
    assert df is not None
    assert len(df) == 1


def test_load_and_validate_aborts_when_required_column_missing(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    csv_path = tmp_path / "no_x.csv"
    pd.DataFrame(
        {
            "y (nm)": [2.0],
            "channelIndex": [0],
            "frameIndex": [1],
        }
    ).to_csv(csv_path, index=False)

    with caplog.at_level("ERROR"):
        df, reason = load_and_validate_data(str(csv_path), _REQUIRED)

    assert df is None
    assert reason is not None
    assert "Missing required columns" in reason
    assert "x (nm)" in reason
    assert "Missing required columns" in caplog.text
