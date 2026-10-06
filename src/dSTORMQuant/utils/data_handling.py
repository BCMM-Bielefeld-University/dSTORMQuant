from __future__ import annotations

from pathlib import Path

import pandas as pd

from dSTORMQuant.utils.logger import get_logger

logger = get_logger()

# Common axial-coordinate headers seen in dSTORMQuant exports.
_AXIAL_COLUMN_CANDIDATES = (
    "z (nm)",
    "z(nm)",
    "z [nm]",
    "z[nm]",
    "z",
    "Z",
    "Z (nm)",
    "position_z",
    "Position Z [nm]",
)


def read_localization_csv(file_path: str | Path) -> pd.DataFrame:
    """Read a localization CSV using pandas' default (C) parser."""
    return pd.read_csv(Path(file_path))


def find_axial_columns(columns: pd.Index | list[str]) -> list[str]:
    """Return input column names that look like axial (z) coordinates.

    Matching is case-insensitive after stripping whitespace.

    Args:
        columns: DataFrame column index or sequence of header names.

    Returns:
        List of original column names that are treated as axial coordinates.
    """
    aliases = {c.strip().lower() for c in _AXIAL_COLUMN_CANDIDATES}
    found: list[str] = []
    for col in columns:
        key = str(col).strip().lower()
        if key in aliases:
            found.append(str(col))
    return found


def warn_if_axial_coordinates_present(
    columns: pd.Index | list[str],
    *,
    context: str = "input data",
) -> list[str]:
    """Warn when axial coordinates are present but will not be used (2D-only pipeline).

    Args:
        columns: Column names to inspect.
        context: Short label included in the warning message.

    Returns:
        Detected axial column names (may be empty).
    """
    axial_cols = find_axial_columns(columns)
    if axial_cols:
        logger.warning(
            "dSTORMQuant currently supports 2D analysis only. "
            f"Axial coordinate column(s) {axial_cols} were found in {context} and "
            "will be ignored; localizations are analyzed as a 2D (x–y) projection. "
            "Full 3D support is planned for a future release."
        )
    return axial_cols


def load_data(file_path: str | Path) -> pd.DataFrame:
    """
    Load localization data from a CSV file and return a pandas DataFrame.

    The current release analyzes data in 2D. If axial coordinates are present
    (e.g. ``z (nm)``), a warning is emitted and those columns are dropped.

    Parameters
    ----------
    file_path : str | Path
        Path to the input CSV file.

    Returns
    -------
    pd.DataFrame
        Loaded localization data.
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    df: pd.DataFrame = read_localization_csv(file_path)

    axial_cols = warn_if_axial_coordinates_present(
        df.columns, context=f"'{file_path.name}'"
    )
    if axial_cols:
        df = df.drop(columns=axial_cols)

    # Optional non-coordinate column sometimes present in NimOS exports
    if "channelName" in df.columns:
        df = df.drop(columns=["channelName"])

    return df


def save_df_to_csv(
    df: pd.DataFrame, filename: str | Path, index: bool = False, *args, **kwargs
) -> None:
    """
    Save a pandas DataFrame to a CSV file, supporting additional positional and keyword arguments.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to save.
    filename : str | Path
        Path to save the CSV file (e.g., "output.csv").
    index : bool, optional
        Whether to write row names (index). Default is False.
    *args
        Additional positional arguments for `pd.DataFrame.to_csv`.
    **kwargs
        Additional keyword arguments for `pd.DataFrame.to_csv`.
    """
    try:
        df.to_csv(filename, index=index, *args, **kwargs)
        logger.info(f"DataFrame saved successfully to {filename}")
    except Exception as e:
        logger.error(f"Failed to save DataFrame: {e}")
