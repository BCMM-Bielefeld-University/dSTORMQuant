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


class AxialDataNotSupportedError(ValueError):
    """Raised when the CSV contains non-zero axial coordinates (true 3D data)."""


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


def axial_column_has_numeric_values(series: pd.Series) -> bool:
    """Return True if the series has any finite non-zero values (true 3D data).

    Missing values and zeros are treated as a 2D placeholder (common in NimOS
    and other exporters that always write a ``z`` column filled with 0).
    """
    numeric = pd.to_numeric(series, errors="coerce")
    # NaN / missing → 0 so empty and all-zero columns count as 2D placeholders.
    return bool((numeric.fillna(0) != 0).any())


def warn_if_axial_coordinates_present(
    columns: pd.Index | list[str],
    *,
    context: str = "input data",
) -> list[str]:
    """Warn when axial coordinate columns are present (names only; no value check).

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
            f"Axial coordinate column(s) {axial_cols} were found in {context}."
        )
    return axial_cols


def finalize_loaded_dataframe(
    df: pd.DataFrame, *, context: str
) -> pd.DataFrame:
    """Handle axial / unused columns for the 2D release.

    - Axial column with **non-zero** numeric values → raise
      ``AxialDataNotSupportedError`` (3D data is not supported).
    - Axial column **empty / all missing / all zero** → warn, drop, continue
      (zeros are treated as a 2D export placeholder).
    - ``channelName`` is dropped when present.
    """
    axial_cols = find_axial_columns(df.columns)
    if axial_cols:
        cols_with_values = [
            c for c in axial_cols if axial_column_has_numeric_values(df[c])
        ]
        if cols_with_values:
            raise AxialDataNotSupportedError(
                "dSTORMQuant currently supports 2D analysis only. "
                f"Axial column(s) {cols_with_values} in {context} contain "
                "non-zero numeric values (3D localizations). Remove the z column "
                "or clear / zero its values for a 2D run. Full 3D support is "
                "planned for a future release."
            )
        logger.warning(
            "dSTORMQuant currently supports 2D analysis only. "
            f"Axial column(s) {axial_cols} in {context} are empty or all-zero "
            "(2D placeholder) and will be dropped."
        )
        df = df.drop(columns=axial_cols)

    if "channelName" in df.columns:
        df = df.drop(columns=["channelName"])

    return df


def load_data(file_path: str | Path) -> pd.DataFrame:
    """
    Load localization data from a CSV file and return a pandas DataFrame.

    The current release analyzes data in 2D. Axial columns with non-zero values
    raise ``AxialDataNotSupportedError``. Empty or all-zero axial columns are
    dropped (common 2D export placeholder).

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
    return finalize_loaded_dataframe(df, context=f"'{file_path.name}'")


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
