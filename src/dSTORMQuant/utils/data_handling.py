from __future__ import annotations

from pathlib import Path

import pandas as pd

from dSTORMQuant.utils.logger import get_logger

logger = get_logger()


def read_localization_csv(file_path: str | Path) -> pd.DataFrame:
    """Read a localization CSV using pandas' default (C) parser."""
    return pd.read_csv(Path(file_path))


def load_data(file_path: str | Path) -> pd.DataFrame:
    """
    Load localization data from a CSV file and return a pandas DataFrame.

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

    # Remove unwanted columns if they exist
    for col in ["z (nm)", "channelName"]:
        if col in df.columns:
            df.drop(col, axis=1, inplace=True)

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
