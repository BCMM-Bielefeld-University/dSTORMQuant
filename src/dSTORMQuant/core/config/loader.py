from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from dSTORMQuant.utils.logger import get_logger
from dSTORMQuant.utils.utils import normalize_metadata_header

from .models import SMMLConfig

logger = get_logger()

# Global configuration instance
_config_instance: SMMLConfig | None = None


def load_config(config_file: str | Path) -> dict[str, Any]:
    """
    Load and validate configuration from YAML file.

    This function maintains backward compatibility by returning a dictionary
    while using Pydantic for validation.

    Parameters
    ----------
    config_file : str or Path
        Path to the configuration YAML file

    Returns
    -------
    Dict[str, Any]
        Validated configuration as a dictionary

    Raises
    ------
    FileNotFoundError
        If configuration file doesn't exist
    yaml.YAMLError
        If YAML parsing fails
    pydantic.ValidationError
        If configuration validation fails
    """
    global _config_instance

    config_path = Path(config_file)

    try:
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file '{config_file}' not found.")

        # Load YAML
        with open(config_path) as f:
            config_data = yaml.safe_load(f)

        # Validate using Pydantic
        _config_instance = SMMLConfig(**config_data)

        logger.info("✅ Configuration loaded and validated successfully.")

        # Return as dictionary for backward compatibility
        return _config_instance.dict()

    except FileNotFoundError:
        logger.error(f"Configuration file '{config_file}' not found.")
        raise
    except yaml.YAMLError as e:
        logger.error(f"YAML loading error: {e}")
        raise
    except Exception as e:
        logger.error(f"Configuration validation error: {e}")
        raise


def get_config_params(config: dict[str, Any], section: str) -> dict[str, Any]:
    """
    Extract parameters from a configuration section.

    This helper eliminates the repetitive pattern of:
        section_config = config.get('section', {})

    Parameters
    ----------
    config : Dict[str, Any]
        The main configuration dictionary
    section : str
        The configuration section to extract (e.g., 'dbscan', 'drift_correction')

    Returns
    -------
    Dict[str, Any]
        Dictionary with all parameters from the section

    Examples
    --------
    >>> config = {'dbscan': {'eps': 0.1, 'min_samples': 5}}
    >>> params = get_config_params(config, 'dbscan')
    >>> params
    {'eps': 0.1, 'min_samples': 5}
    """
    return config[section]


def load_metadata(
    xlsx_file: str | Path, required_columns: list[str] | dict[str, str] | None = None
) -> pd.DataFrame:
    """
    Load metadata from Excel file.

    Column names in the Excel are normalized (strip, single space, lower case)
    before validation. Pass required_columns in that normalized form.

    Parameters
    ----------
    xlsx_file : str or Path
        Path to the Excel metadata file
    required_columns : list of str or dict of str, optional
        Column names that must be present (after normalization). May be a list
        or the required_metadata_columns dict (its values are used). If None, no validation.

    Returns
    -------
    pd.DataFrame
        Metadata DataFrame with normalized column names

    Raises
    ------
    FileNotFoundError
        If Excel file doesn't exist
    ValueError
        If required_columns is provided and any column is missing
    """
    xlsx_path = Path(xlsx_file)

    try:
        if not xlsx_path.exists():
            raise FileNotFoundError(f"Metadata file '{xlsx_file}' not found.")

        df = pd.read_excel(xlsx_path)
        # Normalize column names (strip, single space, lower case)
        df.columns = (
            df.columns.str.strip().str.replace(r"\s+", " ", regex=True).str.lower()
        )

        if required_columns:
            names = (
                list(required_columns.values())
                if isinstance(required_columns, dict)
                else required_columns
            )
            missing = [c for c in names if c not in df.columns]
            if missing:
                raise ValueError(
                    f"Metadata is missing required columns: {missing}. "
                    f"Available columns: {list(df.columns)}"
                )

        logger.info(f"✅ Loaded metadata from {xlsx_file}")
        return df

    except FileNotFoundError:
        logger.error(f"Metadata file '{xlsx_file}' not found.")
        raise
    except Exception as e:
        logger.error(f"Error loading metadata file: {e}")
        raise


def get_metadata_for_file(
    metadata_df: pd.DataFrame,
    input_file_name: str,
    required_metadata_columns: dict[str, str] | None = None,
) -> pd.Series:
    """
    Get metadata row for a specific input file.

    Parameters
    ----------
    metadata_df : pd.DataFrame
        Metadata DataFrame (with normalized column names)
    input_file_name : str
        Name of the input file to find (without path; extension optional)
    required_metadata_columns : dict of str, optional
        Config mapping of role -> column name (e.g. {'file_name': 'file name', ...}).
        If None, uses 'file name' for the file name column.

    Returns
    -------
    pd.Series
        Metadata row for the specified file

    Raises
    ------
    ValueError
        If no metadata found for the file
    """
    file_name_col = (required_metadata_columns).get("file_name", "file name")
    file_name_key = normalize_metadata_header(file_name_col)
    if file_name_key not in metadata_df.columns:
        raise ValueError(
            f"Metadata has no column '{file_name_col}' (normalized: '{file_name_key}'). "
            f"Available: {list(metadata_df.columns)}"
        )
    row = metadata_df[
        metadata_df[file_name_key].astype(str).str.strip()
        == str(input_file_name).strip()
    ]
    if row.empty:
        raise ValueError(f"No metadata found for file '{input_file_name}'")
    return row.iloc[0]
