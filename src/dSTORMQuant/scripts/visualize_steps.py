from __future__ import annotations

import os
import sys
import zipfile
import pandas as pd

from pathlib import Path
from typing import Any, Dict, Optional, Literal, Union

from dSTORMQuant.utils.data_handling import load_data
from dSTORMQuant.utils.logger import setup_logger
from dSTORMQuant.visualization.visualization import visualize_napari
from dSTORMQuant.core.config.loader import load_config, load_metadata, get_metadata_for_file

def get_channel_data(
    data_dir: Optional[Union[str, os.PathLike]] = None,
    config: Optional[Union[str, os.PathLike]] = None
) -> tuple[Dict[str, Any], pd.DataFrame, zipfile.Path, Dict[str, tuple]]:
    """
    Initialize project configuration, load metadata, and build CHANNELS dictionary.
    
    Parameters:
    ----------
    data_dir : str or Path, optional
        Custom data directory path. If None, defaults to PROJECT_ROOT / "data"
    config : str or Path, optional
        Custom configuration file path. If None, defaults to PROJECT_ROOT / "config.yaml"
        
    Returns:
    -------
    tuple
        A tuple containing:
        - channels : dict
            Dictionary mapping channel names to (color, label, index) tuples
            
    Raises:
    ------
    SystemExit
        If any critical error occurs during initialization
    """
    
    logger = setup_logger()

    input_filename = config['post_analysis']['input_filename']
    
    # Load metadata
    xlsx_file: Path = data_dir / "metadata" / config['data']['input'].get('xlsx_filename')
    metadata_df = load_metadata(xlsx_file)
    metadata_row = get_metadata_for_file(metadata_df, input_filename)
    is_single_channel: bool = pd.isna(metadata_row.get('2nd channel')) or metadata_row.get('2nd channel') == 'None'
    
    # Extract channel labels from filename
    input_filename = Path(input_filename)
    filename_parts = input_filename.stem.split('_')
    if len(filename_parts) < 5:
        logger.error(f"Filename '{input_filename}' does not contain enough underscores to extract channel labels.")
        sys.exit(1)
    
    ch1_label, ch2_label = filename_parts[3], filename_parts[4]
    
    # Build channels dictionary
    channels: Dict[str, tuple] = {
        "Ch1": (
            config['channels']['Ch1']['color'],
            ch1_label,
            int(metadata_row['1st channel index'])
        )
    }
    
    if not is_single_channel:
        channels["Ch2"] = (
            config['channels']['Ch2']['color'],
            ch2_label,
            int(metadata_row['2nd channel index'])
        )

    return channels


StepName = Literal["Initial", "Drift Correction", "Filtering", "Temporal Grouping", "Cell Detection"]

logger = setup_logger()

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
OUTPUT_BASE_DIR: Path = DATA_DIR / "output"
TEST_DATA_DIR: Path = OUTPUT_BASE_DIR / "test_data"
CONFIG_FILE: Path = PROJECT_ROOT / "config" / "config.yaml"
SAVE_DIR: Path = OUTPUT_BASE_DIR / "test_images"

# Load configuration
config = load_config(CONFIG_FILE)
input_filename = config.get('input_filename')
CHANNELS = get_channel_data(DATA_DIR, config)

STEP_SUFFIXES: Dict[str, str] = {
    "Drift Correction": "after_drift_correction.csv",
    "Filtering": "after_filtering.csv",
    "Temporal Grouping": "after_temporal_grouping.csv",
    "Cell Detection": "after_cell_detection.csv",
}

def list_available_steps() -> Dict[StepName, Path]:
    """Scan ``TEST_DATA_DIR`` and return available pipeline step CSV paths.

    Returns:
        Dict mapping step label (e.g. ``"Initial"``) to the CSV path for that step.
    """
    available: Dict[StepName, Path] = {"Initial": f"{input_filename}.csv"}

    if not TEST_DATA_DIR.exists():
        logger.warning(f"Test data directory '{TEST_DATA_DIR}' not found.")
        return available

    files: Dict[str, Path] = {f.name: f for f in TEST_DATA_DIR.iterdir() if f.is_file()}

    for step, suffix in STEP_SUFFIXES.items():
        matched_file = next((p for name, p in files.items() if name.endswith(suffix)), None)
        if matched_file:
            available[step] = matched_file
        else:
            logger.debug(f"No file found for step '{step}' with suffix '{suffix}'.")

    return available


def prompt_choice() -> Optional[int]:
    """Read a numeric menu choice from stdin.

    Returns:
        Parsed integer choice, or ``None`` if input is not a valid number.
    """
    try:
        return int(input("\nSelect a step to visualize: "))
    except ValueError:
        logger.error("❌ Invalid input. Please enter a number.")
        return None


def menu() -> None:
    """Interactive terminal menu to open Napari views for pipeline step CSVs.

    Returns:
        None. Exits when the user selects ``0``.
    """
    available_steps = list_available_steps()
    if not available_steps:
        logger.error("No step files found in test_data_dir.")
        sys.exit(1)

    steps_list = list(available_steps.keys())

    while True:
        print("\n=== Visualization Menu ===")
        for i, step in enumerate(steps_list, start=1):
            print(f"{i}. {step}")
        print("0. Exit")

        choice = prompt_choice()
        if choice is None:
            continue
        if choice == 0:
            logger.info("Exiting.")
            break
        if 1 <= choice <= len(steps_list):
            selected_step = steps_list[choice - 1]
            file_path = available_steps[selected_step]
            if selected_step == "Initial":
                file_path = DATA_DIR / "input" / f"{input_filename}.csv"

            logger.info(f"Loading {selected_step} data from {file_path}")
            df = load_data(file_path)

            step_save_dir = SAVE_DIR / selected_step.replace(" ", "_")
            step_save_dir.mkdir(parents=True, exist_ok=True)

            visualize_napari(df, channels=CHANNELS)
        else:
            logger.error("❌ Invalid choice.")


if __name__ == "__main__":
    menu()
