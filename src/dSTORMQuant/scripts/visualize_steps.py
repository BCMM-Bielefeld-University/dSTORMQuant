"""Interactive Napari viewer for pipeline-stage CSVs in existing experiment outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from dSTORMQuant.core.config.loader import (
    get_metadata_for_file,
    load_config,
    load_metadata,
)
from dSTORMQuant.scripts.output_discovery import (
    ensure_output_folders_from_archives,
    list_output_folders,
)
from dSTORMQuant.utils.data_handling import load_data
from dSTORMQuant.utils.logger import setup_logger
from dSTORMQuant.utils.utils import (
    is_dual_channel_metadata_row,
    resolve_channel_labels,
)
from dSTORMQuant.visualization.visualization import visualize_napari

logger = setup_logger()

# src/dSTORMQuant/scripts/this_file.py -> parents[3] = project root
PROJECT_ROOT: Path = Path(__file__).resolve().parents[3]
DATA_DIR: Path = PROJECT_ROOT / "data"
OUTPUT_BASE_DIR: Path = DATA_DIR / "output"
CONFIG_FILE: Path = PROJECT_ROOT / "config" / "config.yaml"

StepName = Literal[
    "Initial",
    "Drift Correction",
    "Filtering",
    "Temporal Grouping",
    "Cell Detection",
]

STEP_SUFFIXES: dict[StepName, str] = {
    "Initial": "_initial.csv",
    "Drift Correction": "_after_drift_correction.csv",
    "Filtering": "_after_filtering.csv",
    "Temporal Grouping": "_after_temporal_grouping.csv",
    "Cell Detection": "_after_cell_detection.csv",
}


def get_channel_data(
    data_dir: Path, config: dict[str, Any], input_filename: str
) -> dict[str, tuple]:
    """Resolve channel colors, labels, and indices from experiment metadata."""
    xlsx_file = data_dir / "metadata" / config["data"]["input"].get("xlsx_filename")
    meta_cols = config["data"]["input"]["required_metadata_columns"]
    metadata_df = load_metadata(xlsx_file, required_columns=meta_cols)
    base_name = str(input_filename).strip()
    metadata_row = get_metadata_for_file(
        metadata_df, base_name, required_metadata_columns=meta_cols
    )
    is_single_channel = not is_dual_channel_metadata_row(metadata_row, meta_cols)
    omc = config["data"]["input"].get("optional_metadata_columns") or {}
    ch1_label, ch2_label = resolve_channel_labels(base_name, metadata_row, omc)

    channels: dict[str, tuple] = {
        "Ch1": (
            config["channels"]["Ch1"]["color"],
            ch1_label,
            int(metadata_row[meta_cols["first_channel_index"]]),
        )
    }
    if not is_single_channel:
        channels["Ch2"] = (
            config["channels"]["Ch2"]["color"],
            ch2_label,
            int(metadata_row[meta_cols["second_channel_index"]]),
        )
    return channels


def list_available_steps(test_data_dir: Path) -> dict[StepName, Path]:
    """Return step label → CSV path for files present under ``test_data``."""
    available: dict[StepName, Path] = {}
    if not test_data_dir.is_dir():
        return available

    files = {f.name: f for f in test_data_dir.iterdir() if f.is_file()}
    for step, suffix in STEP_SUFFIXES.items():
        matched = next((p for name, p in files.items() if name.endswith(suffix)), None)
        if matched is not None:
            available[step] = matched
    return available


def _prompt_int(prompt: str) -> int | None:
    try:
        return int(input(prompt))
    except ValueError:
        logger.error("Invalid input. Please enter a number.")
        return None


def _pick_experiment(output_dirs: list[Path]) -> Path | None:
    print("\n=== Experiments ===")
    for i, path in enumerate(output_dirs, start=1):
        print(f"{i}. {path.name}")
    print("0. Exit")
    choice = _prompt_int("\nSelect an experiment: ")
    if choice is None:
        return None
    if choice == 0:
        return None
    if 1 <= choice <= len(output_dirs):
        return output_dirs[choice - 1]
    logger.error("Invalid choice.")
    return None


def menu() -> None:
    """Interactive terminal menu to open Napari views for pipeline step CSVs."""
    if not CONFIG_FILE.is_file():
        logger.error(f"Configuration file '{CONFIG_FILE}' not found.")
        raise SystemExit(1)

    config = load_config(CONFIG_FILE)
    ensure_output_folders_from_archives(OUTPUT_BASE_DIR, logger=logger)
    output_dirs = list_output_folders(OUTPUT_BASE_DIR, order="newest_first")
    if not output_dirs:
        logger.error(
            f"No experiment outputs with test_data found under '{OUTPUT_BASE_DIR}'. "
            "Run the pipeline first (or place extracted ZIP folders there)."
        )
        raise SystemExit(1)

    while True:
        experiment_dir = _pick_experiment(output_dirs)
        if experiment_dir is None:
            logger.info("Exiting.")
            break

        test_data_dir = experiment_dir / "test_data"
        available_steps = list_available_steps(test_data_dir)
        if not available_steps:
            logger.error(f"No step CSVs found in '{test_data_dir}'.")
            continue

        try:
            channels = get_channel_data(DATA_DIR, config, experiment_dir.name)
        except Exception as exc:
            logger.error(f"Could not resolve channels for '{experiment_dir.name}': {exc}")
            continue

        steps_list = list(available_steps.keys())
        while True:
            print(f"\n=== Steps ({experiment_dir.name}) ===")
            for i, step in enumerate(steps_list, start=1):
                print(f"{i}. {step}")
            print("0. Back to experiments")

            choice = _prompt_int("\nSelect a step to visualize: ")
            if choice is None:
                continue
            if choice == 0:
                break
            if not (1 <= choice <= len(steps_list)):
                logger.error("Invalid choice.")
                continue

            selected_step = steps_list[choice - 1]
            file_path = available_steps[selected_step]
            logger.info(f"Loading {selected_step} data from {file_path}")
            try:
                df = load_data(file_path)
                visualize_napari(df, channels=channels)
            except Exception as exc:
                logger.error(f"Failed to visualize {selected_step}: {exc}")


if __name__ == "__main__":
    menu()
