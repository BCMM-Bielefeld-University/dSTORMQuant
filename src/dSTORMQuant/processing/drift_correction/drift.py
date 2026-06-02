import os
from pathlib import Path
from typing import Any

import pandas as pd
from numpy import savetxt

from dSTORMQuant.utils.logger import get_logger
from dSTORMQuant.visualization.visualization import plot_drift_data

from .AIM import aim, csv2hdf, hdf2csv, load_locs, save_locs

logger = get_logger()


def apply_aim_drift(
    input_path: str,
    input_file_name: str,
    output_dir: Path,
    pixel_size: float,
    segmentation: int,
    intersect_d: float,
    roi_r: float,
    sanity_config: dict[str, Any] | None = None,
    drift_validation_config: dict[str, Any] | None = None,
) -> tuple[str, Any]:
    """
    Apply AIM drift correction to a localization dataset.

    Args:
        input_path (str): Path to the input CSV file
        input_file_name (str): Input file name
        output_dir (str): Directory to save corrected data
        pixel_size (float): Pixel size in nm
        segmentation (int): Temporal interval for drift tracking
        intersect_d (float): Intersect distance in pixels
        roi_r (float): Search region radius in pixels
        sanity_config (dict, optional): Configuration for sanity checks
        drift_validation_config (dict, optional): Configuration for drift validation

    Returns:
        tuple: (Path to the corrected CSV file, drift array)
    """
    try:
        logger.info(f"Starting AIM drift correction for file: {input_file_name}")
        hdf5_path: str = csv2hdf(input_path, pixel_size, sanity_config=sanity_config)

        locs, info = load_locs(hdf5_path)
        logger.info(f"Loaded localization data from: {hdf5_path}")

        locs, new_info, drift, drift_tables = aim(
            locs,
            info,
            segmentation=segmentation,
            intersect_d=intersect_d,
            roi_r=roi_r,
            drift_validation_config=drift_validation_config,
        )

        base, _ = os.path.splitext(input_file_name)
        drift_corrected_dir: str = os.path.join(output_dir, "drift_corrected")
        os.makedirs(drift_corrected_dir, exist_ok=True)

        drift_corrected_hdf5_path: str = os.path.join(
            drift_corrected_dir, f"{base}_after_drift_correction.hdf5"
        )
        drift_txt_path: str = os.path.join(drift_corrected_dir, f"{base}_aimdrift.txt")

        # Save segment drift table
        drift_segment_table_path: str = os.path.join(
            drift_corrected_dir, f"{base}_drift_segments.csv"
        )

        # Force removal of invalid values regardless of config file presence
        if sanity_config is None:
            sanity_config = {}
        sanity_config["remove_invalid_values"] = True

        save_locs(
            drift_corrected_hdf5_path, locs, [new_info], sanity_config=sanity_config
        )
        savetxt(drift_txt_path, drift, header="dx\tdy", newline="\r\n")

        logger.info(
            f"Saved drift corrected localization data to: {drift_corrected_hdf5_path}"
        )
        logger.info(f"Saved drift data text file to: {drift_txt_path}")

        # Save drift segment table with both cumulative and relative drift
        drift_df = pd.DataFrame(
            {
                "segment": range(len(drift_tables["segment_centers"])),
                "segment_center_frame": drift_tables["segment_centers"],
                "cumulative_drift_x_nm": drift_tables["segment_drift_x_nm"],
                "cumulative_drift_y_nm": drift_tables["segment_drift_y_nm"],
                "cumulative_drift_magnitude_nm": drift_tables[
                    "segment_drift_magnitude_nm"
                ],
                "relative_drift_x_nm": drift_tables["relative_drift_x_nm"],
                "relative_drift_y_nm": drift_tables["relative_drift_y_nm"],
                "relative_drift_magnitude_nm": drift_tables[
                    "relative_drift_magnitude_nm"
                ],
            }
        )
        drift_df.to_csv(drift_segment_table_path, index=False)
        logger.info(f"Saved drift segment table to: {drift_segment_table_path}")

        if drift_tables["n_replaced"] > 0:
            logger.info(
                f"📊 Drift validation: {drift_tables['n_replaced']} segments were replaced"
            )

        plot_drift_data(drift_txt_path, output_dir, base, pixel_size)

        out_path: str = hdf2csv(drift_corrected_hdf5_path, new_info)
        os.remove(drift_corrected_hdf5_path)

        return out_path, drift

    except Exception as e:
        logger.error(f"Error during AIM drift correction: {e}", exc_info=True)
        raise
