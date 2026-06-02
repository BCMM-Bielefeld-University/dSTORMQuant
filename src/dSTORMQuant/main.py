import os
import re
import shutil
import time
from pathlib import Path

from dSTORMQuant.core.config.loader import load_config, load_metadata
from dSTORMQuant.core.pipeline import process_single_file, setup_directories
from dSTORMQuant.utils.logger import get_logger

logger = get_logger()


def main() -> None:
    """Run the dSTORMQuant pipeline on all localization CSVs in the input folder.

    Loads configuration and metadata, iterates input files, calls
    :func:`dSTORMQuant.core.pipeline.process_single_file` for each, then removes
    the temporary working directory.
    """
    logger.info("🚀 Starting dSTORMQuant...")

    # Setup directories
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    dirs = setup_directories(PROJECT_ROOT)

    # Load configuration and metadata
    config = load_config(dirs["config_file"])
    metadata_file = os.path.join(
        dirs["metadata_dir"], config["data"]["input"]["xlsx_filename"]
    )
    required_meta_cols = config["data"]["input"]["required_metadata_columns"]
    metadata_df = load_metadata(metadata_file, required_columns=required_meta_cols)

    # Get input files
    required_columns = config["data"]["input"]["required_columns"]
    input_file_format = config["data"]["input"]["file_format"]
    ch_suffix_re = re.compile(
        r"_ch\d+\." + re.escape(input_file_format) + r"$", re.IGNORECASE
    )
    input_files = [
        f
        for f in os.listdir(dirs["input_dir"])
        if f.endswith(f".{input_file_format}")
        and not ch_suffix_re.search(f)
        and "_locs_" not in f
    ]

    if not input_files:
        logger.info(f"No input CSV files found in '{dirs['input_dir']}'.")
        return

    # Process all files
    total_start_time = time.time()

    for input_file_name in input_files:
        try:
            process_single_file(
                input_file_name,
                dirs["input_dir"],
                dirs["output_dir"],
                dirs["temp_dir"],
                metadata_df,
                config,
                required_columns,
            )
        except Exception as e:
            logger.exception(f"❌ Error during processing {input_file_name}: {e}")
            continue

    # Cleanup and final logging
    shutil.rmtree(dirs["temp_dir"])
    total_time = time.time() - total_start_time
    logger.info(f"⏱️ Total execution time for all files: {total_time:.2f} seconds")
    logger.info("✅ All files processed.")


if __name__ == "__main__":
    main()
