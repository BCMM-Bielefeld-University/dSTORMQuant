from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from dSTORMQuant.core.config.loader import (
    get_metadata_for_file,
    load_config,
    load_metadata,
)
from dSTORMQuant.scripts.output_discovery import list_output_folders
from dSTORMQuant.utils.data_handling import load_data
from dSTORMQuant.utils.utils import (
    is_dual_channel_metadata_row,
    resolve_channel_labels,
)
from dSTORMQuant.utils.logger import get_logger
from dSTORMQuant.visualization.visualization import (
    save_napari_points_screenshot,
    visualize_clusters,
)

os.environ["QT_API"] = "pyqt6"
logger = get_logger()

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
OUTPUT_BASE_DIR: Path = DATA_DIR / "output"
CONFIG_FILE: Path = PROJECT_ROOT / "config" / "config.yaml"

# Load configuration
config = load_config(CONFIG_FILE)
IS_CELL_DETECTION: bool = config["cell_detection"]["use"]
USE_NAPARI: bool = config.get("visualization", {}).get("use_napari", True)

# Define processing steps
PROCESSING_STEPS = [
    "initial",
    "after_drift_correction",
    "after_filtering",
    "after_temporal_grouping",
    "after_cell_detection",
    "after_clustering",
]

if not IS_CELL_DETECTION:
    PROCESSING_STEPS = [s for s in PROCESSING_STEPS if s != "after_cell_detection"]


def create_stacked_napari_images(
    output_folder: str | os.PathLike,
    output_filename: str,
    font_path: str | None = None,
) -> None:
    """Stack Napari step screenshots vertically with step titles.

    Args:
        output_folder: Directory containing ``napari_view_*.png`` step images.
        output_filename: Filename for the combined PNG written to ``output_folder``.
        font_path: Optional TrueType font path for step titles.

    Returns:
        None
    """
    images = []

    steps = [
        ("napari_view_initial.png", "0. Initial"),
        ("napari_view_after_drift_correction.png", "1. After Drift Correction"),
        ("napari_view_after_filtering.png", "2. After Filter"),
        ("napari_view_after_temporal_grouping.png", "3. After Temporal Grouping"),
    ]

    if IS_CELL_DETECTION:
        steps.append(
            ("napari_view_after_cell_detection.png", "4. After Cell Detection")
        )

    for filename, title in steps:
        path = os.path.join(output_folder, filename)
        if not os.path.exists(path):
            logger.warning(f"⚠️ Skipping: {path} (not found)")
            continue

        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            logger.warning(f"❌ Failed to open {path}: {e}")
            continue

        title_bar_height = 50
        label_img = Image.new("RGB", (img.width, title_bar_height), (255, 255, 255))
        draw = ImageDraw.Draw(label_img)
        try:
            font = ImageFont.truetype(font_path if font_path else "arial.ttf", 20)
        except Exception:
            font = ImageFont.load_default()
        draw.text((10, 10), title, font=font, fill=(0, 0, 0))

        combined = Image.new(
            "RGB", (img.width, title_bar_height + img.height), (255, 255, 255)
        )
        combined.paste(label_img, (0, 0))
        combined.paste(img, (0, title_bar_height))
        images.append(combined)

    if images:
        total_height = sum(img.height for img in images)
        max_width = max(img.width for img in images)
        final_image = Image.new("RGB", (max_width, total_height), (255, 255, 255))
        y_offset = 0
        for img in images:
            final_image.paste(img, (0, y_offset))
            y_offset += img.height

        save_path = os.path.join(output_folder, output_filename)
        final_image.save(save_path, format="PNG")
        logger.info(f"✅ Saved high-quality stacked image: {save_path}")
    else:
        logger.warning("❌ No valid images found to stack.")


def get_channel_data(
    data_dir: Path, config: dict[str, Any], input_filename: str
) -> dict:
    """Resolve channel colors, labels, and indices from experiment metadata.

    Args:
        data_dir: Project data root containing ``metadata/``.
        config: Loaded pipeline configuration dict.
        input_filename: Experiment stem used to look up the metadata row.

    Returns:
        Dict mapping ``Ch1``/``Ch2`` to ``(color, label, channel_index)``.
    """
    xlsx_file: Path = (
        data_dir / "metadata" / config["data"]["input"].get("xlsx_filename")
    )
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


def process_csv_files(
    test_data_dir: Path,
    channels: dict[str, tuple],
    folder_name: str,
    test_images_dir: Path,
) -> dict[str, int]:
    """Generate Napari screenshots for each pipeline step in one output folder.

    Args:
        test_data_dir: Folder with ``*_<step>.csv`` localization exports.
        channels: Channel configuration from :func:`get_channel_data`.
        folder_name: Experiment folder name (for logging).
        test_images_dir: Directory where PNG screenshots are written.

    Returns:
        Dict with ``processed`` and ``skipped`` counts.
    """
    results = {"processed": 0, "skipped": 0}

    for step in PROCESSING_STEPS:
        if step == "after_clustering":
            continue

        matching_files = list(test_data_dir.glob(f"*_{step}.csv"))
        if not matching_files:
            logger.debug(f"No {step} file found in {folder_name}/test_data")
            continue

        csv_file = matching_files[0]
        logger.info(f"Processing {step}: {csv_file.name}")

        try:
            df = load_data(csv_file)
            logger.debug(f"Loaded {len(df)} rows from {csv_file.name}")

            screenshot_filename = f"napari_view_{step}.png"
            if USE_NAPARI:
                save_napari_points_screenshot(
                    df=df,
                    channels=channels,
                    output_dir=test_images_dir,
                    filename=screenshot_filename,
                    scale=1,
                    first=(step == "initial"),
                    use_napari=True,
                )
                logger.info(f"✓ Saved visualization: {screenshot_filename}")
                results["processed"] += 1
            else:
                logger.info(
                    f"Skipping Napari screenshot for {step} (visualization.use_napari: false)"
                )

        except Exception as e:
            logger.error(f"✗ Error processing {csv_file.name}: {e}")
            results["skipped"] += 1
            continue

    # Create **one stacked image** per folder
    try:
        stacked_name = "napari_view_stacked.png"
        create_stacked_napari_images(test_images_dir, stacked_name)
        logger.info("✓ Created single stacked Napari image for folder")
    except Exception as e:
        logger.warning(
            f"Could not create stacked Napari image for folder {folder_name}: {e}"
        )

    # Clustering visualization
    clustering_files = list(test_data_dir.glob("*_after_clustering.csv"))
    if clustering_files and USE_NAPARI:
        try:
            clustering_file = clustering_files[0]
            df = load_data(clustering_file)
            visualize_clusters(
                df=df,
                save_dir=test_images_dir,
                channels=channels,
                show=False,
                use_napari=True,
            )
            logger.info("✓ Created cluster visualization")
        except Exception as e:
            logger.error(f"✗ Error creating cluster visualization: {e}")
    elif clustering_files:
        logger.info(
            "Skipping cluster Napari views (visualization.use_napari: false)"
        )

    return results


def main():
    """Regenerate Napari visualizations for all pipeline output folders.

    Discovers valid output directories, loads config/metadata per experiment, and
    calls :func:`process_csv_files` for each.

    Returns:
        None
    """
    if not OUTPUT_BASE_DIR.exists():
        logger.error(f"Output directory does not exist: {OUTPUT_BASE_DIR}")
        return

    folders = list_output_folders(OUTPUT_BASE_DIR)
    if not folders:
        logger.warning(f"No analysis folders found in {OUTPUT_BASE_DIR}")
        return

    logger.info(f"Found {len(folders)} folders to process")

    total_processed = 0
    total_skipped = 0
    folder_processed = 0
    folder_skipped = 0

    for folder in folders:
        folder_name = folder.name
        test_data_dir = folder / "test_data"
        test_images_dir = folder / "test_images"
        test_images_dir.mkdir(parents=True, exist_ok=True)

        if not test_data_dir.exists() or not test_data_dir.is_dir():
            logger.warning(f"No test_data folder found in: {folder_name}")
            folder_skipped += 1
            continue

        try:
            logger.info(f"\n{'=' * 60}")
            logger.info(f"Processing folder: {folder_name}")
            logger.info(f"{'=' * 60}")

            channels = get_channel_data(DATA_DIR, config, folder_name)
            logger.info(f"Channels configured: {list(channels.keys())}")

            results = process_csv_files(
                test_data_dir, channels, folder_name, test_images_dir
            )

            total_processed += results["processed"]
            total_skipped += results["skipped"]
            folder_processed += 1

            logger.info(
                f"Folder summary - Processed: {results['processed']}, Skipped: {results['skipped']}"
            )

        except Exception as e:
            logger.error(f"Error processing folder '{folder_name}': {e}")
            folder_skipped += 1
            continue

    logger.info(f"\n{'=' * 60}")
    logger.info("FINAL SUMMARY")
    logger.info(f"{'=' * 60}")
    logger.info(f"Folders processed: {folder_processed}")
    logger.info(f"Folders skipped: {folder_skipped}")
    logger.info(f"Total files processed: {total_processed}")
    logger.info(f"Total files skipped: {total_skipped}")
    logger.info(f"{'=' * 60}")


if __name__ == "__main__":
    main()
