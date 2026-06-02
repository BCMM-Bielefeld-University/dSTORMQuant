import contextlib
import os
import shutil
import string
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull

from dSTORMQuant.utils.data_handling import save_df_to_csv
from dSTORMQuant.utils.logger import get_logger

logger = get_logger()


def move_all_files(
    src_dir: str | os.PathLike,
    dst_dir: str | os.PathLike,
    file_extension: str | None = None,
) -> None:
    """
    Moves all files from the source directory to the destination directory.

    Parameters:
    ----------
    src_dir : str
        Source directory path containing files to move.
    dst_dir : str
        Destination directory path where files will be moved.
    file_extension : str, optional
        If provided, only move files with this extension (e.g., ".csv", ".txt")

    Returns:
    -------
    None
    """
    if not os.path.isdir(src_dir):
        logger.info(f"⏭️ Skipping missing directory: {src_dir}")
        return

    os.makedirs(dst_dir, exist_ok=True)
    for filename in os.listdir(src_dir):
        src_path = os.path.join(src_dir, filename)
        dst_path = os.path.join(dst_dir, filename)
        if os.path.isfile(src_path):
            # Filter by extension if provided
            if file_extension is None or filename.endswith(file_extension):
                shutil.move(src_path, dst_path)
                logger.info(f"✅ Moved: {filename}")


def move_all_images(src_root: str | os.PathLike, dst_folder: str | os.PathLike) -> None:
    """
    Recursively moves all image files (by common extensions) from the source directory tree
    into the destination folder.

    Parameters:
    ----------
    src_root : str
        Root directory to search for image files.
    dst_folder : str
        Destination directory to move the image files to.

    Returns:
    -------
    None
    """
    if not os.path.isdir(src_root):
        logger.info(f"⏭️ Skipping missing image source directory: {src_root}")
        return

    os.makedirs(dst_folder, exist_ok=True)
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".svg"}
    for root, _, files in os.walk(src_root):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in image_exts:
                src_file = os.path.join(root, file)
                dst_file = os.path.join(dst_folder, file)
                try:
                    shutil.move(src_file, dst_file)
                    logger.info(f"✅ Moved: {src_file} → {dst_file}")
                except PermissionError:
                    logger.warning(
                        f"❌ Permission denied: {src_file} (file might be open)"
                    )
                except Exception as e:
                    logger.error(f"❌ Failed to move {src_file}: {e}")


def get_unique_filename(base_name: str, ext: str, target_dir: str | os.PathLike) -> str:
    """
    Generates a unique filename in the target directory to avoid overwriting existing files.

    Parameters:
    ----------
    base_name : str
        Base name for the file (without extension).
    ext : str
        File extension (e.g., '.txt', '.zip').
    target_dir : str
        Directory in which to check for filename uniqueness.

    Returns:
    -------
    str
        Full path of a unique file name.
    """
    filename = f"{base_name}{ext}"
    file_path = os.path.join(target_dir, filename)
    if not os.path.exists(file_path):
        return file_path
    for letter in string.ascii_lowercase:
        filename = f"{base_name}_{letter}{ext}"
        file_path = os.path.join(target_dir, filename)
        if not os.path.exists(file_path):
            return file_path
    i = 1
    while True:
        filename = f"{base_name}_{i}{ext}"
        file_path = os.path.join(target_dir, filename)
        if not os.path.exists(file_path):
            return file_path
        i += 1


def zip_and_rename_folder(
    output_dir: str | os.PathLike, base_name: str, target_dir: str | os.PathLike
) -> str:
    """
    Compresses the contents of a folder into a uniquely named ZIP file in the target directory.

    Parameters:
    ----------
    output_dir : str
        Directory whose contents will be zipped.
    base_name : str
        Base name for the resulting zip file.
    target_dir : str
        Destination directory to save the zip file.

    Returns:
    -------
    str
        Path to the created zip file.
    """
    zip_path = get_unique_filename(base_name, ".zip", target_dir)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(output_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, output_dir)
                zipf.write(file_path, arcname)
    logger.info(f"📦 Zipped folder: {zip_path}")
    return zip_path


def clean_folder(directory_path: str | os.PathLike) -> None:
    """
    Removes all files and subdirectories within the specified folder, but does not delete the folder itself.

    Parameters:
    ----------
    directory_path : str
        Path to the folder to be cleaned.

    Returns:
    -------
    None
    """
    if os.path.exists(directory_path):
        for root, dirs, files in os.walk(directory_path, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                with contextlib.suppress(OSError):
                    os.rmdir(os.path.join(root, name))
        logger.info(f"🧹 Cleaned folder: {directory_path}")


def ensure_directory(directory_path: str | os.PathLike, clean: bool = False) -> str:
    """
    Ensure a directory exists, creating it if necessary.
    Optionally clean the directory if it already exists.

    Parameters:
    ----------
    directory_path : str | os.PathLike
        Path to the directory to create or ensure exists.
    clean : bool, optional
        If True, clean the directory contents if it exists. Default is False.

    Returns:
    -------
    str
        The absolute path to the directory.
    """
    dir_path = os.path.abspath(directory_path)

    if clean and os.path.exists(dir_path):
        clean_folder(dir_path)

    os.makedirs(dir_path, exist_ok=True)
    logger.debug(f"📁 Ensured directory exists: {dir_path}")

    return dir_path


def ensure_directories(
    *directory_paths: str | os.PathLike, clean: bool = False
) -> list[str]:
    """
    Ensure multiple directories exist, creating them if necessary.
    Optionally clean the directories if they already exist.

    Parameters:
    ----------
    *directory_paths : str | os.PathLike
        Variable number of directory paths to create or ensure exist.
    clean : bool, optional
        If True, clean the directory contents if they exist. Default is False.

    Returns:
    -------
    List[str]
        List of absolute paths to the directories.
    """
    return [ensure_directory(path, clean=clean) for path in directory_paths]


def filter_by_channel(
    df: pd.DataFrame,
    channel_index: int,
    additional_filter: pd.Series | None = None,
    columns: list[str] | None = None,
    min_rows: int = 0,
) -> pd.DataFrame:
    """
    Filter DataFrame by channel index with optional additional filters and column selection.

    Parameters:
    ----------
    df : pd.DataFrame
        Input DataFrame with 'channelIndex' column.
    channel_index : int
        Channel index to filter by.
    additional_filter : pd.Series, optional
        Additional boolean filter to apply (e.g., df["cluster"] != -1).
    columns : List[str], optional
        Specific columns to return. If None, returns all columns.
    min_rows : int, optional
        Minimum number of rows required. Logs warning if fewer rows found.
        Default is 0 (no minimum).

    Returns:
    -------
    pd.DataFrame
        Filtered DataFrame (or subset of columns).

    """
    # Build filter
    channel_filter = df["channelIndex"] == channel_index

    if additional_filter is not None:
        combined_filter = channel_filter & additional_filter
    else:
        combined_filter = channel_filter

    # Apply filter
    filtered_df = df[combined_filter]

    # Check minimum rows
    if min_rows > 0 and len(filtered_df) < min_rows:
        logger.warning(
            f"Channel {channel_index} has only {len(filtered_df)} rows "
            f"(minimum {min_rows} expected)"
        )

    # Select columns if specified
    if columns is not None:
        filtered_df = filtered_df[columns]

    return filtered_df


def append_processing_summary_stats(
    file_name: str, processing_summary: dict[str, Any], output_dir: Path
) -> None:
    """Append per-step processing statistics for one input file to a summary CSV.

    Args:
        file_name: Input localization filename being summarized.
        processing_summary: Map of step name -> dataframe(s) with row counts/metrics.
        output_dir: Directory containing ``processing_summary_stats.csv``.

    Returns:
        None
    """

    processing_rows = []
    for step_name, df in processing_summary.items():
        if df is not None:  # Skip None values (e.g., when cell detection is disabled)
            df_list = df if isinstance(df, list) else [df]
            for single_df in df_list:
                for _, row in single_df.reset_index(drop=True).iterrows():
                    row_data = row.to_dict()
                    row_data["File"] = file_name
                    row_data["ProcessingStep"] = step_name
                    processing_rows.append(row_data)

    if processing_rows:
        processing_df = pd.DataFrame(processing_rows)
        # Ensure expected columns exist
        for expected_col in ["File", "ProcessingStep"]:
            if expected_col not in processing_df.columns:
                processing_df[expected_col] = pd.NA

        # Reorder columns safely
        leading = ["File", "ProcessingStep"]
        tail = [c for c in processing_df.columns if c not in leading]
        processing_df = processing_df[leading + tail]

        # Append to existing file or create new one
        csv_path = os.path.join(output_dir, "processing_summary_stats.csv")
        if os.path.exists(csv_path):
            save_df_to_csv(processing_df, csv_path, index=False, mode="a", header=False)
        else:
            save_df_to_csv(processing_df, csv_path, index=False)
        logger.info(f"Appended processing summary for {file_name}")


def append_nn_summary_stats(
    file_name: str, nn_summary_data: dict[str, Any], output_dir: Path
) -> None:
    """Append nearest-neighbor histogram summary stats for one file to a CSV.

    Args:
        file_name: Input localization filename being summarized.
        nn_summary_data: Nested dict of summary type -> metric -> stats dataframe.
        output_dir: Directory containing ``nn_summary_stats.csv``.

    Returns:
        None
    """

    nn_rows = []
    for summary_type, metrics_dict in nn_summary_data.items():
        for metric_name, df in metrics_dict.items():
            for _, row in df.reset_index(drop=True).iterrows():
                row_data = row.to_dict()
                row_data["File"] = file_name
                row_data["SummaryType"] = summary_type
                row_data["Metric"] = metric_name
                nn_rows.append(row_data)

    if nn_rows:
        nn_df = pd.DataFrame(nn_rows)
        for expected_col in ["File", "SummaryType", "Metric"]:
            if expected_col not in nn_df.columns:
                nn_df[expected_col] = pd.NA
        leading = ["File", "SummaryType", "Metric"]
        tail = [c for c in nn_df.columns if c not in leading]
        nn_df = nn_df[leading + tail]

        # Append to existing file or create new one
        csv_path = os.path.join(output_dir, "nn_summary_stats.csv")
        if os.path.exists(csv_path):
            save_df_to_csv(nn_df, csv_path, index=False, mode="a", header=False)
        else:
            save_df_to_csv(nn_df, csv_path, index=False)
        logger.info(f"Appended NN summary for {file_name}")


def append_clustering_summary_stats(
    file_name: str, clustering_summary_results: list[pd.DataFrame], output_dir: Path
) -> None:
    """Append clustering summary statistics for one file to a CSV.

    Args:
        file_name: Input localization filename being summarized.
        clustering_summary_results: List of per-channel cluster stats dataframes.
        output_dir: Directory containing ``clustering_summary_stats.csv``.

    Returns:
        None
    """

    clustering_rows = []
    for df in clustering_summary_results:
        for _, row in df.reset_index(drop=True).iterrows():
            row_data = row.to_dict()
            row_data["File"] = file_name
            clustering_rows.append(row_data)

    if clustering_rows:
        clustering_df = pd.DataFrame(clustering_rows)
        if "File" not in clustering_df.columns:
            clustering_df["File"] = pd.NA
        leading = ["File"]
        tail = [c for c in clustering_df.columns if c not in leading]
        clustering_df = clustering_df[leading + tail]

        # Append to existing file or create new one
        csv_path = os.path.join(output_dir, "clustering_summary_stats.csv")
        if os.path.exists(csv_path):
            save_df_to_csv(clustering_df, csv_path, index=False, mode="a", header=False)
        else:
            save_df_to_csv(clustering_df, csv_path, index=False)
        logger.info(f"Appended clustering summary for {file_name}")


def append_coloc_summary_stats(
    file_name: str, coloc_summary_data: dict[str, Any], output_dir: Path
) -> None:
    """Append colocalization summary statistics for one file to a CSV.

    Args:
        file_name: Input localization filename being summarized.
        coloc_summary_data: Map of summary type -> stats dataframe.
        output_dir: Directory containing ``coloc_summary_stats.csv``.

    Returns:
        None
    """

    coloc_rows = []
    for summary_type, df in coloc_summary_data.items():
        for _, row in df.reset_index(drop=True).iterrows():
            row_data = row.to_dict()
            row_data["File"] = file_name
            row_data["SummaryType"] = summary_type
            coloc_rows.append(row_data)

    if coloc_rows:
        coloc_df = pd.DataFrame(coloc_rows)
        for expected_col in ["File", "SummaryType"]:
            if expected_col not in coloc_df.columns:
                coloc_df[expected_col] = pd.NA
        leading = ["File", "SummaryType"]
        tail = [c for c in coloc_df.columns if c not in leading]
        coloc_df = coloc_df[leading + tail]

        # Append to existing file or create new one
        csv_path = os.path.join(output_dir, "coloc_summary_stats.csv")
        if os.path.exists(csv_path):
            save_df_to_csv(coloc_df, csv_path, index=False, mode="a", header=False)
        else:
            save_df_to_csv(coloc_df, csv_path, index=False)
        logger.info(f"Appended co-localization summary for {file_name}")


def get_cluster_border_points(x: np.ndarray | list, y: np.ndarray | list) -> np.ndarray:
    """
    Returns the border points of a 2D cluster using Convex Hull.

    If the cluster has fewer than 3 points, returns the points themselves.

    Parameters:
    - x, y : array-like (np.ndarray or list)
        Coordinates of cluster points.

    Returns:
    - border_points : np.ndarray
        Array of border points (Nx2)
    """
    points = np.column_stack((x, y))

    if len(points) < 3:
        return points

    hull = ConvexHull(points)
    border_points = points[hull.vertices]
    return border_points


def normalize_metadata_header(name: str) -> str:
    """Normalize a metadata column header for lookup.

    Args:
        name: Raw Excel column header.

    Returns:
        Lowercase string with collapsed whitespace (matches :func:`load_metadata`).
    """
    import re

    return re.sub(r"\s+", " ", str(name).strip()).lower()


def extract_channel_labels_from_stem(stem: str) -> tuple[str, str]:
    """
    Derive two channel label strings from the file stem (no extension).

    Supports short names (e.g. ``Test_PEX14_TOMM20`` → ``PEX14``, ``TOMM20``) and the
    legacy long form with at least five underscore-separated segments (uses the
    4th and 5th segments as labels).
    """
    parts = stem.split("_")
    if len(parts) >= 5:
        return str(parts[3]), str(parts[4])
    if len(parts) >= 3:
        return str(parts[-2]), str(parts[-1])
    if len(parts) == 2:
        return str(parts[0]), str(parts[1])
    if len(parts) == 1:
        return str(parts[0]), "Ch2"
    return "Ch1", "Ch2"


def extract_channels_from_filename(filename: str) -> tuple[str, str]:
    """Derive two channel labels from a localization filename.

    Args:
        filename: Input path or basename (extension is stripped).

    Returns:
        Two channel label strings; see :func:`extract_channel_labels_from_stem`.
    """
    stem = os.path.splitext(filename)[0]
    return extract_channel_labels_from_stem(stem)


def metadata_cell(
    metadata_row: pd.Series,
    column_header_from_yaml: str,
) -> object | None:
    """Read one metadata cell by YAML-configured Excel header name.

    Args:
        metadata_row: Single experiment metadata row (normalized columns).
        column_header_from_yaml: Excel header string as written in config YAML.

    Returns:
        Cell value, or ``None`` if the column is missing or blank.
    """
    import pandas as pd

    key = normalize_metadata_header(column_header_from_yaml)
    if key not in metadata_row.index:
        return None
    val = metadata_row[key]
    if pd.isna(val):
        return None
    if isinstance(val, str) and not val.strip():
        return None
    return val


def is_dual_channel_metadata_row(
    metadata_row: pd.Series, required_metadata_columns: dict[str, str]
) -> bool:
    """
    True when the metadata row defines a usable second channel (index + last frame).

    The legacy ``second_channel`` text column is no longer required; absence of a
    second acquisition is inferred from blank second-channel index / last frame.
    """
    import pandas as pd

    mc = required_metadata_columns
    v_idx = metadata_cell(metadata_row, mc["second_channel_index"])
    v_last = metadata_cell(metadata_row, mc["second_ch_frame_last"])
    if v_idx is None or v_last is None:
        return False
    if isinstance(v_idx, str) and v_idx.strip().lower() in ("none", ""):
        return False
    if isinstance(v_last, str) and v_last.strip().lower() in ("none", ""):
        return False
    if pd.isna(v_last):
        return False
    try:
        int(v_last)
    except (TypeError, ValueError):
        return False
    return True


def resolve_channel_labels(
    input_file_name: str,
    metadata_row: pd.Series,
    optional_metadata_columns: dict[str, str] | None,
) -> tuple[str, str]:
    """Resolve human-readable channel labels for plots and exports.

    Args:
        input_file_name: Localization filename stem (used only for fallback parsing).
        metadata_row: Metadata row for this experiment.
        optional_metadata_columns: Config map of optional label column headers.

    Returns:
        Tuple ``(channel1_label, channel2_label)``.
    """
    d1, d2 = "Channel1", "Channel2"
    omc = optional_metadata_columns or {}

    if omc.get("first_channel_label"):
        v = metadata_cell(metadata_row, omc["first_channel_label"])
        if v is not None:
            d1 = str(v).strip()

    if omc.get("second_channel_label"):
        v = metadata_cell(metadata_row, omc["second_channel_label"])
        if v is not None:
            d2 = str(v).strip()

    return d1, d2
