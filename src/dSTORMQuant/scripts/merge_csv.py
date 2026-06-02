from __future__ import annotations

import glob
import os
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from dSTORMQuant.scripts.output_discovery import list_output_folders
from dSTORMQuant.utils.data_handling import read_localization_csv
from dSTORMQuant.utils.logger import get_logger

logger = get_logger()

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
OUTPUT_BASE_DIR: Path = DATA_DIR / "output"


def normalize_channel_name(channel_name: str) -> str:
    """Normalize a channel name by stripping trailing numeric suffixes.

    Args:
        channel_name: Raw channel label (e.g. ``AST0063-2``).

    Returns:
        Base channel name without a ``-N`` suffix (e.g. ``AST0063``).
    """
    return re.sub(r"-\d+$", "", channel_name)


def extract_cell_line_from_folder(folder_name: str) -> str:
    """Extract the cell-line token from a pipeline output folder name.

    Args:
        folder_name: Experiment folder stem, expected format
            ``AHA_<number>_<cell_line>_<channel1>_<channel2>_...``.

    Returns:
        Cell line string, or ``"Unknown"`` when the name has fewer than three parts.
    """
    parts = folder_name.split("_")
    if len(parts) >= 3:
        return parts[2]  # Cell line is the third part
    return "Unknown"


def extract_channel_name_from_csv(filename: str, prefix: str) -> str | None:
    """Extract the channel suffix from a prefixed CSV filename.

    Args:
        filename: Basename such as ``intra_RST0063.csv``.
        prefix: Leading prefix including trailing underscore (e.g. ``intra_``).

    Returns:
        Channel name without prefix or extension, or ``None`` if the pattern does not match.
    """
    if filename.startswith(prefix) and filename.endswith(".csv"):
        return filename[len(prefix) : -4]
    return None


def extract_mean_distance_channel_name(filename: str) -> str | None:
    """Parse the channel name from an intra mean-distance CSV filename.

    Args:
        filename: Basename matching ``mean_distance_radius_<r>_<channel>.csv``.

    Returns:
        Channel name, or ``None`` if the pattern does not match.
    """
    pattern = r"^mean_distance_radius_\d+\.\d+_(.+)\.csv$"
    match = re.match(pattern, filename)
    return match.group(1) if match else None


def extract_mean_distance_inter_channels(filename: str) -> tuple[str, str] | None:
    """Parse source and target channels from an inter mean-distance CSV filename.

    Args:
        filename: Basename matching ``mean_distance_radius_<r>_<src>_to_<dst>.csv``.

    Returns:
        ``(source_channel, target_channel)`` or ``None`` if the pattern does not match.
    """
    pattern = r"^mean_distance_radius_\d+\.\d+_(.+)_to_(.+)\.csv$"
    match = re.match(pattern, filename)
    return (match.group(1), match.group(2)) if match else None


def extract_inter_channels_from_filename(filename: str) -> tuple[str, str] | None:
    """Parse source and target channels from an inter-channel CSV filename.

    Args:
        filename: Basename matching ``inter_<src>_to_<dst>.csv`` or the cluster-prefixed variant.

    Returns:
        ``(source_channel, target_channel)`` or ``None`` if the pattern does not match.
    """
    pattern = r"^(?:cluster_)?inter_(.+)_to_(.+)\.csv$"
    match = re.match(pattern, filename)
    return (match.group(1), match.group(2)) if match else None


def merge_csv_files_by_channel(
    output_dir: Path = OUTPUT_BASE_DIR,
    file_type: str = "cluster_stats",
    cluster_prefix: bool = False,
) -> dict:
    """
    Merge all CSV files with the same channel name across all analysis folders.
    Now separates files by both cell line and channel name.

    Args:
        output_dir: Directory containing analysis folders
        file_type: Type of files to merge - "cluster_stats", "intra", "inter", "mean_distance", or "mean_distance_inter"
        cluster_prefix: Whether to merge files with "cluster_" prefix (True) or without (False)

    Returns:
        Dictionary with merge summary
    """

    if not output_dir.exists():
        logger.error(f"Output directory does not exist: {output_dir}")
        return {}

    # Define prefix_str at the beginning to avoid UnboundLocalError
    prefix_str = "cluster_" if cluster_prefix else ""

    # Define file patterns based on type and cluster prefix
    if cluster_prefix:
        # Only include cluster_intra and cluster_inter files
        if file_type == "intra":
            pattern = output_dir / "*" / "test_data" / "cluster_intra_*.csv"
            prefix = "cluster_intra_"
        elif file_type == "inter":
            pattern = output_dir / "*" / "test_data" / "cluster_inter_*.csv"
            prefix = "cluster_inter_"
        elif file_type == "mean_distance_inter":
            pattern = (
                output_dir
                / "*"
                / "test_data"
                / "cluster_mean_distance_radius_*_to_*.csv"
            )
            prefix = None
        else:
            # Skip cluster_stats and mean_distance for cluster_prefix=True
            logger.info(f"Skipping {file_type} for cluster_prefix=True as requested")
            return {}
    else:
        # Regular files (no cluster_ prefix)
        if file_type == "cluster_stats":
            pattern = output_dir / "*" / "test_data" / "cluster_stats_*.csv"
            prefix = "cluster_stats_"
        elif file_type == "intra":
            pattern = output_dir / "*" / "test_data" / "intra_*.csv"
            prefix = "intra_"
        elif file_type == "inter":
            pattern = output_dir / "*" / "test_data" / "inter_*.csv"
            prefix = "inter_"
        elif file_type == "mean_distance":
            pattern = output_dir / "*" / "test_data" / "mean_distance_radius_*.csv"
            prefix = None
        elif file_type == "mean_distance_inter":
            pattern = output_dir / "*" / "test_data" / "mean_distance_radius_*_to_*.csv"
            prefix = None
        else:
            logger.error(f"Unknown file type: {file_type}")
            return {}

    # Find all CSV files
    csv_files = glob.glob(str(pattern))

    # Filter out unwanted files for regular patterns
    if not cluster_prefix:
        if file_type == "cluster_stats":
            csv_files = [
                f
                for f in csv_files
                if not Path(f).name.startswith("cluster_cluster_stats_")
            ]
        elif file_type == "mean_distance":
            csv_files = [
                f
                for f in csv_files
                if not Path(f).name.startswith("cluster_mean_distance_radius_")
            ]

    if not csv_files:
        logger.warning(f"No {prefix_str}{file_type} CSV files found in {output_dir}")
        return {}

    logger.info(f"Found {len(csv_files)} {prefix_str}{file_type} CSV files")

    # Create merged output directory
    merged_output_dir = output_dir / "merged_channel_data"
    merged_output_dir.mkdir(parents=True, exist_ok=True)

    if file_type == "inter":
        return merge_inter_files(csv_files, merged_output_dir, cluster_prefix)
    elif file_type == "mean_distance":
        return merge_mean_distance_files(csv_files, merged_output_dir, cluster_prefix)
    elif file_type == "mean_distance_inter":
        return merge_mean_distance_inter_files(
            csv_files, merged_output_dir, cluster_prefix
        )
    else:
        # At this point, prefix should never be None based on the logic above
        if prefix is None:
            logger.error(
                f"Prefix is None for file_type {file_type}, this should not happen"
            )
            return {}
        return merge_single_channel_files(
            csv_files, file_type, prefix, merged_output_dir, cluster_prefix
        )


def merge_single_channel_files(
    csv_files: list,
    file_type: str,
    prefix: str,
    merged_output_dir: Path,
    cluster_prefix: bool = False,
) -> dict:
    """Merge per-channel CSVs across experiments, grouped by cell line.

    Args:
        csv_files: Paths to single-channel CSV files discovered under ``test_data``.
        file_type: Merge category (``cluster_stats`` or ``intra``).
        prefix: Filename prefix identifying the file type (e.g. ``cluster_stats_``).
        merged_output_dir: Directory for merged output CSVs.
        cluster_prefix: If ``True``, write filenames with a ``cluster_`` prefix.

    Returns:
        Summary dict mapping ``(cell_line, channel)`` keys to merge metadata.
    """
    # Group files by (cell_line, normalized_channel_name) for file naming
    cell_line_channel_files = defaultdict(list)

    for csv_file in csv_files:
        csv_path = Path(csv_file)
        filename = csv_path.name
        if filename.startswith(prefix) and filename.endswith(".csv"):
            original_channel_name = filename[len(prefix) : -4]
            normalized_channel_name = normalize_channel_name(original_channel_name)
        else:
            original_channel_name = None
            normalized_channel_name = None

        if original_channel_name and normalized_channel_name:
            analysis_folder = csv_path.parent.parent.name
            cell_line = extract_cell_line_from_folder(analysis_folder)
            key = (cell_line, normalized_channel_name)
            cell_line_channel_files[key].append(
                (csv_path, analysis_folder, original_channel_name)
            )
        else:
            logger.warning(f"Could not extract channel name from: {filename}")

    if not cell_line_channel_files:
        prefix_str = "cluster_" if cluster_prefix else ""
        logger.warning(
            f"No valid channel names found in {prefix_str}{file_type} CSV files"
        )
        return {}

    # Merge files for each (cell_line, channel) combination
    merge_summary = {}

    for (cell_line, channel_name), file_list in cell_line_channel_files.items():
        dataframes = []
        successful_files = 0

        for csv_path, source_folder, original_name in file_list:
            try:
                df = read_localization_csv(csv_path)

                # Filter out noise (cluster=-1) for cluster_stats files
                if file_type == "cluster_stats" and "cluster" in df.columns:
                    df = df[df["cluster"] != -1]

                if (
                    "channel_name" not in df.columns
                    or file_type == "cluster_stats"
                    and df["channel_name"].iloc[0] != original_name
                ):
                    df["channel_name"] = original_name

                df["source_folder"] = source_folder
                df["source_file"] = csv_path.name
                df["cell_line"] = cell_line
                dataframes.append(df)
                successful_files += 1

            except Exception as e:
                logger.warning(
                    f"  ✗ Error reading {source_folder}/{csv_path.name}: {e}"
                )

        if dataframes:
            # Merge all dataframes for this (cell_line, channel) combination
            merged_df = pd.concat(dataframes, ignore_index=True)

            # Save merged CSV with cell line in filename
            prefix_str = "cluster_" if cluster_prefix else ""
            if file_type == "cluster_stats":
                output_filename = (
                    merged_output_dir
                    / f"merged_{prefix_str}cluster_stats_{cell_line}_{channel_name}.csv"
                )
            else:  # intra
                output_filename = (
                    merged_output_dir
                    / f"merged_{prefix_str}intra_{cell_line}_{channel_name}.csv"
                )

            merged_df.to_csv(output_filename, index=False)

            merge_summary[f"{cell_line}_{channel_name}"] = {
                "cell_line": cell_line,
                "channel_name": channel_name,
                "files_merged": successful_files,
                "total_rows": len(merged_df),
                "output_file": output_filename,
            }

            logger.info(f"  ✅ Saved: {output_filename.name}")
            logger.info(f"  Total rows: {len(merged_df)}")

            # Print column summary for the first file of each channel
            if dataframes:
                sample_df = dataframes[0]
                logger.info(f"  Columns: {list(sample_df.columns)}")
        else:
            logger.warning(
                f"  No valid data to merge for cell line: {cell_line}, channel: {channel_name}"
            )

    return merge_summary


def merge_mean_distance_files(
    csv_files: list, merged_output_dir: Path, cluster_prefix: bool = False
) -> dict:
    """
    Merge mean_distance CSV files with cell line separation.
    These files contain single column with mean distance values.
    Files are grouped by (cell_line, channel_name).
    Preserves original channel names (no normalization).
    """
    # Group files by (cell_line, channel_name) - NO normalization
    cell_line_channel_files = defaultdict(list)

    for csv_file in csv_files:
        csv_path = Path(csv_file)
        filename = csv_path.name

        if "_to_" in filename:
            continue

        channel_name = extract_mean_distance_channel_name(filename)

        if channel_name:
            analysis_folder = csv_path.parent.parent.name
            cell_line = extract_cell_line_from_folder(analysis_folder)

            normalized_channel = normalize_channel_name(channel_name)
            key = (cell_line, normalized_channel)
            cell_line_channel_files[key].append(
                (csv_path, analysis_folder, channel_name)
            )
        else:
            logger.warning(f"Could not extract channel name from: {filename}")

    if not cell_line_channel_files:
        prefix_str = "cluster_" if cluster_prefix else ""
        logger.warning(
            f"No valid channel names found in {prefix_str}mean_distance CSV files"
        )
        return {}

    # Merge files for each (cell_line, channel) combination
    merge_summary = {}

    for (cell_line, normalized_channel), file_list in cell_line_channel_files.items():
        dataframes = []
        successful_files = 0

        for csv_path, source_folder, original_name in file_list:
            try:
                df = read_localization_csv(csv_path)

                df["source_folder"] = source_folder
                df["source_file"] = csv_path.name
                df["cell_line"] = cell_line
                df["channel_name"] = original_name
                dataframes.append(df)
                successful_files += 1

            except Exception as e:
                logger.warning(
                    f"  ✗ Error reading {source_folder}/{csv_path.name}: {e}"
                )

        if dataframes:
            merged_df = pd.concat(dataframes, ignore_index=True)

            prefix_str = "cluster_" if cluster_prefix else ""
            output_filename = (
                merged_output_dir
                / f"merged_{prefix_str}mean_distance_{cell_line}_{normalized_channel}.csv"
            )
            merged_df.to_csv(output_filename, index=False)

            merge_summary[f"{cell_line}_{normalized_channel}"] = {
                "cell_line": cell_line,
                "channel_name": normalized_channel,
                "files_merged": successful_files,
                "total_rows": len(merged_df),
                "output_file": output_filename,
            }

            logger.info(f"  ✅ Saved: {output_filename.name}")
        else:
            logger.warning(
                f"  No valid data to merge for cell line: {cell_line}, channel: {normalized_channel}"
            )

    return merge_summary


def merge_mean_distance_inter_files(
    csv_files: list, merged_output_dir: Path, cluster_prefix: bool = False
) -> dict:
    """
    Merge inter-channel mean_distance CSV files with cell line separation.
    Normalizes target channel names so variants are combined.
    Preserves original target channel names in 'target_channel_name' column.
    """
    cell_line_channel_pair_files = defaultdict(list)

    for csv_file in csv_files:
        csv_path = Path(csv_file)
        filename = csv_path.name

        # For cluster_mean_distance_inter files, we need to handle the prefix
        if cluster_prefix:
            pattern = r"^cluster_mean_distance_radius_\d+\.\d+_(.+)_to_(.+)\.csv$"
        else:
            pattern = r"^mean_distance_radius_\d+\.\d+_(.+)_to_(.+)\.csv$"

        match = re.match(pattern, filename)
        if match:
            channel1 = match.group(1)
            channel2 = match.group(2)

            # Extract cell line from analysis folder name
            analysis_folder = csv_path.parent.parent.name
            cell_line = extract_cell_line_from_folder(analysis_folder)

            # Normalize both source and target channels for grouping
            normalized_channel1 = normalize_channel_name(channel1)
            normalized_channel2 = normalize_channel_name(channel2)

            key = (cell_line, normalized_channel1, normalized_channel2)
            # Store original channel names to preserve in data
            cell_line_channel_pair_files[key].append(
                (csv_path, analysis_folder, (channel1, channel2))
            )
        else:
            logger.warning(f"Could not extract channel pair from: {filename}")

    if not cell_line_channel_pair_files:
        prefix_str = "cluster_" if cluster_prefix else ""
        logger.warning(
            f"No valid channel pairs found in {prefix_str}mean_distance inter CSV files"
        )
        return {}

    # Merge files for each (cell_line, normalized_channel1, normalized_channel2) combination
    merge_summary = {}

    for (
        cell_line,
        normalized_channel1,
        normalized_channel2,
    ), file_list in cell_line_channel_pair_files.items():
        pair_name = f"{cell_line}_{normalized_channel1}_to_{normalized_channel2}"
        dataframes = []
        successful_files = 0

        for csv_path, source_folder, (orig_ch1, orig_ch2) in file_list:
            try:
                df = read_localization_csv(csv_path)

                # Normalize column names to combine variants (e.g., distance_to_ast0063-2 -> distance_to_ast0063)
                df = normalize_column_names(df)

                df["source_folder"] = source_folder
                df["source_file"] = csv_path.name
                df["cell_line"] = cell_line
                df["source_channel_name"] = orig_ch1
                df["target_channel_name"] = orig_ch2
                dataframes.append(df)
                successful_files += 1

            except Exception as e:
                logger.warning(
                    f"  ✗ Error reading {source_folder}/{csv_path.name}: {e}"
                )

        if dataframes:
            merged_df = pd.concat(dataframes, ignore_index=True)

            # For columns that appear multiple times (due to different channel variants),
            # combine them by taking non-null values
            for col in merged_df.columns:
                # Check if there are duplicate-like columns (those with variations in channel names)
                if any(
                    pattern in col.lower()
                    for pattern in ["distance_to_", "inter_", "_to_"]
                ):
                    # Count occurrences of this column pattern
                    similar_cols = [
                        c
                        for c in merged_df.columns
                        if c == col
                        or (
                            c.replace("-1", "").replace("-2", "").replace("-3", "")
                            == col.replace("-1", "").replace("-2", "").replace("-3", "")
                        )
                    ]

                    if len(similar_cols) > 1 and col == similar_cols[0]:
                        # Combine multiple similar columns into one
                        combined = merged_df[col].copy()
                        for other_col in similar_cols[1:]:
                            # Fill NaN values in combined with values from other_col
                            mask = combined.isna()
                            combined[mask] = merged_df.loc[mask, other_col]

                        # Update the first column and drop the others
                        merged_df[col] = combined
                        for other_col in similar_cols[1:]:
                            merged_df = merged_df.drop(columns=[other_col])

            prefix_str = "cluster_" if cluster_prefix else ""
            output_filename = (
                merged_output_dir
                / f"merged_{prefix_str}mean_distance_inter_{cell_line}_{normalized_channel1}_to_{normalized_channel2}.csv"
            )
            merged_df.to_csv(output_filename, index=False)

            merge_summary[pair_name] = {
                "cell_line": cell_line,
                "channel1": normalized_channel1,
                "channel2": normalized_channel2,
                "files_merged": successful_files,
                "total_rows": len(merged_df),
                "output_file": output_filename,
            }

            logger.info(f"  ✅ Saved: {output_filename.name}")
        else:
            logger.warning(
                f"  No valid data to merge for cell line: {cell_line}, channel pair: {normalized_channel1} -> {normalized_channel2}"
            )

    return merge_summary


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize column names by applying channel name normalization to columns
    that contain channel names (like distance_to_ast0063-2_nm -> distance_to_ast0063_nm).
    This ensures that different variants of the same channel are consolidated.
    """
    new_columns = {}
    for col in df.columns:
        # Check if column contains channel name patterns
        if any(
            pattern in col.lower() for pattern in ["distance_to_", "inter_", "_to_"]
        ):
            # Try to find and normalize channel names in the column name
            normalized_col = col
            for prefix in ["ast", "pst", "rst"]:
                # Match patterns like "distance_to_ast0063-2_nm" or "_to_ast0063-2"
                pattern = f"({prefix}\\d+)(?:-\\d+)?"
                normalized_col = re.sub(pattern, lambda m: m.group(1), normalized_col)
            new_columns[col] = normalized_col
        else:
            new_columns[col] = col

    return df.rename(columns=new_columns)


def merge_inter_files(
    csv_files: list, merged_output_dir: Path, cluster_prefix: bool = False
) -> dict:
    """
    Merge inter-channel files with cell line separation.
    Normalizes target channel names so variants are combined.
    Preserves original target channel names in 'target_channel_name' column.
    """
    cell_line_channel_pair_files = defaultdict(list)

    for csv_file in csv_files:
        csv_path = Path(csv_file)
        filename = csv_path.name
        channel_pair = extract_inter_channels_from_filename(filename)

        if channel_pair:
            channel1, channel2 = channel_pair

            analysis_folder = csv_path.parent.parent.name
            cell_line = extract_cell_line_from_folder(analysis_folder)

            normalized_channel1 = normalize_channel_name(channel1)
            normalized_channel2 = normalize_channel_name(channel2)

            key = (cell_line, normalized_channel1, normalized_channel2)
            cell_line_channel_pair_files[key].append(
                (csv_path, analysis_folder, (channel1, channel2))
            )
        else:
            logger.warning(f"Could not extract channel pair from: {filename}")

    if not cell_line_channel_pair_files:
        prefix_str = "cluster_" if cluster_prefix else ""
        logger.warning(f"No valid channel pairs found in {prefix_str}inter CSV files")
        return {}

    # Merge files for each (cell_line, channel1, normalized_channel2) combination
    merge_summary = {}

    for (
        cell_line,
        normalized_channel1,
        normalized_channel2,
    ), file_list in cell_line_channel_pair_files.items():
        pair_name = f"{cell_line}_{normalized_channel1}_to_{normalized_channel2}"
        dataframes = []
        successful_files = 0

        for csv_path, source_folder, (orig_ch1, orig_ch2) in file_list:
            try:
                df = read_localization_csv(csv_path)

                # Normalize column names to combine variants (e.g., distance_to_ast0063-2 -> distance_to_ast0063)
                df = normalize_column_names(df)

                df["source_folder"] = source_folder
                df["source_file"] = csv_path.name
                df["cell_line"] = cell_line
                df["source_channel_name"] = orig_ch1
                df["target_channel_name"] = orig_ch2
                dataframes.append(df)
                successful_files += 1

            except Exception as e:
                logger.warning(
                    f"  ✗ Error reading {source_folder}/{csv_path.name}: {e}"
                )

        if dataframes:
            # Merge all dataframes
            merged_df = pd.concat(dataframes, ignore_index=True)

            # For columns that appear multiple times (due to different channel variants),
            # combine them by taking non-null values
            for col in merged_df.columns:
                # Check if there are duplicate-like columns (those with variations in channel names)
                if any(
                    pattern in col.lower()
                    for pattern in ["distance_to_", "inter_", "_to_"]
                ):
                    # Count occurrences of this column pattern
                    similar_cols = [
                        c
                        for c in merged_df.columns
                        if c == col
                        or (
                            c.replace("-1", "").replace("-2", "").replace("-3", "")
                            == col.replace("-1", "").replace("-2", "").replace("-3", "")
                        )
                    ]

                    if len(similar_cols) > 1 and col == similar_cols[0]:
                        # Combine multiple similar columns into one
                        combined = merged_df[col].copy()
                        for other_col in similar_cols[1:]:
                            # Fill NaN values in combined with values from other_col
                            mask = combined.isna()
                            combined[mask] = merged_df.loc[mask, other_col]

                        # Update the first column and drop the others
                        merged_df[col] = combined
                        for other_col in similar_cols[1:]:
                            merged_df = merged_df.drop(columns=[other_col])

            prefix_str = "cluster_" if cluster_prefix else ""
            output_filename = (
                merged_output_dir
                / f"merged_{prefix_str}inter_{cell_line}_{normalized_channel1}_to_{normalized_channel2}.csv"
            )
            merged_df.to_csv(output_filename, index=False)

            merge_summary[pair_name] = {
                "cell_line": cell_line,
                "channel1": normalized_channel1,
                "channel2": normalized_channel2,
                "files_merged": successful_files,
                "total_rows": len(merged_df),
                "output_file": output_filename,
            }

            logger.info(f"  ✅ Saved: {output_filename.name}")
        else:
            logger.warning(
                f"  No valid data to merge for cell line: {cell_line}, channel pair: {normalized_channel1} -> {normalized_channel2}"
            )

    return merge_summary


def analyze_channel_distribution():
    """
    Analyze how many folders have channel files for all file types (excluding cluster_cluster_stats and cluster_mean_distance)
    Now includes cell line information in the analysis.
    """
    analysis_folders = [f for f in OUTPUT_BASE_DIR.iterdir() if f.is_dir()]

    # Regular file counts by cell line
    cluster_stats_counts = defaultdict(lambda: defaultdict(int))
    intra_counts = defaultdict(lambda: defaultdict(int))
    inter_counts = defaultdict(lambda: defaultdict(int))
    mean_distance_counts = defaultdict(lambda: defaultdict(int))
    mean_distance_inter_counts = defaultdict(lambda: defaultdict(int))

    # Cluster-prefixed file counts by cell line (only intra and inter)
    cluster_intra_counts = defaultdict(lambda: defaultdict(int))
    cluster_inter_counts = defaultdict(lambda: defaultdict(int))
    cluster_mean_distance_inter_counts = defaultdict(lambda: defaultdict(int))

    folder_details = []

    for folder in analysis_folders:
        test_data_dir = folder / "test_data"
        if test_data_dir.exists():
            cell_line = extract_cell_line_from_folder(folder.name)

            # Regular files (excluding cluster_ prefixed ones)
            cluster_stats_files = list(test_data_dir.glob("cluster_stats_*.csv"))
            cluster_stats_files = [
                f
                for f in cluster_stats_files
                if not f.name.startswith("cluster_cluster_stats_")
            ]
            cluster_stats_channels = [
                extract_channel_name_from_csv(f.name, "cluster_stats_")
                for f in cluster_stats_files
            ]
            cluster_stats_channels = [
                c for c in cluster_stats_channels if c is not None
            ]

            intra_files = list(test_data_dir.glob("intra_*.csv"))
            intra_files = [
                f for f in intra_files if not f.name.startswith("cluster_intra_")
            ]
            intra_channels = [
                extract_channel_name_from_csv(f.name, "intra_") for f in intra_files
            ]
            intra_channels = [c for c in intra_channels if c is not None]

            inter_files = list(test_data_dir.glob("inter_*.csv"))
            inter_files = [
                f for f in inter_files if not f.name.startswith("cluster_inter_")
            ]
            inter_pairs = [
                extract_inter_channels_from_filename(f.name) for f in inter_files
            ]
            inter_pairs = [p for p in inter_pairs if p is not None]

            mean_distance_files = list(test_data_dir.glob("mean_distance_radius_*.csv"))
            mean_distance_files = [
                f
                for f in mean_distance_files
                if not f.name.startswith("cluster_mean_distance_radius_")
            ]
            mean_distance_single_files = [
                f for f in mean_distance_files if "_to_" not in f.name
            ]
            mean_distance_channels = [
                extract_mean_distance_channel_name(f.name)
                for f in mean_distance_single_files
            ]
            mean_distance_channels = [
                c for c in mean_distance_channels if c is not None
            ]

            mean_distance_inter_files = [
                f for f in mean_distance_files if "_to_" in f.name
            ]
            mean_distance_inter_pairs = [
                extract_mean_distance_inter_channels(f.name)
                for f in mean_distance_inter_files
            ]
            mean_distance_inter_pairs = [
                p for p in mean_distance_inter_pairs if p is not None
            ]

            # Cluster-prefixed files (only intra and inter as requested)
            cluster_intra_files = list(test_data_dir.glob("cluster_intra_*.csv"))
            cluster_inter_files = list(test_data_dir.glob("cluster_inter_*.csv"))
            cluster_mean_distance_inter_files = list(
                test_data_dir.glob("cluster_mean_distance_radius_*_to_*.csv")
            )

            folder_details.append(
                {
                    "folder": folder.name,
                    "cell_line": cell_line,
                    "cluster_stats_channels": cluster_stats_channels,
                    "intra_channels": intra_channels,
                    "inter_pairs": inter_pairs,
                    "mean_distance_channels": mean_distance_channels,
                    "mean_distance_inter_pairs": mean_distance_inter_pairs,
                    "cluster_stats_count": len(cluster_stats_files),
                    "intra_count": len(intra_files),
                    "inter_count": len(inter_files),
                    "mean_distance_count": len(mean_distance_single_files),
                    "mean_distance_inter_count": len(mean_distance_inter_files),
                    "cluster_intra_count": len(cluster_intra_files),
                    "cluster_inter_count": len(cluster_inter_files),
                    "cluster_mean_distance_inter_count": len(
                        cluster_mean_distance_inter_files
                    ),
                }
            )

            # Update counts by cell line for regular files
            cluster_stats_counts[cell_line][len(cluster_stats_files)] += 1
            intra_counts[cell_line][len(intra_files)] += 1
            inter_counts[cell_line][len(inter_files)] += 1
            mean_distance_counts[cell_line][len(mean_distance_single_files)] += 1
            mean_distance_inter_counts[cell_line][len(mean_distance_inter_files)] += 1

            # Update counts by cell line for cluster-prefixed files
            cluster_intra_counts[cell_line][len(cluster_intra_files)] += 1
            cluster_inter_counts[cell_line][len(cluster_inter_files)] += 1
            cluster_mean_distance_inter_counts[cell_line][
                len(cluster_mean_distance_inter_files)
            ] += 1


# Post Merging Part
def extract_channel_from_filename(filename):
    """Extract a channel token from a filename.

    Args:
        filename: Basename or path containing ``RST``, ``AST``, or ``PST`` plus digits.

    Returns:
        Uppercase channel label (e.g. ``RST0063``), or ``None`` if no match.
    """
    match = re.search(r"(RST|AST|PST)(\d+)", filename, re.IGNORECASE)
    if match:
        return f"{match.group(1).upper()}{match.group(2)}"
    return None


def extract_channels_from_inter_filename(filename):
    """Extract source and target channels from an inter-cluster filename.

    Args:
        filename: Basename matching ``_<src>_to_<dst>.csv`` with RST/AST/PST tokens.

    Returns:
        ``(source_channel, target_channel)`` tuple, or ``(None, None)`` if unmatched.
    """
    match = re.search(
        r"_(rst|ast|pst)(\d+)_to_(rst|ast|pst)(\d+)\.csv", filename, re.IGNORECASE
    )
    if match:
        source = f"{match.group(1).upper()}{match.group(2)}"
        target = f"{match.group(3).upper()}{match.group(4)}"
        return source, target
    return None, None


def merge_cluster_data(folder_path):
    """Join merged cluster statistics with intra- and inter-distance CSVs.

    For each ``merged_cluster_stats_<cell_line>_<channel>.csv``, looks up matching
    intra and inter files and adds distance columns aligned by ``source_folder``.

    Args:
        folder_path: Directory containing ``merged_cluster_*.csv`` files.

    Returns:
        None. Writes augmented CSV files beside the input stats files.
    """

    # Find all merged cluster files
    all_files = glob.glob(os.path.join(folder_path, "merged_cluster_*.csv"))

    # Categorize files by type
    stats_files = [f for f in all_files if "stats" in f.lower()]
    intra_files = [f for f in all_files if "intra" in f.lower()]
    inter_files = [
        f
        for f in all_files
        if "inter" in f.lower() and "mean_distance" not in f.lower()
    ]

    # Process each stats file
    for stats_file in stats_files:
        print(f"\nProcessing: {os.path.basename(stats_file)}")

        # Extract channel from filename
        channel = extract_channel_from_filename(os.path.basename(stats_file))
        if not channel:
            print(f"ERROR: Could not extract channel from {stats_file}")
            continue

        print(f"Channel: {channel}")

        # Load stats data
        stats_df = read_localization_csv(stats_file)
        stats_df.columns = [col.strip().lower() for col in stats_df.columns]

        # Create a copy to hold merged results
        merged_df = stats_df.copy()

        # 1. Merge intra-cluster data
        intra_file = None
        for f in intra_files:
            intra_channel = extract_channel_from_filename(os.path.basename(f))
            if intra_channel and intra_channel.lower() == channel.lower():
                intra_file = f
                break

        if intra_file:
            intra_df = read_localization_csv(intra_file)
            intra_df.columns = [col.strip().lower() for col in intra_df.columns]

            # Find the distance column
            distance_col = None
            for col in intra_df.columns:
                if "distance" in col.lower() or "nn" in col.lower():
                    distance_col = col
                    break

            if distance_col:
                # Create a dictionary to map (source_folder, position) to distance
                intra_dict = {}
                for source_folder in intra_df["source_folder"].unique():
                    folder_data = intra_df[intra_df["source_folder"] == source_folder]
                    for pos, (idx, row) in enumerate(folder_data.iterrows()):
                        intra_dict[(source_folder, pos)] = row[distance_col]

                # Add intra distances to merged_df
                intra_distances = []
                for idx, row in merged_df.iterrows():
                    source_folder = row["source_folder"]
                    folder_rows = merged_df[merged_df["source_folder"] == source_folder]
                    folder_pos = folder_rows.index.get_loc(idx)
                    key = (source_folder, folder_pos)
                    intra_distances.append(intra_dict.get(key, np.nan))

                merged_df["intra_distance_nm"] = intra_distances
            else:
                merged_df["intra_distance_nm"] = np.nan
        else:
            merged_df["intra_distance_nm"] = np.nan

        # 2. Merge inter-cluster data
        for inter_file in inter_files:
            source_channel, target_channel = extract_channels_from_inter_filename(
                os.path.basename(inter_file)
            )

            if source_channel and source_channel.lower() == channel.lower():
                inter_df = read_localization_csv(inter_file)
                inter_df.columns = [col.strip().lower() for col in inter_df.columns]

                # Find the distance column
                distance_col = None
                for col in inter_df.columns:
                    if "distance" in col.lower():
                        distance_col = col
                        break

                if distance_col:
                    col_name = f"inter_distance_to_{target_channel.lower()}_nm"

                    if col_name not in merged_df.columns:
                        merged_df[col_name] = np.nan

                        # Create a dictionary to map (source_folder, position) to inter distance
                        inter_dict = {}
                        for source_folder in inter_df["source_folder"].unique():
                            folder_data = inter_df[
                                inter_df["source_folder"] == source_folder
                            ]
                            for pos, (idx, row) in enumerate(folder_data.iterrows()):
                                inter_dict[(source_folder, pos)] = row[distance_col]

                        # Match inter distances to stats
                        for source_folder in merged_df["source_folder"].unique():
                            stats_indices = merged_df[
                                merged_df["source_folder"] == source_folder
                            ].index.tolist()
                            inter_count = len(
                                inter_df[inter_df["source_folder"] == source_folder]
                            )

                            if inter_count > 0:
                                for pos, stats_idx in enumerate(stats_indices):
                                    if pos < inter_count:
                                        key = (source_folder, pos)
                                        if key in inter_dict:
                                            merged_df.loc[stats_idx, col_name] = (
                                                inter_dict[key]
                                            )

        # Save to CSV
        output_filename = f"merged_complete_{channel}.csv"
        output_path = os.path.join(folder_path, output_filename)
        merged_df.to_csv(output_path, index=False)
        print(f"Saved: {output_filename}")

    print("\nProcessing complete!")


def main():
    """Merge pipeline CSV exports across experiments and print a summary.

    Scans :data:`OUTPUT_BASE_DIR`, merges cluster stats, intra/inter kNN, and
    mean-distance files, then runs :func:`merge_cluster_data` on merged outputs.

    Returns:
        None
    """
    logger.info("Starting CSV merge process...")

    # Check if output directory exists
    if not OUTPUT_BASE_DIR.exists():
        logger.info("Output directory does not exist. Nothing to merge.")
        return

    # First analyze the distribution
    analyze_channel_distribution()

    # Check if there are any valid analysis folders
    valid_folders = list_output_folders(OUTPUT_BASE_DIR)
    if not valid_folders:
        logger.info("No valid analysis folders found. Exiting merge process.")
        return

    # Merge all file types
    summaries = {
        "cluster_stats": merge_csv_files_by_channel(
            OUTPUT_BASE_DIR, "cluster_stats", cluster_prefix=False
        ),
        "intra": merge_csv_files_by_channel(
            OUTPUT_BASE_DIR, "intra", cluster_prefix=False
        ),
        "inter": merge_csv_files_by_channel(
            OUTPUT_BASE_DIR, "inter", cluster_prefix=False
        ),
        "mean_distance": merge_csv_files_by_channel(
            OUTPUT_BASE_DIR, "mean_distance", cluster_prefix=False
        ),
        "mean_distance_inter": merge_csv_files_by_channel(
            OUTPUT_BASE_DIR, "mean_distance_inter", cluster_prefix=False
        ),
        "cluster_intra": merge_csv_files_by_channel(
            OUTPUT_BASE_DIR, "intra", cluster_prefix=True
        ),
        "cluster_inter": merge_csv_files_by_channel(
            OUTPUT_BASE_DIR, "inter", cluster_prefix=True
        ),
        "cluster_mean_distance_inter": merge_csv_files_by_channel(
            OUTPUT_BASE_DIR, "mean_distance_inter", cluster_prefix=True
        ),
    }

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("MERGE SUMMARY")
    logger.info("=" * 60)
    for file_type, summary in summaries.items():
        if summary:
            total_files = sum(s["files_merged"] for s in summary.values())
            total_rows = sum(s["total_rows"] for s in summary.values())
            logger.info(f"\n{file_type}:")
            logger.info(f"  Total merges: {len(summary)}")
            logger.info(f"  Total files merged: {total_files}")
            logger.info(f"  Total rows: {total_rows}")

    logger.info(
        f"\nAll merged files saved to: {OUTPUT_BASE_DIR / 'merged_channel_data'}"
    )
    logger.info("Merge process completed!")

    # Check if any merging actually occurred
    if not any(summaries.values()):
        logger.info("No data was merged. Skipping final merge step.")
        return

    # Get folder path
    folder_path = OUTPUT_BASE_DIR / "merged_channel_data"

    if not folder_path.exists():
        print(f"Error: Folder '{folder_path}' does not exist!")
        return

    print(f"\nProcessing files in: {folder_path}")

    # Merge the data
    merge_cluster_data(folder_path)


if __name__ == "__main__":
    main()
