import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from dSTORMQuant.analysis.clustering.cluster_analysis import (
    perform_clustering_analysis,
)
from dSTORMQuant.analysis.colocalization.CBC import (
    compute_coordinate_based_colocalization,
)
from dSTORMQuant.analysis.colocalization.relative_enrichment_colocalization import (
    compute_relative_enrichment,
)
from dSTORMQuant.analysis.nearest_neighbor.nearest_neighbor_analysis import (
    analyze_nearest_neighbors,
)
from dSTORMQuant.core.config.loader import get_config_params, get_metadata_for_file
from dSTORMQuant.processing.cell_detection.cell_detection import (
    grid_cell_detection,
    voronoi_cell_detection,
)
from dSTORMQuant.processing.drift_correction.drift import apply_aim_drift
from dSTORMQuant.processing.filtering.filtering import apply_filters
from dSTORMQuant.processing.filtering.temporal_grouping import (
    duration_filtering,
    run_spatiotemporal_grouping,
)
from dSTORMQuant.utils.data_handling import load_data, save_df_to_csv
from dSTORMQuant.utils.logger import get_logger
from dSTORMQuant.utils.utils import (
    append_clustering_summary_stats,
    append_coloc_summary_stats,
    append_nn_summary_stats,
    append_processing_summary_stats,
    clean_folder,
    ensure_directories,
    ensure_directory,
    is_dual_channel_metadata_row,
    move_all_files,
    move_all_images,
    resolve_channel_labels,
    zip_and_rename_folder,
)
from dSTORMQuant.visualization.visualization import (
    create_stacked_histogram,
    create_stacked_napari_images,
    plot_cbc,
    plot_cell_detection,
    plot_cluster_histogram_log_npoints,
    plot_histogram_cluster_stats,
    plot_metrics,
    plot_re_histograms,
    plot_voronoi_cells,
    save_napari_points_screenshot,
    visualize_clusters,
)

logger = get_logger()


def setup_directories(project_root: Path) -> dict[str, Path]:
    """
    Set up and return all required directory paths.

    Args:
        project_root: Root directory of the project

    Returns:
        Dictionary containing all directory paths
    """
    dirs = {
        "config_file": project_root / "config" / "config.yaml",
        "data_dir": project_root / "data",
        "input_dir": project_root / "data" / "input",
        "output_dir": project_root / "data" / "output",
        "temp_dir": project_root / "data" / "temp",
        "metadata_dir": project_root / "data" / "metadata",
    }

    ensure_directory(dirs["output_dir"])

    return dirs


def setup_temp_directories(temp_dir: Path) -> None:
    """
    Clean and set up temporary processing directories.

    Args:
        temp_dir: Base temporary directory path
    """
    clean_folder(temp_dir)
    ensure_directories(
        os.path.join(temp_dir, "initial"),
        os.path.join(temp_dir, "drift_corrected"),
        os.path.join(temp_dir, "filtered"),
        os.path.join(temp_dir, "temporal_grouped"),
        os.path.join(temp_dir, "cell_detected"),
        os.path.join(temp_dir, "quantification_results"),
        os.path.join(temp_dir, "test_images"),
        os.path.join(temp_dir, "test_data"),
    )


def load_and_validate_data(
    input_file_path: str, required_columns: list[str]
) -> pd.DataFrame | None:
    """
    Load data and validate required columns exist.

    Args:
        input_file_path: Path to input CSV file
        required_columns: List of required column names

    Returns:
        Loaded DataFrame or None if validation fails
    """
    logger.info("🔍 Loading localization data...")
    df = load_data(input_file_path)

    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        logger.error(f"Missing required columns: {missing}")
        return None

    return df


def extract_and_configure_channels(
    df: pd.DataFrame,
    input_file_name: str,
    metadata_row: pd.Series,
    config: dict[str, Any],
    input_dir: Path,
) -> tuple[pd.DataFrame, dict[str, tuple[str, str, int]], bool]:
    """
    Extract channel data and configure channel information.

    Args:
        df: Full localization DataFrame
        input_file_name: Name of input file
        metadata_row: Metadata row for this file
        config: Configuration dictionary
        input_dir: Input directory path

    Returns:
        Tuple of (combined_df, channels_dict, is_single_channel)
    """
    mc = config["data"]["input"]["required_metadata_columns"]
    omc = config["data"]["input"].get("optional_metadata_columns") or {}
    ch1_label, ch2_label = resolve_channel_labels(
        input_file_name, metadata_row, omc
    )

    ch1_index = metadata_row[mc["first_channel_index"]]
    ch1_last_frame = int(metadata_row[mc["first_ch_frame_last"]])
    is_single_channel = not is_dual_channel_metadata_row(metadata_row, mc)

    skip_frames = config["data"]["input"]["skip_frames"]

    if ch1_last_frame <= 0:
        logger.error(
            f"1st channel last frame ({ch1_last_frame}) must be positive for {input_file_name}."
        )
        return None, None, None
    if ch1_last_frame <= skip_frames:
        logger.error(
            f"1st channel last frame ({ch1_last_frame}) must be greater than skip_frames ({skip_frames}) for {input_file_name}."
        )
        return None, None, None

    start_ch1 = 0

    channels = {"Ch1": (config["channels"]["Ch1"]["color"], ch1_label, ch1_index)}

    # Extract channel 1 data (skip first skip_frames of channel range)
    df_ch1 = df[
        (df["channelIndex"] == ch1_index)
        & (df["frameIndex"].between(start_ch1 + skip_frames, ch1_last_frame - 1))
    ]
    save_df_to_csv(
        df_ch1,
        os.path.join(input_dir, f"{os.path.splitext(input_file_name)[0]}_ch1.csv"),
    )

    if not is_single_channel:
        ch2_index = metadata_row[mc["second_channel_index"]]
        ch2_last_frame = int(metadata_row[mc["second_ch_frame_last"]])
        # Ch2 starts where Ch1 ends
        start_ch2 = ch1_last_frame

        if ch2_last_frame <= start_ch2:
            logger.error(
                f"2nd channel last frame ({ch2_last_frame}) must be greater than 1st channel last frame ({ch1_last_frame}) for {input_file_name}."
            )
            return None, None, None
        if ch2_last_frame <= start_ch2 + skip_frames:
            logger.error(
                f"2nd channel last frame ({ch2_last_frame}) must be greater than start_ch2 + skip_frames ({start_ch2 + skip_frames}) for {input_file_name}."
            )
            return None, None, None

        channels["Ch2"] = (config["channels"]["Ch2"]["color"], ch2_label, ch2_index)

        # Extract channel 2 data (skip first skip_frames of channel range)
        df_ch2 = df[
            (df["channelIndex"] == ch2_index)
            & (df["frameIndex"].between(start_ch2 + skip_frames, ch2_last_frame - 1))
        ]
        save_df_to_csv(
            df_ch2,
            os.path.join(input_dir, f"{os.path.splitext(input_file_name)[0]}_ch2.csv"),
        )

        new_df = pd.concat([df_ch1, df_ch2], ignore_index=True)
        logger.info(
            f"First channel: {len(df_ch1['frameIndex'].unique())} frames (from {start_ch1 + skip_frames} to {ch1_last_frame - 1})"
        )
        logger.info(
            f"Second channel: {len(df_ch2['frameIndex'].unique())} frames (from {start_ch2 + skip_frames} to {ch2_last_frame - 1})"
        )
    else:
        new_df = df_ch1.copy()
        logger.info(
            f"First channel: {len(df_ch1['frameIndex'].unique())} frames (from {start_ch1 + skip_frames} to {ch1_last_frame - 1})"
        )

    return new_df, channels, is_single_channel


def visualize_initial_data(
    df: pd.DataFrame,
    channels: dict[str, tuple[str, str, int]],
    temp_dir: Path,
    input_file_name: str,
    *,
    use_napari: bool = True,
) -> dict[str, Any]:
    """
    Create initial visualizations and save metrics.

    Args:
        df: Localization DataFrame
        channels: Channel configuration
        temp_dir: Temporary directory path
        input_file_name: Name of input file

    Returns:
        Dictionary of initial summary statistics
    """
    logger.info(f"Initial rows: {len(df)}")

    save_napari_points_screenshot(
        df,
        channels,
        os.path.join(temp_dir, "initial"),
        "napari_input_view.png",
        first=True,
        scale=1,
        use_napari=use_napari,
    )

    initial_summary_stats = plot_metrics(
        df, channels, os.path.join(temp_dir, "initial"), prefix="initial",
        log_scale_metrics={"lp", "pvalue", "intensity", "bg"}
    )

    try:
        base_name = os.path.splitext(input_file_name)[0]
        temp_save_path = os.path.join(temp_dir, "test_data", f"{base_name}_initial.csv")
        save_df_to_csv(df, temp_save_path)
    except Exception as e:
        logger.error(f"Failed to save initial new_df to temp test_data: {e}")

    return initial_summary_stats


def apply_drift_correction(
    input_file_name: str,
    input_dir: Path,
    temp_dir: Path,
    config: dict[str, Any],
    is_single_channel: bool,
) -> pd.DataFrame:
    """
    Apply drift correction to channel data.

    Args:
        input_file_name: Name of input file
        input_dir: Input directory path
        temp_dir: Temporary directory path
        config: Configuration dictionary
        is_single_channel: Whether data is single channel

    Returns:
        Drift-corrected DataFrame
    """
    logger.info("🔁 Applying drift correction...")

    drift_correction_config = config["drift_correction"]
    pixel_size = drift_correction_config["pixel_size"]
    intersect_d_nm = drift_correction_config["intersect_d_nm"]
    roi_r_nm = drift_correction_config["roi_r_nm"]

    sanity_config = drift_correction_config["sanity_checks"]
    drift_validation_config = drift_correction_config["drift_validation"]

    intersect_d_pixels = intersect_d_nm / pixel_size
    roi_r_pixels = roi_r_nm / pixel_size

    logger.info(
        f"Drift correction parameters: intersect_d={intersect_d_nm}nm ({intersect_d_pixels:.4f} pixels), roi_r={roi_r_nm}nm ({roi_r_pixels:.4f} pixels)"
    )

    if drift_validation_config.get("use"):
        logger.info(
            f"Drift validation enabled: max_segment_drift={drift_validation_config['max_segment_drift_nm']}nm"
        )

    # Get metadata-specific parameters
    ch1_segmentation = config["drift_correction"]["segmentation"]

    # Apply drift correction to Ch1
    logger.info(
        f"Step 1: Drift correcting Ch1 (first-acquired channel) with segmentation={ch1_segmentation}..."
    )
    undrift_df_ch1_path, _ = apply_aim_drift(
        os.path.join(input_dir, f"{os.path.splitext(input_file_name)[0]}_ch1.csv"),
        f"{os.path.splitext(input_file_name)[0]}_ch1.csv",
        temp_dir,
        pixel_size,
        ch1_segmentation,
        intersect_d_pixels,
        roi_r_pixels,
        sanity_config=sanity_config,
        drift_validation_config=drift_validation_config,
    )
    df_ch1_drift_corrected = load_data(undrift_df_ch1_path)

    if not is_single_channel:
        ch2_segmentation = config["drift_correction"]["segmentation"]

        # Apply drift correction to Ch2
        logger.info(
            f"Step 2: Independently drift correcting Ch2 (second-acquired channel) with segmentation={ch2_segmentation}..."
        )
        undrift_df_ch2_path, _ = apply_aim_drift(
            os.path.join(input_dir, f"{os.path.splitext(input_file_name)[0]}_ch2.csv"),
            f"{os.path.splitext(input_file_name)[0]}_ch2.csv",
            temp_dir,
            pixel_size,
            ch2_segmentation,
            intersect_d_pixels,
            roi_r_pixels,
            sanity_config=sanity_config,
            drift_validation_config=drift_validation_config,
        )
        df_ch2_drift_corrected = load_data(undrift_df_ch2_path)

        # Calculate and apply cumulative Ch1 drift to Ch2
        logger.info(
            "Step 3: Calculating cumulative Ch1 drift and subtracting from Ch2..."
        )

        ch1_drift_txt_path = os.path.join(
            temp_dir,
            "drift_corrected",
            f"{os.path.splitext(input_file_name)[0]}_ch1_aimdrift.txt",
        )

        drift_data = np.loadtxt(ch1_drift_txt_path)
        total_drift_x_pixels = drift_data[-1, 0] - drift_data[0, 0]
        total_drift_y_pixels = drift_data[-1, 1] - drift_data[0, 1]

        total_drift_x_nm = total_drift_x_pixels * pixel_size
        total_drift_y_nm = total_drift_y_pixels * pixel_size

        logger.info(
            f"Cumulative Ch1 drift: x={total_drift_x_nm:.2f} nm ({total_drift_x_pixels:.4f} pixels), "
            + f"y={total_drift_y_nm:.2f} nm ({total_drift_y_pixels:.4f} pixels)"
        )

        df_ch2_drift_corrected["x"] -= total_drift_x_nm
        df_ch2_drift_corrected["y"] -= total_drift_y_nm

        logger.info("Subtracted cumulative Ch1 drift from all Ch2 localizations")

        df_drift_corrected = pd.concat(
            [df_ch1_drift_corrected, df_ch2_drift_corrected], ignore_index=True
        )
    else:
        df_drift_corrected = df_ch1_drift_corrected.copy()

    # Clean up temporary channel files
    os.remove(
        os.path.join(input_dir, f"{os.path.splitext(input_file_name)[0]}_ch1.csv")
    )
    if not is_single_channel:
        os.remove(
            os.path.join(input_dir, f"{os.path.splitext(input_file_name)[0]}_ch2.csv")
        )

    return df_drift_corrected


def visualize_drift_corrected_data(
    df: pd.DataFrame,
    channels: dict[str, tuple[str, str, int]],
    temp_dir: Path,
    input_file_name: str,
    *,
    use_napari: bool = True,
) -> dict[str, Any]:
    """
    Visualize drift-corrected data and save metrics.

    Args:
        df: Drift-corrected DataFrame
        channels: Channel configuration
        temp_dir: Temporary directory path
        input_file_name: Name of input file

    Returns:
        Dictionary of summary statistics
    """
    save_napari_points_screenshot(
        df,
        channels,
        os.path.join(temp_dir, "drift_corrected"),
        "napari_view_after_drift_correction.png",
        scale=1,
        use_napari=use_napari,
    )

    summary_stats = plot_metrics(
        df,
        channels,
        os.path.join(temp_dir, "drift_corrected"),
        prefix="after_drift_correction",
        log_scale_metrics={"lp", "pvalue", "intensity", "bg"},
    )

    save_df_to_csv(
        df,
        os.path.join(
            temp_dir,
            "drift_corrected",
            f"{os.path.splitext(input_file_name)[0]}_after_drift_correction.csv",
        ),
    )

    return summary_stats


def apply_filtering_pipeline(
    df: pd.DataFrame,
    config: dict[str, Any],
    channels: dict[str, tuple[str, str, int]],
    temp_dir: Path,
    input_file_name: str,
    *,
    use_napari: bool = True,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """
    Apply filtering pipeline and visualize results.

    Args:
        df: Input DataFrame
        config: Configuration dictionary
        channels: Channel configuration
        temp_dir: Temporary directory path
        input_file_name: Name of input file

    Returns:
        Tuple of (filtered_df, list of summary statistics)
    """
    logger.info("🧼 Filtering data...")

    filter_results, applied_flags = apply_filters(df, config)
    filtered_path = ensure_directory(os.path.join(temp_dir, "filtered"))

    filter_steps = [
        ("after_sigma_filter", filter_results["after_sigma_filter"]),
        ("after_photons_count_filter", filter_results["after_photons_count_filter"]),
        (
            "after_localization_precision_filter",
            filter_results["after_localization_precision_filter"],
        ),
        ("after_pvalue_filter", filter_results["after_pvalue_filter"]),
    ]

    after_filtering_summary_stats = []
    for step_name, filtered_df in filter_steps:
        if not applied_flags.get(step_name, False):
            logger.info(
                f"📌 Skipping {step_name} visualization (filter not applied or disabled)."
            )
            continue
        logger.info(f"📌 Processing {step_name}...")
        save_napari_points_screenshot(
            filtered_df,
            channels,
            filtered_path,
            filename=f"napari_view_{step_name}.png",
            scale=1,
            use_napari=use_napari,
        )
        summary_stats = plot_metrics(
            filtered_df, channels, filtered_path, prefix=f"{step_name}",
            log_scale_metrics={"lp", "pvalue", "intensity", "bg"}
        )
        after_filtering_summary_stats.append(summary_stats)

    final_df = filter_steps[-1][1]
    save_df_to_csv(
        final_df,
        os.path.join(
            filtered_path, f"{os.path.splitext(input_file_name)[0]}_after_filtering.csv"
        ),
    )

    return final_df, after_filtering_summary_stats


def apply_temporal_grouping(
    df: pd.DataFrame,
    config: dict[str, Any],
    channels: dict[str, tuple[str, str, int]],
    temp_dir: Path,
    input_file_name: str,
    *,
    use_napari: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any] | None]:
    """
    Apply temporal grouping to localizations.

    Args:
        df: Filtered DataFrame
        config: Configuration dictionary
        channels: Channel configuration
        temp_dir: Temporary directory path
        input_file_name: Name of input file

    Returns:
        Tuple of (grouped_df, summary_statistics)
    """
    temporal_grouping_params = get_config_params(config, "temporal_grouping")
    if not temporal_grouping_params.get("use", True):
        logger.info("⏭️ Temporal grouping disabled (use: false). Skipping.")
        temporal_grouped_path = ensure_directory(
            os.path.join(temp_dir, "temporal_grouped")
        )
        save_df_to_csv(
            df,
            os.path.join(
                temporal_grouped_path,
                f"{os.path.splitext(input_file_name)[0]}_after_temporal_grouping.csv",
            ),
        )
        return df, None  # None so processing summary skips this step

    logger.info("⏳ Temporal grouping...")
    temporal_grouped_path = ensure_directory(os.path.join(temp_dir, "temporal_grouped"))

    max_frame_gap = temporal_grouping_params["max_frame_gap"]
    max_distance_nm = temporal_grouping_params["max_distance_nm"]
    min_duration = temporal_grouping_params["min_duration"]
    max_duration = temporal_grouping_params["max_duration"]

    channel_groups = list(df.groupby("channelIndex"))
    with ThreadPoolExecutor() as executor:
        grouped_results = list(
            executor.map(
                lambda tup: run_spatiotemporal_grouping(
                    tup, max_frame_gap=max_frame_gap, max_distance_nm=max_distance_nm
                ),
                channel_groups,
            )
        )
    combined_df = pd.concat(grouped_results, ignore_index=True)
    combined_df = duration_filtering(
        combined_df, min_duration=min_duration, max_duration=max_duration
    )

    summary_stats = plot_metrics(
        combined_df, channels, temporal_grouped_path, prefix="after_temporal_grouping",
        log_scale_metrics={"lp", "pvalue", "intensity", "bg"}
    )
    save_napari_points_screenshot(
        combined_df,
        channels,
        temporal_grouped_path,
        filename="napari_view_after_temporal_grouping.png",
        scale=1,
        use_napari=use_napari,
    )
    save_df_to_csv(
        combined_df,
        os.path.join(
            temporal_grouped_path,
            f"{os.path.splitext(input_file_name)[0]}_after_temporal_grouping.csv",
        ),
    )
    return combined_df, summary_stats


def perform_cell_detection(
    df: pd.DataFrame,
    config: dict[str, Any],
    channels: dict[str, tuple[str, str, int]],
    temp_dir: Path,
    input_file_name: str,
    *,
    use_napari: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any] | None]:
    """
    Perform cell detection.

    Args:
        df: Input DataFrame
        config: Configuration dictionary
        channels: Channel configuration
        temp_dir: Temporary directory path
        input_file_name: Name of input file

    Returns:
        Tuple of (cell_df, summary_statistics or None)
    """
    cell_cfg = config["cell_detection"]

    if cell_cfg["use"]:
        logger.info("🔬 Detecting cells...")
        cell_detection_approach = cell_cfg["approach"]

        if cell_detection_approach == "grid":
            grid_cfg = cell_cfg[cell_detection_approach]
            cell_df = grid_cell_detection(
                df,
                bin_size=grid_cfg["grid_size"],
                high_density_threshold=grid_cfg["high_density_threshold"],
            )
            ensure_directory(os.path.join(temp_dir, "cell_detected"))
            plot_cell_detection(
                df_all=df,
                df_cells=cell_df,
                bin_size=grid_cfg["grid_size"],
                channels=channels,
                output_folder_path=os.path.join(temp_dir, "cell_detected"),
                file_name="combined",
                log_scale=True,
            )
        else:
            vornoi_cfg = cell_cfg[cell_detection_approach]
            cell_df, vor = voronoi_cell_detection(
                df, clip_percentile=vornoi_cfg["percentile"]
            )
            plot_voronoi_cells(
                df_all=df,
                df_cells=cell_df,
                vor=vor,
                save_path=os.path.join(temp_dir, "cell_detected", "CD_combined.png"),
            )

        summary_stats = plot_metrics(
            cell_df,
            channels,
            os.path.join(temp_dir, "cell_detected"),
            prefix="after_cell_detection",
            log_scale_metrics={"lp", "pvalue", "intensity", "bg"},
        )

        save_napari_points_screenshot(
            cell_df,
            channels,
            os.path.join(temp_dir, "cell_detected"),
            filename="napari_view_after_cell_detection.png",
            scale=1,
            use_napari=use_napari,
        )

        save_df_to_csv(
            cell_df,
            os.path.join(
                temp_dir,
                "cell_detected",
                f"{os.path.splitext(input_file_name)[0]}_after_cell_detection.csv",
            ),
        )

        return cell_df, summary_stats
    else:
        logger.info("⏭️ Skipping cell detection (disabled in config)")
        return df, None


def perform_nearest_neighbor_analysis(
    df: pd.DataFrame,
    config: dict[str, Any],
    channels: dict[str, tuple[str, str, int]],
    temp_dir: Path,
    input_file_name: str,
) -> dict[str, Any]:
    """
    Perform nearest neighbor analysis on localizations.

    Args:
        df: Input DataFrame
        config: Configuration dictionary
        channels: Channel configuration
        temp_dir: Temporary directory path
        input_file_name: Name of input file

    Returns:
        Dictionary containing nearest neighbor summary data
    """
    nn_summary_data = {}

    if config["nearest_neighbor_analysis"]["use"]:
        logger.info("📏 Nearest Neighbor Analysis...")
        nn_output = ensure_directory(
            os.path.join(temp_dir, "quantification_results", "distance_based")
        )

        nn_df, nn_summary_results = analyze_nearest_neighbors(
            data=df,
            output_dir=nn_output,
            channels=channels,
            nn_config=config["nearest_neighbor_analysis"],
        )

        nn_summary_data["locs_nn_summary"] = nn_summary_results

        save_df_to_csv(
            nn_df,
            os.path.join(
                nn_output,
                f"{os.path.splitext(input_file_name)[0]}_after_neighbors_analysis.csv",
            ),
        )

    return nn_summary_data


def run_clustering_analysis(
    df: pd.DataFrame,
    config: dict[str, Any],
    channels: dict[str, tuple[str, str, int]],
    temp_dir: Path,
    input_file_name: str,
    output_dir: Path,
    nn_summary_data: dict[str, Any],
    *,
    use_napari: bool = True,
) -> None:
    """
    Perform clustering analysis and related distance calculations.

    Args:
        df: Input DataFrame
        config: Configuration dictionary
        channels: Channel configuration
        temp_dir: Temporary directory path
        input_file_name: Name of input file
        output_dir: Output directory path for saving summary stats
        nn_summary_data: Dictionary to update with NN summary data
    """
    if not config["clustering"]["use"]:
        logger.info("⏭️ Skipping clustering")
        return

    logger.info("🔗 Running clustering...")
    clustering_dir = ensure_directory(
        os.path.join(temp_dir, "quantification_results", "clustering")
    )

    # Run clustering
    clustered_df, stats_pairs = perform_clustering_analysis(
        df, clustering_dir, config["clustering"], channels=channels
    )
    stats_df_list = [pair[1] for pair in stats_pairs]

    save_df_to_csv(
        clustered_df,
        os.path.join(
            clustering_dir,
            f"{os.path.splitext(input_file_name)[0]}_after_clustering.csv",
        ),
    )

    visualize_clusters(
        clustered_df, clustering_dir, channels, show=False, use_napari=use_napari
    )

    # Plot clustering summaries
    clustering_summary_results = plot_histogram_cluster_stats(
        stats_df_list, channels, clustering_dir, log_scale=True
    )
    cluster_npoints_summary_results = plot_cluster_histogram_log_npoints(
        stats_df_list, clustering_dir, channels
    )

    # Append clustering summary stats to output
    append_clustering_summary_stats(
        input_file_name,
        [clustering_summary_results, cluster_npoints_summary_results],
        output_dir,
    )

    # Filter noise clusters and add channel prefix
    filtered_stats = []
    ch_labels_for_clustering: list[str] = []
    for ch_key, df_ch in stats_pairs:
        df_ch = df_ch[df_ch["cluster"] != -1].copy()
        ch_labels_for_clustering.append(ch_key)
        df_ch["cluster"] = f"ch{channels[ch_key][2]}_" + df_ch["cluster"].astype(str)
        filtered_stats.append(df_ch)

    # Perform cluster nearest neighbor analysis only when we have cluster stats.
    # This keeps the pipeline safe when channel-level clustering is disabled.
    if (
        config["clustering"].get("use_cluster_knn")
        and len(ch_labels_for_clustering) > 0
        and any(not df_ch.empty for df_ch in filtered_stats)
    ):
        _perform_cluster_knn_analysis(
            filtered_stats,
            channels,
            clustering_dir,
            config,
            nn_summary_data,
        )
    elif config["clustering"].get("use_cluster_knn"):
        logger.info(
            "⏭️ Skipping cluster kNN analysis: no clustered channels or no non-noise clusters."
        )


def _perform_cluster_knn_analysis(
    filtered_stats: list[pd.DataFrame],
    channels: dict[str, tuple[str, str, int]],
    clustering_dir: str,
    config: dict[str, Any],
    nn_summary_data: dict[str, Any],
) -> None:
    """
    Perform cluster-based nearest neighbor analysis (CSV / summaries; no scatter plots).

    Args:
        filtered_stats: List of filtered cluster statistics DataFrames
        channels: Channel configuration
        clustering_dir: Output directory for clustering results
        config: Configuration dictionary
        nn_summary_data: Dictionary to update with NN summary data
    """
    combined_stats_df = pd.concat(filtered_stats, ignore_index=True)

    _, nn_clustering_summary_results = analyze_nearest_neighbors(
        combined_stats_df,
        clustering_dir,
        channels=channels,
        nn_config=config["nearest_neighbor_analysis"],
        x_col="x_center",
        y_col="y_center",
        is_clustered=True,
    )

    nn_summary_data["clusters_nn_summary"] = nn_clustering_summary_results


def perform_colocalization_analysis(
    df: pd.DataFrame,
    config: dict[str, Any],
    channels: dict[str, tuple[str, str, int]],
    temp_dir: Path,
    is_single_channel: bool,
) -> dict[str, Any] | None:
    """
    Perform co-localization analysis for dual-channel data.

    Args:
        df: Input DataFrame
        config: Configuration dictionary
        channels: Channel configuration
        temp_dir: Temporary directory path
        is_single_channel: Whether data is single channel

    Returns:
        Dictionary of co-localization summary data or None
    """
    if is_single_channel:
        return None

    coloc_config = config["colocalization"]

    if not coloc_config["use"]:
        return None

    logger.info("📊 Running co-localization analysis...")
    coloc_dir = ensure_directory(
        os.path.join(temp_dir, "quantification_results", "colocalization")
    )

    # CBC Method
    cbc_config = coloc_config["cbc"]
    logger.info("🔹 Running CBC co-localization")
    cbc_plot_data = compute_coordinate_based_colocalization(
        df,
        channels_to_analyze=["Ch1", "Ch2"],
        channels=channels,
        radius=cbc_config["radius"],
        n_steps=cbc_config["n_steps"],
    )
    cbc_summary_stats = plot_cbc(
        cbc_plot_data, channels=channels, save_dir=coloc_dir, log_scale=False
    )

    # RE Method
    logger.info("🔹 Running RE co-localization")
    RE_per_localization_df = compute_relative_enrichment(
        df=df,
        channels_to_analyze=[channels["Ch1"][2], channels["Ch2"][2]],
        channel_labels={
            channels["Ch1"][2]: channels["Ch1"][1],
            channels["Ch2"][2]: channels["Ch2"][1],
        },
        save_dir=coloc_dir,
    )
    RE_summary_df = plot_re_histograms(RE_per_localization_df, channels, coloc_dir, log_scale=True)
    coloc_summary_data = {
        "cbc_summary": cbc_summary_stats,
        "re_summary": RE_summary_df,
    }

    return coloc_summary_data


def export_final_results(
    temp_dir: Path, output_dir: Path, input_file_name: str
) -> None:
    """
    Export and organize all final results.

    Args:
        temp_dir: Temporary directory path
        output_dir: Output directory path
        input_file_name: Name of input file
    """
    logger.info("📦 Exporting final results...")

    final_data_output_path = os.path.join(temp_dir, "test_data")
    final_img_output_path = os.path.join(temp_dir, "test_images")

    # Create stacked visualizations
    metrics = ["lp", "bg", "sx", "sy", "pvalue", "intensity"]
    for metric in metrics:
        create_stacked_histogram(temp_dir, metric)

    create_stacked_napari_images(temp_dir, "napari_view_all_steps.png")

    # Move files to final locations
    move_all_images(temp_dir, final_img_output_path)
    move_all_files(temp_dir, final_data_output_path, file_extension=".csv")

    move_all_files(os.path.join(temp_dir, "drift_corrected"), final_data_output_path)
    move_all_files(
        os.path.join(temp_dir, "filtered"),
        final_data_output_path,
        file_extension=".csv",
    )
    move_all_files(
        os.path.join(temp_dir, "temporal_grouped"),
        final_data_output_path,
        file_extension=".csv",
    )
    move_all_files(
        os.path.join(temp_dir, "cell_detected"),
        final_data_output_path,
        file_extension=".csv",
    )
    move_all_files(
        os.path.join(temp_dir, "quantification_results", "clustering"),
        final_data_output_path,
        file_extension=".csv",
    )
    move_all_files(
        os.path.join(temp_dir, "quantification_results", "distance_based"),
        final_data_output_path,
        file_extension=".csv",
    )

    move_all_images(
        os.path.join(temp_dir, "quantification_results", "clustering"),
        final_img_output_path,
    )
    move_all_images(
        os.path.join(temp_dir, "quantification_results", "distance_based"),
        final_img_output_path,
    )
    move_all_images(
        os.path.join(temp_dir, "quantification_results", "colocalization"),
        final_img_output_path,
    )

    # Zip and save
    zip_and_rename_folder(temp_dir, os.path.splitext(input_file_name)[0], output_dir)
    logger.info(f"🎉 Done. Zipped results saved in: {output_dir}")


def process_single_file(
    input_file_name: str,
    input_dir: Path,
    output_dir: Path,
    temp_dir: Path,
    metadata_df: pd.DataFrame,
    config: dict[str, Any],
    required_columns: list[str],
) -> dict[str, float]:
    """
    Process a single input file through the entire pipeline.

    Args:
        input_file_name: Name of input file
        input_dir: Input directory path
        output_dir: Output directory path
        temp_dir: Temporary directory path
        metadata_df: Metadata DataFrame
        config: Configuration dictionary
        required_columns: List of required column names

    Returns:
        Dictionary of step execution times
    """
    input_file_path = os.path.join(input_dir, input_file_name)
    logger.info(f"\n📂 Processing {input_file_path}...")

    file_start_time = time.time()
    step_times = {}

    meta_cols = config["data"]["input"]["required_metadata_columns"]
    metadata_row = get_metadata_for_file(
        metadata_df,
        os.path.splitext(input_file_name)[0],
        required_metadata_columns=meta_cols,
    )
    setup_temp_directories(temp_dir)

    use_napari = get_config_params(config, "visualization").get("use_napari", True)
    if not use_napari:
        logger.info(
            "visualization.use_napari: false — skipping Napari point/cluster screenshots."
        )

    if not os.path.exists(input_file_path):
        logger.error(f"Input file '{input_file_path}' not found.")
        return step_times

    # Step 1: Load and validate data
    t0 = time.time()
    df = load_and_validate_data(input_file_path, required_columns)
    if df is None:
        return step_times
    step_times["load_data"] = time.time() - t0

    # Extract and configure channels
    new_df, channels, is_single_channel = extract_and_configure_channels(
        df, input_file_name, metadata_row, config, input_dir
    )
    if new_df is None:
        return step_times

    # Step 2: Initial visualization
    t0 = time.time()
    initial_summary_stats = visualize_initial_data(
        new_df, channels, temp_dir, input_file_name, use_napari=use_napari
    )
    step_times["initial_visualization"] = time.time() - t0

    # Step 3: Drift correction
    t0 = time.time()
    df_drift_corrected = apply_drift_correction(
        input_file_name, input_dir, temp_dir, config, is_single_channel
    )
    step_times["drift_correction"] = time.time() - t0

    after_drift_correction_summary_stats = visualize_drift_corrected_data(
        df_drift_corrected, channels, temp_dir, input_file_name, use_napari=use_napari
    )

    # Step 4: Filtering
    t0 = time.time()
    final_df, after_filtering_summary_stats = apply_filtering_pipeline(
        df_drift_corrected,
        config,
        channels,
        temp_dir,
        input_file_name,
        use_napari=use_napari,
    )
    step_times["filtering"] = time.time() - t0

    # Step 5: Temporal grouping
    t0 = time.time()
    combined_df, after_temporal_grouping_summary_stats = apply_temporal_grouping(
        final_df, config, channels, temp_dir, input_file_name, use_napari=use_napari
    )
    step_times["temporal_grouping"] = time.time() - t0

    # Step 6: Cell detection
    t0 = time.time()
    cell_df, after_cell_detection_summary_stats = perform_cell_detection(
        combined_df, config, channels, temp_dir, input_file_name, use_napari=use_napari
    )
    step_times["cell_detection"] = time.time() - t0

    # Save processing summary
    processing_summary = {
        "initial": initial_summary_stats,
        "after_drift_correction": after_drift_correction_summary_stats,
        "after_filtering": after_filtering_summary_stats,
        "after_temporal_grouping": after_temporal_grouping_summary_stats,
        "after_cell_detection": after_cell_detection_summary_stats,
    }
    append_processing_summary_stats(input_file_name, processing_summary, output_dir)

    # Step 7: Nearest neighbor analysis
    t0 = time.time()
    nn_summary_data = perform_nearest_neighbor_analysis(
        cell_df, config, channels, temp_dir, input_file_name
    )
    step_times["nearest_neighbor_analysis"] = time.time() - t0

    # Step 8: Clustering
    t0 = time.time()
    run_clustering_analysis(
        cell_df,
        config,
        channels,
        temp_dir,
        input_file_name,
        output_dir,
        nn_summary_data,
        use_napari=use_napari,
    )
    step_times["clustering"] = time.time() - t0

    # Save NN summary stats
    append_nn_summary_stats(input_file_name, nn_summary_data, output_dir)

    # Step 9: Co-localization analysis
    t0 = time.time()
    coloc_summary_data = perform_colocalization_analysis(
        cell_df, config, channels, temp_dir, is_single_channel
    )
    if coloc_summary_data:
        append_coloc_summary_stats(input_file_name, coloc_summary_data, output_dir)
    step_times["colocalization"] = time.time() - t0

    # Step 10: Final export
    t0 = time.time()
    export_final_results(temp_dir, output_dir, input_file_name)
    step_times["final_export"] = time.time() - t0

    # Log execution times
    file_total_time = time.time() - file_start_time
    logger.info(
        f"⏱️ Execution time for {input_file_name}: {file_total_time:.2f} seconds"
    )
    for step, duration in step_times.items():
        logger.info(f"    Step '{step}' took {duration:.2f} seconds")

    return step_times
