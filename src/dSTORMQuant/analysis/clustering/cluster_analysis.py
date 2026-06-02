import json
import os
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
from sklearn.cluster import DBSCAN, HDBSCAN
from tqdm import tqdm

from dSTORMQuant.core.config.loader import get_config_params
from dSTORMQuant.utils.logger import get_logger
from dSTORMQuant.utils.utils import filter_by_channel

logger = get_logger()


def cluster_with_finder(
    coordinates: np.ndarray,
    threshold: float = 10.0,
    points_per_dimension: int = 15,
    algorithm: str = "dbscan",
    min_threshold: int = 5,
    max_threshold: int = 21,
    decay: float = 0.5,
) -> dict:
    """
    Run the FINDER algorithm on 2D localization data using native C++ bindings.

    Parameters
    ----------
    coordinates : np.ndarray
        2D array of shape (n_points, 2) for x, y coordinates.
    threshold : float
        Minimum points for cluster (minPts)
    points_per_dimension : int
        Points per dimension for grid
    algorithm : str
        Algorithm: "dbscan" or "DbscanLoop"
    min_threshold : int
        Minimum threshold to search
    max_threshold : int
        Maximum threshold to search
    decay : float
        Decay parameter

    Returns
    -------
    np.ndarray
        Cluster labels for each point
    """
    try:
        import finder_cpp

        if coordinates.shape[1] != 2:
            raise ValueError(
                f"2D clustering requires 2 columns, got {coordinates.shape[1]}"
            )

        coordinates_float32 = coordinates.astype(np.float32, copy=False)
        results = {}
        results: dict = finder_cpp.run_finder_2d(
            coordinates_float32,
            threshold=int(threshold),
            points_per_dimension=points_per_dimension,
            algorithm=algorithm,
            min_threshold=min_threshold,
            max_threshold=max_threshold,
            decay=decay,
        )
        logger.info("FINDER 2D C++ binding completed successfully")
        return results

    except Exception as e:
        logger.error(f"FINDER 2D C++ binding failed: {e}")
        raise


def perform_clustering_analysis(
    data: pd.DataFrame,
    output_dir: str,
    clustering_config: dict[str, Any],
    channels: dict[str, tuple[str, str, int]],
) -> tuple[pd.DataFrame, list[tuple[str, pd.DataFrame]]]:
    """
    Run clustering on localization data and save results.

    Parameters
    ----------
    data : pd.DataFrame
        Localization data.
    output_dir : str
        Directory to save clustering results.
    clustering_config : dict
        Configuration dictionary containing clustering parameters.
    channels : dict
        Channel definitions in format:
        {
            "Ch1": (color, label, index),
            "Ch2": (color, label, index)
        }

    Returns
    -------
    pd.DataFrame
        DataFrame with cluster assignments.
    list of tuple[str, pd.DataFrame]
        Per-channel cluster statistics for channels where clustering ran
        (each tuple is ``("Ch1"|"Ch2", stats_df)``).
    """
    clustered_localizations: pd.DataFrame = data.copy()
    clustered_localizations["cluster"] = -1
    clustered_localizations["channel_color"] = None

    channel_cluster_statistics: list[tuple[str, pd.DataFrame]] = []

    for channel_name, (channel_color, channel_label, channel_index) in tqdm(
        channels.items(), desc="Processing channels"
    ):
        channel_localizations: pd.DataFrame = filter_by_channel(data, channel_index)

        channel_cfg = clustering_config[channel_name.lower()]
        if not channel_cfg.get("use", True):
            logger.info(
                f"⏭️ Skipping clustering for {channel_name} ({channel_label}) "
                "(disabled in config)"
            )
            continue

        if not channel_localizations.empty:
            spatial_coordinates: np.ndarray = channel_localizations[
                ["x", "y"]
            ].to_numpy()

            # Get the method for this specific channel
            channel_config = channel_cfg
            clustering_method: str = channel_config["method"].lower()

            logger.info(
                f"Clustering {channel_label} ({channel_name}, {channel_color}) with {clustering_method.upper()}..."
            )

            if clustering_method == "dbscan":
                dbscan_params: dict[str, Any] = get_config_params(
                    channel_config, "dbscan"
                )
                with tqdm(
                    total=1, desc=f"DBSCAN clustering for {channel_label}", leave=False
                ) as pbar:
                    clusterer = DBSCAN(
                        eps=dbscan_params["eps"],
                        min_samples=dbscan_params["min_samples"],
                    ).fit(spatial_coordinates)
                    pbar.update(1)
                cluster_labels: np.ndarray = clusterer.labels_
                num_clusters: int = len(set(cluster_labels)) - (
                    1 if -1 in cluster_labels else 0
                )
                logger.info(
                    f"Found {num_clusters} clusters in {channel_label} using DBSCAN"
                )

            elif clustering_method == "hdbscan":
                hdbscan_params: dict[str, Any] = get_config_params(
                    channel_config, "hdbscan"
                )
                with tqdm(
                    total=1, desc=f"HDBSCAN clustering for {channel_label}", leave=False
                ) as pbar:
                    clusterer = HDBSCAN(
                        min_cluster_size=hdbscan_params["min_cluster_size"],
                        min_samples=hdbscan_params["min_samples"],
                        allow_single_cluster=True,
                    ).fit(spatial_coordinates)
                    pbar.update(1)
                cluster_labels = clusterer.labels_
                num_clusters = len(set(cluster_labels)) - (
                    1 if -1 in cluster_labels else 0
                )
                logger.info(
                    f"Found {num_clusters} clusters in {channel_label} using HDBSCAN"
                )

            elif clustering_method == "finder":
                finder_params: dict[str, Any] = get_config_params(
                    channel_config, "finder"
                )

                with tqdm(
                    total=1,
                    desc=f"FINDER C++ clustering for {channel_label}",
                    leave=False,
                ) as pbar:
                    results = cluster_with_finder(
                        spatial_coordinates,
                        threshold=finder_params["threshold"],
                        points_per_dimension=finder_params["points_per_dimension"],
                        algorithm=finder_params["algorithm"],
                        min_threshold=finder_params["min_threshold"],
                        max_threshold=finder_params["max_threshold"],
                        decay=finder_params["decay"],
                    )
                    pbar.update(1)

                cluster_labels = results["labels"]
                sigma = results["sigma"]
                thr = results["threshold"]

                params = {"threshold": thr, "sigma": sigma}
                saving_params_path = os.path.join(
                    output_dir, f"finder_params_{channel_label}.json"
                )
                with open(saving_params_path, "w") as f:
                    json.dump(params, f)

                num_clusters: int = len(set(cluster_labels)) - (
                    1 if -1 in cluster_labels else 0
                )
                logger.info(
                    f"Found {num_clusters} clusters in {channel_label} using FINDER"
                )

            else:
                raise ValueError(
                    f"Unknown clustering method: '{clustering_method}'. Supported: 'dbscan', 'hdbscan', 'finder'."
                )

            # Offset to avoid cluster ID collisions across channels
            cluster_offset: int = 0
            if channel_name == "Ch2" and "Ch1" in channels:
                ch1_index = channels["Ch1"][2]
                ch1_max_cluster: int = clustered_localizations.loc[
                    clustered_localizations["channelIndex"] == ch1_index, "cluster"
                ].max()
                if ch1_max_cluster >= 0:
                    cluster_offset = ch1_max_cluster + 1

            adjusted_cluster_labels: np.ndarray = np.array(
                [
                    label if label == -1 else label + cluster_offset
                    for label in cluster_labels
                ]
            )
            clustered_localizations.loc[
                clustered_localizations["channelIndex"] == channel_index, "cluster"
            ] = adjusted_cluster_labels
            clustered_localizations.loc[
                clustered_localizations["channelIndex"] == channel_index,
                "channel_color",
            ] = channel_color

            cluster_statistics: pd.DataFrame = save_cluster_stats(
                channel_localizations,
                cluster_labels,
                channel_index,
                channel_label,
                channel_color,
                output_dir,
            )
            channel_cluster_statistics.append((channel_name, cluster_statistics))

    return clustered_localizations, channel_cluster_statistics


def save_cluster_stats(
    df: pd.DataFrame,
    labels: np.ndarray,
    channel_index: int,
    channel_label: str,
    channel_color: str,
    output_dir: str,
) -> pd.DataFrame:
    """
    Save statistics about clusters including convex hull area.

    Parameters
    ----------
    df : pd.DataFrame
        Data points of the channel.
    labels : array-like
        Cluster labels for points.
    channel_index : int
        Channel index.
    channel_label : str
        Channel label/name.
    channel_color : str
        Channel color (for reference/visualization).
    output_dir : str
        Directory to save stats CSV.

    Returns
    -------
    pd.DataFrame
        DataFrame of cluster statistics.
    """
    stats: list[dict[str, Any]] = []
    unique_labels: set = set(labels)
    if -1 in unique_labels:
        unique_labels.remove(-1)

    for k in unique_labels:
        cluster_points: pd.DataFrame = df[labels == k]
        n_points: int = len(cluster_points)
        x_mean: float = cluster_points["x"].mean()
        y_mean: float = cluster_points["y"].mean()
        x_std: float = cluster_points["x"].std()
        y_std: float = cluster_points["y"].std()

        area: float = 0.0
        if n_points >= 3:
            points: np.ndarray = cluster_points[["x", "y"]].to_numpy()
            try:
                hull: ConvexHull = ConvexHull(points)
                area = hull.volume  # 2D points, volume gives area
            except Exception as e:
                logger.warning(
                    f"ConvexHull calculation failed for cluster {k} in channel {channel_label}: {e}"
                )
                area = 0.0

        density: float = n_points / area if area > 0 else np.nan

        stats.append(
            {
                "cluster": k,
                "channelIndex": channel_index,
                "channel_name": channel_label,
                "channel_color": channel_color,
                "n_points": n_points,
                "x_center": x_mean,
                "y_center": y_mean,
                "x_std": x_std,
                "y_std": y_std,
                "area_nm2": area,
                "density_points_per_nm2": density,
            }
        )

    # Add noise stats
    noise_points: pd.DataFrame = df[labels == -1]
    n_noise: int = len(noise_points)

    stats.append(
        {
            "cluster": -1,
            "channelIndex": channel_index,
            "channel_name": channel_label,
            "channel_color": channel_color,
            "n_points": n_noise,
            "x_center": noise_points["x"].mean() if n_noise > 0 else np.nan,
            "y_center": noise_points["y"].mean() if n_noise > 0 else np.nan,
            "x_std": noise_points["x"].std() if n_noise > 0 else np.nan,
            "y_std": noise_points["y"].std() if n_noise > 0 else np.nan,
            "area_nm2": np.nan,
            "density_points_per_nm2": np.nan,
        }
    )

    stats_df: pd.DataFrame = pd.DataFrame(stats)
    stats_path: str = os.path.join(output_dir, f"cluster_stats_{channel_label}.csv")
    stats_df.to_csv(stats_path, index=False)
    logger.info(f"Saved cluster stats to {stats_path}")

    return stats_df
