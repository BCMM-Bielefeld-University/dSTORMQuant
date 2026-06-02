import os
from typing import Any

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from dSTORMQuant.utils.logger import get_logger
from dSTORMQuant.visualization.visualization import (
    plot_distance_histogram,
    plot_mean_distance_histogram,
)

logger = get_logger()


def calculate_nearest_neighbor_distances(
    data1: pd.DataFrame,
    data2: pd.DataFrame | None = None,
    k: int = 1,
    algorithm: str = "auto",
    metric: str = "euclidean",
    x_col: str = "x",
    y_col: str = "y",
) -> np.ndarray:
    """Return the nearest-neighbor distance for each point in ``data1``.

    Args:
        data1: Query localizations with ``x_col`` and ``y_col``.
        data2: Optional target set for inter-channel distances; ``None`` for intra.
        k: Number of neighbors (only the first neighbor distance is returned).
        algorithm: sklearn ``NearestNeighbors`` algorithm.
        metric: Distance metric passed to sklearn.
        x_col: Column name for x coordinates in nanometers.
        y_col: Column name for y coordinates in nanometers.

    Returns:
        1D array of k=1 distances in nanometers, or empty if inputs are invalid.
    """
    dist = calculate_knn_sklearn(
        data1,
        data2=data2,
        k=k,
        algorithm=algorithm,
        metric=metric,
        x_col=x_col,
        y_col=y_col,
    )
    if dist.size == 0:
        return np.array([])
    return dist[:, 0]


def calculate_knn_sklearn(
    data1: pd.DataFrame,
    data2: pd.DataFrame | None = None,
    k: int = 3,
    algorithm: str = "auto",
    metric: str = "euclidean",
    x_col: str = "x",
    y_col: str = "y",
) -> np.ndarray:
    """Compute k-nearest-neighbor distances with sklearn.

    For intra-channel analysis (``data2`` is ``None``), each point's self-match is
    excluded. For inter-channel analysis, neighbors are drawn from ``data2``.
    Short neighbor lists are padded with ``nan``.

    Args:
        data1: Query localizations with ``x_col`` and ``y_col``.
        data2: Optional target localizations; ``None`` searches within ``data1``.
        k: Number of nearest neighbors per query point.
        algorithm: sklearn ``NearestNeighbors`` algorithm.
        metric: Distance metric passed to sklearn.
        x_col: Column name for x coordinates in nanometers.
        y_col: Column name for y coordinates in nanometers.

    Returns:
        ``(n_query, k)`` distance array in nanometers; empty or NaN-filled when
        inputs are insufficient.
    """
    if data1.empty or x_col not in data1.columns or y_col not in data1.columns:
        logger.warning("Input data1 is empty or missing required coordinate columns.")
        return np.empty((0, k))

    coords1 = data1[[x_col, y_col]].values

    if data2 is not None:
        if data2.empty or x_col not in data2.columns or y_col not in data2.columns:
            logger.warning(
                "Input data2 is empty or missing required coordinate columns."
            )
            return np.empty((coords1.shape[0], k))

        coords2 = data2[[x_col, y_col]].values
        n_neighbors = min(k, coords2.shape[0])

        knn = NearestNeighbors(
            n_neighbors=n_neighbors, algorithm=algorithm, metric=metric
        ).fit(coords2)
        distances, _ = knn.kneighbors(coords1)

        if n_neighbors < k:
            pad = np.full((coords1.shape[0], k - n_neighbors), np.nan)
            distances = np.hstack([distances, pad])
    else:
        if coords1.shape[0] < 2:
            logger.warning(
                "Input data1 has fewer than 2 points for intra-distance calculation."
            )
            return np.full((coords1.shape[0], k), np.nan)

        n_neighbors = min(k + 1, coords1.shape[0])
        knn = NearestNeighbors(
            n_neighbors=n_neighbors, algorithm=algorithm, metric=metric
        ).fit(coords1)
        distances, _ = knn.kneighbors(coords1)

        if n_neighbors > 1:
            distances = distances[:, 1:]
        else:
            distances = np.full((coords1.shape[0], k), np.nan)

        if distances.shape[1] < k:
            pad = np.full((coords1.shape[0], k - distances.shape[1]), np.nan)
            distances = np.hstack([distances, pad])

    logger.info(
        f"Calculated kNN distances using sklearn NearestNeighbors with k={k}, algorithm={algorithm}, metric={metric}"
    )
    return distances


def calculate_mean_distance_within_radius(
    data1: pd.DataFrame,
    radius: float = 100,
    data2: pd.DataFrame | None = None,
    x_col: str = "x",
    y_col: str = "y",
) -> np.ndarray:
    """Mean distance to all neighbors within a fixed radius.

    Args:
        data1: Query localizations with ``x_col`` and ``y_col``.
        radius: Search radius in nanometers.
        data2: Optional target set; ``None`` uses neighbors within ``data1`` (self excluded).
        x_col: Column name for x coordinates in nanometers.
        y_col: Column name for y coordinates in nanometers.

    Returns:
        1D array of mean neighbor distances per query point; ``0`` where no
        neighbors were found (NaN replaced).
    """
    if data1.empty or x_col not in data1.columns or y_col not in data1.columns:
        logger.warning("Input data1 is empty or missing required coordinate columns.")
        return np.array([])

    coords1 = data1[[x_col, y_col]].values

    if data2 is not None:
        if data2.empty or x_col not in data2.columns or y_col not in data2.columns:
            logger.warning(
                "Input data2 is empty or missing required coordinate columns."
            )
            return np.array([])
        coords2 = data2[[x_col, y_col]].values
        nn = NearestNeighbors(radius=radius, algorithm="auto", metric="euclidean").fit(
            coords2
        )
        distances, _ = nn.radius_neighbors(coords1)
    else:
        nn = NearestNeighbors(radius=radius, algorithm="auto", metric="euclidean").fit(
            coords1
        )
        distances, indices = nn.radius_neighbors(coords1)

    mean_distances = np.zeros(len(coords1))
    for i, dist in enumerate(distances):
        if data2 is None:
            dist = dist[indices[i] != i]  # remove self distance
        mean_distances[i] = np.mean(dist) if len(dist) > 0 else np.nan

    logger.info(
        f"Calculated mean distances within radius={radius} nm using kNN (radius_neighbors)."
    )
    return np.nan_to_num(mean_distances, nan=0)


def analyze_nearest_neighbors(
    data: pd.DataFrame,
    output_dir: str,
    channels: dict[str, tuple[str, str, int]],
    nn_config: dict[str, Any],
    x_col: str = "x",
    y_col: str = "y",
    is_clustered: bool = False,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Run intra- and inter-channel kNN and mean-distance analyses.

    Writes per-channel CSV and PNG outputs under ``output_dir`` and annotates
    ``data`` with distance columns.

    Args:
        data: Localization dataframe with ``channelIndex`` and coordinate columns.
        output_dir: Directory for histograms and CSV exports.
        channels: Mapping ``Ch1``/``Ch2`` -> ``(color, label, channel_index)``.
        nn_config: Settings with keys ``radius``, ``k``, ``algorithm``, ``metric``.
        x_col: Column name for x coordinates in nanometers.
        y_col: Column name for y coordinates in nanometers.
        is_clustered: If ``True``, prefix outputs with ``cluster_`` and include
            cluster IDs in CSV files.

    Returns:
        Tuple of the annotated dataframe and a dict of per-plot summary statistics
        from histogram helpers.
    """
    df = data.copy()

    results = {}
    summary_results: dict[str, pd.DataFrame] = {}

    radius = nn_config["radius"]
    k = nn_config["k"]
    algorithm = nn_config["algorithm"]
    metric = nn_config["metric"]

    # Add cluster prefix to file names if is_clustered is True
    cluster_prefix = "cluster_" if is_clustered else ""

    logger.info("Starting intra-channel nearest neighbor analysis using kNN.")

    # -------------------
    # Intra-channel analysis
    # -------------------
    for _, (ch_color, ch_label, ch_index) in channels.items():
        ch_df = df[df["channelIndex"] == ch_index]
        if ch_df.empty:
            continue

        distances = calculate_knn_sklearn(
            ch_df, k=k, algorithm=algorithm, metric=metric, x_col=x_col, y_col=y_col
        )
        # Histogram
        summary_results[f"{ch_label}_intra"] = plot_distance_histogram(
            distances[:, 0],
            f"kNN (k=1) Distances within {ch_label}",
            os.path.join(output_dir, f"{cluster_prefix}intra_{ch_label.lower()}.png"),
            ch_color,
            log_scale=True,
        )
        # Save CSV with cluster IDs and all kNN columns
        csv_dict = {
            "nn_distance_nm": distances[:, i] for i in range(distances.shape[1])
        }
        if is_clustered:
            csv_dict["cluster"] = ch_df["cluster"].values
        csv_data = pd.DataFrame(csv_dict)
        csv_data.to_csv(
            os.path.join(output_dir, f"{cluster_prefix}intra_{ch_label.lower()}.csv"),
            index=False,
        )
        # Add first nearest neighbor distance to df
        df.loc[df["channelIndex"] == ch_index, "nn_distance_nm"] = distances[:, 0]

    # -------------------
    # Inter-channel analysis
    # -------------------
    ch_keys = list(channels.keys())
    if all(df[df["channelIndex"] == channels[ch][2]].shape[0] > 0 for ch in ch_keys):
        logger.info("Starting inter-channel nearest neighbor analysis.")
        for i, ch1_key in enumerate(ch_keys):
            for j, ch2_key in enumerate(ch_keys):
                if i == j:
                    continue
                color1, label1, index1 = channels[ch1_key]
                _, label2, index2 = channels[ch2_key]

                data1 = df[df["channelIndex"] == index1]
                data2 = df[df["channelIndex"] == index2]
                col_name = f"nn_distance_to_{label2.lower()}_nm"

                distances = calculate_knn_sklearn(
                    data1,
                    data2,
                    k=k,
                    algorithm=algorithm,
                    metric=metric,
                    x_col=x_col,
                    y_col=y_col,
                )
                # Histogram
                summary_results[f"{label1}_to_{label2}_inter"] = (
                    plot_distance_histogram(
                        distances[:, 0],
                        f"kNN (k=1) Distances from {label1} to {label2}",
                        os.path.join(
                            output_dir,
                            f"{cluster_prefix}inter_{label1.lower()}_to_{label2.lower()}.png",
                        ),
                        color1,
                        log_scale=True,
                    )
                )
                # Save CSV with cluster IDs and all kNN columns
                csv_dict = {
                    f"distance_to_{label2.lower()}_nm": distances[:, i]
                    for i in range(distances.shape[1])
                }
                if is_clustered:
                    csv_dict["cluster"] = data1["cluster"].values
                csv_data = pd.DataFrame(csv_dict)
                csv_data.to_csv(
                    os.path.join(
                        output_dir,
                        f"{cluster_prefix}inter_{label1.lower()}_to_{label2.lower()}.csv",
                    ),
                    index=False,
                )
                df.loc[df["channelIndex"] == index1, col_name] = distances[:, 0]

    # -------------------
    # Mean distance within radius
    # -------------------
    logger.info("Starting mean distance within radius analysis.")

    # Intra-channel mean distance
    for _, (ch_color, ch_label, ch_index) in channels.items():
        ch_df = df[df["channelIndex"] == ch_index]
        if ch_df.empty:
            continue

        mean_distances = calculate_mean_distance_within_radius(
            ch_df, radius=radius, x_col=x_col, y_col=y_col
        )
        if mean_distances.size > 0:
            valid_distances = mean_distances[mean_distances > 0]
            if valid_distances.size > 0:
                results[f"{ch_label}_mean_distances_radius_{radius}"] = valid_distances
                summary_results[f"{ch_label}_intra_mean_radius_{radius}"] = (
                    plot_mean_distance_histogram(
                        valid_distances,
                        f"{ch_label} Mean Displacement (r = {radius} nm)",
                        os.path.join(
                            output_dir,
                            f"{cluster_prefix}mean_distance_radius_{radius}_{ch_label.lower()}.png",
                        ),
                        ch_color,
                        log_scale=False,
                        radius=radius,
                    )
                )
                # Save CSV with cluster IDs
                valid_mask = mean_distances > 0
                csv_dict = {"mean_distance_nm": valid_distances}
                if is_clustered:
                    valid_clusters = ch_df.loc[valid_mask, "cluster"].values
                    csv_dict["cluster"] = valid_clusters
                csv_data = pd.DataFrame(csv_dict)
                csv_data.to_csv(
                    os.path.join(
                        output_dir,
                        f"{cluster_prefix}mean_distance_radius_{radius}_{ch_label.lower()}.csv",
                    ),
                    index=False,
                )
                df.loc[
                    df["channelIndex"] == ch_index, f"mean_distance_radius_{radius}_nm"
                ] = mean_distances

    # Inter-channel mean distance
    if len(ch_keys) == 2:
        ch1 = channels[ch_keys[0]]
        ch2 = channels[ch_keys[1]]
        inter_pairs = [(ch1, ch2), (ch2, ch1)]
        for (color1, label1, index1), (_color2, label2, index2) in inter_pairs:
            data1 = df[df["channelIndex"] == index1]
            data2 = df[df["channelIndex"] == index2]
            col_name = f"mean_distance_to_{label2.lower()}_radius_{radius}_nm"

            mean_distances = calculate_mean_distance_within_radius(
                data1, radius=radius, data2=data2, x_col=x_col, y_col=y_col
            )
            if mean_distances.size > 0:
                valid_distances = mean_distances[mean_distances > 0]
                if valid_distances.size > 0:
                    results[f"{label1}_to_{label2}_mean_distances_radius_{radius}"] = (
                        valid_distances
                    )
                    summary_results[
                        f"{label1}_to_{label2}_inter_mean_radius_{radius}"
                    ] = plot_mean_distance_histogram(
                        valid_distances,
                        f"{label1} to {label2} Mean Displacement (r = {radius} nm)",
                        os.path.join(
                            output_dir,
                            f"{cluster_prefix}mean_distance_radius_{radius}_{label1.lower()}_to_{label2.lower()}.png",
                        ),
                        color1,
                        log_scale=is_clustered,
                    )
                    # Save CSV with cluster IDs
                    valid_mask = mean_distances > 0
                    csv_dict = {"mean_distance_nm": valid_distances}
                    if is_clustered:
                        valid_clusters = data1.loc[valid_mask, "cluster"].values
                        csv_dict["cluster"] = valid_clusters
                    csv_data = pd.DataFrame(csv_dict)
                    csv_data.to_csv(
                        os.path.join(
                            output_dir,
                            f"{cluster_prefix}mean_distance_radius_{radius}_{label1.lower()}_to_{label2.lower()}.csv",
                        ),
                        index=False,
                    )
                    df.loc[df["channelIndex"] == index1, col_name] = mean_distances

    logger.info("Completed mean distance within radius analysis.")
    return df, summary_results
