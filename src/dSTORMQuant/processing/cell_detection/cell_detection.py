import numpy as np
import pandas as pd
from scipy.spatial import QhullError, Voronoi
from skimage.filters import threshold_otsu

from dSTORMQuant.utils.logger import get_logger

logger = get_logger()


def grid_cell_detection(
    df: pd.DataFrame, bin_size: int, high_density_threshold: int
) -> pd.DataFrame:
    """
    Filter SMLM localization data to keep only the dense region (the cell) and remove background/noise.
    Grid-based: divides FOV into bins and keeps points in bins with count >= high_density_threshold.

    Args:
        df (pd.DataFrame): Input dataframe with 'x', 'y', and optionally 'channelIndex'.
        bin_size (int): Number of bins along each axis for 2D histogram.
        high_density_threshold (int): Minimum number of localizations per bin to consider as part of the cell.

    Returns:
        pd.DataFrame: Filtered dataframe containing only points in the cell. Adds columns 'x_bin', 'y_bin', 'is_cell'.
    """
    if df.empty:
        logger.warning("Input dataframe is empty. Returning empty results.")
        df["is_cell"] = False
        return df

    # Compute bin edges
    x_edges = np.linspace(df["x"].min(), df["x"].max(), bin_size + 1)
    y_edges = np.linspace(df["y"].min(), df["y"].max(), bin_size + 1)

    # Assign bins
    df["x_bin"] = np.digitize(df["x"], x_edges) - 1
    df["y_bin"] = np.digitize(df["y"], y_edges) - 1

    # Keep only valid bins
    df = df[
        (df["x_bin"] >= 0)
        & (df["x_bin"] < bin_size)
        & (df["y_bin"] >= 0)
        & (df["y_bin"] < bin_size)
    ].copy()

    # Initialize is_cell column
    df["is_cell"] = False

    # Identify high-density bins
    bin_counts = df.groupby(["x_bin", "y_bin"]).size()
    high_density_bins = bin_counts[bin_counts >= high_density_threshold].index
    logger.info(
        f"Total bins: {len(bin_counts)}, High-density bins: {len(high_density_bins)}"
    )

    # Mark points in high-density bins
    df["is_cell"] = [
        b in high_density_bins for b in zip(df["x_bin"], df["y_bin"], strict=False)
    ]

    # Filter dataframe to keep only points in the cell
    df_filtered = df[df["is_cell"]].copy()
    return df_filtered


def voronoi_cell_detection(
    df: pd.DataFrame, clip_percentile: float = 99.0
) -> tuple[pd.DataFrame, Voronoi]:
    """
    Voronoi-based background removal for single-cell SMLM data.
    Filters out sparse background points using Voronoi area thresholding.

    Args:
        df (pd.DataFrame): Input dataframe with 'x' and 'y' columns.
        clip_percentile (float, optional): Percentile to clip extreme Voronoi areas
                                           before thresholding (default 99.0).

    Returns:
        tuple:
            - pd.DataFrame: Filtered dataframe containing only points considered
                            part of the cell. Adds columns 'voronoi_area' and 'is_cell'.
            - Voronoi | None: Voronoi object computed from the original points
                              (None if Voronoi failed).
    """
    if df.empty:
        df["voronoi_area"] = np.nan
        df["is_cell"] = False
        logger.warning("Input dataframe is empty")
        return df, None

    coords = df[["x", "y"]].values

    # Try Voronoi computation
    try:
        vor = Voronoi(coords)
    except QhullError as e:
        logger.error(f"Voronoi computation failed: {e}")
        df["voronoi_area"] = np.nan
        df["is_cell"] = False
        return df, None

    # Compute Voronoi cell areas
    areas = []
    for region_index in vor.point_region:
        vertices = vor.regions[region_index]
        if -1 in vertices or len(vertices) == 0:  # infinite region or empty
            areas.append(np.inf)
            continue
        polygon = vor.vertices[vertices]
        x, y = polygon[:, 0], polygon[:, 1]
        area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        areas.append(area)

    df["voronoi_area"] = areas

    finite_areas = df["voronoi_area"][np.isfinite(df["voronoi_area"])]
    if len(finite_areas) == 0:
        df["is_cell"] = False
        logger.warning("No finite Voronoi areas found")
        return df, vor

    # Clip extreme Voronoi areas to avoid skewing threshold
    clip_value = np.percentile(finite_areas, clip_percentile)
    finite_areas_clipped = finite_areas[finite_areas <= clip_value]

    if len(finite_areas_clipped) == 0:
        df["is_cell"] = False
        logger.warning("All finite Voronoi areas clipped away")
        return df, vor

    # Otsu threshold on clipped Voronoi areas
    thresh = threshold_otsu(finite_areas_clipped.to_numpy())
    df["is_cell"] = df["voronoi_area"] <= thresh
    logger.info(
        f"Voronoi threshold (Otsu) = {thresh:.3f}, "
        f"points considered part of cell = {df['is_cell'].sum()}"
    )

    # Filter background points
    df_filtered = df[df["is_cell"]].copy()

    return df_filtered, vor
