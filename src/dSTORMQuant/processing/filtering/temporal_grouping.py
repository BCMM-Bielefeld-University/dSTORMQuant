import numpy as np
import pandas as pd

from tqdm import tqdm
from typing import Any
from scipy.spatial import cKDTree

from dSTORMQuant.utils.logger import get_logger

logger = get_logger()


def _aggregate_optional_column(col: str, series: pd.Series) -> float:
    """Aggregate one optional column for a spatiotemporal group."""
    lower = col.lower()
    if any(token in lower for token in ("photon", "intensity")):
        return float(series.sum())
    if any(token in lower for token in ("sigma", "sx", "sy", "lp", "pvalue")):
        return float(np.sqrt((series**2).mean()))
    return float(series.mean())


def spatiotemporal_grouping(
    df: pd.DataFrame,
    max_frame_gap: int = 2,
    max_distance_nm: float = 60,
    channel_index: int | None = None,
    preserve_columns: list[str] | tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Group localizations into spatiotemporal tracks and aggregate per track.

    Neighboring points within ``max_distance_nm`` and ``max_frame_gap`` frames are
    merged; each group is summarized by mean position, frame span, and optional
    photophysical columns when present.

    Args:
        df: Localization table with columns ``x``, ``y``, and ``frame``.
        max_frame_gap: Maximum absolute frame difference to link two points.
        max_distance_nm: Spatial linking radius in nanometers.
        channel_index: Channel index stored on each output row, or ``None``.
        preserve_columns: Extra numeric columns to aggregate (e.g. LP config
            headers carried through drift).

    Returns:
        One row per track with columns ``x``, ``y``, ``track_ID``, ``n_locs``,
        ``first_frame``, ``last_frame``, ``duration``, ``channelIndex``, and any
        aggregated optional columns (``lp``, ``photons``, ``bg``, ``sx``, ``sy``,
        ``pvalue``, plus ``preserve_columns``) that were valid in the input.
    """
    df = df.dropna(subset=["x", "y", "frame"]).copy()
    df.sort_values(by="frame", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Identify which optional columns are actually present and numeric
    optional_cols = ["lp", "photons", "bg", "sx", "sy", "pvalue"]
    for col in preserve_columns or ():
        if col not in optional_cols:
            optional_cols.append(col)
    available_cols: dict[str, bool] = {}
    for col in optional_cols:
        if col in df.columns and pd.to_numeric(df[col], errors="coerce").notna().any():
            df[col] = pd.to_numeric(df[col], errors="coerce")
            available_cols[col] = True
        else:
            available_cols[col] = False
            if col in df.columns:
                logger.warning(f"Column '{col}' present but has no valid numeric data; skipping in grouping.")
            else:
                logger.info(f"Column '{col}' not found; skipping in grouping.")

    coords = df[["x", "y"]].values
    frames = df["frame"].values

    tree = cKDTree(coords)
    visited = np.zeros(len(df), dtype=bool)
    group_id = 0
    grouped_data: list[dict[str, Any]] = []

    for i in tqdm(range(len(df)), desc="Temporal Grouping Progress"):
        if visited[i]:
            continue

        group = [i]
        visited[i] = True
        queue = [i]

        while queue:
            idx = queue.pop()
            neighbors = tree.query_ball_point(coords[idx], r=max_distance_nm)
            for j in neighbors:
                if not visited[j] and abs(frames[j] - frames[idx]) <= max_frame_gap:
                    visited[j] = True
                    group.append(j)
                    queue.append(j)

        group_df = df.iloc[group]

        # Always-present fields
        record: dict[str, Any] = {
            "x": group_df["x"].mean(),
            "y": group_df["y"].mean(),
            "track_ID": group_id,
            "n_locs": len(group_df),
            "first_frame": group_df["frame"].min(),
            "last_frame": group_df["frame"].max(),
            "duration": group_df["frame"].max() - group_df["frame"].min() + 1,
            "channelIndex": channel_index,
        }

        for col, is_available in available_cols.items():
            if is_available:
                record[col] = _aggregate_optional_column(col, group_df[col])

        grouped_data.append(record)
        group_id += 1

    return pd.DataFrame(grouped_data)


def duration_filtering(
    grouped_df: pd.DataFrame, min_duration: int = 2, max_duration: int | None = 50
) -> pd.DataFrame:
    """Keep tracks whose ``duration`` lies within the configured frame range.

    Args:
        grouped_df: Output of :func:`spatiotemporal_grouping` with a ``duration`` column.
        min_duration: Minimum inclusive track length in frames.
        max_duration: Maximum inclusive track length, or ``None`` for no upper bound.

    Returns:
        Filtered copy of ``grouped_df``.
    """
    df = grouped_df.copy()
    if max_duration is not None:
        df = df[(df["duration"] >= min_duration) & (df["duration"] <= max_duration)]
    else:
        df = df[df["duration"] >= min_duration]
    return df


def run_spatiotemporal_grouping(
    channel_tuple: tuple[int, pd.DataFrame],
    max_frame_gap: int = 2,
    max_distance_nm: float = 60,
    preserve_columns: list[str] | tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Apply :func:`spatiotemporal_grouping` to one channel's localizations.

    Args:
        channel_tuple: ``(channel_index, localization_dataframe)`` pair.
        max_frame_gap: Passed to :func:`spatiotemporal_grouping`.
        max_distance_nm: Passed to :func:`spatiotemporal_grouping`.
        preserve_columns: Passed to :func:`spatiotemporal_grouping`.

    Returns:
        Grouped track dataframe for the channel.
    """
    channel, channel_df = channel_tuple
    return spatiotemporal_grouping(
        channel_df,
        max_frame_gap=max_frame_gap,
        max_distance_nm=max_distance_nm,
        channel_index=channel,
        preserve_columns=preserve_columns,
    )
