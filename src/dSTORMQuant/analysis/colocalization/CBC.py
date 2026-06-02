from typing import Any

import locan as lc
import pandas as pd

from dSTORMQuant.utils.logger import get_logger

logger = get_logger()


def compute_coordinate_based_colocalization(
    localizations_df: pd.DataFrame,
    channels_to_analyze: list,
    channels: dict[str, tuple[str, str, int]],
    radius: float = 100,
    n_steps: int = 10,
) -> dict[str, dict[str, Any]]:
    """Compute coordinate-based colocalization (CBC) for both channel directions.

    Args:
        localizations_df: Localization table with ``x``, ``y``, and ``channelIndex``.
        channels_to_analyze: Channel keys to include (e.g. ``["Ch1", "Ch2"]``).
        channels: Mapping channel key -> ``(color, label, channel_index)``.
        radius: CBC search radius in nanometers.
        n_steps: Number of distance steps for the CBC computation.

    Returns:
        Dict keyed by direction label (e.g. ``"PEX14→TOMM20"``) with keys
        ``coordinates``, ``cbc_values``, and ``color`` for plotting. Empty if a
        channel has no localizations.
    """
    # Rename for locan compatibility
    localizations_df = localizations_df.rename(
        columns={"x": "position_x", "y": "position_y"}
    )

    channel_indices = [channels[channel][2] for channel in channels_to_analyze]
    channel_dataframes = [
        localizations_df[localizations_df["channelIndex"] == idx]
        for idx in channel_indices
    ]

    if any(channel_df.empty for channel_df in channel_dataframes):
        logger.warning(
            "❌ One of the channels is empty, skipping coordinate-based colocalization."
        )
        return {}

    localization_datasets = [
        lc.LocData.from_dataframe(channel_df) for channel_df in channel_dataframes
    ]
    channel_labels = [channels[channel][1] for channel in channels_to_analyze]
    channel_colors = [channels[channel][0] for channel in channels_to_analyze]

    cbc_plot_data: dict[str, dict[str, Any]] = {}

    for i in range(2):  # Two directions: ch1->ch2 and ch2->ch1
        locdata_primary, locdata_reference = (
            localization_datasets[i],
            localization_datasets[1 - i],
        )
        label_primary, label_reference = channel_labels[i], channel_labels[1 - i]
        color_primary = channel_colors[i]

        # Compute coordinate-based colocalization (CBC)
        cbc_result = lc.CoordinateBasedColocalization(
            radius=radius, n_steps=n_steps
        ).compute(locdata=locdata_primary, other_locdata=locdata_reference)
        coordinates = locdata_primary.coordinates
        cbc_column = [
            col
            for col in cbc_result.results.columns
            if col.startswith("colocalization_cbc_")
        ][0]
        cbc_values = cbc_result.results[cbc_column].values

        # Store data for plotting only
        cbc_plot_data[f"{label_primary}→{label_reference}"] = {
            "coordinates": coordinates,
            "cbc_values": cbc_values,
            "color": color_primary,
        }

    return cbc_plot_data
