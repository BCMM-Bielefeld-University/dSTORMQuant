import os
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dSTORMQuant.analysis.colocalization.RE_function import (
    relative_enrichment,
)
from dSTORMQuant.utils.utils import filter_by_channel, get_logger

logger = get_logger()

def process_direction(
        df: pd.DataFrame,
        primary_idx: int,
        reference_idx: int,
        coords: Dict[int, np.ndarray],
        labels: Dict[int, str],
        save_dir: str
    ) -> pd.DataFrame:
    """Compute relative enrichment for one primary/reference channel pair.

    Args:
        df: Full localization dataframe (used for FOV extent normalization).
        primary_idx: ``channelIndex`` of the primary species.
        reference_idx: ``channelIndex`` of the reference Voronoi tessellation.
        coords: Pre-extracted ``(N, 2)`` coordinate arrays per channel index.
        labels: Human-readable channel labels per channel index.
        save_dir: Directory for RE scatter plot PNG/SVG exports.

    Returns:
        DataFrame with columns ``Channel``, ``Reference_Channel``, and
        ``RE_per_localization``.
    """
    
    coords_primary = coords[primary_idx]
    coords_reference = coords[reference_idx]

    n_points, area_vals, _, bool_index = relative_enrichment(coords_primary, coords_reference)

    # Calculate relative enrichment
    n_exp: np.ndarray = n_points / (coords_reference.shape[0] * area_vals / (max(df['x']) - min(df['x']))**2)
    n_exp[np.isinf(n_exp)] = 0.1
    n_exp[n_exp == 0] = 0.1
    
    # Scatter plot colored by RE
    fig, ax = plt.subplots(figsize=(5, 6), dpi=300)
    sc = ax.scatter(coords_primary[bool_index, 0], coords_primary[bool_index, 1],
                    c=n_exp, marker='o', s=3, lw=0, alpha=1,
                    cmap='viridis', vmin=0, vmax=10)
    ax.set_title(
        f"Relative enrichment map: {labels[primary_idx]} vs {labels[reference_idx]}",
        fontsize=10,
    )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.invert_yaxis()
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_ticks([0, 5, 10])
    cbar.set_label("Relative enrichment", rotation=90, labelpad=10, fontsize=10)
    scatter_base = os.path.join(save_dir, f"RE_scatter_{labels[reference_idx]}_ref_by_{labels[primary_idx]}")
    plt.savefig(scatter_base + ".png", dpi=300, bbox_inches="tight")
    plt.savefig(scatter_base + ".svg", bbox_inches="tight")
    plt.close(fig)
    logger.info(f"✅ Saved: {scatter_base}.png, {scatter_base}.svg")

    return pd.DataFrame({
        "Channel": labels[primary_idx],
        "Reference_Channel": labels[reference_idx],
        "RE_per_localization": n_exp
    })


def compute_relative_enrichment(
    df: pd.DataFrame,
    channels_to_analyze: List[int],
    channel_labels: Optional[Dict[int, str]] = None,
    channel_colors: Optional[Dict[int, str]] = None,
    save_dir: str = "."
) -> pd.DataFrame:
    """Compute relative enrichment in both channel directions and save scatter plots.

    Args:
        df: Localization dataframe with ``x``, ``y``, and ``channelIndex``.
        channels_to_analyze: Two channel indices ``[ch1_idx, ch2_idx]``.
        channel_labels: Optional map ``channel_index -> label`` for plots and output.
        channel_colors: Optional map of colors (reserved for future use).
        save_dir: Output directory for RE scatter figures.

    Returns:
        Concatenated per-localization RE table for both directions.
    """
    ch1_idx, ch2_idx = channels_to_analyze

    coords: Dict[int, np.ndarray] = {
        ch1_idx: filter_by_channel(df, ch1_idx, columns=['x', 'y']).to_numpy(),
        ch2_idx: filter_by_channel(df, ch2_idx, columns=['x', 'y']).to_numpy()
    }

    labels: Dict[int, str] = {
        ch1_idx: channel_labels[ch1_idx] if channel_labels and ch1_idx in channel_labels else f"Channel {ch1_idx}",
        ch2_idx: channel_labels[ch2_idx] if channel_labels and ch2_idx in channel_labels else f"Channel {ch2_idx}"
    }

    # Process both directions
    df1 = process_direction(df, primary_idx=ch1_idx, reference_idx=ch2_idx,
                            coords=coords, labels=labels, save_dir=save_dir)
    df2 = process_direction(df, primary_idx=ch2_idx, reference_idx=ch1_idx,
                            coords=coords, labels=labels, save_dir=save_dir)

    RE_per_localization_df = pd.concat([df1, df2], ignore_index=True)

    return RE_per_localization_df