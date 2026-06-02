import os
import re
from pathlib import Path
from typing import Dict, Union

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import napari
import numpy as np
import pandas as pd
from matplotlib import cm
from matplotlib.colors import to_hex
from matplotlib.patches import Patch
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial import Voronoi, voronoi_plot_2d

from dSTORMQuant.utils.logger import get_logger
from dSTORMQuant.utils.utils import filter_by_channel

logger = get_logger()

plt.rcParams["svg.fonttype"] = "none"


def _napari_points_style(face_color) -> dict:
    """Keyword args for Napari points layers without visible marker borders.

    Napari defaults are ``border_color='dimgray'`` and ``border_width_is_relative=True``.
    Passing only ``border_width=0`` is not reliable on all platforms (Linux vs Windows,
    VisPy/OpenGL); borders can still render as a gray halo around dense SMLM data.

    Args:
        face_color: Point fill color (hex string, name, or per-point color sequence).

    Returns:
        Dict of kwargs to pass to :meth:`napari.Viewer.add_points`.
    """
    return {
        "face_color": face_color,
        "border_width": 0,
        "border_width_is_relative": False,
        "border_color": face_color,
        "symbol": "disc",
    }


############# Filtering Analysis Visualization ##############


def sanitize_filename(s: str) -> str:
    """Replace unsafe characters in a string so it can be used as a filename.

    Args:
        s: Raw label or title string.

    Returns:
        Sanitized string containing only alphanumeric characters, underscores,
        and dots.
    """
    return re.sub(r"[^A-Za-z0-9_.]+", "_", s)


def _save_figure_variants(fig, base_path: str, dpi: int = 300) -> list[str]:
    """Save a Matplotlib figure as both PNG and SVG."""
    root, ext = os.path.splitext(base_path)
    if ext.lower() not in {".png", ".svg"}:
        root = base_path

    saved_paths: list[str] = []
    for suffix in ("png", "svg"):
        path = f"{root}.{suffix}"
        save_kwargs = {"bbox_inches": "tight"}
        if suffix == "png":
            save_kwargs["dpi"] = dpi
        fig.savefig(path, **save_kwargs)
        saved_paths.append(path)

    return saved_paths


def plot_metrics(
    df: pd.DataFrame,
    channels: dict[str, tuple[str, str, int]],
    output_dir: str | os.PathLike,
    prefix: str = "initial",
    log_scale_metrics: set[str] | None = None,
) -> pd.DataFrame:
    """
    Generates histograms for various localization metrics using np.histogram (counts),
    saves them as PNGs, and stores summary statistics (mean, std, median, num localizations)
    and highest-frequency bin in a CSV. Uses a single legend per plot showing stats.
    Colors are set based on the `channels` dictionary.

    Args:
        df: DataFrame with localization data
        channels: Channel configuration dict
        output_dir: Output directory for PNG files
        prefix: Prefix for output filenames
        log_scale_metrics: Set of metric keys to plot with log scale (e.g., {"lp", "pvalue", "intensity", "bg"})
                          If None, no metrics use log scale

    Returns:
        DataFrame of per-channel summary statistics written alongside the PNG plots.
    """
    if log_scale_metrics is None:
        log_scale_metrics = set()
    summary_rows = []

    is_raw = prefix == "initial"

    # Collect only available channels
    dfs_to_concat = []
    for _, (_, ch_label, ch_idx) in channels.items():
        ch_df = filter_by_channel(df, ch_idx)
        if not ch_df.empty:
            ch_copy = ch_df.copy()
            ch_copy["Channel"] = ch_label
            dfs_to_concat.append(ch_copy)

    if not dfs_to_concat:
        logger.warning("⚠️ No valid channels found in dataframe. Skipping plots.")
        return pd.DataFrame()

    df = pd.concat(dfs_to_concat, ignore_index=True)

    def _pick_raw_col(*candidates: str) -> str | None:
        for c in candidates:
            if c in df.columns:
                return c
        return None

    # Map internal metric keys -> CSV column (raw) or post-process column name
    if is_raw:
        col_map: dict[str, str | None] = {
            "sx": _pick_raw_col("sigmaX (nm)"),
            "sy": _pick_raw_col("sigmaY (nm)"),
            "bg": _pick_raw_col("background (photons/nm^2)"),
            "intensity": _pick_raw_col("intensity (photons)"),
            "pvalue": _pick_raw_col("p-value", "pvalue"),
            "lp": _pick_raw_col("localization precision (nm)"),
        }
    else:
        col_map = {
            "sx": "sx",
            "sy": "sy",
            "bg": "bg",
            "intensity": "photons",
            "pvalue": "pvalue",
            "lp": "lp",
        }

    # Metrics to plot
    metrics = [
        ("sx", "σx (nm)", "σx (nm)"),
        ("sy", "σy (nm)", "σy (nm)"),
        ("lp", "Localization Precision (nm)", "Localization Precision (nm)"),
        ("pvalue", "pvalue", "pvalue"),
        ("intensity", "Intensity (photons)", "Intensity (photons)"),
        ("bg", "Background (photons/nm²)", "Background (photons/nm²)"),
    ]

    for key, title, xlabel in metrics:
        src = col_map.get(key)
        if src is None or (isinstance(src, str) and src not in df.columns):
            logger.info(f"⏭️ Skipping {title} histogram ({prefix}): column not in input.")
            continue

        fig, ax = plt.subplots(figsize=(10, 7), dpi=300)
        y_max = 0  # For adjusting y-axis limit
        custom_handles = []

        for _, (ch_label, group) in enumerate(df.groupby("Channel")):
            if src not in group.columns:
                continue
            data_vals = pd.to_numeric(group[src], errors="coerce").dropna().values
            if len(data_vals) == 0:
                logger.info(f"⏭️ Skipping {title} for {ch_label}: no valid numeric data.")
                continue

            mean_val = data_vals.mean()
            std_val = data_vals.std()
            median_val = np.median(data_vals)

            # Use log-spaced bins for specified metrics
            if key in log_scale_metrics:
                # Filter out non-positive values for log scale
                valid_vals = data_vals[data_vals > 0]
                if len(valid_vals) == 0:
                    logger.warning(f"⚠️ No valid positive values for {title} ({prefix}). Skipping.")
                    continue
                min_val = max(valid_vals.min(), 1e-10)
                max_val = valid_vals.max()
                bins = np.logspace(np.log10(min_val), np.log10(max_val), 70)
                counts, bin_edges = np.histogram(valid_vals, bins=bins)
            else:
                # Histogram using np.histogram
                counts, bin_edges = np.histogram(data_vals, bins="auto", density=False)
            
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

            # Highest-frequency bin
            max_idx = np.argmax(counts)
            highest_freq_bin = (bin_edges[max_idx], bin_edges[max_idx + 1])

            # Get channel color from channels dictionary
            ch_color = next((v[0] for k, v in channels.items() if v[1] == ch_label))

            # Plot bars without label (legend handled separately)
            ax.bar(
                bin_centers,
                counts,
                width=np.diff(bin_edges),
                alpha=0.6,
                color=ch_color,
                edgecolor="none",
            )
            y_max = max(y_max, counts.max())

            # Prepare custom legend handle with stats
            handle = Patch(
                facecolor=ch_color,
                alpha=0.6,
                label=f"{ch_label}: μ={mean_val:.4f}, σ={std_val:.4f}, Md={median_val:.4f}",
            )
            custom_handles.append(handle)

            # Add to summary table
            summary_rows.append(
                {
                    "PlotName": f"{sanitize_filename(key)}_{prefix}",
                    "Channel": ch_label,
                    "Num Localizations": len(data_vals),
                    "Mean": mean_val,
                    "Std": std_val,
                    "Median": median_val,
                    "Highest Freq Bin": f"[{highest_freq_bin[0]:.4f}, {highest_freq_bin[1]:.4f}]",
                }
            )

        if not ax.patches:  # skip if no data to plot
            logger.info(f"⏭️ Skipping {title} histogram (no data).")
            continue

        axis_label = f"{xlabel} [log scale]" if key in log_scale_metrics else xlabel
        ax.set_xlabel(axis_label)
        ax.set_ylabel("Frequency")
        ax.set_title(f"{title} Histogram - {prefix}")
        ax.set_ylim(0, y_max * 1.1)
        
        # Apply log scale to x-axis if metric is in log_scale_metrics
        if key in log_scale_metrics:
            ax.set_xscale("log")
        
        ax.grid(True, linestyle="--", alpha=0.4)

        # Add single combined legend
        ax.legend(handles=custom_handles, fontsize=10)

        output_path = os.path.join(output_dir, f"{sanitize_filename(key)}_{prefix}.png")
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close(fig)
        logger.info(f"✅ Saved PNG: {output_path}")

    summary_stats = pd.DataFrame(summary_rows)
    logger.info("🎉 All PNG plots and CSV summary generated successfully.")

    return summary_stats


############# General Visualization ##############


def visualize_napari(
    df: pd.DataFrame, channels: dict[str, tuple], *, use_napari: bool = True
) -> None:
    """Open an interactive Napari viewer for one or two localization channels.

    Args:
        df: Localization dataframe with coordinate columns.
        channels: Mapping channel key -> ``(color, label, channel_index)``.
        use_napari: If ``False``, log and return without opening a viewer.

    Returns:
        None
    """
    if not use_napari:
        logger.info(
            "visualization.use_napari is false — skipping Napari "
        )
        return
    rename_map = {}
    if "x (nm)" in df.columns:
        rename_map["x (nm)"] = "x"
    if "y (nm)" in df.columns:
        rename_map["y (nm)"] = "y"
    if rename_map:
        df = df.rename(columns=rename_map)

    viewer = napari.Viewer()
    viewer.scale_bar.visible = True
    viewer.scale_bar.unit = "nm"

    for _, (color, label, ch_index) in channels.items():
        coords = filter_by_channel(df, ch_index, columns=["y", "x"]).to_numpy()
        if coords.size > 0:
            viewer.add_points(
                coords,
                name=label,
                size=5,
                **_napari_points_style(color),
            )
        else:
            logger.warning(
                f"Channel {label} has no localizations, skipping visualization."
            )

    napari.run()


def save_napari_points_screenshot(
    df: pd.DataFrame,
    channels: dict[str, tuple],
    output_dir: str | os.PathLike,
    filename: str,
    scale: int = 1,
    first: bool = False,
    legend_pos: tuple[int, int] = (10, 10),
    font_path: str | None = None,
    font_size: int = 20,
    *,
    use_napari: bool = True,
) -> None:
    """Save a Napari screenshot with per-channel localization counts in a legend.

    Args:
        df: Localization dataframe for the current pipeline step.
        channels: Mapping channel key -> ``(color, label, channel_index)``.
        output_dir: Directory for the output PNG.
        filename: Output image filename.
        scale: Screenshot scale factor passed to Napari.
        first: If ``True``, read ``x (nm)``/``y (nm)`` columns instead of ``x``/``y``.
        legend_pos: ``(x, y)`` pixel offset for the legend text.
        font_path: Optional TrueType font for the legend.
        font_size: Legend font size in points.
        use_napari: If ``False``, skip rendering (headless runs).

    Returns:
        None
    """
    if not use_napari:
        logger.info(
            f"Skipping Napari point screenshot {filename!r} (visualization.use_napari: false)."
        )
        return

    x_col, y_col = ("x (nm)", "y (nm)") if first else ("x", "y")

    viewer = napari.Viewer(show=True)
    viewer.scale_bar.visible = True
    viewer.scale_bar.unit = "nm"
    viewer.scale_bar.position = "bottom_right"

    # Collect coordinates and legend info
    legend_texts = []
    for _, (color, label, ch_idx) in channels.items():
        coords = filter_by_channel(df, ch_idx, columns=[y_col, x_col]).to_numpy()
        if len(coords):
            viewer.add_points(
                coords,
                name=label,
                size=5,
                **_napari_points_style(color),
            )
        legend_texts.append((label, len(coords), color))

    viewer.reset_view()

    screenshot_path = os.path.join(output_dir, filename)
    screenshot = viewer.screenshot(canvas_only=True, scale=scale)
    viewer.close()

    # Draw legend on image
    image = Image.fromarray(screenshot)
    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype(font_path if font_path else "arial.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()

    x, y = legend_pos
    for i, (label, count, color) in enumerate(legend_texts):
        draw.text(
            (x, y + i * (font_size + 10)),
            f"{label}: {count} locs",
            fill=color,
            font=font,
        )

    image.save(screenshot_path)
    logger.info(f"Screenshot with legend saved at: {screenshot_path}")


def create_stacked_histogram(output_folder: str | os.PathLike, metric: str) -> None:
    """Stack per-step metric histogram PNGs into one vertical composite image.

    Args:
        output_folder: Root folder containing step subfolders with ``{metric}_*.png`` files.
        metric: Metric key used in filenames (e.g. ``photons``, ``lp``).

    Returns:
        None
    """
    # The plot_metrics function saves files named as: "{metric}_{prefix}.png"
    # where prefix is e.g. 'initial', 'after_drift_correction', 'after_sigma_filter', etc.
    steps = [
        ("initial", f"{metric}_initial.png"),
        ("drift_corrected", f"{metric}_after_drift_correction.png"),
        ("filtered", f"{metric}_after_sigma_filter.png"),
        ("filtered", f"{metric}_after_photons_count_filter.png"),
        ("filtered", f"{metric}_after_localization_precision_filter.png"),
        ("filtered", f"{metric}_after_pvalue_filter.png"),
        ("temporal_grouped", f"{metric}_after_temporal_grouping.png"),
        ("cell_detected", f"{metric}_after_cell_detection.png"),
    ]

    images = []
    for folder, filename in steps:
        path = os.path.join(output_folder, folder, filename)
        if os.path.exists(path):
            try:
                img = Image.open(path).convert("RGB")
                images.append(img)
            except Exception as e:
                logger.warning(f"⚠️ Failed to open image {path}: {e}")
        else:
            logger.debug(f"File not found (will skip): {path}")

    if not images:
        logger.warning(f"❌ No images found for metric '{metric}' to stack.")
        return

    widths, heights = zip(*(img.size for img in images), strict=False)
    total_height = sum(heights)
    max_width = max(widths)

    stacked_img = Image.new("RGB", (max_width, total_height), color=(255, 255, 255))
    y_offset = 0
    for img in images:
        stacked_img.paste(img, (0, y_offset))
        y_offset += img.size[1]

    output_path = os.path.join(output_folder, f"{metric}_hist_all_steps.png")
    try:
        stacked_img.save(output_path)
        logger.info(f"✅ Stacked histogram saved: {output_path}")
    except Exception as e:
        logger.error(f"Failed to save stacked histogram '{output_path}': {e}")


def create_stacked_napari_images(
    output_folder: str | os.PathLike,
    output_filename: str,
    font_path: str | None = None,
) -> None:
    """Stack Napari step screenshots vertically with step titles.

    Args:
        output_folder: Directory containing ``napari_view_*.png`` files.
        output_filename: Combined image filename written under ``output_folder``.
        font_path: Optional TrueType font for step titles.

    Returns:
        None
    """
    images = []
    steps = [
        ("initial", "napari_input_view.png", "0. Initial"),
        (
            "drift_corrected",
            "napari_view_after_drift_correction.png",
            "1. After Drift Correction",
        ),
        ("filtered", "napari_view_after_sigma_filter.png", "2. After Sigma Filter"),
        (
            "filtered",
            "napari_view_after_photons_count_filter.png",
            "3. After Photon Count Filter",
        ),
        (
            "filtered",
            "napari_view_after_localization_precision_filter.png",
            "4. After Localization Precision Filter",
        ),
        ("filtered", "napari_view_after_pvalue_filter.png", "5. After P-Value Filter"),
        (
            "temporal_grouped",
            "napari_view_after_temporal_grouping.png",
            "6. After Temporal Grouping",
        ),
        (
            "cell_detected",
            "napari_view_after_cell_detection.png",
            "7. After Cell Detection",
        ),
    ]

    for folder, filename, title in steps:
        path = os.path.join(output_folder, folder, filename)
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
        logger.info(f"✅ Saved high-quality image: {save_path}")
    else:
        logger.warning("❌ No valid images found to stack.")


############# Nearest Neighbors Analysis Visualization ##############


def plot_mean_distance_histogram(
    distances: np.ndarray,
    title: str,
    output_path: str | Path,
    color: str,
    log_scale: bool = False,
    radius: float = 100,
) -> pd.DataFrame:
    """
    Plot histogram of mean distances using np.histogram (counts, no density) and highlight
    the highest-frequency interval. Calculates mean, std, median, and interval.

    Parameters
    ----------
    distances : np.ndarray
        Array of distances.
    title : str
        Plot title.
    output_path : str or Path
        Path to save PNG.
    log_scale : bool
        If True, use log scale on x-axis.
    radius : float
        Radius used in analysis (for labeling).
    color : str
        Hex code or matplotlib color for histogram bars.

    Returns
    -------
    pd.DataFrame
        Summary statistics including highest-frequency interval.
    """
    # Clean distances
    distances = distances[~np.isnan(distances)]
    if len(distances) == 0:
        logger.warning(f"No valid distances to plot for {title}. Skipping.")
        return pd.DataFrame()

    # For log scale, ensure all values are positive
    if log_scale:
        distances = distances[distances > 0]
        if len(distances) == 0:
            logger.warning(f"No positive distances to plot for {title}. Skipping.")
            return pd.DataFrame()

    # Compute stats
    mu = np.mean(distances)
    sigma = np.std(distances)
    median_val = np.median(distances)

    num_bins = 70
    # Create appropriate bins based on scale
    if log_scale:
        # Use log-spaced bins for log scale
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        bins = np.logspace(np.log10(min_dist), np.log10(max_dist), num_bins)
    else:
        # Use auto bins for linear scale
        bins = "auto"

    # Compute histogram
    counts, bin_edges = np.histogram(distances, bins=bins, density=False)

    # Filter out empty bins to avoid visualization issues
    non_zero_mask = counts > 0
    counts = counts[non_zero_mask]

    # Use proper bin centers and widths
    if len(bin_edges) > 1:
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_centers = bin_centers[non_zero_mask]
        bin_widths = np.diff(bin_edges)[non_zero_mask]
    else:
        # Fallback for single bin
        bin_centers = bin_edges[:1]
        bin_widths = [bin_edges[0] * 0.1] if len(bin_edges) > 0 else [1.0]

    if len(counts) == 0:
        logger.warning(f"No valid bins to plot for {title}. Skipping.")
        return pd.DataFrame()

    # Highest-frequency bin
    max_idx = np.argmax(counts)
    highest_freq_bin = (bin_edges[max_idx], bin_edges[max_idx + 1])

    # Plot
    fig, ax = plt.subplots(figsize=(10, 7), dpi=300)

    # Use bar plot with calculated widths and proper centers
    ax.bar(
        bin_centers - bin_widths / 2,
        counts,
        width=bin_widths,
        color=color,
        edgecolor="none",
        alpha=0.7,
        align="edge",
    )

    if log_scale:
        ax.set_xscale("log")

    ax.set_xlabel(
        f"Mean Displacement (r = {radius} nm) (nm)"
        + (" [log scale]" if log_scale else ""),
        fontsize=12,
    )
    ax.set_ylabel("Counts", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, linestyle="--", alpha=0.4)

    # Set appropriate x-axis limits
    if len(bin_edges) > 0:
        if log_scale:
            ax.set_xlim(bin_edges[0] * 0.9, bin_edges[-1] * 1.1)
        else:
            ax.set_xlim(
                bin_edges[0] - bin_widths[0] * 0.1, bin_edges[-1] + bin_widths[-1] * 1.1
            )

    # Annotate stats and highest-frequency bin
    stats_text = (
        f"μ = {mu:.2f} nm\n"
        f"σ = {sigma:.2f} nm\n"
        f"Md = {median_val:.2f} nm\n"
        f"Peak: [{highest_freq_bin[0]:.2f}, {highest_freq_bin[1]:.2f}]"
    )
    ax.text(
        0.95,
        0.95,
        stats_text,
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment="top",
        horizontalalignment="right",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.7},
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    logger.info(f"📊 Saved mean distance histogram: {output_path}")

    # Return summary stats
    summary_stats = pd.DataFrame(
        [
            {
                "PlotName": Path(output_path).stem,
                "Mean": mu,
                "Std": sigma,
                "Median": median_val,
                "Highest Freq Interval": f"[{highest_freq_bin[0]:.2f}, {highest_freq_bin[1]:.2f}]",
            }
        ]
    )

    return summary_stats


def plot_distance_histogram(
    distances: np.ndarray,
    title: str,
    output_path: str | Path,
    color: str,
    log_scale: bool = False,
) -> pd.DataFrame:
    """
    Plot histogram of nearest neighbor distances using np.histogram (counts, no density),
    annotate μ, σ, Md, and highlight the highest-frequency interval.

    Parameters
    ----------
    distances : np.ndarray
        Array of distances.
    title : str
        Plot title.
    output_path : str or Path
        Path to save PNG.
    log_scale : bool
        If True, x-axis is log scale.
    color : str
        Hex code or matplotlib color for histogram bars.

    Returns
    -------
    pd.DataFrame
        Summary statistics including highest-frequency interval.
    """
    # Filter invalid distances
    distances = distances[~np.isnan(distances)]
    if len(distances) == 0:
        logger.warning(f"No valid distances to plot for {title}. Skipping.")
        return pd.DataFrame()

    # For log scale, ensure all values are positive
    if log_scale:
        distances = distances[distances > 0]
        if len(distances) == 0:
            logger.warning(f"No positive distances to plot for {title}. Skipping.")
            return pd.DataFrame()

    # Compute stats
    mu = np.mean(distances)
    sigma = np.std(distances)
    median_val = np.median(distances)

    # Create appropriate bins based on scale
    if log_scale:
        # Use log-spaced bins for log scale
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        bins = np.logspace(np.log10(min_dist), np.log10(max_dist), 70)
    else:
        # Use auto bins for linear scale
        bins = "auto"

    # Histogram
    counts, bin_edges = np.histogram(distances, bins=bins, density=False)

    # Filter out empty bins to avoid visualization issues
    non_zero_mask = counts > 0
    counts = counts[non_zero_mask]

    # Use proper bin centers and widths
    if len(bin_edges) > 1:
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_centers = bin_centers[non_zero_mask]
        bin_widths = np.diff(bin_edges)[non_zero_mask]
    else:
        # Fallback for single bin
        bin_centers = bin_edges[:1]
        bin_widths = [bin_edges[0] * 0.1] if len(bin_edges) > 0 else [1.0]

    if len(counts) == 0:
        logger.warning(f"No valid bins to plot for {title}. Skipping.")
        return pd.DataFrame()

    # Highest-frequency bin - use original bin_edges for accurate interval
    max_idx = np.argmax(counts)
    highest_freq_bin = (bin_edges[max_idx], bin_edges[max_idx + 1])

    # Plot
    fig, ax = plt.subplots(figsize=(10, 7), dpi=300)

    # Use bar plot with calculated widths and proper centers
    ax.bar(
        bin_centers - bin_widths / 2,
        counts,
        width=bin_widths,
        color=color,
        edgecolor="none",
        alpha=0.7,
        align="edge",
    )

    if log_scale:
        ax.set_xscale("log")

    ax.set_xlabel(
        "Min. Displacement (nm)" + (" (log scale)" if log_scale else ""), fontsize=12
    )
    ax.set_ylabel("Counts", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, linestyle="--", alpha=0.4)

    # Set appropriate x-axis limits
    if len(bin_edges) > 0:
        if log_scale:
            ax.set_xlim(bin_edges[0], bin_edges[-1])
        else:
            ax.set_xlim(bin_edges[0], bin_edges[-1])

    # Annotate stats and highest-frequency bin
    stats_text = (
        f"μ = {mu:.2f} nm\n"
        f"σ = {sigma:.2f} nm\n"
        f"Md = {median_val:.2f} nm\n"
        f"Peak: [{highest_freq_bin[0]:.2f}, {highest_freq_bin[1]:.2f}]"
    )
    ax.text(
        0.95,
        0.95,
        stats_text,
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment="top",
        horizontalalignment="right",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.7},
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    logger.info(f"📊 Saved nearest neighbor histogram: {output_path}")

    # Return summary stats
    summary_stats = pd.DataFrame(
        [
            {
                "PlotName": Path(output_path).stem,
                "Mean": mu,
                "Std": sigma,
                "Median": median_val,
                "Highest Freq Interval": f"[{highest_freq_bin[0]:.2f}, {highest_freq_bin[1]:.2f}]",
            }
        ]
    )

    return summary_stats


############# Cluster Visualization ##############


def plot_histogram_cluster_stats(
    dataframes: list[pd.DataFrame],
    channels: dict[str, tuple[str, str, int]],
    save_dir: str | os.PathLike,
    log_scale: bool = False,
) -> pd.DataFrame:
    """
    Plots and saves histograms for each cluster feature and channel using matplotlib + np.histogram.
    Highlights the highest-probable histogram interval.
    """
    summary_rows = []

    features = [
        ("area_nm2", "Cluster Area (nm²)"),
        ("density_points_per_nm2", "Cluster Density (points/nm²)"),
    ]

    if not isinstance(dataframes, list):
        dataframes = [dataframes]

    # Map channel index to (channel label, color)
    index_to_label_color = {v[2]: (v[1], v[0]) for v in channels.values()}

    for df in dataframes:
        if df.empty:
            logger.warning("Empty DataFrame provided, skipping plotting.")
            continue

        for ch_index in df["channelIndex"].unique():
            df_ch = filter_by_channel(df, ch_index)
            if df_ch.empty:
                continue

            channel_label, color = index_to_label_color.get(
                ch_index, (f"Channel_{ch_index}", "dodgerblue")
            )
            channel_name = channel_label.replace(" ", "_")

            for feature, label in features:
                if feature not in df_ch.columns:
                    logger.warning(f"Feature '{feature}' not in DataFrame, skipping.")
                    continue

                values = df_ch[feature].dropna().values
                if len(values) == 0:
                    continue

                use_log_scale = log_scale or (feature == "area_nm2")
                xlabel = label + (" (log scale)" if use_log_scale else "")

                # Compute statistics
                mu = np.mean(values)
                sigma = np.std(values)
                median_val = np.median(values)

                # Create appropriate bins based on scale
                if use_log_scale:
                    # Filter positive values for log scale
                    values = values[values > 0]
                    if len(values) == 0:
                        continue
                    # Use log-spaced bins for log scale
                    min_val = np.min(values)
                    max_val = np.max(values)
                    bins = np.logspace(np.log10(min_val), np.log10(max_val), 70)
                else:
                    # Use auto bins for linear scale
                    bins = "auto"

                # Compute histogram using np.histogram
                counts, bin_edges = np.histogram(values, bins=bins, density=False)

                # Filter out empty bins to avoid visualization issues
                non_zero_mask = counts > 0
                counts = counts[non_zero_mask]

                # Use proper bin centers and widths
                if len(bin_edges) > 1:
                    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                    bin_centers = bin_centers[non_zero_mask]
                    bin_widths = np.diff(bin_edges)[non_zero_mask]
                else:
                    # Fallback for single bin
                    bin_centers = bin_edges[:1]
                    bin_widths = [bin_edges[0] * 0.1] if len(bin_edges) > 0 else [1.0]

                if len(counts) == 0:
                    continue

                # Highest-frequency bin - use original bin_edges for accurate interval
                max_idx = np.argmax(counts)
                highest_frequency_bin = (bin_edges[max_idx], bin_edges[max_idx + 1])

                # Plot using matplotlib
                plt.figure(figsize=(10, 7))

                # Use bar plot with calculated widths and proper centers
                plt.bar(
                    bin_centers - bin_widths / 2,
                    counts,
                    width=bin_widths,
                    color=color,
                    edgecolor="none",
                    alpha=0.7,
                    align="edge",
                )

                if use_log_scale:
                    plt.xscale("log")

                stats_text = (
                    f"μ = {mu:.4f}\n"
                    f"σ = {sigma:.4f}\n"
                    f"Md = {median_val:.4f}\n"
                    f"Peak: [{highest_frequency_bin[0]:.4f}, {highest_frequency_bin[1]:.4f}]"
                )

                plt.xlabel(xlabel)
                plt.ylabel("Frequency")
                plt.title(f"{channel_label} - {label}", wrap=True)
                plt.grid(True)

                # Set appropriate x-axis limits
                if len(bin_edges) > 0:
                    if use_log_scale:
                        plt.xlim(bin_edges[0] * 0.9, bin_edges[-1] * 1.1)
                    else:
                        plt.xlim(
                            bin_edges[0] - bin_widths[0] * 0.1,
                            bin_edges[-1] + bin_widths[-1] * 1.1,
                        )

                plt.text(
                    0.95,
                    0.95,
                    stats_text,
                    transform=plt.gca().transAxes,
                    fontsize=12,
                    verticalalignment="top",
                    horizontalalignment="right",
                    bbox={
                        "boxstyle": "round,pad=0.3",
                        "facecolor": "white",
                        "alpha": 0.7,
                    },
                )

                # Save plot
                filename = f"{feature}_hist_{channel_name.lower()}.png"
                save_path = os.path.join(save_dir, filename)
                plt.tight_layout()
                plt.savefig(save_path, dpi=300)
                plt.close()
                logger.info(f"📊 Saved: {save_path}")

                # Save summary
                summary_rows.append(
                    {
                        "PlotName": filename,
                        "Channel": channel_name,
                        "Feature": feature,
                        "Mean": mu,
                        "Std": sigma,
                        "Median": median_val,
                        "High Frequency Bin": f"[{highest_frequency_bin[0]:.4f}, {highest_frequency_bin[1]:.4f}]",
                    }
                )

    return pd.DataFrame(summary_rows)


def plot_cluster_histogram_log_npoints(
    dataframes: list[pd.DataFrame],
    save_dir: str | os.PathLike,
    channels: dict[str, tuple[str, str, int]],
) -> pd.DataFrame:
    """
    Plots histogram of cluster n_points for each channel with log-scaled x-axis (for visualization only).
    Uses channel colors and labels from the `channels` dictionary.
    Also saves summary stats (μ, σ, Md, Mo, Highest_Prob_Interval) to a DataFrame.
    """
    combined_df = pd.concat(
        [df for df in dataframes if not df.empty], ignore_index=True
    )
    if combined_df.empty:
        logger.warning("No data to plot in plot_cluster_histogram_log_npoints.")
        return pd.DataFrame()

    channel_colors = {info[1]: info[0] for info in channels.values()}

    filename = "npoints_hist_clustering.png"
    save_path = os.path.join(save_dir, filename)

    plt.figure(figsize=(10, 6))

    summary_rows = []

    for i, (ch_name, group_df) in enumerate(combined_df.groupby("channel_name")):
        values = group_df["n_points"].dropna().values
        if len(values) == 0:
            mu = sigma = median_val = np.nan
            highest_frequency_interval_str = "[nan, nan]"
            counts = np.array([])
            bin_edges = np.array([])
        else:
            # Filter positive values for log scale
            values = values[values > 0]
            if len(values) == 0:
                mu = sigma = median_val = np.nan
                highest_frequency_interval_str = "[nan, nan]"
                counts = np.array([])
                bin_edges = np.array([])
            else:
                # Compute statistics
                mu = np.mean(values)
                sigma = np.std(values)
                median_val = np.median(values)

                # Create log-spaced bins for log scale
                min_val = np.min(values)
                max_val = np.max(values)
                bins = np.logspace(np.log10(min_val), np.log10(max_val), 70)

                # Histogram using np.histogram
                counts, bin_edges = np.histogram(values, bins=bins, density=False)

                # Filter out empty bins
                non_zero_mask = counts > 0
                counts = counts[non_zero_mask]

                # Use proper bin centers and widths
                if len(bin_edges) > 1:
                    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                    bin_centers = bin_centers[non_zero_mask]
                    bin_widths = np.diff(bin_edges)[non_zero_mask]
                else:
                    # Fallback for single bin
                    bin_centers = bin_edges[:1]
                    bin_widths = [bin_edges[0] * 0.1] if len(bin_edges) > 0 else [1.0]

                if len(counts) > 0:
                    # Highest-frequency bin - use original bin_edges for accurate interval
                    max_idx = np.argmax(counts)
                    highest_frequency_interval = (
                        bin_edges[max_idx],
                        bin_edges[max_idx + 1],
                    )
                    highest_frequency_interval_str = f"[{highest_frequency_interval[0]:.2f}, {highest_frequency_interval[1]:.2f}]"

                    # Plot using matplotlib bar with proper centers and widths
                    plt.bar(
                        bin_centers - bin_widths / 2,
                        counts,
                        width=bin_widths,
                        color=channel_colors[ch_name],
                        alpha=0.6,
                        edgecolor="black",
                        label=ch_name,
                        align="edge",
                    )
                else:
                    highest_frequency_interval_str = "[nan, nan]"

        # Save summary stats
        summary_rows.append(
            {
                "PlotName": filename,
                "Channel": ch_name,
                "Feature": "number of localizations in cluster",
                "Mean": mu,
                "Std": sigma,
                "Median": median_val,
                "High Frequency Bin": highest_frequency_interval_str,
            }
        )

        # Stats text box - only if we have valid data
        if len(values) > 0 and len(counts) > 0:
            stats_text = (
                f"{ch_name}:\n"
                f"μ = {mu:.2f}\n"
                f"σ = {sigma:.2f}\n"
                f"Md = {median_val:.2f}\n"
                f"Peak: {highest_frequency_interval_str}"
            )
            ax = plt.gca()
            y_frac = 0.95 - (i * 0.14)
            if y_frac < 0.05:
                y_frac = 0.05
            ax.text(
                1.02,
                y_frac,
                stats_text,
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment="top",
                horizontalalignment="left",
                bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.8},
                color=channel_colors[ch_name],
            )

    plt.xscale("log")  # only for visualization
    plt.title("Histogram of Cluster Size (n_points) per Channel", wrap=True)
    plt.xlabel("Number of Points in Cluster [log scale]")
    plt.ylabel("Frequency")
    plt.legend(loc="lower left", fontsize=10)

    # Set appropriate x-axis limits
    ax = plt.gca()
    if "bin_edges" in locals() and len(bin_edges) > 0:
        ax.set_xlim(bin_edges[0] * 0.9, bin_edges[-1] * 1.1)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"📊 Saved: {save_path}")

    return pd.DataFrame(summary_rows)


def _cluster_color_map(valid_clusters: list) -> dict:
    num_clusters = len(valid_clusters)
    if num_clusters == 1:
        return {valid_clusters[0]: "#1f77b4"}
    if num_clusters <= 20:
        cmap = cm.get_cmap("tab20")
        return {c: to_hex(cmap(i)) for i, c in enumerate(valid_clusters)}
    cluster_min, cluster_max = min(valid_clusters), max(valid_clusters)
    cmap = cm.get_cmap("viridis")
    return {
        c: to_hex(cmap((c - cluster_min) / max(cluster_max - cluster_min, 1)))
        for c in valid_clusters
    }


def visualize_clusters(
    df: pd.DataFrame,
    save_dir: str | os.PathLike,
    channels: dict,
    show: bool = False,
    *,
    use_napari: bool = True,
) -> None:
    """
    Visualizes localization points for one or more channels with per-cluster coloring.
    Saves combined and per-channel images using channel labels.
    Works with single or multiple channels.
    """
    if not use_napari:
        logger.info(
            "Skipping Napari cluster screenshots (visualization.use_napari: false)."
        )
        if show:
            logger.warning(
                "visualize_clusters: show=True has no effect when use_napari=False."
            )
        return

    viewer = napari.Viewer()
    viewer.scale_bar.visible = True
    viewer.scale_bar.unit = "nm"

    # Map channel index to label
    index_to_label = {v[2]: v[1] for v in channels.values()}

    # Add points for all channels present in df
    for ch in sorted(df["channelIndex"].unique()):
        df_ch = df[(df["channelIndex"] == ch) & (df["cluster"] != -1)]
        if df_ch.empty:
            continue

        coords = df_ch[["y", "x"]].to_numpy()
        valid_clusters = sorted(df_ch["cluster"].unique())
        cluster_colors = _cluster_color_map(valid_clusters)
        point_colors = df_ch["cluster"].map(cluster_colors).tolist()
        layer_name = index_to_label.get(
            ch, f"Channel {ch}"
        )  # fallback if label missing
        viewer.add_points(
            coords,
            size=5,
            name=layer_name,
            **_napari_points_style(point_colors),
        )

    # Save combined view
    combined_path = os.path.join(save_dir, "clusters_napari_view.png")
    viewer.reset_view()
    viewer.screenshot(combined_path, canvas_only=True)
    logger.info(f"🖼️ Saved combined cluster view: {combined_path}")

    # Save per-channel views using channel label (only for channels with layers)
    # Get list of layer names that actually exist in the viewer
    existing_layer_names = [layer.name for layer in viewer.layers]

    for ch in sorted(df["channelIndex"].unique()):
        layer_name = index_to_label.get(ch, f"Channel {ch}")

        # Skip if no layer exists for this channel (e.g., all data was filtered out or no clusters)
        if layer_name not in existing_layer_names:
            logger.warning(
                f"⚠️ Skipping screenshot for {layer_name}: no clustered data found after filtering"
            )
            continue

        for layer in viewer.layers:
            layer.visible = False
        viewer.layers[layer_name].visible = True
        per_channel_path = os.path.join(
            save_dir, f"{layer_name}_clusters_napari_view.png"
        )
        viewer.reset_view()
        viewer.screenshot(per_channel_path, canvas_only=True)
        logger.info(f"🖼️ Saved channel {layer_name} cluster view: {per_channel_path}")

    # Restore visibility
    for layer in viewer.layers:
        layer.visible = True

    if show:
        logger.info("Launching napari viewer...")
        napari.run()
    else:
        viewer.close()
        logger.info("Napari viewer closed.")


###################### Drift Correction #########################
def plot_drift_data(
    drift_file_path: str, output_dir: Path, base: str, pixel_size: float
) -> None:
    """
    Plot drift data from the AIM drift correction text file.

    Args:
        drift_file_path (str): Path to the drift correction text file
        output_dir (str): Directory to save the plot
        base (str): Base name for the output plot file
        pixel_size (float): Pixel size in nm to convert drift values
    """
    try:
        logger.info(f"Loading drift data from: {drift_file_path}")
        drift_data = np.loadtxt(drift_file_path, skiprows=1)  # Skip header line
        x_values: np.ndarray = drift_data[:, 0] * pixel_size
        y_values: np.ndarray = drift_data[:, 1] * pixel_size
        time_points: np.ndarray = np.arange(len(x_values))

        fig, ax = plt.subplots(figsize=(12, 8), constrained_layout=True)
        ax.plot(time_points, x_values, "b-", linewidth=2, label="x (X-direction)")
        ax.plot(time_points, y_values, "r-", linewidth=2, label="y (Y-direction)")
        ax.set_xlabel("Time Frame")
        ax.set_ylabel("Drift (nm)")
        ax.set_title("Drift Correction - X and Y Directions", wrap=True)
        ax.grid(True, alpha=0.3)
        ax.legend()

        plot_dir: str = os.path.join(output_dir, "drift_corrected")
        plot_path: str = os.path.join(plot_dir, f"drift_correction_plot_{base}.png")
        saved_paths = _save_figure_variants(fig, plot_path)
        plt.close(fig)

        logger.info(f"Drift correction plot saved at: {', '.join(saved_paths)}")

    except Exception as e:
        logger.error(f"Error plotting drift data: {e}", exc_info=True)


###################### Cell Detection #########################


def plot_voronoi_cells(
    df_all: pd.DataFrame,
    df_cells: pd.DataFrame,
    vor: Voronoi,
    channels: dict[str, tuple[str, str, int]] | None = None,
    save_path: str | None = None,
    figsize: tuple[int, int] = (6, 6),
):
    """Plot Voronoi tessellation with detected cell points highlighted.

    Args:
        df_all: All localizations (background, grey).
        df_cells: Localizations marked as cell (``is_cell``).
        vor: Voronoi diagram from cell detection.
        channels: Optional channel color map for multi-channel overlays.
        save_path: Base path for PNG/SVG output (extension added by helper).
        figsize: Matplotlib figure size in inches.

    Returns:
        None
    """
    if df_all.empty or df_cells.empty or vor is None:
        logger.info("Warning: Empty data or Voronoi object for plotting")
        return

    fig, ax = plt.subplots(figsize=figsize)

    # Voronoi lines
    voronoi_plot_2d(
        vor,
        ax=ax,
        show_vertices=False,
        line_colors="grey",
        line_width=0.5,
        line_alpha=0.5,
        point_size=0,
    )

    # Plot all points in light grey
    ax.scatter(df_all["x"], df_all["y"], c="lightgrey", s=10, label="Background")

    # Plot filtered cell points per channel
    if channels:
        for _, (color, label, ch_index) in channels.items():
            if "channelIndex" in df_cells:
                df_ch = df_cells[df_cells["channelIndex"] == ch_index]
            else:
                df_ch = df_cells  # fallback if no channelIndex
            if not df_ch.empty:
                ax.scatter(df_ch["x"], df_ch["y"], c=color, s=25, label=label)
    else:
        ax.scatter(df_cells["x"], df_cells["y"], c="red", s=25, label="Cell")

    ax.set_title("Voronoi-based Background Removal")
    ax.set_xlabel("X (nm)")
    ax.set_ylabel("Y (nm)")
    ax.invert_yaxis()
    ax.legend()
    plt.tight_layout()

    if save_path:
        saved_paths = _save_figure_variants(fig, save_path)
        logger.info(f"Voronoi plot saved to: {', '.join(saved_paths)}")
    plt.close()


def plot_cell_detection(
    df_all: pd.DataFrame,
    df_cells: pd.DataFrame,
    bin_size: int,
    channels: dict[str, tuple[str, str, int]],
    output_folder_path: str,
    file_name: str,
    log_scale: bool = True,
):
    """
    Plot 2D density map of localizations and overlay detected cell regions.

    Args:
        df_all (pd.DataFrame): Original dataframe with all points.
        df_cells (pd.DataFrame): Filtered dataframe containing only points in the cell.
        bin_size (int): Number of bins along each axis for 2D histogram.
        channels (dict): Channel info for plotting, e.g., {'Ch1': ('red', 'Label1', 0)}.
        output_folder_path (str): Folder to save plots.
        file_name (str): Base filename for saved plots.
        log_scale (bool, optional): Apply log scale to 2D histogram.
    """

    # Compute bin edges
    x_edges = np.linspace(df_all["x"].min(), df_all["x"].max(), bin_size + 1)
    y_edges = np.linspace(df_all["y"].min(), df_all["y"].max(), bin_size + 1)

    # 2D histogram of all points
    hist, _, _ = np.histogram2d(df_all["y"], df_all["x"], bins=[y_edges, x_edges])
    hist_to_plot = np.log1p(hist) if log_scale else hist

    # Plot 2D density map
    fig, ax = plt.subplots(figsize=(6, 8), constrained_layout=True)
    ax.imshow(
        hist_to_plot,
        extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
        cmap="Greys",
        origin="lower",
        alpha=0.5,
        aspect="auto",
    )

    # Overlay cell points per channel
    for _, (color, label, ch_index) in channels.items():
        df_ch = df_cells[df_cells["channelIndex"] == ch_index]
        if df_ch.empty:
            continue

        # Compute bin indices dynamically
        x_bins = np.digitize(df_ch["x"], x_edges) - 1
        y_bins = np.digitize(df_ch["y"], y_edges) - 1
        ch_bins = set(zip(x_bins, y_bins, strict=False))  # unique bins

        ax.scatter(
            [(x_edges[x] + x_edges[x + 1]) / 2 for x, y in ch_bins],
            [(y_edges[y] + y_edges[y + 1]) / 2 for x, y in ch_bins],
            c=color,
            s=25,
            label=f"{label}",
        )

    ax.set_xlim(x_edges[0], x_edges[-1])
    ax.set_ylim(y_edges[0], y_edges[-1])
    ax.invert_yaxis()
    ax.set_title("Filtered Cell Localization Map")
    ax.legend()
    plot_path = os.path.join(output_folder_path, f"CD_{file_name}.png")
    saved_paths = _save_figure_variants(fig, plot_path)
    plt.close(fig)
    logger.info(f"Filtered cell localization map saved to: {', '.join(saved_paths)}")

    # 1D histogram of counts per bin
    counts = hist.flatten()
    mean_val = np.mean(counts)
    std_val = np.std(counts)
    median_val = np.median(counts)

    fig_hist, ax_hist = plt.subplots(figsize=(6, 5), constrained_layout=True)
    ax_hist.hist(counts, bins=30, color="gray", edgecolor="none", density=True)
    ax_hist.set_xlabel("Counts per bin")
    ax_hist.set_ylabel("Probability")
    ax_hist.set_title("Histogram of localizations per bin")
    legend_text = f"μ={mean_val:.4f}, σ={std_val:.4f}\nMd={median_val:.4f}"
    ax_hist.legend([legend_text], loc="upper right")
    hist_path = os.path.join(output_folder_path, f"CD_hist_{file_name}.png")
    saved_hist_paths = _save_figure_variants(fig_hist, hist_path)
    plt.close(fig_hist)
    logger.info(f"Histogram of bin counts saved to: {', '.join(saved_hist_paths)}")


###################### Co-localization #########################

def plot_cbc(
    cbc_plot_data: dict[str, dict[str, any]],
    channels: dict[str, tuple[str, str, int]],
    save_dir: str | None = None,
    log_scale: bool = False,
) -> pd.DataFrame:
    """
    Plot CBC results using pre-computed CBC values.
    Colors and labels are dynamically inferred from channels dict.
    Computes statistics for each direction and returns a summary DataFrame.
    log_scale: If True, x-axis is displayed in log scale (visualization only).
    """
    summary_rows = []

    for direction, data in cbc_plot_data.items():
        points = data["coordinates"]
        cbc_values = data["cbc_values"]

        # Only consider valid (non-NaN) values
        valid_values = cbc_values[~np.isnan(cbc_values)]
        if len(valid_values) == 0:
            logger.warning(f"No valid CBC values for {direction}, skipping.")
            continue

        # Compute statistics
        mu = np.mean(valid_values)
        sigma = np.std(valid_values)
        median_val = np.median(valid_values)

        # Create appropriate bins based on scale
        if log_scale:
            # Filter positive values for log scale (CBC values can be negative, so we need to handle this)
            positive_values = valid_values[valid_values > 0]
            if len(positive_values) == 0:
                logger.warning(
                    f"No positive CBC values for {direction} with log scale, using linear bins."
                )
                bins = "auto"
            else:
                # Use log-spaced bins for positive values
                min_val = np.min(positive_values)
                max_val = np.max(positive_values)
                bins = np.logspace(np.log10(min_val), np.log10(max_val), 70)
        else:
            # Use auto bins for linear scale
            bins = "auto"

        # Histogram and highest-probable interval
        counts, bin_edges = np.histogram(valid_values, bins=bins, density=False)

        # Filter out empty bins to avoid visualization issues
        non_zero_mask = counts > 0
        counts = counts[non_zero_mask]

        # Use proper bin centers and widths
        if len(bin_edges) > 1:
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            bin_centers = bin_centers[non_zero_mask]
            bin_widths = np.diff(bin_edges)[non_zero_mask]
        else:
            # Fallback for single bin
            bin_centers = bin_edges[:1]
            bin_widths = [bin_edges[0] * 0.1] if len(bin_edges) > 0 else [1.0]

        if len(counts) > 0:
            # Highest-frequency bin - use original bin_edges for accurate interval
            max_idx = np.argmax(counts)
            highest_frequency_bin = (bin_edges[max_idx], bin_edges[max_idx + 1])
        else:
            highest_frequency_bin = (np.nan, np.nan)
            bin_edges = np.array([])
            bin_widths = np.array([])

        # Determine color dynamically
        ch_label = direction.split("→")[0]
        color = next((v[0] for v in channels.values() if v[1] == ch_label), "blue")

        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

        # Scatter plot
        sc = axes[0].scatter(
            points[:, 0],
            points[:, 1],
            c=cbc_values,
            cmap="coolwarm",
            norm=colors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1),
            s=10,
            alpha=0.8,
            edgecolors="none",
        )
        axes[0].set_title(f"CBC: {direction}", fontsize=14)
        axes[0].set_xlabel("X (nm)")
        axes[0].set_ylabel("Y (nm)")
        axes[0].set_aspect("equal")
        axes[0].invert_yaxis()
        fig.colorbar(sc, ax=axes[0], label="CBC coefficient")

        # Histogram - only plot if we have valid bins
        if len(counts) > 0 and len(bin_centers) > 0:
            axes[1].bar(
                bin_centers - bin_widths / 2,
                counts,
                width=bin_widths,
                color=color,
                alpha=0.6,
                edgecolor="none",
                align="edge",
            )

            # Set appropriate x-axis limits
            if log_scale and len(bin_edges) > 0:
                axes[1].set_xlim(bin_edges[0] * 0.9, bin_edges[-1] * 1.1)
            elif len(bin_edges) > 0:
                axes[1].set_xlim(
                    bin_edges[0] - bin_widths[0] * 0.1,
                    bin_edges[-1] + bin_widths[-1] * 1.1,
                )

        if log_scale:
            axes[1].set_xscale("log")

        axes[1].set_title(f"CBC distribution: {direction}", fontsize=14)
        axes[1].set_xlabel("CBC coefficient" + (" (log scale)" if log_scale else ""))
        axes[1].set_ylabel("Frequency")
        axes[1].grid(True, linestyle="--", alpha=0.4)

        stats_text = (
            f"μ = {mu:.4f}\n"
            f"σ = {sigma:.4f}\n"
            f"Md = {median_val:.4f}\n"
            f"Peak: [{highest_frequency_bin[0]:.4f}, {highest_frequency_bin[1]:.4f}]"
        )
        axes[1].text(
            0.95,
            0.95,
            stats_text,
            transform=axes[1].transAxes,
            fontsize=12,
            verticalalignment="top",
            horizontalalignment="right",
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.8},
        )

        if save_dir:
            path = os.path.join(save_dir, f"CBC_{direction}.png")
            saved_paths = _save_figure_variants(fig, path)
            logger.info(f"✅ Saved: {', '.join(saved_paths)}")
        plt.close(fig)

        # Save stats
        summary_rows.append(
            {
                "Direction": direction,
                "Mean": mu,
                "Std": sigma,
                "Median": median_val,
                "High Frequency Bin": f"[{highest_frequency_bin[0]:.4f}, {highest_frequency_bin[1]:.4f}]",
            }
        )

    return pd.DataFrame(summary_rows)

def plot_re_histograms(
    re_per_loc_df: pd.DataFrame,
    channels: Dict[str, tuple[str, str, int]],
    save_dir: Union[str, os.PathLike] = ".",
    log_scale: bool = False
) -> pd.DataFrame:
    """
    Plot histograms of RE_per_localization per direction and save plots.
    Highlights highest-probable interval and returns a summary DataFrame.

    Expects `re_per_loc_df` with columns: ['Channel', 'Reference_Channel', 'RE_per_localization'].
    """
    summary_rows = []

    # Build color map from label -> color
    label_to_color = {channels[ch][1]: channels[ch][0] for ch in channels}

    # Group by direction
    re_per_loc_df["Direction"] = re_per_loc_df["Channel"] + "→" + re_per_loc_df["Reference_Channel"]

    for direction, group in re_per_loc_df.groupby("Direction"):
        values = group["RE_per_localization"].replace([np.inf, -np.inf], np.nan).dropna().values
        
        # Filter out zeros and negative values for log scale
        if log_scale:
            values = values[values > 0]
        
        if len(values) == 0:
            logger.warning(f"No valid RE values for {direction}, skipping.")
            continue

        # Color based on primary channel
        primary_label = direction.split("→")[0]
        color = label_to_color.get(primary_label, "steelblue")

        # Compute stats
        mu = np.mean(values)
        sigma = np.std(values)
        median_val = np.median(values)

        # Create figure and axis
        fig, ax = plt.subplots(figsize=(8, 5), dpi=300)

        # Handle binning differently for log vs linear scale
        if log_scale:
            # Use log-spaced bins for log scale
            min_val = max(values.min(), 1e-10)  # Avoid log(0)
            max_val = values.max()
            bins = np.logspace(np.log10(min_val), np.log10(max_val), 70)
            
            # Plot histogram
            counts, bin_edges, _ = ax.hist(values, bins=bins, color=color, alpha=0.7, 
                                         edgecolor='white', linewidth=0.5)
            
            ax.set_xscale("log")
            ax.set_xlabel("Relative Enrichment (RE) (log scale)")
            
        else:
            # Use auto bins for linear scale
            counts, bin_edges, _ = ax.hist(values, bins='auto', color=color, alpha=0.7,
                                         edgecolor='white', linewidth=0.5)
            ax.set_xlabel("Relative Enrichment (RE)")

        # Find highest frequency bin
        if len(counts) > 0:
            max_idx = np.argmax(counts)
            highest_frequency_bin = (bin_edges[max_idx], bin_edges[max_idx + 1])
        else:
            highest_frequency_bin = (np.nan, np.nan)

        # Set labels and title
        ax.set_ylabel("Frequency")
        ax.set_title(f"RE per localization - {direction}")
        
        # Improve grid
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)

        # Annotate stats with better positioning
        stats_text = (
            f"μ = {mu:.4f}\n"
            f"σ = {sigma:.4f}\n"
            f"Md = {median_val:.4f}\n"
            f"High-freq: [{highest_frequency_bin[0]:.4f}, {highest_frequency_bin[1]:.4f}]"
        )
        
        # Position text based on data distribution
        x_pos = 0.95  # Right side
        y_pos = 0.95  # Top
        
        ax.text(
            x_pos, y_pos, stats_text,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9)
        )

        plt.tight_layout()
        out_path = os.path.join(save_dir, f"RE_per_localization_hist_{sanitize_filename(direction)}.png")
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"✅ Saved: {out_path}")

        # Append summary
        summary_rows.append({
            "Direction": direction,
            "Mean": mu,
            "Std": sigma,
            "Median": median_val,
            "High Frequency Bin": f"[{highest_frequency_bin[0]:.4f}, {highest_frequency_bin[1]:.4f}]"
        })

    return pd.DataFrame(summary_rows)