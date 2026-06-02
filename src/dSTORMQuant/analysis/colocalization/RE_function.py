# MIT License
#
# Copyright (c) 2022 Aske Lykke Ejdrup
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from itertools import compress

import matplotlib.path as path
import numpy as np
from scipy.spatial import ConvexHull, Delaunay, Voronoi, cKDTree
from shapely.geometry import Polygon


def in_hull(hull, p):
    """Count how many points in ``p`` lie inside a Delaunay hull.

    Args:
        hull: :class:`scipy.spatial.Delaunay` instance or ``(M, K)`` array of hull
            vertices (triangulation is computed when an array is passed).
        p: ``(N, K)`` array of query point coordinates.

    Returns:
        Number of query points inside the hull (simplex index >= 0).

    See Also:
        https://stackoverflow.com/questions/16750618/whats-an-efficient-way-to-find-if-a-point-lies-in-the-convex-hull-of-a-point-cl
    """
    if not isinstance(hull, Delaunay):
        hull = Delaunay(hull)

    return sum(hull.find_simplex(p) >= 0)


def get_neighbours_list(voronoi):
    """Build a per-point list of first-order Voronoi neighbors.

    Args:
        voronoi: :class:`scipy.spatial.Voronoi` tessellation.

    Returns:
        List of length ``len(voronoi.points)``; each entry lists indices of
        points sharing a Voronoi ridge with that site.
    """
    neighbours_list = [[] for point_index in range(len(voronoi.points))]
    counter = 0
    for count, neighbour_pair in enumerate(voronoi.ridge_points):
        if count % int((len(voronoi.ridge_points) - 1) / 5) == 0:
            # print('Looping every voronoi region - progress: {}%'.format(20*counter))
            counter += 1
        neighbours_list[neighbour_pair[0]].append(neighbour_pair[1])
        neighbours_list[neighbour_pair[1]].append(neighbour_pair[0])
    return neighbours_list


def relative_enrichment(ch1_points, ch2_points, verbose=False):
    """Compute 2D relative enrichment per finite reference Voronoi region.

    Args:
        ch1_points: Reference species coordinates, shape ``(N, 2)``.
        ch2_points: Primary species coordinates, shape ``(M, 2)``.
        verbose: If ``True``, print progress messages.

    Returns:
        Tuple of:

        - ``n_points_ch2_in_region``: Primary localizations per reference region.
        - ``sorted_region_area``: Area of each finite reference region.
        - ``first_order_mean_distance``: Mean distance to first-order neighbors.
        - ``bool_index``: Mask of reference sites with finite Voronoi cells.
    """

    # Compute the voronoi tessellation for each channel
    if verbose:
        print("Computing 2D voronoi tessellation.")
    ch1_vor = Voronoi(ch1_points)
    ch2_vor = Voronoi(ch2_points)

    unfiltered_region = [ch1_vor.regions[i] for i in ch1_vor.point_region]
    bool_index = np.ones((len(unfiltered_region)), dtype=bool)
    for i in range(0, len(bool_index)):
        if -1 in unfiltered_region[i]:
            bool_index[i] = False
    sorted_region = list(compress(unfiltered_region, bool_index))
    sorted_vertices = [ch1_vor.vertices[i] for i in sorted_region]
    sorted_region_path = [path.Path(i) for i in sorted_vertices]
    sorted_region_area = [Polygon(i).area for i in sorted_vertices]
    sorted_points_ch1 = ch1_vor.points[bool_index]

    ckd_tree_ch2 = cKDTree(ch2_points)  # Construct a knn tree
    # Figure out how to circumvent k = max
    if len(ch2_vor.points) < 1000:
        dist_ch2, idx_ch2 = ckd_tree_ch2.query(
            sorted_points_ch1, k=len(ch2_vor.points)
        )  # query closest neighbor
        n_points_ch2_in_region = [
            np.count_nonzero(i.contains_points(ch2_vor.points[idx_ch2[j, :], :]))
            for j, i in enumerate(sorted_region_path)
        ]
    else:
        dist_ch2, idx_ch2 = ckd_tree_ch2.query(
            sorted_points_ch1, k=1000
        )  # query closest neighbor
        n_points_ch2_in_region = [
            np.count_nonzero(i.contains_points(ch2_vor.points[idx_ch2[j, :], :]))
            for j, i in enumerate(sorted_region_path)
        ]

    # Construct dictionary of points in adjacent regions for each point
    # print("Constructing dictionary of adjacent regions.")

    # Compute first order mean distance
    neighbours_list = get_neighbours_list(ch1_vor)
    neighbour_indices = list(compress(neighbours_list, bool_index))
    neighbour_points = [ch1_vor.points[i] for i in neighbour_indices]

    # Compute mean ditance to first order neighbors
    neighbour_distances = [
        np.sqrt(np.sum((np.array(j) - sorted_points_ch1[i]) ** 2, axis=1))
        for i, j in enumerate(neighbour_points)
    ]
    first_order_mean_distance = [np.nanmean(i) for i in neighbour_distances]

    return (
        np.array(n_points_ch2_in_region),
        np.array(sorted_region_area),
        np.array(first_order_mean_distance),
        bool_index,
    )


def RE3D(ch1_points, ch2_points, verbose=False):
    """Compute 3D relative enrichment per finite reference Voronoi region.

    Args:
        ch1_points: Reference species coordinates, shape ``(N, 3)``.
        ch2_points: Primary species coordinates, shape ``(M, 3)``.
        verbose: If ``True``, print progress messages.

    Returns:
        Tuple of:

        - ``n_points_ch2_in_region``: Primary localizations per reference region.
        - ``sorted_region_volume``: Convex-hull volume of each finite region.
        - ``first_order_mean_distance``: Mean distance to first-order neighbors.
        - ``bool_index``: Mask of reference sites with finite Voronoi cells.
    """

    # Compute the voronoi tessellation for each channel
    if verbose:
        print("Computing 3D voronoi tessellation.")
    ch1_vor = Voronoi(ch1_points)
    ch2_vor = Voronoi(ch2_points)

    unfiltered_region = [ch1_vor.regions[i] for i in ch1_vor.point_region]

    bool_index = np.ones((len(unfiltered_region)), dtype=bool)
    for i in range(0, len(bool_index)):
        if -1 in unfiltered_region[i]:
            bool_index[i] = False
    sorted_region = list(compress(unfiltered_region, bool_index))

    sorted_vertices = [ch1_vor.vertices[i] for i in sorted_region]
    sorted_region_hull = [ConvexHull(i) for i in sorted_vertices]
    sorted_region_volume = [i.volume for i in sorted_region_hull]
    sorted_points_ch1 = ch1_vor.points[bool_index]

    # Construct and query a knn tree
    if verbose:
        print("Constructing a knn tree")
    ckd_tree_ch2 = cKDTree(ch2_points)
    dist_ch2, idx_ch2 = ckd_tree_ch2.query(ch1_vor.points[bool_index], k=100)

    # Compare the old and new hull for differences
    if verbose:
        print("Comparing hulls")
    n_points_ch2_in_region = [
        in_hull(i.points, ch2_vor.points[idx_ch2[j, :], :])
        for j, i in enumerate(sorted_region_hull)
    ]

    # Compute first order mean distance
    if verbose:
        print("Identifying neighboring points")
    neighbours_list = get_neighbours_list(ch1_vor)
    neighbour_indices = list(compress(neighbours_list, bool_index))
    neighbour_points = [ch1_vor.points[i] for i in neighbour_indices]

    # Compute mean distance to first order neighbors
    if verbose:
        print("Computing mean distance to neighbors")
    neighbour_distances = [
        np.sqrt(np.sum((np.array(j) - sorted_points_ch1[i]) ** 2, axis=1))
        for i, j in enumerate(neighbour_points)
    ]
    first_order_mean_distance = [np.nanmean(i) for i in neighbour_distances]

    return (
        np.array(n_points_ch2_in_region),
        np.array(sorted_region_volume),
        np.array(first_order_mean_distance),
        bool_index,
    )


def bin_relative_enrichment(
    n_points,
    areas_ch1,
    first_ord_dist,
    max_dist,
    step_size,
    total_volume="None",
    size_threshold=99.5,
):
    """Bin relative-enrichment results by log10 mean first-neighbor distance.

    Args:
        n_points: Primary counts per region from :func:`relative_enrichment` or
            :func:`RE3D`.
        areas_ch1: Region areas (2D) or volumes (3D) from the same functions.
        first_ord_dist: First-order mean neighbor distances per region.
        max_dist: Upper log10 bin edge for neighbor-distance binning.
        step_size: Log10 bin width.
        total_volume: Normalization volume; ``"None"`` estimates from regions below
            ``size_threshold`` percentile of area/volume.
        size_threshold: Percentile cutoff when ``total_volume`` is ``"None"``.

    Returns:
        Tuple ``(RE_values, no_regions, no_loc_per_region, area_per_bins)`` — binned
        enrichment ratios and per-bin region counts, localization counts, and areas.
    """

    # Try to bin
    if total_volume == "None":
        threshold = np.percentile(areas_ch1, size_threshold)
        total_volume = np.sum(areas_ch1[areas_ch1 < threshold])
    sorted_density_ch1 = np.sort(1 / areas_ch1)
    sorted_density_ch1_idx = np.argsort(1 / areas_ch1)
    first_ord_dist_sorted = first_ord_dist[sorted_density_ch1_idx]
    maxbin = int(max_dist / step_size)
    dr = step_size
    density = sum(n_points) / total_volume
    pts_ratio = np.zeros((maxbin + 1, 1))
    no_regions = np.zeros((maxbin + 1, 1))
    no_loc_per_region = np.zeros((maxbin + 1, 1))
    area_per_bins = np.zeros((maxbin + 1, 1))
    for i in range(0, len(sorted_density_ch1)):
        if np.isfinite(first_ord_dist_sorted[i]):
            bin_no = int(np.log10(first_ord_dist_sorted[i]) / dr)
            if bin_no <= maxbin:
                obs_dens = n_points[sorted_density_ch1_idx[i]] / (
                    1 / sorted_density_ch1[i] * density
                )
                pts_ratio[bin_no] += obs_dens
                no_regions[bin_no] += 1
                no_loc_per_region[bin_no] += n_points[sorted_density_ch1_idx[i]]
                area_per_bins[bin_no] += areas_ch1[sorted_density_ch1_idx[i]]
    RE_values = np.divide(pts_ratio, no_regions)
    return RE_values, no_regions, no_loc_per_region, area_per_bins


def bin_relative_enrichment_area(
    n_points,
    areas_ch1,
    first_ord_dist,
    max_dist,
    step_size,
    total_volume="None",
    size_threshold=99.5,
):
    """Bin relative-enrichment results by log10 reference region area or volume.

    Args:
        n_points: Primary counts per region from :func:`relative_enrichment` or
            :func:`RE3D`.
        areas_ch1: Region areas (2D) or volumes (3D) from the same functions.
        first_ord_dist: First-order mean neighbor distances per region.
        max_dist: Upper log10 bin edge for area/volume binning.
        step_size: Log10 bin width.
        total_volume: Normalization volume; ``"None"`` estimates from regions below
            ``size_threshold`` percentile of area/volume.
        size_threshold: Percentile cutoff when ``total_volume`` is ``"None"``.

    Returns:
        Tuple ``(RE_values, no_regions, no_loc_per_region, area_per_bins)`` — binned
        enrichment ratios and per-bin region counts, localization counts, and areas.
    """
    if total_volume == "None":
        threshold = np.percentile(areas_ch1, size_threshold)
        total_volume = np.sum(areas_ch1[areas_ch1 < threshold])
    sorted_density_ch1 = np.sort(1 / areas_ch1)
    sorted_density_ch1_idx = np.argsort(1 / areas_ch1)
    first_ord_dist_sorted = first_ord_dist[sorted_density_ch1_idx]
    area_sorted = areas_ch1[sorted_density_ch1_idx]
    maxbin = int(max_dist / step_size)
    dr = step_size
    density = sum(n_points) / total_volume
    pts_ratio = np.zeros((maxbin + 1, 1))
    no_regions = np.zeros((maxbin + 1, 1))
    no_loc_per_region = np.zeros((maxbin + 1, 1))
    area_per_bins = np.zeros((maxbin + 1, 1))
    for i in range(0, len(sorted_density_ch1)):
        if np.isfinite(first_ord_dist_sorted[i]):
            bin_no = int(np.log10(area_sorted[i]) / dr)
            if bin_no <= maxbin:
                obs_dens = n_points[sorted_density_ch1_idx[i]] / (
                    1 / sorted_density_ch1[i] * density
                )
                pts_ratio[bin_no] += obs_dens
                no_regions[bin_no] += 1
                no_loc_per_region[bin_no] += n_points[sorted_density_ch1_idx[i]]
                area_per_bins[bin_no] += areas_ch1[sorted_density_ch1_idx[i]]
    RE_values = np.divide(pts_ratio, no_regions)
    return RE_values, no_regions, no_loc_per_region, area_per_bins
