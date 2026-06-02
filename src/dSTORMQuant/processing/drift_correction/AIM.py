"""
picasso.aim
~~~~~~~~~~~

Picasso implementation of Adaptive Intersection Maximization (AIM)
for fast undrifting in 2D and 3D.

Adapted from: Ma, H., et al. Science Advances. 2024.

:author: Hongqiang Ma, Maomao Chen, Phuong Nguyen, Yang Liu,
    Rafal Kowalewski, 2024
:copyright: Copyright (c) 2016-2024 Jungmann Lab, MPI of Biochemistry


Modified by: Suraj Karki
"""

import os
import os.path as _ospath
from concurrent.futures import ThreadPoolExecutor as ThreadPoolExecutor

import h5py
import numpy as _np
import pandas as pd
import yaml as _yaml

from dSTORMQuant.utils.data_handling import read_localization_csv
from scipy.interpolate import make_interp_spline
from tqdm import tqdm as tqdm

from dSTORMQuant.utils.logger import get_logger

logger = get_logger()


def validate_and_replace_drift(
    relative_drift_x,
    relative_drift_y,
    pixelsize,
    max_drift_nm,
    n_previous_segments=10,
):
    """
    Validate segment drift values and replace unrealistic ones with the mean
    of previous N valid segments based on relative displacements.

    Parameters
    ----------
    relative_drift_x : _np.array
        Relative (frame-to-frame) drift in x-direction for each segment (in pixels).
    relative_drift_y : _np.array
        Relative (frame-to-frame) drift in y-direction for each segment (in pixels).
    pixelsize : float
        Pixel size in nm.
    max_drift_nm : float
        Maximum allowed drift per segment in nanometers.
    n_previous_segments : int, optional
        Number of previous valid segments to average for replacement (default: 10).

    Returns
    -------
    corrected_segment_drift_x : _np.array
        Corrected cumulative drift in x-direction.
    corrected_segment_drift_y : _np.array
        Corrected cumulative drift in y-direction.
    corrected_relative_drift_x : _np.array
        Corrected relative drift in x-direction.
    corrected_relative_drift_y : _np.array
        Corrected relative drift in y-direction.
    n_replaced : int
        Number of segments that were replaced.
    """

    n_segments = len(relative_drift_x)

    # Copy arrays to avoid modifying originals
    corrected_relative_x = relative_drift_x.copy()
    corrected_relative_y = relative_drift_y.copy()

    n_replaced = 0

    # Single-pass sequential replacement: process segments in order
    # This allows corrected segments to be used in subsequent averages
    for s in range(n_segments):
        # Calculate the magnitude of relative drift for this segment
        drift_magnitude_pixels = _np.sqrt(
            corrected_relative_x[s] ** 2 + corrected_relative_y[s] ** 2
        )
        drift_magnitude_nm = drift_magnitude_pixels * pixelsize

        # Check if drift exceeds threshold
        if drift_magnitude_nm > max_drift_nm:
            logger.warning(
                f"⚠️ Segment {s}: Drift magnitude {drift_magnitude_nm:.2f} nm "
                f"exceeds threshold {max_drift_nm:.2f} nm. Marked for replacement."
            )

            # Look back at most n_previous_segments positions
            start_idx = max(0, s - n_previous_segments)

            if s > 0:
                # Calculate mean from previous segments (includes any already corrected)
                mean_rel_x = _np.mean(corrected_relative_x[start_idx:s])
                mean_rel_y = _np.mean(corrected_relative_y[start_idx:s])

                corrected_relative_x[s] = mean_rel_x
                corrected_relative_y[s] = mean_rel_y
                n_replaced += 1

                n_used = s - start_idx
                logger.info(
                    f"✅ Segment {s}: Replaced with mean of {n_used} "
                    f"previous segments (mean_x={mean_rel_x:.4f} px, mean_y={mean_rel_y:.4f} px)"
                )
            else:
                # First segment - set to zero
                corrected_relative_x[s] = 0
                corrected_relative_y[s] = 0
                n_replaced += 1

                logger.warning(
                    f"⚠️ Segment {s}: No previous segments available. Set to zero."
                )

    if n_replaced > 0:
        logger.info(
            f"✅ Drift validation complete: {n_replaced}/{n_segments} segments replaced "
            f"using mean of up to {n_previous_segments} previous segments"
        )

    # Recalculate cumulative drift from corrected relative drift
    corrected_segment_drift_x = _np.zeros(n_segments)
    corrected_segment_drift_y = _np.zeros(n_segments)

    cumulative_x = 0
    cumulative_y = 0
    for s in range(n_segments):
        # Relative drift is negative of the shift, so we subtract
        cumulative_x -= corrected_relative_x[s]
        cumulative_y -= corrected_relative_y[s]
        corrected_segment_drift_x[s] = cumulative_x
        corrected_segment_drift_y[s] = cumulative_y

    if n_replaced > 0:
        logger.info(
            f"✅ Drift validation complete: {n_replaced}/{n_segments} segments replaced"
        )
    else:
        logger.info(
            f"✅ Drift validation complete: All segments within threshold ({max_drift_nm:.2f} nm)"
        )

    return (
        corrected_segment_drift_x,
        corrected_segment_drift_y,
        corrected_relative_x,
        corrected_relative_y,
        n_replaced,
    )


def intersect1d(a, b):
    """Slightly faster implementation of _np.intersect1d without
    unnecessary checks, etc.

    Finds the indices of common elements in two 1D arrays (a and b).
    Both a and b are assumed to be sorted and contain only unique
    values.

    Parameters
    ----------
    a : _np.array
        1D array of integers.
    b : _np.array
        1D array of integers.

    Returns
    -------
    a_indices : _np.array
        Indices of common elements in a.
    b_indices : _np.array
        Indices of common elements in b.
    """

    aux = _np.concatenate((a, b))
    aux_sort_indices = _np.argsort(aux, kind="mergesort")
    aux = aux[aux_sort_indices]

    mask = aux[1:] == aux[:-1]
    a_indices = aux_sort_indices[:-1][mask]
    b_indices = aux_sort_indices[1:][mask] - a.size

    return a_indices, b_indices


def count_intersections(l0_coords, l0_counts, l1_coords, l1_counts):
    """Counts the number of intersected localizations between the two
    datasets. We assume that the intersection distance is 1 and since
    the coordinates are expressed in the units of intersection distance,
    we require the coordinates to be exactly the same to count as
    intersection. Also, coordinates are converted to 1D arrays
    (x + y * width).

    Parameters
    ----------
    l0_coords : _np.array
        Unique coordinates of the reference localizations.
    l0_counts : _np.array
        Counts of the unique values of reference localizations.
    l1_coords : _np.array
        Unique coordinates of the target localizations.
    l1_counts : _np.array
        Counts of the unique values of target localizations.

    Returns
    -------
    n_intersections : int
        Number of intersections.
    """

    # indices of common elements
    idx0, idx1 = intersect1d(l0_coords, l1_coords)
    # extract the counts of these elements
    l0_counts_subset = l0_counts[idx0]
    l1_counts_subset = l1_counts[idx1]
    # for each overlapping coordinate, take the minimum count from l0
    # and l1, sum up across all overlapping coordinates
    n_intersections = _np.sum(_np.minimum(l0_counts_subset, l1_counts_subset))
    return n_intersections


def run_intersections(l0_coords, l0_counts, l1_coords, l1_counts, shifts_xy, box):
    """Run intersection counting across the local search region. Returns
    the 2D array with number of intersections across the local search
    region. Uses multithreading.

    Parameters
    ----------
    l0_coords : _np.array
        Unique coordinates of the reference localizations.
    l0_counts : _np.array
        Counts of the reference localizations.
    l1_coords : _np.array
        Unique coordinates of the target localizations.
    l1_counts : _np.array
        Counts of the target localizations.
    shifts_xy : _np.array
        1D array with x and y shifts.
    box : int
        Side length of the local search region.

    Returns
    -------
    roi_cc : _np.2darray
        2D array with number of intersections across the local search
        region.
    """

    # shift target coordinates
    l1_coords_shifted = l1_coords[:, _np.newaxis] + shifts_xy
    # run multiple threads
    n_workers = len(shifts_xy)
    executor = ThreadPoolExecutor(n_workers)
    f = [
        executor.submit(
            count_intersections,
            l0_coords,
            l0_counts,
            l1_coords_shifted[:, i],
            l1_counts,
        )
        for i in range(len(shifts_xy))
    ]
    executor.shutdown(wait=True)
    if box == 1:  # z intersection only, for z undrifting
        roi_cc = _np.array([_.result() for _ in f])
    else:  # 2D intersection
        roi_cc = _np.array([_.result() for _ in f]).reshape(box, box)
    return roi_cc


def point_intersect_2d(
    l0_coords, l0_counts, x1, y1, intersect_d, width_units, shifts_xy, box
):
    """Converts target coordinates into a 1D array in units of
    intersect_d and counts the number of intersections in the local
    search region.

    Parameters
    ----------
    l0_coords : _np.array
        Unique values of the reference localizations.
    l0_counts : _np.array
        Counts of the unique values of reference localizations.
    x1 : _np.array
        x-coordinates of the target (currently undrifted) localizations.
    y1 : _np.array
        y-coordinates of the target (currently undrifted) localizations.
    intersect_d : float
        Intersect distance in camera pixels.
    width_units : int
        Width of the camera image in units of intersect_d.
    shifts_xy : _np.array
        1D array with x and y shifts.
    box : int
        Final side length of the local search region.

    Returns
    -------
    roi_cc : _np.2darray
        2D array with numbers of intersections in the local search
        region.
    """

    # convert target coordinates to a 1D array in intersect_d units
    x1_units = _np.round(x1 / intersect_d)
    y1_units = _np.round(y1 / intersect_d)
    l1 = _np.int32(x1_units + y1_units * width_units)  # 1d list
    # get unique values and counts of the target localizations
    l1_coords, l1_counts = _np.unique(l1, return_counts=True)
    # run the intersections counting
    roi_cc = run_intersections(
        l0_coords, l0_counts, l1_coords, l1_counts, shifts_xy, box
    )
    return roi_cc


def get_fft_peak(roi_cc, roi_size):
    """Estimate the precise sub-pixel position of the peak of roi_cc
    with FFT.

    Parameters
    ----------
    roi_cc : _np.2darray
        2D array with numbers of intersections in the local search region.
    roi_size : int
        Size of the local search region.

    Returns
    -------
    px : float
        Estimated x-coordinate of the peak.
    py : float
        Estimated y-coordinate of the peak.
    """

    fft_values = _np.fft.fft2(roi_cc.T)
    ang_x = _np.angle(fft_values[0, 1])
    ang_x = ang_x - 2 * _np.pi * (ang_x > 0)  # normalize
    px = (
        _np.abs(ang_x) / (2 * _np.pi / roi_cc.shape[0]) - (roi_cc.shape[0] - 1) / 2
    )  # peak in x
    px *= roi_size / roi_cc.shape[0]  # convert to intersect_d units
    ang_y = _np.angle(fft_values[1, 0])
    ang_y = ang_y - 2 * _np.pi * (ang_y > 0)  # normalize
    py = (
        _np.abs(ang_y) / (2 * _np.pi / roi_cc.shape[1]) - (roi_cc.shape[1] - 1) / 2
    )  # peak in y
    py *= roi_size / roi_cc.shape[1]  # convert to intersect_d units
    return px, py


def intersection_max(
    x,
    y,
    ref_x,
    ref_y,
    frame,
    seg_bounds,
    intersect_d,
    roi_r,
    width,
    aim_round=1,
):
    """Maximize intersection (undrift) for 2D localizations.

    Parameters
    ----------
    x : _np.array
        x-coordinates of the localizations.
    y : _np.array
        y-coordinates of the localizations.
    ref_x_list : _np.array
        x-coordinates of the reference localizations.
    ref_y_list : _np.array
        y-coordinates of the reference localizations.
    frame : _np.array
        Frame indices of localizations.
    seg_bounds : np.array
        Frame indices of the segmentation bounds. Defines temporal
        intervals used to estimate drift.
    intersect_d : float
        Intersect distance in camera pixels.
    roi_r : float
        Radius of the local search region in camera pixels. Should be
        higher than the maximum expected drift within one segment.
    width : int
        Width of the camera image in camera pixels.
    aim_round : {1, 2}
        Round of AIM algorithm. The first round uses the first interval
        as reference, the second round uses the entire dataset as
        reference. The impact is that in the second round, the first
        interval is also undrifted.

    Returns
    -------
    x_pdc : _np.array
        Undrifted x-coordinates.
    y_pdc : _np.array
        Undrifted y-coordinates.
    drift_x : _np.array
        Drift in x-direction.
    drift_y : _np.array
        Drift in y-direction.
    segment_drift_x : _np.array
        Segment drift in x-direction (before interpolation).
    segment_drift_y : _np.array
        Segment drift in y-direction (before interpolation).
    relative_drift_x : _np.array
        Relative (frame-to-frame) drift in x-direction.
    relative_drift_y : _np.array
        Relative (frame-to-frame) drift in y-direction.
    """

    assert aim_round in [1, 2], "aim_round must be 1 or 2."

    # number of segments
    n_segments = len(seg_bounds) - 1
    rel_drift_x = 0  # adaptive drift (updated at each interval)
    rel_drift_y = 0

    # drift in x and y
    drift_x = _np.zeros(n_segments)
    drift_y = _np.zeros(n_segments)

    # Store relative drift (frame-to-frame shift) for each segment
    relative_drift_x = _np.zeros(n_segments)
    relative_drift_y = _np.zeros(n_segments)

    # find shifts for the local search region (in units of intersect_d)
    roi_units = int(_np.ceil(roi_r / intersect_d))
    steps = _np.arange(-roi_units, roi_units + 1, 1)
    box = len(steps)
    shifts_xy = _np.zeros((box, box), dtype=_np.int32)
    width_units = width / intersect_d
    for i, shift_x in enumerate(steps):
        for j, shift_y in enumerate(steps):
            shifts_xy[i, j] = shift_x + shift_y * width_units
    shifts_xy = shifts_xy.reshape(box**2)

    # convert reference to a 1D array in units of intersect_d and find
    # unique values and counts
    x0_units = _np.round(ref_x / intersect_d)
    y0_units = _np.round(ref_y / intersect_d)
    l0 = _np.int32(x0_units + y0_units * width_units)  # 1d list
    l0_coords, l0_counts = _np.unique(l0, return_counts=True)

    # initialize progress such that if GUI is used, tqdm is omitted
    start_idx = 1 if aim_round == 1 else 0

    iterator = tqdm(
        range(start_idx, n_segments),
        desc=f"Undrifting ({aim_round}/2)",
        unit="segment",
    )

    # run across each segment
    for s in iterator:
        # get the target localizations within the current segment
        min_frame_idx = frame > seg_bounds[s]
        max_frame_idx = frame <= seg_bounds[s + 1]
        x1 = x[min_frame_idx & max_frame_idx]
        y1 = y[min_frame_idx & max_frame_idx]

        # skip if no reference localizations
        if len(x1) == 0:
            drift_x[s] = drift_x[s - 1]
            drift_y[s] = drift_y[s - 1]
            relative_drift_x[s] = 0
            relative_drift_y[s] = 0
            continue

        # undrifting from the previous round
        x1 += rel_drift_x
        y1 += rel_drift_y

        # count the number of intersected localizations
        roi_cc = point_intersect_2d(
            l0_coords,
            l0_counts,
            x1,
            y1,
            intersect_d,
            width_units,
            shifts_xy,
            box,
        )

        # estimate the precise sub-pixel position of the peak of roi_cc
        # with FFT
        px, py = get_fft_peak(roi_cc, 2 * roi_r)

        # Store relative drift (the frame-to-frame shift)
        relative_drift_x[s] = -px
        relative_drift_y[s] = -py

        # update the relative drift reference for the subsequent
        # segmented subset (interval) and save the drifts
        rel_drift_x += px
        rel_drift_y += py
        drift_x[s] = -rel_drift_x
        drift_y[s] = -rel_drift_y

        iterator.update(s - iterator.n)

    # Store segment drift before interpolation
    segment_drift_x = drift_x.copy()
    segment_drift_y = drift_y.copy()

    # Interpolate the drifts (cubic spline) for all frames
    # MATLAB-style boundary extension: extend drift arrays with extrapolated points
    # drift_extended = [2*drift[0]-drift[1], drift, 2*drift[-1]-drift[-2]]
    drift_x_extended = _np.concatenate(
        [[2 * drift_x[0] - drift_x[1]], drift_x, [2 * drift_x[-1] - drift_x[-2]]]
    )
    drift_y_extended = _np.concatenate(
        [[2 * drift_y[0] - drift_y[1]], drift_y, [2 * drift_y[-1] - drift_y[-2]]]
    )

    # Extend time points: center of each segment
    t = (seg_bounds[1:] + seg_bounds[:-1]) / 2
    # Extended time points: (-0.5, t, n_segments+0.5) * segmentation
    # MATLAB: interp1((-0.5:(trackNUM+0.5))*trackInterval, ...)
    segmentation = seg_bounds[1] - seg_bounds[0]  # assuming uniform segmentation
    t_extended = _np.concatenate(
        [[-0.5 * segmentation], t, [(n_segments + 0.5) * segmentation]]
    )

    # Interpolate using extended arrays
    drift_x_pol = make_interp_spline(
        t_extended, drift_x_extended, k=3, bc_type="not-a-knot"
    )
    drift_y_pol = make_interp_spline(
        t_extended, drift_y_extended, k=3, bc_type="not-a-knot"
    )
    t_inter = _np.arange(seg_bounds[-1]) + 1
    drift_x = drift_x_pol(t_inter)
    drift_y = drift_y_pol(t_inter)

    # undrift the localizations
    x_pdc = x - drift_x[frame - 1]
    y_pdc = y - drift_y[frame - 1]

    return (
        x_pdc,
        y_pdc,
        drift_x,
        drift_y,
        segment_drift_x,
        segment_drift_y,
        relative_drift_x,
        relative_drift_y,
    )


def aim(
    locs,
    info,
    segmentation,
    intersect_d,
    roi_r,
    drift_validation_config=None,
):
    """Apply AIM undrifting to the localizations.

    Parameters
    ----------
    locs : _np.rec.array
        Localizations list to be undrifted.
    info : list of dicts
        Localizations list's metadata.
    intersect_d : float
        Intersect distance in camera pixels.
    segmentation : int
        Time interval for drift tracking, unit: frames.
    roi_r : float
        Radius of the local search region in camera pixels. Should be
        larger than the  maximum expected drift within segmentation.
    drift_validation_config : dict, optional
        Configuration for drift validation. Should contain:
        - enable: bool
        - max_segment_drift_nm: float
    progress : picasso.lib.ProgressDialog (default=None)
        Progress dialog. If None, progress is displayed with into the
        console.

    Returns
    -------
    locs : _np.rec.array
        Undrifted localizations.
    new_info : list of 1 dict
        Updated metadata.
    drift : _np.rec.array
        Drift in x and y directions (and z if applicable).
    drift_tables : dict
        Dictionary containing segment drift and relative drift tables.
    """

    # extract metadata
    width = _np.nan
    height = _np.nan
    pixelsize = _np.nan
    n_frames = _np.nan
    info = info[0]

    if val := info.get("Width"):
        width = val
    if val := info.get("Height"):
        height = val
    if val := info.get("Frames"):
        n_frames = val - locs["frame"].min()
    if val := info.get("PixelSize"):
        pixelsize = val
    if _np.isnan(width * height * pixelsize * n_frames):
        raise KeyError(
            "Insufficient metadata available. Please specify 'Width', 'Height',"
            " 'Frames' and 'Pixelsize' in the metadata .yaml."
        )

    # frames should start at 1
    frame = locs["frame"] + 1 - locs["frame"].min()

    # find the segmentation bounds (temporal intervals)
    seg_bounds = _np.concatenate((_np.arange(0, n_frames, segmentation), [n_frames]))

    # get the reference localizations (first interval or first temporal segment)
    ref_x = locs["x"][frame <= segmentation]
    ref_y = locs["y"][frame <= segmentation]

    ### RUN AIM TWICE ###
    # the first run is with the first interval as reference
    (
        x_pdc,
        y_pdc,
        drift_x1,
        drift_y1,
        seg_drift_x1,
        seg_drift_y1,
        rel_drift_x1,
        rel_drift_y1,
    ) = intersection_max(
        locs.x,
        locs.y,
        ref_x,
        ref_y,
        frame,
        seg_bounds,
        intersect_d,
        roi_r,
        width,
        aim_round=1,
    )
    # the second run is with the entire dataset as reference
    (
        x_pdc,
        y_pdc,
        drift_x2,
        drift_y2,
        seg_drift_x2,
        seg_drift_y2,
        rel_drift_x2,
        rel_drift_y2,
    ) = intersection_max(
        x_pdc,
        y_pdc,
        x_pdc,
        y_pdc,
        frame,
        seg_bounds,
        intersect_d,
        roi_r,
        width,
        aim_round=2,
    )

    # add the drifts together from the two rounds
    drift_x = drift_x1 + drift_x2
    drift_y = drift_y1 + drift_y2

    # Zero the drift (set first frame as reference)
    # Comment the following lines to disable drift zeroing
    # This sets drift[0] = 0, making all drift relative to the first frame
    drift_x = drift_x - drift_x[0]
    drift_y = drift_y - drift_y[0]

    # Combine segment drifts from both rounds
    segment_drift_x = seg_drift_x1 + seg_drift_x2
    segment_drift_y = seg_drift_y1 + seg_drift_y2

    # Combine relative drifts from both rounds
    relative_drift_x = rel_drift_x1 + rel_drift_x2
    relative_drift_y = rel_drift_y1 + rel_drift_y2

    # Apply drift validation if enabled
    n_replaced = 0
    logger.info(f"INFO: drift_validation_config = {drift_validation_config}")
    if drift_validation_config and drift_validation_config.get("use"):
        logger.info("🔍 Validating and replacing unrealistic drift values...")

        (
            segment_drift_x,
            segment_drift_y,
            relative_drift_x,
            relative_drift_y,
            n_replaced,
        ) = validate_and_replace_drift(
            relative_drift_x,
            relative_drift_y,
            pixelsize,
            max_drift_nm=drift_validation_config.get("max_segment_drift_nm"),
            n_previous_segments=drift_validation_config.get("n_previous_segments", 10),
        )

        # If segments were replaced, recalculate interpolated drift
        if n_replaced > 0:
            logger.info(
                "♻️ Recalculating interpolated drift with corrected segment values..."
            )

            # Interpolate the corrected drifts (cubic spline) for all frames
            # MATLAB-style boundary extension
            drift_x_extended = _np.concatenate(
                [
                    [2 * segment_drift_x[0] - segment_drift_x[1]],
                    segment_drift_x,
                    [2 * segment_drift_x[-1] - segment_drift_x[-2]],
                ]
            )
            drift_y_extended = _np.concatenate(
                [
                    [2 * segment_drift_y[0] - segment_drift_y[1]],
                    segment_drift_y,
                    [2 * segment_drift_y[-1] - segment_drift_y[-2]],
                ]
            )

            # Extend time points
            t = (seg_bounds[1:] + seg_bounds[:-1]) / 2
            n_segments = len(seg_bounds) - 1
            segmentation = seg_bounds[1] - seg_bounds[0]
            t_extended = _np.concatenate(
                [[-0.5 * segmentation], t, [(n_segments + 0.5) * segmentation]]
            )

            drift_x_pol = make_interp_spline(
                t_extended, drift_x_extended, k=3, bc_type="not-a-knot"
            )
            drift_y_pol = make_interp_spline(
                t_extended, drift_y_extended, k=3, bc_type="not-a-knot"
            )
            t_inter = _np.arange(seg_bounds[-1]) + 1
            drift_x = drift_x_pol(t_inter)
            drift_y = drift_y_pol(t_inter)

            # Re-apply drift to localizations
            x_pdc = locs.x - drift_x[frame - 1]
            y_pdc = locs.y - drift_y[frame - 1]

    drift_x = _np.asarray(drift_x)
    drift_y = _np.asarray(drift_y)

    # combine to Picasso format
    drift = _np.rec.array((drift_x, drift_y), dtype=[("x", "f"), ("y", "f")])

    # apply the drift to localizations
    locs["x"] = x_pdc
    locs["y"] = y_pdc

    # Prepare drift tables dictionary
    # Calculate segment centers (frame numbers)
    segment_centers = (seg_bounds[1:] + seg_bounds[:-1]) / 2

    # Convert to nanometers
    segment_drift_x_nm = segment_drift_x * pixelsize
    segment_drift_y_nm = segment_drift_y * pixelsize
    relative_drift_x_nm = relative_drift_x * pixelsize
    relative_drift_y_nm = relative_drift_y * pixelsize

    # Calculate magnitudes
    segment_drift_magnitude_nm = _np.sqrt(segment_drift_x_nm**2 + segment_drift_y_nm**2)
    relative_drift_magnitude_nm = _np.sqrt(
        relative_drift_x_nm**2 + relative_drift_y_nm**2
    )

    drift_tables = {
        "segment_centers": segment_centers,
        "segment_drift_x_nm": segment_drift_x_nm,
        "segment_drift_y_nm": segment_drift_y_nm,
        "segment_drift_magnitude_nm": segment_drift_magnitude_nm,
        "relative_drift_x_nm": relative_drift_x_nm,
        "relative_drift_y_nm": relative_drift_y_nm,
        "relative_drift_magnitude_nm": relative_drift_magnitude_nm,
        "n_replaced": n_replaced,
    }

    new_info = {
        **info,
        "Generated by": "AIM undrift",
        "Intersect distance (nm)": intersect_d * pixelsize,
        "Segmentation": segmentation,
        "Search regions radius (nm)": roi_r * pixelsize,
    }

    return locs, new_info, drift, drift_tables


def save_info(path, info):
    """Write localization metadata records to a YAML sidecar file.

    Args:
        path: Base path; ``.yaml`` is appended when missing from the extension.
        info: Iterable of metadata dicts dumped with PyYAML.
    """
    with open(path, "w") as file:
        _yaml.dump_all(info, file)


def load_info(path):
    """Load localization metadata from the YAML sidecar next to an HDF5 file.

    Args:
        path: Path to the ``.hdf5`` or base path; ``.yaml`` is resolved alongside it.

    Returns:
        List of metadata dicts, or an empty list if the YAML file is missing.
    """
    path_base, _ = _ospath.splitext(path)
    filename = path_base + ".yaml"
    try:
        with open(filename) as info_file:
            info = list(_yaml.load_all(info_file, Loader=_yaml.UnsafeLoader))
    except FileNotFoundError:
        logger.error(f"\nAn error occurred. Could not find metadata file:\n{filename}")
    return info


def load_locs(path, _qt_parent=None):
    """Load localizations and metadata from an HDF5 file.

    Args:
        path: Path to the HDF5 localization file.
        _qt_parent: Unused; kept for API compatibility with Picasso callers.

    Returns:
        Tuple of a numpy record array of localizations and the metadata list from
        :func:`load_info`.
    """
    with h5py.File(path, "r") as locs_file:
        locs = locs_file["locs"][...]
    locs = _np.rec.array(
        locs, dtype=locs.dtype
    )  # Convert to rec array with fields as attributes
    info = load_info(path)
    return locs, info


def save_locs(path, locs, info, sanity_config=None):
    """Apply sanity filters and persist localizations to HDF5 plus YAML metadata.

    Args:
        path: Output ``.hdf5`` path (YAML written alongside with the same stem).
        locs: Localization record array.
        info: Metadata records passed to :func:`save_info`.
        sanity_config: Optional dict with keys ``remove_invalid_values``,
            ``filter_boundary``, and ``filter_parameters`` for :func:`ensure_sanity`.

    Returns:
        None
    """
    if sanity_config is None:
        sanity_config = {
            "remove_invalid_values": True,
            "filter_boundary": True,
            "filter_parameters": True,
        }

    # Create path for removed localizations log
    base, _ = _ospath.splitext(path)
    removed_locs_path = base + "_removed_locs.csv"

    locs = ensure_sanity(
        locs,
        remove_invalid=sanity_config["remove_invalid_values"],
        filter_boundary=sanity_config["filter_boundary"],
        filter_parameters=sanity_config["filter_parameters"],
        removed_locs_path=removed_locs_path,
    )
    with h5py.File(path, "w") as locs_file:
        locs_file.create_dataset("locs", data=locs)
    info_path = base + ".yaml"
    save_info(info_path, info)


def ensure_sanity(
    locs,
    remove_invalid=True,
    filter_boundary=True,
    filter_parameters=True,
    removed_locs_path=None,
):
    """Ensures that localizations are within the image dimensions
    and have positive localization precisions.

    Parameters
    ----------
    locs : np.rec.array
        Localizations list.
    remove_invalid : bool, optional
        Remove localizations with inf/nan values (default=True, STRONGLY RECOMMENDED)
    filter_boundary : bool, optional
        Remove localizations with x<=0 or y<=0 (default=True)
    filter_parameters : bool, optional
        Remove localizations with sx<=0, sy<=0, or lp<=0 (default=True)
    removed_locs_path : str, optional
        Path to save removed localizations for debugging (default=None)

    Returns
    -------
    locs : np.rec.array
        Localizations that pass the sanity checks.
    """

    initial_count = len(locs)
    removed_locs_list = []
    removal_reasons = []

    # Remove any rows with inf or nan in any field
    if remove_invalid:
        valid_mask = _np.all(
            _np.array([_np.isfinite(locs[_]) for _ in locs.dtype.names]),
            axis=0,
        )
        invalid_mask = ~valid_mask
        invalid_count = _np.sum(invalid_mask)
        if invalid_count > 0:
            logger.warning(
                f"⚠️ Removed {invalid_count} localizations with inf/nan values"
            )
            removed_locs_list.append(locs[invalid_mask])
            removal_reasons.extend(["inf_or_nan"] * invalid_count)
        locs = locs[valid_mask]

    # Boundary checks
    if filter_boundary:
        boundary_mask = (locs.x > 0) & (locs.y > 0)
        boundary_removed_mask = ~boundary_mask
        boundary_removed = _np.sum(boundary_removed_mask)
        if boundary_removed > 0:
            logger.warning(
                f"⚠️ Removed {boundary_removed} localizations outside image bounds (x<=0 or y<=0)"
            )
            removed_locs_list.append(locs[boundary_removed_mask])
            removal_reasons.extend(["boundary_x_or_y<=0"] * boundary_removed)
        locs = locs[boundary_mask]

    # Parameter checks
    if filter_parameters:
        param_mask = _np.ones(len(locs), dtype=bool)  # start with all valid
        for field in ["sx", "sy", "lp"]:
            if field in locs.dtype.names:
                param_mask &= (locs[field] > 0)
            else:
                logger.info(f"ℹ️ Sanity check: field '{field}' not present in data; skipping '{field} > 0' check.")
    
        param_removed_mask = ~param_mask
        param_removed = _np.sum(param_removed_mask)
        if param_removed > 0:
            logger.warning(
                f"⚠️ Removed {param_removed} localizations with invalid parameters (sx<=0, sy<=0, or lp<=0)"
            )
            removed_locs_list.append(locs[param_removed_mask])
            removal_reasons.extend(["invalid_sx_sy_or_lp"] * param_removed)
        locs = locs[param_mask]

    total_removed = initial_count - len(locs)
    if total_removed > 0:
        logger.info(
            f"📊 Sanity check summary: {total_removed}/{initial_count} localizations removed ({100 * total_removed / initial_count:.2f}%)"
        )

        # Save removed localizations if path is provided
        if removed_locs_path and removed_locs_list:
            try:
                import pandas as pd

                removed_locs_combined = _np.concatenate(removed_locs_list)

                # Create DataFrame with all fields from locs plus removal reason
                removed_df = pd.DataFrame(removed_locs_combined)
                removed_df["removal_reason"] = removal_reasons
            except Exception as e:
                logger.error(f"❌ Failed to save removed localizations: {e}")
    else:
        logger.info(f"✅ All {initial_count} localizations passed sanity checks")

    return locs


def _csv_numeric_column(
    data: pd.DataFrame,
    column_names: tuple[str, ...],
    *,
    default: float,
) -> pd.Series:
    """Use the first present column among ``column_names``; coerce numeric; NaN -> ``default``."""
    for name in column_names:
        if name in data.columns:
            s = pd.to_numeric(data[name], errors="coerce")
            return s.fillna(default)
    return pd.Series(_np.full(len(data), default), index=data.index, dtype=_np.float64)


def csv2hdf(path, pixelsize, sanity_config=None):
    """Convert a Picasso-style localization CSV to HDF5 for AIM drift correction.

    Args:
        path: Input CSV path; output is ``<stem>_locs.hdf5`` beside the input.
        pixelsize: Pixel size in nanometers used to convert coordinates to pixels.
        sanity_config: Optional sanity-check settings forwarded to :func:`save_locs`.

    Returns:
        Path to the written HDF5 file.

    Raises:
        Exception: Re-raised when required columns cannot be parsed.
    """
    data = read_localization_csv(path)
    try:
        frames = data["frameIndex"].astype(int)
        # make sure frames start at zero:
        frames = frames - _np.min(frames)
        x = data["x (nm)"] / pixelsize
        y = data["y (nm)"] / pixelsize

        # Build optional columns only if raw input contains them (avoid creating phantom fields)
        arrays = [frames, x, y]
        dtype = [("frame", "u4"), ("x", "f4"), ("y", "f4")]

        # photons / intensity
        if any(name in data.columns for name in ("intensity (photons)", "intensity", "photons")):
            photons = _csv_numeric_column(data, ("intensity (photons)", "intensity", "photons"), default=_np.nan)
            arrays.append(photons)
            dtype.append(("photons", "f4"))

        # sx
        if any(name in data.columns for name in ("sigmaX (nm)", "sigmaX", "sx")):
            sx = _csv_numeric_column(data, ("sigmaX (nm)", "sigmaX", "sx"), default=_np.nan) / pixelsize
            arrays.append(sx)
            dtype.append(("sx", "f4"))

        # sy
        if any(name in data.columns for name in ("sigmaY (nm)", "sigmaY", "sy")):
            sy = _csv_numeric_column(data, ("sigmaY (nm)", "sigmaY", "sy"), default=_np.nan) / pixelsize
            arrays.append(sy)
            dtype.append(("sy", "f4"))

        # bg
        if any(name in data.columns for name in ("background (photons/nm^2)", "bg", "background")):
            bg = _csv_numeric_column(data, ("background (photons/nm^2)", "bg", "background"), default=_np.nan)
            arrays.append(bg)
            dtype.append(("bg", "f4"))

        # p-value
        if any(name in data.columns for name in ("p-value", "pvalue")):
            p_value = _csv_numeric_column(data, ("p-value", "pvalue"), default=_np.nan)
            arrays.append(p_value)
            dtype.append(("pvalue", "f4"))

        # channelIndex (required)
        channel_index = data["channelIndex"]
        arrays.append(channel_index)
        dtype.append(("channelIndex", "u4"))

        # lp
        if any(name in data.columns for name in ("localization precision (nm)", "lp", "localization_precision")):
            lp = _csv_numeric_column(data, ("localization precision (nm)", "lp", "localization_precision"), default=_np.nan)
            arrays.append(lp)
            dtype.append(("lp", "f4"))

        LOCS_DTYPE = dtype

        # Create record array using only available fields
        locs = _np.rec.array(tuple(arrays), dtype=LOCS_DTYPE)

        # Sort by channelIndex first, then by frame
        locs.sort(kind="mergesort", order=["channelIndex", "frame"])

        img_info = {}
        img_info["Generated by"] = "Picasso csv2hdf"
        img_info["Frames"] = int(_np.max(frames)) + 1
        img_info["Height"] = int(_np.ceil(_np.max(y)))
        img_info["Width"] = int(_np.ceil(_np.max(x)))
        img_info["PixelSize"] = pixelsize

        info = []
        info.append(img_info)

        base, _ = os.path.splitext(path)
        out_path = base + "_locs.hdf5"
        save_locs(out_path, locs, info, sanity_config=sanity_config)
        logger.info(f"Saved to {out_path}.")
        return out_path
    except Exception as e:
        logger.error("Error. Datatype not understood.")
        raise e


def hdf2csv(path, new_info):
    """Export drift-corrected HDF5 localizations back to CSV in nanometers.

    Args:
        path: HDF5 localization file produced by the AIM pipeline.
        new_info: Image metadata dict containing ``PixelSize`` for nm conversion.

    Returns:
        Path to the written CSV file.
    """
    locs = load_locs(path)[0]
    base, _ = os.path.splitext(path)
    out_path = base + ".csv"
    # Select only fields that actually exist in the HDF5 dataset
    available_fields = [f for f in ["x", "y", "photons", "frame", "sx", "sy", "bg", "pvalue", "channelIndex", "lp"] if f in locs.dtype.names]
    locs = locs[available_fields].copy()
    pixel_size = new_info["PixelSize"]
    if "x" in locs.dtype.names:
        locs.x *= pixel_size
    if "y" in locs.dtype.names:
        locs.y *= pixel_size
    if "sx" in locs.dtype.names:
        locs.sx *= pixel_size
    if "sy" in locs.dtype.names:
        locs.sy *= pixel_size

    df = pd.DataFrame(locs)
    df.to_csv(out_path, sep=",", encoding="utf-8")
    logger.info(f"A total of {len(locs)} rows loaded.")

    return out_path
