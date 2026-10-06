"""Tests for nearest-neighbor distance helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from dSTORMQuant.analysis.nearest_neighbor.nearest_neighbor_analysis import (
    calculate_nearest_neighbor_distances,
)


def test_intra_channel_nearest_neighbor_distance() -> None:
    data = pd.DataFrame({"x": [0.0, 3.0, 10.0], "y": [0.0, 4.0, 10.0]})
    distances = calculate_nearest_neighbor_distances(data, data2=None, k=1)

    assert distances.shape == (3,)
    # Point (0,0) -> (3,4) is distance 5
    assert np.isclose(distances[0], 5.0)
    assert np.isclose(distances[1], 5.0)


def test_inter_channel_nearest_neighbor_distance() -> None:
    ch1 = pd.DataFrame({"x": [0.0, 10.0], "y": [0.0, 0.0]})
    ch2 = pd.DataFrame({"x": [3.0], "y": [4.0]})
    distances = calculate_nearest_neighbor_distances(ch1, data2=ch2, k=1)

    assert distances.shape == (2,)
    assert np.isclose(distances[0], 5.0)
    assert np.isclose(distances[1], np.hypot(7.0, 4.0))


def test_nearest_neighbor_empty_input() -> None:
    empty = pd.DataFrame(columns=["x", "y"])
    distances = calculate_nearest_neighbor_distances(empty, data2=None, k=1)
    assert distances.size == 0
