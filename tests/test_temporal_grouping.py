"""Tests for spatiotemporal grouping."""

from __future__ import annotations

import pandas as pd

from dSTORMQuant.processing.filtering.temporal_grouping import spatiotemporal_grouping


def test_spatiotemporal_grouping_merges_nearby_points() -> None:
    df = pd.DataFrame(
        {
            "x": [0.0, 5.0, 1000.0],
            "y": [0.0, 5.0, 1000.0],
            "frame": [1, 2, 50],
        }
    )
    grouped = spatiotemporal_grouping(
        df, max_frame_gap=2, max_distance_nm=50.0, channel_index=1
    )

    assert len(grouped) == 2
    assert grouped["n_locs"].sum() == 3
    assert set(grouped["channelIndex"].unique()) == {1}
    assert {"x", "y", "track_ID", "n_locs", "first_frame", "last_frame", "duration"}.issubset(
        grouped.columns
    )


def test_spatiotemporal_grouping_keeps_isolated_points_separate() -> None:
    df = pd.DataFrame(
        {
            "x": [0.0, 500.0],
            "y": [0.0, 500.0],
            "frame": [1, 1],
        }
    )
    grouped = spatiotemporal_grouping(
        df, max_frame_gap=2, max_distance_nm=10.0, channel_index=None
    )

    assert len(grouped) == 2
    assert (grouped["n_locs"] == 1).all()
