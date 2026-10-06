"""Tests for small utility helpers."""

from __future__ import annotations

from dSTORMQuant.utils.utils import (
    extract_channel_labels_from_stem,
    normalize_metadata_header,
)


def test_normalize_metadata_header_collapses_whitespace() -> None:
    assert normalize_metadata_header("  File   Name ") == "file name"
    assert normalize_metadata_header("first_channel_index") == "first_channel_index"


def test_extract_channel_labels_from_short_stem() -> None:
    ch1, ch2 = extract_channel_labels_from_stem("Test_PEX14_TOMM20")
    assert ch1 == "PEX14"
    assert ch2 == "TOMM20"


def test_extract_channel_labels_from_long_stem() -> None:
    ch1, ch2 = extract_channel_labels_from_stem(
        "AHA_24022_U2OS-WT_AST0126_AST0063-2_posXY1"
    )
    assert ch1 == "AST0126"
    assert ch2 == "AST0063-2"
