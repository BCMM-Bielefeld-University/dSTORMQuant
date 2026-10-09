"""Tests for batch run_summary.txt / run_summary.json."""

from __future__ import annotations

import json
from pathlib import Path

from dSTORMQuant.utils.run_summary import (
    FileRunRecord,
    format_run_summary_text,
    write_batch_run_summary,
)


def test_format_and_write_run_summary(tmp_path: Path) -> None:
    records = [
        FileRunRecord(
            input_file="ok.csv",
            status="ok",
            duration_seconds=12.5,
            filters_applied=["sigma_filter", "photons_count_filter"],
            filters_skipped_missing_columns=["pvalue_filter"],
            expected_csv_columns_for_skipped={
                "pvalue_filter": ["p-value", "pvalue"],
            },
            skipped_missing_columns=True,
        ),
        FileRunRecord(
            input_file="bad.csv",
            status="failed",
            reason="Missing required columns: ['x (nm)']",
            duration_seconds=0.2,
        ),
    ]

    text = format_run_summary_text(records, total_seconds=12.7)
    assert "OK: 1    Failed: 1" in text
    assert "File: ok.csv" in text
    assert "pvalue_filter" in text
    assert "File: bad.csv" in text
    assert "Missing required columns" in text

    txt_path, json_path = write_batch_run_summary(
        records, tmp_path, total_seconds=12.7
    )
    assert txt_path.exists()
    assert json_path.exists()
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["ok"] == 1
    assert payload["failed"] == 1
    assert payload["files"][0]["input_file"] == "ok.csv"
