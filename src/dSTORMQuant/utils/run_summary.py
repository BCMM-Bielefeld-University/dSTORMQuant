"""Batch run summary for all processed localization CSVs."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dSTORMQuant.processing.filtering.filtering import FilterStepReport
from dSTORMQuant.utils.logger import get_logger

logger = get_logger()


@dataclass
class FileRunRecord:
    """Outcome of processing one input CSV."""

    input_file: str
    status: str  # ok | failed
    reason: str | None = None
    duration_seconds: float = 0.0
    filters_applied: list[str] = field(default_factory=list)
    filters_disabled: list[str] = field(default_factory=list)
    filters_skipped_missing_columns: list[str] = field(default_factory=list)
    expected_csv_columns_for_skipped: dict[str, list[str]] = field(
        default_factory=dict
    )
    skipped_missing_columns: bool = False


def filtering_summary_from_reports(
    reports: list[FilterStepReport],
) -> dict[str, Any]:
    """Build compact filter fields for a :class:`FileRunRecord`."""
    applied: list[str] = []
    disabled: list[str] = []
    skipped: list[str] = []
    expected: dict[str, list[str]] = {}
    for r in reports:
        if not r.enabled:
            disabled.append(r.step_name)
        elif r.applied:
            applied.append(r.step_name)
        elif r.skipped_missing_columns:
            skipped.append(r.step_name)
            expected[r.step_name] = list(r.expected_csv_columns)
    return {
        "filters_applied": applied,
        "filters_disabled": disabled,
        "filters_skipped_missing_columns": skipped,
        "expected_csv_columns_for_skipped": expected,
        "skipped_missing_columns": bool(skipped),
    }


def format_run_summary_text(
    records: list[FileRunRecord],
    *,
    total_seconds: float,
) -> str:
    """Human-readable batch summary."""
    n_ok = sum(1 for r in records if r.status == "ok")
    n_failed = sum(1 for r in records if r.status == "failed")
    lines = [
        "dSTORMQuant run summary",
        f"Generated (UTC): {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}",
        f"Total files: {len(records)}",
        f"OK: {n_ok}    Failed: {n_failed}",
        f"Total wall time: {total_seconds:.2f} s",
        "",
    ]
    for r in records:
        lines.append("-" * 60)
        lines.append(f"File: {r.input_file}")
        lines.append(f"Status: {r.status}")
        lines.append(f"Duration: {r.duration_seconds:.2f} s")
        if r.reason:
            lines.append(f"Reason: {r.reason}")
        if r.status == "ok" or r.filters_applied or r.filters_skipped_missing_columns:
            lines.append("Filtering:")
            if r.filters_applied:
                lines.append(f"  applied: {', '.join(r.filters_applied)}")
            if r.filters_disabled:
                lines.append(f"  disabled in config: {', '.join(r.filters_disabled)}")
            if r.filters_skipped_missing_columns:
                lines.append(
                    "  skipped (CSV headers must match documented names): "
                    + ", ".join(r.filters_skipped_missing_columns)
                )
                for step, cols in r.expected_csv_columns_for_skipped.items():
                    named = ", ".join(f"'{c}'" for c in cols)
                    lines.append(f"    - {step}: expected {named}")
            if (
                not r.filters_applied
                and not r.filters_disabled
                and not r.filters_skipped_missing_columns
                and r.status == "failed"
            ):
                lines.append("  (filtering not reached)")
        lines.append("")
    lines.append("-" * 60)
    lines.append(
        "Per-file details also in each output zip under test_data/filtering_report.json."
    )
    return "\n".join(lines).rstrip() + "\n"


def write_batch_run_summary(
    records: list[FileRunRecord],
    output_dir: str | Path,
    *,
    total_seconds: float,
) -> tuple[Path, Path]:
    """Write ``run_summary.txt`` and ``run_summary.json`` under ``output_dir``."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    txt_path = out / "run_summary.txt"
    json_path = out / "run_summary.json"

    text = format_run_summary_text(records, total_seconds=total_seconds)
    txt_path.write_text(text, encoding="utf-8")

    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "total_files": len(records),
        "ok": sum(1 for r in records if r.status == "ok"),
        "failed": sum(1 for r in records if r.status == "failed"),
        "total_seconds": round(total_seconds, 3),
        "files": [asdict(r) for r in records],
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    logger.info("Wrote batch run summary: %s and %s", txt_path, json_path)
    return txt_path, json_path
