from __future__ import annotations

import zipfile
from pathlib import Path
from shutil import copyfileobj
from typing import Literal, Protocol


class _LoggerLike(Protocol):
    """Minimal logger interface used by output discovery helpers."""

    def info(self, msg: str) -> None:
        """Log an informational message.

        Args:
            msg: Message text.
        """

    def warning(self, msg: str) -> None:
        """Log a warning message.

        Args:
            msg: Message text.
        """

    def error(self, msg: str) -> None:
        """Log an error message.

        Args:
            msg: Message text.
        """


def _extract_zip_to_folder(
    zip_path: Path, target_dir: Path, logger: _LoggerLike | None = None
) -> bool:
    """Extract a pipeline output archive into its folder representation."""
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            for member in zf.infolist():
                normalized_name = member.filename.replace("\\", "/").lstrip("/")

                # Skip potentially unsafe paths and empty names.
                if not normalized_name or ".." in Path(normalized_name).parts:
                    continue

                destination = target_dir / normalized_name
                if member.is_dir():
                    destination.mkdir(parents=True, exist_ok=True)
                    continue

                destination.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(member, "r") as src, open(destination, "wb") as dst:
                    copyfileobj(src, dst)
        if logger:
            logger.info(f"Extracted archive: {zip_path.name} -> {target_dir.name}")
        return True
    except Exception as exc:
        if logger:
            logger.error(f"Failed to extract '{zip_path}': {exc}")
        return False


def ensure_output_folders_from_archives(
    output_base_dir: Path, logger: _LoggerLike | None = None
) -> list[Path]:
    """Ensure post-analysis folder outputs exist by auto-extracting ZIP archives.

    Args:
        output_base_dir: Root directory containing ``*.zip`` exports and folders.
        logger: Optional logger implementing :class:`_LoggerLike`.

    Returns:
        List of output folder paths that contain a non-empty ``test_data``
        subdirectory, sorted by modification time (newest first).
    """
    if not output_base_dir.exists():
        return []

    zip_files = sorted(output_base_dir.glob("*.zip"))
    for zip_path in zip_files:
        target_dir = output_base_dir / zip_path.stem
        test_data_dir = target_dir / "test_data"
        if test_data_dir.exists() and any(test_data_dir.iterdir()):
            continue

        if test_data_dir.exists() and logger:
            logger.warning(
                f"Found empty test_data folder for '{target_dir.name}'. Re-extracting from archive."
            )
        _extract_zip_to_folder(zip_path, target_dir, logger=logger)

    valid_dirs = [
        d
        for d in output_base_dir.iterdir()
        if d.is_dir()
        and not d.name.startswith(".")
        and (d / "test_data").exists()
        and any((d / "test_data").iterdir())
    ]
    valid_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return valid_dirs


def list_output_folders(
    output_base_dir: Path,
    *,
    order: Literal["newest_first", "name"] = "newest_first",
) -> list[Path]:
    """List analysis directories that already contain non-empty ``test_data``.

    This helper never extracts archives; it only inspects existing folders.

    Args:
        output_base_dir: Root directory containing per-experiment output folders.
        order: ``"newest_first"`` sorts by mtime descending; ``"name"`` sorts
            alphabetically by folder name.

    Returns:
        Paths to valid output directories.
    """
    if not output_base_dir.exists():
        return []

    valid_dirs = [
        d
        for d in output_base_dir.iterdir()
        if d.is_dir()
        and not d.name.startswith(".")
        and (d / "test_data").exists()
        and any((d / "test_data").iterdir())
    ]
    if order == "name":
        valid_dirs.sort(key=lambda p: p.name)
    else:
        valid_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return valid_dirs


def discover_experiment_basename_from_output(output_base_dir: Path) -> str | None:
    """
    Infer experiment base name (metadata / pipeline stem) from exported ``test_data`` files.

    Uses the first output folder when sorted **alphabetically** by folder name, then the
    first matching file in that folder: ``*_initial.csv``, else ``*_after_clustering.csv``
    (each group sorted alphabetically by filename).
    """
    output_dirs = list_output_folders(output_base_dir, order="name")
    if not output_dirs:
        return None

    test_data_dir = output_dirs[0] / "test_data"
    if not test_data_dir.is_dir():
        return None

    initial_files = sorted(
        p
        for p in test_data_dir.iterdir()
        if p.is_file() and p.name.endswith("_initial.csv")
    )
    if initial_files:
        return initial_files[0].name[: -len("_initial.csv")]

    cluster_files = sorted(
        p
        for p in test_data_dir.iterdir()
        if p.is_file() and p.name.endswith("_after_clustering.csv")
    )
    if cluster_files:
        return cluster_files[0].name[: -len("_after_clustering.csv")]

    return None
