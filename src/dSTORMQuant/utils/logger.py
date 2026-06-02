from __future__ import annotations

import logging
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path

# Global logger instance (singleton)
_logger: logging.Logger | None = None


def setup_logger(
    name: str = "smlm_pipeline",
    level: int = logging.INFO,
    log_dir: Path | None = None,
    enable_file: bool = True,
    for_tests: bool = False,
    max_bytes: int = 5_000_000,
    backup_count: int = 5,
) -> logging.Logger:
    """
    Create and configure a logger that logs to both file and console.

    Parameters
    ----------
    name : str
        Name of the logger.
    level : int
        Logging level (e.g., logging.INFO).
    log_dir : Optional[Path]
        Directory to store log files. If None, defaults to project-root/logs.
    enable_file : bool
        If False, no file handler will be added.
    for_tests : bool
        If True, do not create file handlers (useful for unit tests).
    max_bytes : int
        Max bytes per log file before rotation.
    backup_count : int
        Number of rotated backup files to keep.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """

    # Resolve default log directory to project root / logs
    if log_dir is None:
        # Path(__file__) -> .../src/dSTORMQuant/utils/logger.py
        # parents[3] should point to the project root
        log_dir = Path(__file__).resolve().parents[3] / "logs"

    # Create logger
    logger: logging.Logger = logging.getLogger(name)
    logger.setLevel(level)

    # Prevent adding duplicate handlers if logger already configured
    if not logger.handlers:
        # File handler (optional)
        if enable_file and not for_tests:
            try:
                log_dir.mkdir(parents=True, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                log_file = log_dir / f"{name}_{timestamp}.log"

                file_handler = RotatingFileHandler(
                    filename=str(log_file),
                    maxBytes=max_bytes,
                    backupCount=backup_count,
                    encoding="utf-8",
                )
                file_handler.setLevel(level)
                file_formatter = logging.Formatter(
                    "%(asctime)s - %(levelname)s - %(message)s"
                )
                file_handler.setFormatter(file_formatter)
                logger.addHandler(file_handler)
            except Exception:
                # If file handler cannot be created, continue with console logging
                pass

        # Console handler
        console_handler: logging.StreamHandler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_formatter: logging.Formatter = logging.Formatter(
            "%(levelname)s: %(message)s"
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    return logger


def get_logger() -> logging.Logger:
    """
    Get the singleton logger instance.

    Initializes the logger on first call, returns the same instance on subsequent calls.
    This ensures consistent logging configuration across the entire application.

    Returns
    -------
    logging.Logger
        The configured singleton logger instance.
    """
    global _logger
    if _logger is None:
        _logger = setup_logger()
    return _logger
