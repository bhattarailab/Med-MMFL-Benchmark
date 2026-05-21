"""Centralized logging configuration for fed-multimodal-benchmark.

Replaces scattered print() calls with structured Python logging. Supports
configurable verbosity levels and optional file logging.
"""

import logging
import sys
from typing import Optional


_CONFIGURED = False


def get_logger(
    name: str,
    level: int = logging.INFO,
    log_file: Optional[str] = None,
) -> logging.Logger:
    """Get or create a named logger with consistent formatting.

    Args:
        name: Logger name, typically ``__name__`` of the calling module.
        level: Logging level (e.g., ``logging.DEBUG``, ``logging.INFO``).
        log_file: Optional path to a log file. If provided, logs are written
            to both the console and the file.

    Returns:
        Configured ``logging.Logger`` instance.

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Training started")
    """
    global _CONFIGURED

    logger = logging.getLogger(name)

    if not _CONFIGURED:
        _configure_root(level=level, log_file=log_file)
        _CONFIGURED = True

    return logger


def _configure_root(
    level: int = logging.INFO,
    log_file: Optional[str] = None,
) -> None:
    """Configure the root logger with console and optional file handlers.

    Args:
        level: Logging level for the root logger.
        log_file: Optional path to a log file.
    """
    root = logging.getLogger()
    root.setLevel(level)

    if root.handlers:
        return

    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root.addHandler(console_handler)

    if log_file is not None:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)
