"""
utils/logging_config.py
Structured logging setup for ResearchFlow AI.

Configures a consistent log format used by all agents, tools and the
orchestrator. Log files are written to the logs/ directory; the console
mirrors the same output for real-time visibility.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path

from utils.config import config


def setup_logging(name: str = "researchflow") -> logging.Logger:
    """
    Configure and return a named logger.

    Args:
        name: Logger name, typically the module name.

    Returns:
        Configured Logger instance.
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers if called multiple times
    if logger.handlers:
        return logger

    log_level = getattr(logging, config.LOG_LEVEL.upper(), logging.INFO)
    logger.setLevel(log_level)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # --- Console handler ---
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # --- File handler ---
    try:
        log_file = config.LOGS_DIR / f"researchflow_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except (OSError, PermissionError) as exc:
        # Non-fatal: log directory might not be writable in some environments
        logger.warning("Could not create log file: %s", exc)

    # Prevent log records from propagating to the root logger (avoids duplication)
    logger.propagate = False
    return logger


# Default application-wide logger
logger = setup_logging("researchflow")
