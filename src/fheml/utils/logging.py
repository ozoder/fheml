"""Logging configuration utilities."""

import logging
import sys
from datetime import datetime


def setup_logging(
    level: int = logging.INFO,
    log_file: str | None = None,
    format_string: str | None = None,
) -> None:
    """
    Set up logging configuration.

    Args:
        level: Logging level
        log_file: Optional log file path
        format_string: Optional custom format string
    """
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    handlers = [logging.StreamHandler(sys.stdout)]

    if log_file is None:
        # Generate default log filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"fheml_training_{timestamp}.log"

    handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=level,
        format=format_string,
        handlers=handlers,
        force=True,  # Override existing configuration
    )

    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured. Log file: {log_file}")


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance.

    Args:
        name: Logger name

    Returns:
        Logger instance
    """
    return logging.getLogger(name)
