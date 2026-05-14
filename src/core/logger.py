"""
Structured logging configuration using loguru.
Provides clean, readable logs with proper formatting.
"""
import sys
from pathlib import Path
from loguru import logger


def setup_logger(
    log_dir: Path = None,
    log_file: str = "pest_detection.log",
    level: str = "INFO",
    rotation: str = "10 MB",
    retention: str = "7 days"
):
    """
    Configure loguru logger with file and console output.

    Args:
        log_dir: Directory for log files. If None, uses outputs/
        log_file: Log filename
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        rotation: When to rotate log file
        retention: How long to keep old logs
    """
    # Remove default logger
    logger.remove()

    # Console logger with color
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=level,
        colorize=True
    )

    # File logger
    if log_dir is None:
        log_dir = Path("outputs")

    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / log_file

    logger.add(
        log_path,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level=level,
        rotation=rotation,
        retention=retention,
        compression="zip"
    )

    logger.info(f"Logger initialized. Logs saved to: {log_path}")
    return logger


def get_logger():
    """Get the configured logger instance."""
    return logger
