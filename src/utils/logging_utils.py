"""
Logging utilities for the LLM fine-tuning platform.
Uses loguru for structured, colourful logging with configurable level.
"""
from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger


def setup_logger(
    level: str = "INFO",
    log_file: Path | None = None,
    rotation: str = "100 MB",
    retention: str = "7 days",
) -> None:
    """
    Configure loguru logger with console and optional file sink.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR).
        log_file: Optional path to write logs to disk.
        rotation: When to rotate the log file.
        retention: How long to keep rotated files.
    """
    # On Windows the console is often cp1252; model output routinely contains
    # em-dashes / curly quotes / other non-latin1 glyphs. Force UTF-8 so a
    # print() of a generated answer never crashes with UnicodeEncodeError.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except (AttributeError, ValueError):
            pass

    logger.remove()  # Remove default handler
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=level,
        colorize=True,
    )
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            str(log_file),
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            level=level,
            rotation=rotation,
            retention=retention,
            encoding="utf-8",
        )
    logger.info(f"Logger initialized at level={level}")


def get_logger(name: str):
    """Return a loguru logger bound to the given module name."""
    return logger.bind(name=name)
