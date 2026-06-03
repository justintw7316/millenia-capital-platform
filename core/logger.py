"""
core/logger.py — Structured logging for every step in the pipeline.
"""
import logging
import sys
from pathlib import Path
from typing import Optional
from rich.logging import RichHandler
from rich.console import Console

_console = Console()


def get_logger(name: str, log_file: Optional[str] = None, level: str = "INFO") -> logging.Logger:
    """
    Returns a configured logger with Rich console output and optional file output.

    Args:
        name: Logger name (typically __name__ of the calling module).
        log_file: Optional path to write logs to file as well.
        level: Logging level string (DEBUG, INFO, WARNING, ERROR).
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger  # already configured

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Rich console handler
    console_handler = RichHandler(
        console=_console,
        rich_tracebacks=True,
        show_time=True,
        show_path=False,
        markup=True,
    )
    console_handler.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.addHandler(console_handler)

    # File handler (if requested)
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    logger.propagate = False
    return logger


def get_deal_logger(deal_id: str, step: str, output_dir: str = "outputs") -> logging.Logger:
    """
    Returns a logger scoped to a specific deal and step, writing to a deal-specific log file.
    """
    log_dir = Path(output_dir) / deal_id / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = str(log_dir / f"step_{step}.log")
    return get_logger(f"millenia.{deal_id}.step{step}", log_file=log_file)
