"""Process logging: stderr plus an optional custom log file."""

from __future__ import annotations

import logging
from pathlib import Path


LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"


def configure_logging(level: str, log_file: str | None = None) -> None:
    """Send logs to stderr, and also to ``log_file`` when set."""
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file:
        path = Path(log_file).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(path, encoding="utf-8"))

    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=LOG_FORMAT,
        handlers=handlers,
        force=True,
    )
    if log_file:
        logging.getLogger(__name__).info("file_logging path=%s", log_file)
