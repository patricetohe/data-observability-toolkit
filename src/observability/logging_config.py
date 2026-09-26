"""Central logging setup for the toolkit.

Every entry point (CLI, scheduler, dashboard) should call `setup_logging()`
once so log output is consistently formatted, instead of each module
configuring `logging` on its own.
"""
from __future__ import annotations

import logging
import sys

_CONFIGURED = False


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure the root 'observability' logger and return it.

    Safe to call multiple times: only configures handlers once per process.
    """
    logger = logging.getLogger("observability")

    global _CONFIGURED
    if _CONFIGURED:
        logger.setLevel(level)
        return logger

    handler = logging.StreamHandler(stream=sys.stdout)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False

    _CONFIGURED = True
    return logger
