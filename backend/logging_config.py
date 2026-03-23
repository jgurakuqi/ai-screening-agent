"""Configure application logging around Loguru.

Installs the shared console handler, routes standard-library logging through
Loguru, and exposes a single ``logger`` import for the rest of the backend.
"""

import logging
import sys

from loguru import logger


def setup_logging() -> None:
    """Configure Loguru as the application's single logging backend.

    Removes Loguru's default handler, installs the console formatter used by
    this project, and redirects standard-library logging so framework and
    dependency logs appear in the same sink.
    """

    # Remove loguru's default handler
    logger.remove()

    # Add console handler at TRACE level (deepest possible)
    logger.add(
        sys.stderr,
        level="TRACE",
        format=(
            "<green>{time:HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
        colorize=True,
        backtrace=True,      # Full traceback on exceptions
        diagnose=True,        # Show variable values in tracebacks
    )

    # Intercept all stdlib logging → route through loguru
    logging.basicConfig(handlers=[_InterceptHandler()], level=0, force=True)

    # Suppress overly chatty third-party loggers
    for name in ("httpcore", "httpx", "hpack"):
        logging.getLogger(name).setLevel(logging.WARNING)


class _InterceptHandler(logging.Handler):
    """Route standard-library log records through Loguru.

    This keeps FastAPI, Uvicorn, APScheduler, and dependency logs formatted
    consistently with the application's own log messages.
    """

    def emit(self, record: logging.LogRecord) -> None:
        """Forward a stdlib log record to Loguru.

        Args:
            record: Log record emitted by a framework or dependency logger.
        """
        # Map stdlib level to loguru level
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find the caller frame (skip stdlib logging internals)
        frame, depth = logging.currentframe(), 0
        while frame and (depth == 0 or frame.f_code.co_filename == logging.__file__):
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())
