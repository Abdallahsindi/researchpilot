"""Logging setup with a per-run correlation id."""

import logging
from contextvars import ContextVar

run_id_var: ContextVar[str] = ContextVar("run_id", default="-")

LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s [run=%(run_id)s] %(message)s"


class RunIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.run_id = run_id_var.get()
        return True


def configure_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    handler.addFilter(RunIdFilter())

    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level.upper())
    # httpx logs every request at INFO, which drowns out workflow logs.
    logging.getLogger("httpx").setLevel(logging.WARNING)
