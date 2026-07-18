"""Structured (JSON-ish) logging used by every engine.

Every engine calls ``get_logger(__name__)`` and logs its inputs, outputs and
timing through the ``timed`` context manager. Logs go to stderr as one JSON
object per line so they are greppable and machine-parseable, satisfying the
audit's "no monitoring / logging" gap.

Deliberately dependency-free (stdlib only) so it can never break the app.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from contextlib import contextmanager

_CONFIGURED = False


class _JsonFormatter(logging.Formatter):
    """Render each record as a single compact JSON line."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "engine": record.name,
            "msg": record.getMessage(),
        }
        # Attach any structured extras passed via logger.info(..., extra={"kv": {...}}).
        extra = getattr(record, "kv", None)
        if isinstance(extra, dict):
            payload.update(extra)
        return json.dumps(payload, default=str)


def _configure() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(_JsonFormatter())
    root = logging.getLogger("evbrain")
    root.setLevel(logging.INFO)
    root.handlers.clear()
    root.addHandler(handler)
    root.propagate = False
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced structured logger (e.g. ``evbrain.engine_battery``)."""
    _configure()
    short = name.split(".")[-1]
    return logging.getLogger(f"evbrain.{short}")


@contextmanager
def timed(logger: logging.Logger, event: str, **fields):
    """Log the start/end of a unit of work with its wall-clock duration.

    Usage::

        with timed(log, "score_fleet", n=len(df)):
            ...
    """
    start = time.perf_counter()
    logger.info(f"{event}:start", extra={"kv": {"event": event, **fields}})
    try:
        yield
    finally:
        ms = round((time.perf_counter() - start) * 1000, 1)
        logger.info(
            f"{event}:end",
            extra={"kv": {"event": event, "duration_ms": ms, **fields}},
        )
