"""Logging configuration.

Logs go to the console and to a rotating file under `data/logs/`. Rotation is
capped so a long-running tray app can never fill the disk.

`RedactingFilter` strips anything that looks like a Hugging Face or bearer
token from log records, because those can otherwise leak into tracebacks
emitted by third-party libraries.
"""

from __future__ import annotations

import logging
import logging.handlers
import re
from pathlib import Path

_TOKEN_PATTERNS = (
    # Hugging Face read tokens.
    re.compile(r"\bhf_[A-Za-z0-9]{8,}\b"),
    # "Authorization: Bearer xyz", "token=xyz", "api_key: xyz". The optional
    # "Bearer " is what makes this swallow the value rather than stopping at
    # the scheme name.
    re.compile(r"(?i)\b(?:authorization|token|api[-_]?key|secret)\b"
               r"\s*[:=]\s*(?:bearer\s+)?\S+"),
    # A bare bearer credential with no preceding header name.
    re.compile(r"(?i)\bbearer\s+\S+"),
)

_CONFIGURED = False


class RedactingFilter(logging.Filter):
    """Replace credential-looking substrings in log output with ***."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = self._scrub(record.getMessage())
        record.args = ()
        return True

    @staticmethod
    def _scrub(text: str) -> str:
        for pattern in _TOKEN_PATTERNS:
            text = pattern.sub("[redacted]", text)
        return text


def setup_logging(log_dir: Path, level: str = "INFO") -> None:
    """Configure root logging once per process. Safe to call repeatedly."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    log_dir.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)-7s %(name)-28s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    redactor = RedactingFilter()

    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    console.addFilter(redactor)
    root.addHandler(console)

    file_handler = logging.handlers.RotatingFileHandler(
        log_dir / "teams-notes.log",
        maxBytes=2_000_000,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    file_handler.addFilter(redactor)
    root.addHandler(file_handler)

    # These libraries are chatty at INFO and drown out our own messages.
    for noisy in ("httpx", "httpcore", "urllib3", "faster_whisper", "speechbrain",
                  "pyannote", "torch", "numba", "uvicorn.access", "filelock"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
