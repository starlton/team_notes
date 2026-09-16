"""SQLite connection handling.

Choices that matter for a desktop app writing during a live recording:

* **WAL journalling** so the dashboard can read while the pipeline writes.
* **A connection per thread.** sqlite3 connections are not safe to share across
  threads, and this app has a tray thread, a capture thread, a worker thread and
  the web server all touching the database.
* **A busy timeout plus retries** so a brief lock contends rather than fails.
* **Foreign keys on**, which SQLite otherwise leaves off, so deleting a meeting
  really does remove its transcript.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from app.db.schema import MIGRATIONS
from app.errors import StorageError
from app.logging_setup import get_logger

log = get_logger(__name__)

BUSY_TIMEOUT_MS = 10_000
MAX_RETRIES = 4


class Database:
    """Owns the SQLite file and hands out per-thread connections."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._local = threading.local()
        self._init_lock = threading.Lock()
        self._initialised = False

    # --- connections ------------------------------------------------------

    def connect(self) -> sqlite3.Connection:
        """The calling thread's connection, opened on first use."""
        connection: sqlite3.Connection | None = getattr(self._local, "conn", None)
        if connection is not None:
            return connection

        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            connection = sqlite3.connect(
                str(self.path),
                timeout=BUSY_TIMEOUT_MS / 1000,
                isolation_level=None,  # explicit transactions, see transaction()
            )
        except sqlite3.Error as exc:
            raise StorageError(
                f"Could not open the database at {self.path}: {exc}",
                "Check that the data directory exists and is writable.",
            ) from exc

        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA synchronous=NORMAL")
        connection.execute(f"PRAGMA busy_timeout={BUSY_TIMEOUT_MS}")
        self._local.conn = connection
        return connection

    def close(self) -> None:
        """Close this thread's connection."""
        connection: sqlite3.Connection | None = getattr(self._local, "conn", None)
        if connection is not None:
            connection.close()
            self._local.conn = None

    # --- schema -----------------------------------------------------------

    def initialise(self) -> None:
        """Apply any migrations this database has not seen yet."""
        with self._init_lock:
            if self._initialised:
                return
            connection = self.connect()
            current = int(connection.execute("PRAGMA user_version").fetchone()[0])
            for version, script in enumerate(MIGRATIONS, start=1):
                if version <= current:
                    continue
                log.info("Applying database migration %d", version)
                with self.transaction() as cursor:
                    cursor.executescript(script)
                # PRAGMA cannot be parameterised; version is a loop counter.
                connection.execute(f"PRAGMA user_version={version}")
            self._initialised = True

    # --- queries ----------------------------------------------------------

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Cursor]:
        """Run statements in one transaction, retrying on a locked database."""
        connection = self.connect()
        last_error: sqlite3.OperationalError | None = None

        for attempt in range(1, MAX_RETRIES + 1):
            cursor = connection.cursor()
            try:
                cursor.execute("BEGIN IMMEDIATE")
            except sqlite3.OperationalError as exc:
                if "locked" not in str(exc).lower() and "busy" not in str(exc).lower():
                    raise StorageError(f"Database error: {exc}", "") from exc
                last_error = exc
                time.sleep(0.1 * attempt)
                continue

            try:
                yield cursor
            except Exception:
                connection.rollback()
                raise
            else:
                connection.commit()
                return

        raise StorageError(
            f"The database stayed locked after {MAX_RETRIES} attempts "
            f"({last_error}).",
            "Another copy of the app may be running. Close it and try again.",
        )

    def query(self, sql: str, params: tuple[Any, ...] = ()) -> list[sqlite3.Row]:
        try:
            return list(self.connect().execute(sql, params).fetchall())
        except sqlite3.Error as exc:
            raise StorageError(f"Database read failed: {exc}", "") from exc

    def query_one(self, sql: str, params: tuple[Any, ...] = ()) -> sqlite3.Row | None:
        rows = self.query(sql, params)
        return rows[0] if rows else None


_DATABASE: Database | None = None
_DATABASE_LOCK = threading.Lock()


def get_database(path: Path | None = None) -> Database:
    """The process-wide Database, created on first use."""
    global _DATABASE
    with _DATABASE_LOCK:
        if _DATABASE is None:
            if path is None:
                from config import get_settings

                path = get_settings().db_path
            _DATABASE = Database(path)
            _DATABASE.initialise()
        return _DATABASE


def reset_database_singleton() -> None:
    """Drop the cached Database. Used by tests."""
    global _DATABASE
    with _DATABASE_LOCK:
        if _DATABASE is not None:
            _DATABASE.close()
        _DATABASE = None
