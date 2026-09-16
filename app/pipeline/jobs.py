"""A single-worker background job queue.

Processing runs one meeting at a time on purpose: diarization and the LLM each
want most of the machine's cores and several gigabytes of RAM, so running two
at once on a 16GB laptop makes both slower and risks an out-of-memory kill.

Jobs are keyed, so asking twice to process the same meeting queues it once.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from app.logging_setup import get_logger

log = get_logger(__name__)


@dataclass
class Job:
    key: str
    description: str
    run: Callable[[], Any]
    queued_at: float = field(default_factory=time.time)


@dataclass
class JobStatus:
    key: str
    description: str
    state: str  # queued | running | done | failed
    error: str = ""
    started_at: float = 0.0
    finished_at: float = 0.0


class JobQueue:
    """Runs queued callables on one background thread."""

    HISTORY_LIMIT = 50

    def __init__(self) -> None:
        self._queue: queue.Queue[Job | None] = queue.Queue()
        self._lock = threading.Lock()
        self._pending: dict[str, JobStatus] = {}
        self._history: list[JobStatus] = []
        self._current: JobStatus | None = None
        self._worker: threading.Thread | None = None
        self._stopping = threading.Event()

    # --- lifecycle --------------------------------------------------------

    def start(self) -> None:
        with self._lock:
            if self._worker is not None and self._worker.is_alive():
                return
            self._stopping.clear()
            self._worker = threading.Thread(target=self._loop, name="job-worker",
                                            daemon=True)
            self._worker.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stopping.set()
        self._queue.put(None)
        worker = self._worker
        if worker is not None:
            worker.join(timeout=timeout)
        self._worker = None

    # --- submission -------------------------------------------------------

    def submit(self, key: str, description: str, run: Callable[[], Any]) -> bool:
        """Queue a job. Returns False if one with the same key is already queued."""
        with self._lock:
            if key in self._pending:
                return False
            if self._current is not None and self._current.key == key:
                return False
            self._pending[key] = JobStatus(key=key, description=description,
                                           state="queued")
        self._queue.put(Job(key=key, description=description, run=run))
        self.start()
        log.info("Queued job %s (%s)", key, description)
        return True

    # --- observation ------------------------------------------------------

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "running": _status_dict(self._current) if self._current else None,
                "queued": [_status_dict(item) for item in self._pending.values()],
                "recent": [_status_dict(item) for item in reversed(self._history[-10:])],
            }

    def is_busy(self) -> bool:
        with self._lock:
            return self._current is not None or bool(self._pending)

    def job_state(self, key: str) -> str:
        with self._lock:
            if self._current is not None and self._current.key == key:
                return "running"
            if key in self._pending:
                return "queued"
            for item in reversed(self._history):
                if item.key == key:
                    return item.state
        return "unknown"

    # --- worker -----------------------------------------------------------

    def _loop(self) -> None:
        while not self._stopping.is_set():
            try:
                job = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if job is None:
                return

            with self._lock:
                status = self._pending.pop(job.key, None) or JobStatus(
                    key=job.key, description=job.description, state="queued")
                status.state = "running"
                status.started_at = time.time()
                self._current = status

            log.info("Running job %s", job.key)
            try:
                job.run()
                status.state = "done"
            except Exception as exc:  # noqa: BLE001 - a job must never kill the worker
                status.state = "failed"
                status.error = str(exc)
                log.exception("Job %s failed", job.key)
            finally:
                status.finished_at = time.time()
                with self._lock:
                    self._current = None
                    self._history.append(status)
                    del self._history[:-self.HISTORY_LIMIT]


def _status_dict(status: JobStatus) -> dict[str, Any]:
    return {
        "key": status.key,
        "description": status.description,
        "state": status.state,
        "error": status.error,
        "started_at": status.started_at,
        "finished_at": status.finished_at,
    }
