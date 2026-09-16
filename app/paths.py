"""Filesystem helpers.

`resolve_within` is the one that matters for security: anything that turns a
stored path into a file the web layer will open goes through it, so a corrupted
or tampered database row cannot make the dashboard serve `C:\\Windows\\...`.
"""

from __future__ import annotations

import re
from pathlib import Path

from app.errors import StorageError

_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


def slugify(text: str, max_length: int = 60, fallback: str = "meeting") -> str:
    """Reduce arbitrary text to a filename-safe slug."""
    cleaned = _UNSAFE.sub("-", (text or "").strip()).strip("-.")
    cleaned = cleaned[:max_length].strip("-.")
    return cleaned or fallback


_ESCAPE = StorageError(
    "Refusing to use a file outside the app's data directory.",
    "This usually means the database holds a path the app did not create.",
)


def _has_parent_component(candidate: str) -> bool:
    """True if the path steps upwards, under either platform's separator.

    Path.resolve() alone is not enough: on Linux a backslash is an ordinary
    filename character, so "..\\..\\Windows" resolves to a harmless name
    there and to a real traversal on Windows. Checking both separators means
    the same input is rejected wherever the tests happen to run.
    """
    return any(part == ".." for part in candidate.replace("\\", "/").split("/"))


def resolve_within(root: Path, candidate: Path | str) -> Path:
    """Resolve `candidate` and assert it really sits inside `root`.

    Raises StorageError rather than returning a path outside the sandbox.
    """
    root_resolved = Path(root).resolve()
    if _has_parent_component(str(candidate)):
        raise _ESCAPE

    target = Path(candidate)
    if not target.is_absolute():
        target = root_resolved / target
    target = target.resolve()
    if target != root_resolved and root_resolved not in target.parents:
        raise _ESCAPE
    return target


def meeting_audio_dir(audio_root: Path, meeting_id: int) -> Path:
    """Per-meeting audio directory, created on demand."""
    if meeting_id <= 0:
        raise StorageError("Invalid meeting id.", "")
    path = Path(audio_root) / f"meeting-{int(meeting_id):06d}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def unique_path(path: Path) -> Path:
    """Return `path`, or `path` with a numeric suffix if it already exists."""
    if not path.exists():
        return path
    stem, suffix = path.stem, path.suffix
    for index in range(1, 1000):
        candidate = path.with_name(f"{stem}-{index}{suffix}")
        if not candidate.exists():
            return candidate
    raise StorageError(f"Could not find a free filename near {path.name}.", "")
