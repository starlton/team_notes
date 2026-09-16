"""Exception types shared across the app.

Every error that a user could plausibly hit carries a `remedy`: a short, plain
sentence telling them what to do about it. The dashboard and the CLI both show
it verbatim, so the app never dead-ends on a stack trace.
"""

from __future__ import annotations


class TeamsNotesError(Exception):
    """Base class for every error this app raises deliberately."""

    def __init__(self, message: str, remedy: str = "") -> None:
        super().__init__(message)
        self.message = message
        self.remedy = remedy

    def __str__(self) -> str:
        return f"{self.message} {self.remedy}".strip()


class DependencyMissingError(TeamsNotesError):
    """An optional third-party package or external program is not available."""


class AudioDeviceError(TeamsNotesError):
    """Audio devices could not be opened."""


class CaptureError(TeamsNotesError):
    """Something went wrong while recording."""


class TranscriptionError(TeamsNotesError):
    """Whisper could not produce a transcript."""


class DiarizationError(TeamsNotesError):
    """pyannote could not label speakers."""


class OllamaError(TeamsNotesError):
    """The local LLM was unreachable or returned something unusable."""


class StorageError(TeamsNotesError):
    """The database or the audio directory could not be used."""


class NotFoundError(TeamsNotesError):
    """A requested record does not exist."""
