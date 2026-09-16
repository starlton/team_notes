"""Shared test fixtures.

The heavy runtime dependencies (torch, faster-whisper, pyaudiowpatch) are never
imported here. Every stage that would need one is replaced by a fake, so the
suite runs on any platform in seconds and still exercises the real
orchestration, storage, security and merge logic.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.capture.wav_io import write_wav  # noqa: E402
from app.db.database import Database  # noqa: E402
from app.db.repository import Repository  # noqa: E402
from app.intelligence.schemas import MeetingIntelligence  # noqa: E402
from app.pipeline.service import MeetingService  # noqa: E402
from app.transcribe.models import TranscriptResult, TranscriptSegment, Word  # noqa: E402
from config import Secret, Settings  # noqa: E402


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Settings pointed at a throwaway data directory."""
    config = Settings(
        data_dir=tmp_path / "data",
        hf_token=Secret("hf_testtoken000000000000"),
        web_port=8765,
        auto_start_after_seconds=10.0,
        auto_stop_after_seconds=45.0,
        auto_min_meeting_seconds=60.0,
    )
    config.ensure_dirs()
    return config


@pytest.fixture
def database(settings: Settings) -> Database:
    db = Database(settings.db_path)
    db.initialise()
    yield db
    db.close()


@pytest.fixture
def repo(database: Database) -> Repository:
    return Repository(database)


@pytest.fixture
def service(settings: Settings, database: Database) -> MeetingService:
    svc = MeetingService(settings, database=database)
    svc.start()
    svc.acknowledge_consent(True)
    yield svc
    svc.shutdown()


# --- audio helpers ----------------------------------------------------------

def tone(seconds: float, freq: float = 440.0, rate: int = 16000,
         amplitude: float = 0.3) -> np.ndarray:
    """A simple sine wave, used wherever a test needs 'some audio'."""
    t = np.arange(int(seconds * rate), dtype=np.float32) / rate
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


@pytest.fixture
def wav_file(tmp_path: Path) -> Path:
    return write_wav(tmp_path / "sample.wav", tone(2.0), 16000)


# --- fakes ------------------------------------------------------------------

class FakeTranscriber:
    """Stands in for faster-whisper."""

    def __init__(self, segments=None, error: Exception | None = None) -> None:
        self.segments = segments if segments is not None else _default_segments()
        self.error = error
        self.calls = 0

    def transcribe(self, audio_path, progress=None):
        self.calls += 1
        if self.error:
            raise self.error
        if progress:
            progress(0.5, "Transcribing audio")
            progress(1.0, "Transcription complete")
        return TranscriptResult(segments=list(self.segments), language="en",
                                duration_seconds=12.0, model="small.en")


class FakeDiarizer:
    """Stands in for pyannote."""

    def __init__(self, turns=None, available: bool = True,
                 error: Exception | None = None) -> None:
        self.turns = turns or []
        self.available = available
        self.error = error

    def diarize(self, audio_path, progress=None):
        if self.error:
            raise self.error
        if progress:
            progress(1.0, "Speakers identified")
        return list(self.turns)


class FakeEngine:
    """Stands in for the Ollama-backed IntelligenceEngine."""

    def __init__(self, result: MeetingIntelligence | None = None,
                 error: Exception | None = None) -> None:
        self.result = result or _default_intelligence()
        self.error = error
        self.calls: list = []

    def analyse(self, segments, speaker_names=None, progress=None):
        from app.intelligence.pipeline import AnalysisOutcome

        self.calls.append((list(segments), dict(speaker_names or {})))
        if self.error:
            raise self.error
        if progress:
            progress(1.0, "Notes ready")
        return AnalysisOutcome(intelligence=self.result, warnings=[], chunks_processed=1)


class FakeOllamaClient:
    """Returns canned JSON replies, keyed by a substring of the system prompt."""

    def __init__(self, replies: dict[str, dict] | None = None,
                 num_ctx: int = 8192) -> None:
        self.replies = replies or {}
        self.num_ctx = num_ctx
        self.requests: list[tuple[str, str]] = []
        self.ready_calls = 0

    def ensure_ready(self) -> None:
        self.ready_calls += 1

    def chat_json(self, system: str, user: str, temperature: float = 0.2,
                  max_tokens: int | None = None) -> dict:
        self.requests.append((system, user))
        for marker, reply in self.replies.items():
            if marker in system or marker in user:
                if isinstance(reply, Exception):
                    raise reply
                return reply
        return {}


def _default_segments() -> list[TranscriptSegment]:
    return [
        TranscriptSegment(0.0, 4.0, "Right, shall we start with the release?",
                          words=[Word(0.0, 1.0, "Right,"), Word(1.0, 2.0, "shall"),
                                 Word(2.0, 3.0, "we"), Word(3.0, 4.0, "start?")]),
        TranscriptSegment(4.5, 9.0, "Yes. I will send the notes by Friday.",
                          words=[Word(4.5, 5.5, "Yes."), Word(5.5, 6.5, "I"),
                                 Word(6.5, 7.5, "will"), Word(7.5, 9.0, "send.")]),
    ]


def _default_intelligence() -> MeetingIntelligence:
    return MeetingIntelligence.model_validate({
        "title": "Release planning",
        "summary": "The team agreed the release date and who writes the notes.",
        "bullets": ["Release is on Friday", "Notes owed by Friday"],
        "action_items": [{"task": "Send the release notes", "owner": "Speaker 2",
                          "due": "Friday", "priority": "high"}],
        "priorities": [{"point": "Release date", "priority": "high",
                        "reason": "Everything else depends on it"}],
        "speakers": [{"speaker": "Speaker 1", "summary": "Chaired the meeting",
                      "key_points": ["Opened the release discussion"]},
                     {"speaker": "Speaker 2", "summary": "Owns the notes",
                      "key_points": ["Committed to Friday"]}],
        "drafts": [{"kind": "email", "audience": "Everyone", "subject": "Recap",
                    "body": "Here is what we agreed."}],
    })


@pytest.fixture
def fake_transcriber() -> FakeTranscriber:
    return FakeTranscriber()


@pytest.fixture
def fake_engine() -> FakeEngine:
    return FakeEngine()
