"""The service façade: consent gating, recording lifecycle, retention."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.capture.recorder import RecordingResult, TrackResult
from app.capture.wav_io import write_wav
from app.db.repository import (STATUS_COMPLETE, STATUS_DISCARDED, STATUS_FAILED,
                               STATUS_RECORDED, STATUS_RECORDING)
from app.errors import CaptureError, NotFoundError
from app.pipeline.service import MeetingService
from tests.conftest import tone


class FakeRecorder:
    """Stands in for DualRecorder without touching a sound card."""

    instances: list["FakeRecorder"] = []

    def __init__(self, output_dir: Path, **kwargs) -> None:
        self.output_dir = Path(output_dir)
        self.kwargs = kwargs
        self.is_recording = False
        self.elapsed_seconds = 0.0
        self.aborted = False
        self.duration = 120.0
        self.start_error: Exception | None = None
        FakeRecorder.instances.append(self)

    def start(self) -> None:
        if self.start_error:
            raise self.start_error
        self.is_recording = True

    def levels(self) -> dict[str, float]:
        return {"loopback": -20.0, "microphone": float("-inf")}

    def stop(self) -> RecordingResult:
        self.is_recording = False
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = write_wav(self.output_dir / "mixed.wav", tone(1.0), 16000)
        return RecordingResult(
            mixed_path=path, duration_seconds=self.duration, sample_rate=16000,
            tracks=[TrackResult("loopback", path, 16000, self.duration, 0.0)],
            warnings=["microphone was not recorded"],
        )

    def abort(self) -> None:
        self.aborted = True
        self.is_recording = False


@pytest.fixture
def fake_service(settings, database, monkeypatch) -> MeetingService:
    FakeRecorder.instances.clear()
    monkeypatch.setattr("app.pipeline.service.DualRecorder", FakeRecorder)
    svc = MeetingService(settings, database=database)
    svc.start()
    # Processing is exercised in test_processor; here it must not actually run.
    monkeypatch.setattr(svc.jobs, "submit", lambda *a, **k: True)
    yield svc
    svc.shutdown()


# --- consent ----------------------------------------------------------------

def test_recording_is_blocked_until_the_notice_is_acknowledged(fake_service):
    with pytest.raises(CaptureError) as excinfo:
        fake_service.start_recording()
    assert "notice" in str(excinfo.value).lower()
    assert not fake_service.is_recording


def test_acknowledging_the_notice_unblocks_recording(fake_service):
    fake_service.acknowledge_consent(True)
    assert fake_service.start_recording() > 0
    assert fake_service.is_recording


def test_the_notice_survives_a_restart(settings, database):
    first = MeetingService(settings, database=database)
    first.acknowledge_consent(True)
    assert MeetingService(settings, database=database).consent_acknowledged is True


def test_the_notice_can_be_reset(fake_service):
    fake_service.acknowledge_consent(True)
    fake_service.acknowledge_consent(False)
    assert fake_service.consent_acknowledged is False


def test_consent_state_carries_the_notice_text(fake_service):
    state = fake_service.consent_state()
    assert "responsible" in state["notice"]
    assert state["reminder_enabled"] is True


def test_the_per_meeting_informed_flag_is_stored(fake_service):
    fake_service.acknowledge_consent(True)
    meeting_id = fake_service.start_recording(participants_informed=True)
    assert fake_service.repo.get_meeting(meeting_id)["participants_informed"] is True


# --- recording lifecycle ----------------------------------------------------

def test_start_then_stop_records_the_meeting(fake_service):
    fake_service.acknowledge_consent(True)
    meeting_id = fake_service.start_recording(title="Sync")
    assert fake_service.repo.get_meeting(meeting_id)["status"] == STATUS_RECORDING

    result = fake_service.stop_recording(process=True)
    meeting = fake_service.repo.get_meeting(meeting_id)
    assert result["meeting_id"] == meeting_id
    assert meeting["status"] == STATUS_RECORDED
    assert meeting["duration_seconds"] == pytest.approx(120.0)
    assert Path(meeting["audio_path"]).is_file()
    assert "microphone was not recorded" in meeting["warnings"]
    assert not fake_service.is_recording


def test_two_recordings_at_once_are_refused(fake_service):
    fake_service.acknowledge_consent(True)
    fake_service.start_recording()
    with pytest.raises(CaptureError):
        fake_service.start_recording()


def test_stopping_when_nothing_is_recording_is_refused(fake_service):
    with pytest.raises(CaptureError):
        fake_service.stop_recording()


def test_a_failed_start_does_not_leave_a_stuck_meeting(fake_service, monkeypatch):
    fake_service.acknowledge_consent(True)

    def failing_recorder(output_dir, **kwargs):
        recorder = FakeRecorder(output_dir, **kwargs)
        recorder.start_error = CaptureError("device busy", "close the other app")
        return recorder

    monkeypatch.setattr("app.pipeline.service.DualRecorder", failing_recorder)
    with pytest.raises(CaptureError):
        fake_service.start_recording()

    assert not fake_service.is_recording
    meetings = fake_service.repo.list_meetings()
    assert meetings[0]["status"] == STATUS_FAILED


def test_a_short_recording_is_discarded(fake_service):
    fake_service.acknowledge_consent(True)
    meeting_id = fake_service.start_recording(source="auto")
    FakeRecorder.instances[-1].duration = 12.0

    result = fake_service.stop_recording(process=True, min_seconds=60.0)
    assert result["discarded"] is True
    meeting = fake_service.repo.get_meeting(meeting_id)
    assert meeting["status"] == STATUS_DISCARDED
    assert not meeting["audio_path"]
    assert not (fake_service.settings.audio_dir / f"meeting-{meeting_id:06d}").exists()


def test_discarding_a_live_recording_removes_its_audio(fake_service):
    fake_service.acknowledge_consent(True)
    meeting_id = fake_service.start_recording()
    assert fake_service.discard_recording() == meeting_id
    assert FakeRecorder.instances[-1].aborted is True
    assert fake_service.repo.get_meeting(meeting_id)["status"] == STATUS_DISCARDED
    assert not fake_service.is_recording


def test_discarding_when_idle_returns_nothing(fake_service):
    assert fake_service.discard_recording() is None


def test_shutdown_saves_an_in_flight_recording(fake_service):
    fake_service.acknowledge_consent(True)
    meeting_id = fake_service.start_recording()
    fake_service.shutdown()
    assert fake_service.repo.get_meeting(meeting_id)["status"] == STATUS_RECORDED


# --- meetings ---------------------------------------------------------------

def test_deleting_a_meeting_removes_its_audio(fake_service):
    fake_service.acknowledge_consent(True)
    meeting_id = fake_service.start_recording()
    fake_service.stop_recording(process=False)
    audio_dir = fake_service.settings.audio_dir / f"meeting-{meeting_id:06d}"
    assert audio_dir.exists()

    fake_service.delete_meeting(meeting_id)
    assert not audio_dir.exists()
    assert fake_service.repo.find_meeting(meeting_id) is None


def test_deleting_a_missing_meeting_raises(fake_service):
    with pytest.raises(NotFoundError):
        fake_service.delete_meeting(4242)


def test_queueing_a_meeting_without_audio_is_refused(fake_service):
    meeting_id = fake_service.repo.create_meeting("No audio")
    with pytest.raises(NotFoundError):
        fake_service.queue_processing(meeting_id)


def test_audio_path_is_confined_to_the_data_directory(fake_service, tmp_path):
    """A tampered database row must not make the app read an arbitrary file."""
    outside = tmp_path / "secret.wav"
    write_wav(outside, tone(0.2), 16000)
    meeting_id = fake_service.repo.create_meeting("Tampered")
    fake_service.repo.update_meeting(meeting_id, audio_path=str(outside))
    assert fake_service.audio_path_for(meeting_id) is None


def test_audio_path_is_none_when_the_file_is_gone(fake_service):
    meeting_id = fake_service.repo.create_meeting("Gone")
    fake_service.repo.update_meeting(
        meeting_id, audio_path=str(fake_service.settings.audio_dir / "nope.wav"))
    assert fake_service.audio_path_for(meeting_id) is None


# --- status and health ------------------------------------------------------

def test_status_reports_recording_state_and_levels(fake_service):
    fake_service.acknowledge_consent(True)
    assert fake_service.status()["recording"] is False

    fake_service.start_recording()
    status = fake_service.status()
    assert status["recording"] is True
    assert status["levels"]["loopback"] == -20.0
    assert status["levels"]["microphone"] is None       # -inf becomes null
    assert status["consent"]["acknowledged"] is True


def test_health_reports_a_missing_hugging_face_token(settings, database):
    from config import Secret

    object.__setattr__(settings, "hf_token", Secret(""))
    health = MeetingService(settings, database=database).health()
    assert health["diarization"]["ok"] is False
    assert "HF_TOKEN" in health["diarization"]["remedy"]


def test_health_reports_ollama_being_down(fake_service):
    # Nothing is listening on the configured port during tests.
    assert fake_service.health()["ollama"]["ok"] is False


def test_startup_recovers_meetings_left_by_a_crash(settings, database):
    from app.db.repository import Repository

    repo = Repository(database)
    stuck = repo.create_meeting("Crashed")
    MeetingService(settings, database=database).start()
    assert repo.get_meeting(stuck)["status"] == STATUS_FAILED


def test_retention_deletes_old_audio_but_keeps_the_notes(settings, database):
    from app.db.repository import Repository

    object.__setattr__(settings, "audio_retention_days", 30)
    repo = Repository(database)
    meeting_id = repo.create_meeting("Old", started_at="2000-01-01T00:00:00Z")
    audio_dir = settings.audio_dir / f"meeting-{meeting_id:06d}"
    audio_dir.mkdir(parents=True, exist_ok=True)
    write_wav(audio_dir / "mixed.wav", tone(0.2), 16000)
    repo.update_meeting(meeting_id, status=STATUS_COMPLETE,
                        audio_path=str(audio_dir / "mixed.wav"))

    MeetingService(settings, database=database).start()
    assert not audio_dir.exists()
    assert repo.get_meeting(meeting_id)["audio_path"] == ""
    assert repo.get_meeting(meeting_id)["title"] == "Old"


def test_auto_detect_toggle_persists(fake_service):
    fake_service.set_auto_detect(False)
    assert fake_service.auto_detect_enabled is False
    fake_service.set_auto_detect(True)
    assert fake_service.auto_detect_enabled is True
