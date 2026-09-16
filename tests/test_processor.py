"""The post-meeting pass, with every heavy stage faked."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.db.repository import (Repository, STATUS_COMPLETE, STATUS_FAILED,
                               STATUS_RECORDED)
from app.diarize.merge import SpeakerTurn
from app.errors import DiarizationError, OllamaError, TranscriptionError
from app.intelligence.schemas import MeetingIntelligence
from app.pipeline.processor import MeetingProcessor
from config import Secret, Settings
from tests.conftest import FakeDiarizer, FakeEngine, FakeTranscriber


@pytest.fixture
def recorded_meeting(repo: Repository, wav_file: Path) -> int:
    meeting_id = repo.create_meeting("Sync")
    repo.update_meeting(meeting_id, status=STATUS_RECORDED,
                        audio_path=str(wav_file), duration_seconds=12.0)
    return meeting_id


def make_processor(settings: Settings, repo: Repository, *, transcriber=None,
                   diarizer=None, engine=None) -> MeetingProcessor:
    processor = MeetingProcessor(settings, repo)
    processor.make_transcriber = lambda: transcriber or FakeTranscriber()
    processor.make_diarizer = lambda: diarizer or FakeDiarizer(turns=[
        SpeakerTurn(0.0, 4.2, "SPEAKER_01"), SpeakerTurn(4.2, 9.0, "SPEAKER_00")])
    processor.make_engine = lambda: engine or FakeEngine()
    return processor


def test_a_full_pass_stores_everything(settings, repo, recorded_meeting):
    outcome = make_processor(settings, repo).process(recorded_meeting)

    assert not outcome.failed
    meeting = repo.get_meeting(recorded_meeting)
    assert meeting["status"] == STATUS_COMPLETE
    assert meeting["progress"] == 1.0

    transcript = repo.get_transcript(recorded_meeting)
    assert len(transcript) == 2
    assert {segment.speaker for segment in transcript} == {"Speaker 1", "Speaker 2"}

    assert repo.get_summary(recorded_meeting)["summary"]
    assert repo.get_action_items(recorded_meeting)[0]["task"] == "Send the release notes"
    assert repo.get_priorities(recorded_meeting)[0]["priority"] == "high"
    assert repo.get_drafts(recorded_meeting)[0]["subject"] == "Recap"

    speakers = repo.get_speakers(recorded_meeting)
    assert len(speakers) == 2
    assert speakers[0]["summary"]                     # LLM prose attached
    assert sum(s["share"] for s in speakers) == pytest.approx(1.0, abs=1e-6)


def test_progress_is_reported_and_only_moves_forward(settings, repo, recorded_meeting):
    seen: list[float] = []
    make_processor(settings, repo).process(
        recorded_meeting, progress=lambda _stage, value: seen.append(value))
    assert seen == sorted(seen)
    assert seen[-1] == pytest.approx(1.0)


def test_missing_audio_fails_cleanly(settings, repo):
    meeting_id = repo.create_meeting("Gone")
    repo.update_meeting(meeting_id, status=STATUS_RECORDED,
                        audio_path=str(settings.audio_dir / "nope.wav"))
    outcome = make_processor(settings, repo).process(meeting_id)
    assert outcome.failed
    assert repo.get_meeting(meeting_id)["status"] == STATUS_FAILED
    assert "missing" in repo.get_meeting(meeting_id)["error"].lower()


def test_transcription_failure_fails_the_meeting(settings, repo, recorded_meeting):
    processor = make_processor(settings, repo, transcriber=FakeTranscriber(
        error=TranscriptionError("Whisper fell over", "Check the WAV file.")))
    outcome = processor.process(recorded_meeting)
    assert outcome.failed
    assert repo.get_meeting(recorded_meeting)["status"] == STATUS_FAILED


def test_diarization_failure_still_yields_a_transcript(settings, repo, recorded_meeting):
    """Losing speaker labels must not cost the user the whole meeting."""
    processor = make_processor(settings, repo, diarizer=FakeDiarizer(
        error=DiarizationError("Hugging Face said no", "Accept the model terms.")))
    outcome = processor.process(recorded_meeting)

    assert not outcome.failed
    assert repo.get_meeting(recorded_meeting)["status"] == STATUS_COMPLETE
    assert len(repo.get_transcript(recorded_meeting)) == 2
    assert any("Speakers were not identified" in w for w in outcome.warnings)


def test_no_hugging_face_token_warns_rather_than_failing(repo, tmp_path, wav_file):
    settings = Settings(data_dir=tmp_path / "data", hf_token=Secret(""))
    settings.ensure_dirs()
    meeting_id = repo.create_meeting("No token")
    repo.update_meeting(meeting_id, status=STATUS_RECORDED, audio_path=str(wav_file))

    processor = make_processor(settings, repo,
                               diarizer=FakeDiarizer(available=False))
    outcome = processor.process(meeting_id)
    assert not outcome.failed
    assert any("Hugging Face token" in w for w in outcome.warnings)
    assert repo.get_transcript(meeting_id)


def test_diarization_can_be_turned_off(repo, tmp_path, wav_file):
    settings = Settings(data_dir=tmp_path / "data", diarization_enabled=False)
    settings.ensure_dirs()
    meeting_id = repo.create_meeting("No diarization")
    repo.update_meeting(meeting_id, status=STATUS_RECORDED, audio_path=str(wav_file))

    outcome = make_processor(settings, repo).process(meeting_id)
    assert not outcome.failed
    assert repo.get_speakers(meeting_id)[0]["label"] == "Unknown"


def test_the_transcript_survives_an_llm_failure(settings, repo, recorded_meeting):
    """Saving the transcript before the LLM step is the whole point."""
    processor = make_processor(settings, repo, engine=FakeEngine(
        error=OllamaError("Ollama is not running", "Start it with ollama serve.")))
    outcome = processor.process(recorded_meeting)

    assert outcome.failed
    assert repo.get_meeting(recorded_meeting)["status"] == STATUS_FAILED
    assert len(repo.get_transcript(recorded_meeting)) == 2     # kept anyway
    assert repo.get_speakers(recorded_meeting)                 # kept anyway


def test_missing_drafts_fall_back_to_a_plain_recap(settings, repo, recorded_meeting):
    engine = FakeEngine(result=MeetingIntelligence.model_validate(
        {"summary": "We met.", "action_items": [{"task": "Do it", "owner": "Ada"}]}))
    outcome = make_processor(settings, repo, engine=engine).process(recorded_meeting)

    drafts = repo.get_drafts(recorded_meeting)
    assert drafts and "Do it" in drafts[0]["body"]
    assert any("fell back" in w for w in outcome.warnings)


def test_an_untitled_meeting_gets_a_title(settings, repo, wav_file):
    meeting_id = repo.create_meeting("")
    repo.update_meeting(meeting_id, status=STATUS_RECORDED, audio_path=str(wav_file))
    make_processor(settings, repo).process(meeting_id)
    assert repo.get_meeting(meeting_id)["title"]


def test_processing_twice_is_idempotent(settings, repo, recorded_meeting):
    processor = make_processor(settings, repo)
    processor.process(recorded_meeting)
    processor.process(recorded_meeting)
    assert len(repo.get_transcript(recorded_meeting)) == 2
    assert len(repo.get_action_items(recorded_meeting)) == 1


def test_regenerate_uses_the_renamed_speakers(settings, repo, recorded_meeting):
    engine = FakeEngine()
    processor = make_processor(settings, repo, engine=engine)
    processor.process(recorded_meeting)

    repo.rename_speakers(recorded_meeting, {"Speaker 1": "Ada", "Speaker 2": "Grace"})
    processor.regenerate_notes(recorded_meeting)

    segments_seen = engine.calls[-1][0]
    assert {segment.speaker for segment in segments_seen} == {"Ada", "Grace"}
    # The names must survive the regenerate, not be reset to labels.
    assert repo.speaker_name_map(recorded_meeting) == {"Speaker 1": "Ada",
                                                       "Speaker 2": "Grace"}
    assert repo.get_meeting(recorded_meeting)["status"] == STATUS_COMPLETE


def test_regenerate_without_a_transcript_fails_clearly(settings, repo):
    meeting_id = repo.create_meeting("Empty")
    outcome = make_processor(settings, repo).regenerate_notes(meeting_id)
    assert outcome.failed
    assert "no transcript" in outcome.error.lower()


def test_the_real_diarizer_reports_itself_unavailable_without_a_token():
    """The `available` flag the processor branches on is token presence."""
    from app.diarize.pyannote_runner import Diarizer

    assert Diarizer(hf_token="").available is False
    assert Diarizer(hf_token="hf_something").available is True


def test_an_empty_transcript_is_recorded_as_a_warning(settings, repo, recorded_meeting):
    processor = make_processor(settings, repo,
                               transcriber=FakeTranscriber(segments=[]))
    outcome = processor.process(recorded_meeting)
    assert any("No speech" in w for w in outcome.warnings)
