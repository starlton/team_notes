"""Database behaviour: migrations, transactions, and the repository."""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from app.db.database import Database
from app.db.repository import (Repository, STATUS_COMPLETE, STATUS_DISCARDED,
                               STATUS_FAILED, STATUS_PROCESSING, STATUS_RECORDING)
from app.errors import NotFoundError
from app.intelligence.schemas import MeetingIntelligence
from app.transcribe.models import TranscriptSegment


def test_migrations_are_idempotent(tmp_path: Path):
    path = tmp_path / "x.db"
    Database(path).initialise()
    second = Database(path)
    second.initialise()          # must not raise on an existing database
    assert second.query_one("PRAGMA user_version")[0] >= 1


def test_foreign_keys_cascade(repo: Repository):
    meeting_id = repo.create_meeting("Sync")
    repo.replace_transcript(meeting_id, [TranscriptSegment(0, 1, "hi")])
    repo.delete_meeting(meeting_id)
    assert repo.get_transcript(meeting_id) == []


def test_getting_a_missing_meeting_raises(repo: Repository):
    with pytest.raises(NotFoundError):
        repo.get_meeting(9999)
    assert repo.find_meeting(9999) is None


def test_update_meeting_ignores_unknown_columns(repo: Repository):
    """A stray key must never reach the SQL statement."""
    meeting_id = repo.create_meeting("Sync")
    repo.update_meeting(meeting_id, **{"title": "Renamed",
                                       "status; DROP TABLE meetings; --": "x",
                                       "id": 42})
    assert repo.get_meeting(meeting_id)["title"] == "Renamed"
    assert repo.count_meetings() == 1


def test_transcript_round_trips_in_order(repo: Repository):
    meeting_id = repo.create_meeting()
    repo.replace_transcript(meeting_id, [
        TranscriptSegment(0, 2, "one", speaker="Speaker 1", confidence=0.9),
        TranscriptSegment(2, 4, "two", speaker="Speaker 2"),
    ])
    stored = repo.get_transcript(meeting_id)
    assert [s.text for s in stored] == ["one", "two"]
    assert stored[0].confidence == pytest.approx(0.9)


def test_replacing_a_transcript_does_not_duplicate_it(repo: Repository):
    meeting_id = repo.create_meeting()
    for _ in range(3):
        repo.replace_transcript(meeting_id, [TranscriptSegment(0, 1, "hi")])
    assert len(repo.get_transcript(meeting_id)) == 1


def test_speaker_rename_only_touches_known_labels(repo: Repository):
    meeting_id = repo.create_meeting()
    repo.replace_speakers(meeting_id, [{"label": "Speaker 1"}, {"label": "Speaker 2"}])
    updated = repo.rename_speakers(meeting_id, {"Speaker 1": "Ada", "Ghost": "Nobody"})
    assert updated == 1
    assert repo.speaker_name_map(meeting_id) == {"Speaker 1": "Ada",
                                                 "Speaker 2": "Speaker 2"}


def test_renaming_to_blank_keeps_the_label(repo: Repository):
    meeting_id = repo.create_meeting()
    repo.replace_speakers(meeting_id, [{"label": "Speaker 1"}])
    repo.rename_speakers(meeting_id, {"Speaker 1": "   "})
    assert repo.speaker_name_map(meeting_id)["Speaker 1"] == "Speaker 1"


def test_saving_intelligence_replaces_the_previous_run(repo: Repository):
    meeting_id = repo.create_meeting()
    first = MeetingIntelligence.model_validate(
        {"summary": "v1", "action_items": [{"task": "A"}, {"task": "B"}]})
    second = MeetingIntelligence.model_validate(
        {"summary": "v2", "action_items": [{"task": "C"}]})
    repo.save_intelligence(meeting_id, first, "m")
    repo.save_intelligence(meeting_id, second, "m")
    assert repo.get_summary(meeting_id)["summary"] == "v2"
    assert [item["task"] for item in repo.get_action_items(meeting_id)] == ["C"]


def test_action_items_without_text_are_not_stored(repo: Repository):
    meeting_id = repo.create_meeting()
    repo.save_intelligence(meeting_id, MeetingIntelligence.model_validate(
        {"action_items": [{"task": ""}, {"task": "Real one"}]}), "m")
    assert len(repo.get_action_items(meeting_id)) == 1


def test_marking_an_action_done(repo: Repository):
    meeting_id = repo.create_meeting()
    repo.save_intelligence(meeting_id, MeetingIntelligence.model_validate(
        {"action_items": [{"task": "A"}]}), "m")
    action_id = repo.get_action_items(meeting_id)[0]["id"]
    assert repo.set_action_done(meeting_id, action_id, True) is True
    assert repo.get_action_items(meeting_id)[0]["done"] is True


def test_marking_an_action_from_another_meeting_fails(repo: Repository):
    """An action id from meeting A must not be writable through meeting B."""
    first = repo.create_meeting()
    second = repo.create_meeting()
    repo.save_intelligence(first, MeetingIntelligence.model_validate(
        {"action_items": [{"task": "A"}]}), "m")
    action_id = repo.get_action_items(first)[0]["id"]
    assert repo.set_action_done(second, action_id, True) is False


def test_settings_round_trip(repo: Repository):
    assert repo.get_bool_setting("missing", True) is True
    repo.set_bool_setting("flag", False)
    assert repo.get_bool_setting("flag", True) is False
    repo.set_setting("flag", "true")
    assert repo.get_bool_setting("flag") is True


def test_discarded_meetings_are_hidden_by_default(repo: Repository):
    kept = repo.create_meeting("Kept")
    dropped = repo.create_meeting("Dropped")
    repo.update_meeting(dropped, status=STATUS_DISCARDED)
    assert [m["id"] for m in repo.list_meetings()] == [kept]
    assert len(repo.list_meetings(include_discarded=True)) == 2
    assert repo.count_meetings() == 1


def test_list_meetings_clamps_a_silly_limit(repo: Repository):
    repo.create_meeting()
    assert len(repo.list_meetings(limit=10_000)) == 1
    assert repo.list_meetings(limit=0, offset=-5) is not None


def test_recover_interrupted_clears_stuck_meetings(repo: Repository):
    recording = repo.create_meeting("Crashed while recording")
    processing = repo.create_meeting("Crashed while processing")
    repo.update_meeting(processing, status=STATUS_PROCESSING)
    done = repo.create_meeting("Fine")
    repo.update_meeting(done, status=STATUS_COMPLETE)

    assert repo.recover_interrupted() == 2
    assert repo.get_meeting(recording)["status"] == STATUS_FAILED
    assert repo.get_meeting(processing)["status"] == STATUS_FAILED
    assert repo.get_meeting(done)["status"] == STATUS_COMPLETE
    assert "closed before" in repo.get_meeting(recording)["error"]


def test_active_recording_finds_the_live_one(repo: Repository):
    assert repo.active_recording() is None
    meeting_id = repo.create_meeting()
    assert repo.active_recording()["id"] == meeting_id
    repo.update_meeting(meeting_id, status=STATUS_COMPLETE)
    assert repo.active_recording() is None


def test_warnings_round_trip_as_a_list(repo: Repository):
    meeting_id = repo.create_meeting()
    repo.update_meeting(meeting_id, warnings=["a", "b"])
    assert repo.get_meeting(meeting_id)["warnings"] == ["a", "b"]


def test_corrupt_json_columns_degrade_to_empty(repo: Repository):
    meeting_id = repo.create_meeting()
    repo.update_meeting(meeting_id, warnings="not json at all")
    assert repo.get_meeting(meeting_id)["warnings"] == []


def test_progress_is_clamped(repo: Repository):
    meeting_id = repo.create_meeting()
    repo.set_progress(meeting_id, "Way ahead", 7.5)
    assert repo.get_meeting(meeting_id)["progress"] == 1.0
    repo.set_progress(meeting_id, "Behind", -2)
    assert repo.get_meeting(meeting_id)["progress"] == 0.0


def test_concurrent_writers_do_not_corrupt_the_database(database: Database):
    """Several threads write at once; WAL plus retries must cope."""
    errors: list[Exception] = []

    def writer(index: int) -> None:
        try:
            repo = Repository(database)
            for _ in range(5):
                repo.create_meeting(f"thread-{index}")
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=(i,)) for i in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert not errors
    assert Repository(database).count_meetings() == 20


def test_retention_only_lists_old_meetings(repo: Repository):
    meeting_id = repo.create_meeting(started_at="2000-01-01T00:00:00Z")
    repo.update_meeting(meeting_id, audio_path="/tmp/x.wav")
    recent = repo.create_meeting()
    repo.update_meeting(recent, audio_path="/tmp/y.wav")
    old = repo.meetings_with_audio_older_than(30)
    assert [m["id"] for m in old] == [meeting_id]
    assert repo.meetings_with_audio_older_than(0) == []
