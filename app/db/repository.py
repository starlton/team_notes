"""All SQL the app runs.

Every statement is parameterised; no query is ever built by string formatting
with user or model data. Keeping the SQL in one module makes that easy to audit.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from app.db.database import Database
from app.errors import NotFoundError
from app.intelligence.schemas import MeetingIntelligence
from app.transcribe.models import TranscriptSegment

STATUS_RECORDING = "recording"
STATUS_RECORDED = "recorded"
STATUS_PROCESSING = "processing"
STATUS_COMPLETE = "complete"
STATUS_FAILED = "failed"
STATUS_DISCARDED = "discarded"

ALL_STATUSES = (STATUS_RECORDING, STATUS_RECORDED, STATUS_PROCESSING,
                STATUS_COMPLETE, STATUS_FAILED, STATUS_DISCARDED)


def utc_now() -> str:
    """Timestamps are stored as UTC ISO-8601 with a trailing Z."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z")


def _json_loads(raw: Any, default: Any) -> Any:
    if not raw:
        return default
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return default


class Repository:
    """Read and write meetings and everything hanging off them."""

    def __init__(self, database: Database) -> None:
        self.db = database

    # --- meetings ---------------------------------------------------------

    def create_meeting(self, title: str = "", source: str = "manual",
                       participants_informed: bool = False,
                       started_at: str | None = None) -> int:
        now = utc_now()
        with self.db.transaction() as cursor:
            cursor.execute(
                """INSERT INTO meetings
                   (title, started_at, source, status, participants_informed,
                    created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (title.strip()[:200], started_at or now, source, STATUS_RECORDING,
                 1 if participants_informed else 0, now, now),
            )
            return int(cursor.lastrowid or 0)

    def update_meeting(self, meeting_id: int, **fields: Any) -> None:
        """Update a whitelisted set of meeting columns."""
        allowed = {
            "title", "ended_at", "duration_seconds", "audio_path", "status",
            "stage", "progress", "error", "warnings", "participants_informed",
            "consent_note", "source",
        }
        updates = {key: value for key, value in fields.items() if key in allowed}
        if not updates:
            return
        if "warnings" in updates and not isinstance(updates["warnings"], str):
            updates["warnings"] = json.dumps(list(updates["warnings"]))
        if "audio_path" in updates and updates["audio_path"] is not None:
            updates["audio_path"] = str(updates["audio_path"])

        updates["updated_at"] = utc_now()
        # Column names come from the `allowed` set above, never from the caller's
        # data, so this f-string cannot carry injected SQL. Values stay bound.
        assignments = ", ".join(f"{column} = ?" for column in updates)
        with self.db.transaction() as cursor:
            cursor.execute(
                f"UPDATE meetings SET {assignments} WHERE id = ?",
                (*updates.values(), int(meeting_id)),
            )

    def set_progress(self, meeting_id: int, stage: str, progress: float) -> None:
        self.update_meeting(meeting_id, stage=stage[:120],
                            progress=max(0.0, min(1.0, float(progress))))

    def get_meeting(self, meeting_id: int) -> dict[str, Any]:
        row = self.db.query_one("SELECT * FROM meetings WHERE id = ?", (int(meeting_id),))
        if row is None:
            raise NotFoundError(f"Meeting {meeting_id} does not exist.", "")
        return self._meeting_row(row)

    def find_meeting(self, meeting_id: int) -> dict[str, Any] | None:
        row = self.db.query_one("SELECT * FROM meetings WHERE id = ?", (int(meeting_id),))
        return self._meeting_row(row) if row else None

    def list_meetings(self, limit: int = 50, offset: int = 0,
                      include_discarded: bool = False) -> list[dict[str, Any]]:
        limit = max(1, min(500, int(limit)))
        offset = max(0, int(offset))
        if include_discarded:
            rows = self.db.query(
                """SELECT * FROM meetings
                   ORDER BY datetime(started_at) DESC, id DESC LIMIT ? OFFSET ?""",
                (limit, offset))
        else:
            rows = self.db.query(
                """SELECT * FROM meetings WHERE status != ?
                   ORDER BY datetime(started_at) DESC, id DESC LIMIT ? OFFSET ?""",
                (STATUS_DISCARDED, limit, offset))
        return [self._meeting_row(row) for row in rows]

    def count_meetings(self, include_discarded: bool = False) -> int:
        if include_discarded:
            row = self.db.query_one("SELECT COUNT(*) AS n FROM meetings")
        else:
            row = self.db.query_one(
                "SELECT COUNT(*) AS n FROM meetings WHERE status != ?",
                (STATUS_DISCARDED,))
        return int(row["n"]) if row else 0

    def active_recording(self) -> dict[str, Any] | None:
        row = self.db.query_one(
            "SELECT * FROM meetings WHERE status = ? ORDER BY id DESC LIMIT 1",
            (STATUS_RECORDING,))
        return self._meeting_row(row) if row else None

    def delete_meeting(self, meeting_id: int) -> None:
        with self.db.transaction() as cursor:
            cursor.execute("DELETE FROM meetings WHERE id = ?", (int(meeting_id),))

    @staticmethod
    def _meeting_row(row: Any) -> dict[str, Any]:
        data = dict(row)
        data["participants_informed"] = bool(data.get("participants_informed"))
        data["warnings"] = _json_loads(data.get("warnings"), [])
        return data

    # --- transcript -------------------------------------------------------

    def replace_transcript(self, meeting_id: int,
                           segments: Sequence[TranscriptSegment]) -> None:
        rows = [
            (int(meeting_id), position, float(segment.start), float(segment.end),
             segment.speaker, segment.text, segment.confidence)
            for position, segment in enumerate(segments)
        ]
        with self.db.transaction() as cursor:
            cursor.execute("DELETE FROM transcript_segments WHERE meeting_id = ?",
                           (int(meeting_id),))
            if rows:
                cursor.executemany(
                    """INSERT INTO transcript_segments
                       (meeting_id, position, start_seconds, end_seconds,
                        speaker_label, text, confidence)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""", rows)

    def get_transcript(self, meeting_id: int) -> list[TranscriptSegment]:
        rows = self.db.query(
            """SELECT start_seconds, end_seconds, speaker_label, text, confidence
               FROM transcript_segments WHERE meeting_id = ? ORDER BY position""",
            (int(meeting_id),))
        return [
            TranscriptSegment(
                start=float(row["start_seconds"]), end=float(row["end_seconds"]),
                text=row["text"], speaker=row["speaker_label"],
                confidence=row["confidence"],
            )
            for row in rows
        ]

    # --- speakers ---------------------------------------------------------

    def replace_speakers(self, meeting_id: int,
                         speakers: Iterable[dict[str, Any]]) -> None:
        rows = [
            (int(meeting_id), str(entry["label"]),
             str(entry.get("display_name") or entry["label"])[:120],
             float(entry.get("talk_seconds", 0.0)), int(entry.get("word_count", 0)),
             int(entry.get("turn_count", 0)), float(entry.get("share", 0.0)),
             str(entry.get("summary", ""))[:5000],
             json.dumps(list(entry.get("key_points", []))), position)
            for position, entry in enumerate(speakers)
        ]
        with self.db.transaction() as cursor:
            cursor.execute("DELETE FROM speakers WHERE meeting_id = ?",
                           (int(meeting_id),))
            if rows:
                cursor.executemany(
                    """INSERT INTO speakers
                       (meeting_id, label, display_name, talk_seconds, word_count,
                        turn_count, share, summary, key_points, position)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""", rows)

    def get_speakers(self, meeting_id: int) -> list[dict[str, Any]]:
        rows = self.db.query(
            "SELECT * FROM speakers WHERE meeting_id = ? ORDER BY position",
            (int(meeting_id),))
        speakers = []
        for row in rows:
            entry = dict(row)
            entry["key_points"] = _json_loads(entry.get("key_points"), [])
            speakers.append(entry)
        return speakers

    def speaker_name_map(self, meeting_id: int) -> dict[str, str]:
        return {
            speaker["label"]: speaker["display_name"] or speaker["label"]
            for speaker in self.get_speakers(meeting_id)
        }

    def rename_speakers(self, meeting_id: int, names: dict[str, str]) -> int:
        """Set display names for a meeting's speaker labels."""
        known = {speaker["label"] for speaker in self.get_speakers(meeting_id)}
        updates = [
            (str(name).strip()[:120] or label, int(meeting_id), label)
            for label, name in names.items() if label in known
        ]
        if not updates:
            return 0
        with self.db.transaction() as cursor:
            cursor.executemany(
                "UPDATE speakers SET display_name = ? WHERE meeting_id = ? AND label = ?",
                updates)
        return len(updates)

    # --- generated notes --------------------------------------------------

    def save_intelligence(self, meeting_id: int, result: MeetingIntelligence,
                          model: str) -> None:
        """Replace every generated artefact for a meeting, in one transaction."""
        meeting_id = int(meeting_id)
        with self.db.transaction() as cursor:
            cursor.execute("DELETE FROM summaries WHERE meeting_id = ?", (meeting_id,))
            cursor.execute("DELETE FROM action_items WHERE meeting_id = ?", (meeting_id,))
            cursor.execute("DELETE FROM priorities WHERE meeting_id = ?", (meeting_id,))
            cursor.execute("DELETE FROM drafts WHERE meeting_id = ?", (meeting_id,))

            cursor.execute(
                """INSERT INTO summaries (meeting_id, title, summary, bullets, model,
                                          created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (meeting_id, result.title, result.summary,
                 json.dumps(result.bullets), model, utc_now()))

            cursor.executemany(
                """INSERT INTO action_items
                   (meeting_id, position, task, owner, due, priority, context)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                [(meeting_id, position, item.task, item.owner, item.due,
                  item.priority, item.context)
                 for position, item in enumerate(result.action_items) if item.task])

            cursor.executemany(
                """INSERT INTO priorities (meeting_id, position, point, priority, reason)
                   VALUES (?, ?, ?, ?, ?)""",
                [(meeting_id, position, item.point, item.priority, item.reason)
                 for position, item in enumerate(result.priorities) if item.point])

            cursor.executemany(
                """INSERT INTO drafts (meeting_id, position, kind, audience, subject, body)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                [(meeting_id, position, item.kind, item.audience, item.subject,
                  item.body)
                 for position, item in enumerate(result.drafts) if item.body])

    def get_summary(self, meeting_id: int) -> dict[str, Any] | None:
        row = self.db.query_one("SELECT * FROM summaries WHERE meeting_id = ?",
                                (int(meeting_id),))
        if row is None:
            return None
        data = dict(row)
        data["bullets"] = _json_loads(data.get("bullets"), [])
        return data

    def get_action_items(self, meeting_id: int) -> list[dict[str, Any]]:
        rows = self.db.query(
            "SELECT * FROM action_items WHERE meeting_id = ? ORDER BY position",
            (int(meeting_id),))
        items = []
        for row in rows:
            entry = dict(row)
            entry["done"] = bool(entry.get("done"))
            items.append(entry)
        return items

    def set_action_done(self, meeting_id: int, action_id: int, done: bool) -> bool:
        with self.db.transaction() as cursor:
            cursor.execute(
                "UPDATE action_items SET done = ? WHERE id = ? AND meeting_id = ?",
                (1 if done else 0, int(action_id), int(meeting_id)))
            return cursor.rowcount > 0

    def get_priorities(self, meeting_id: int) -> list[dict[str, Any]]:
        return [dict(row) for row in self.db.query(
            "SELECT * FROM priorities WHERE meeting_id = ? ORDER BY position",
            (int(meeting_id),))]

    def get_drafts(self, meeting_id: int) -> list[dict[str, Any]]:
        return [dict(row) for row in self.db.query(
            "SELECT * FROM drafts WHERE meeting_id = ? ORDER BY position",
            (int(meeting_id),))]

    # --- settings ---------------------------------------------------------

    def get_setting(self, key: str, default: str = "") -> str:
        row = self.db.query_one("SELECT value FROM app_settings WHERE key = ?", (key,))
        return str(row["value"]) if row else default

    def set_setting(self, key: str, value: str) -> None:
        with self.db.transaction() as cursor:
            cursor.execute(
                """INSERT INTO app_settings (key, value, updated_at) VALUES (?, ?, ?)
                   ON CONFLICT(key) DO UPDATE SET value = excluded.value,
                                                  updated_at = excluded.updated_at""",
                (key, str(value), utc_now()))

    def get_bool_setting(self, key: str, default: bool = False) -> bool:
        raw = self.get_setting(key, "")
        if not raw:
            return default
        return raw.lower() in {"1", "true", "yes", "on"}

    def set_bool_setting(self, key: str, value: bool) -> None:
        self.set_setting(key, "true" if value else "false")

    # --- maintenance ------------------------------------------------------

    def meetings_with_audio_older_than(self, days: int) -> list[dict[str, Any]]:
        """Meetings whose audio is past the retention window."""
        if days <= 0:
            return []
        rows = self.db.query(
            """SELECT * FROM meetings
               WHERE audio_path IS NOT NULL AND audio_path != ''
                 AND datetime(started_at) < datetime('now', ?)""",
            (f"-{int(days)} days",))
        return [self._meeting_row(row) for row in rows]

    def clear_audio_path(self, meeting_id: int) -> None:
        self.update_meeting(meeting_id, audio_path="")

    def recover_interrupted(self) -> int:
        """Mark meetings left mid-flight by a crash as failed, at startup.

        Without this a meeting whose process died stays 'recording' forever and
        blocks the next recording from starting.
        """
        rows = self.db.query(
            "SELECT id FROM meetings WHERE status IN (?, ?)",
            (STATUS_RECORDING, STATUS_PROCESSING))
        if not rows:
            return 0
        with self.db.transaction() as cursor:
            cursor.executemany(
                """UPDATE meetings
                   SET status = ?, error = ?, stage = '', progress = 0,
                       updated_at = ?
                   WHERE id = ?""",
                [(STATUS_FAILED,
                  "The app closed before this meeting finished processing.",
                  utc_now(), int(row["id"])) for row in rows])
        return len(rows)


def meeting_audio_exists(meeting: dict[str, Any]) -> bool:
    path = meeting.get("audio_path")
    return bool(path) and Path(str(path)).is_file()
