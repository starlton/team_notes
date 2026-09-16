"""The database schema, applied as numbered migrations.

Migrations are append-only: to change the schema, add a new entry to
`MIGRATIONS` rather than editing an old one, so an existing database upgrades
cleanly instead of needing to be thrown away.
"""

from __future__ import annotations

SCHEMA_V1 = """
CREATE TABLE IF NOT EXISTS meetings (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    title                TEXT    NOT NULL DEFAULT '',
    started_at           TEXT    NOT NULL,
    ended_at             TEXT,
    duration_seconds     REAL    NOT NULL DEFAULT 0,
    audio_path           TEXT,
    source               TEXT    NOT NULL DEFAULT 'manual',
    status               TEXT    NOT NULL DEFAULT 'recording',
    stage                TEXT    NOT NULL DEFAULT '',
    progress             REAL    NOT NULL DEFAULT 0,
    error                TEXT    NOT NULL DEFAULT '',
    warnings             TEXT    NOT NULL DEFAULT '[]',
    participants_informed INTEGER NOT NULL DEFAULT 0,
    consent_note         TEXT    NOT NULL DEFAULT '',
    created_at           TEXT    NOT NULL DEFAULT (datetime('now')),
    updated_at           TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_meetings_started ON meetings(started_at DESC);
CREATE INDEX IF NOT EXISTS idx_meetings_status  ON meetings(status);

CREATE TABLE IF NOT EXISTS transcript_segments (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    meeting_id   INTEGER NOT NULL REFERENCES meetings(id) ON DELETE CASCADE,
    position     INTEGER NOT NULL,
    start_seconds REAL   NOT NULL,
    end_seconds  REAL    NOT NULL,
    speaker_label TEXT,
    text         TEXT    NOT NULL,
    confidence   REAL
);

CREATE INDEX IF NOT EXISTS idx_segments_meeting
    ON transcript_segments(meeting_id, position);

CREATE TABLE IF NOT EXISTS speakers (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    meeting_id    INTEGER NOT NULL REFERENCES meetings(id) ON DELETE CASCADE,
    label         TEXT    NOT NULL,
    display_name  TEXT    NOT NULL DEFAULT '',
    talk_seconds  REAL    NOT NULL DEFAULT 0,
    word_count    INTEGER NOT NULL DEFAULT 0,
    turn_count    INTEGER NOT NULL DEFAULT 0,
    share         REAL    NOT NULL DEFAULT 0,
    summary       TEXT    NOT NULL DEFAULT '',
    key_points    TEXT    NOT NULL DEFAULT '[]',
    position      INTEGER NOT NULL DEFAULT 0,
    UNIQUE (meeting_id, label)
);

CREATE TABLE IF NOT EXISTS summaries (
    meeting_id  INTEGER PRIMARY KEY REFERENCES meetings(id) ON DELETE CASCADE,
    title       TEXT NOT NULL DEFAULT '',
    summary     TEXT NOT NULL DEFAULT '',
    bullets     TEXT NOT NULL DEFAULT '[]',
    model       TEXT NOT NULL DEFAULT '',
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS action_items (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    meeting_id INTEGER NOT NULL REFERENCES meetings(id) ON DELETE CASCADE,
    position   INTEGER NOT NULL DEFAULT 0,
    task       TEXT    NOT NULL,
    owner      TEXT    NOT NULL DEFAULT 'Unassigned',
    due        TEXT    NOT NULL DEFAULT '',
    priority   TEXT    NOT NULL DEFAULT 'medium',
    context    TEXT    NOT NULL DEFAULT '',
    done       INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_actions_meeting ON action_items(meeting_id, position);

CREATE TABLE IF NOT EXISTS priorities (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    meeting_id INTEGER NOT NULL REFERENCES meetings(id) ON DELETE CASCADE,
    position   INTEGER NOT NULL DEFAULT 0,
    point      TEXT    NOT NULL,
    priority   TEXT    NOT NULL DEFAULT 'medium',
    reason     TEXT    NOT NULL DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_priorities_meeting ON priorities(meeting_id, position);

CREATE TABLE IF NOT EXISTS drafts (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    meeting_id INTEGER NOT NULL REFERENCES meetings(id) ON DELETE CASCADE,
    position   INTEGER NOT NULL DEFAULT 0,
    kind       TEXT    NOT NULL DEFAULT 'email',
    audience   TEXT    NOT NULL DEFAULT '',
    subject    TEXT    NOT NULL DEFAULT '',
    body       TEXT    NOT NULL DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_drafts_meeting ON drafts(meeting_id, position);

CREATE TABLE IF NOT EXISTS app_settings (
    key        TEXT PRIMARY KEY,
    value      TEXT NOT NULL,
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);
"""

MIGRATIONS: tuple[str, ...] = (SCHEMA_V1,)
