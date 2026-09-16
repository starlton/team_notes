"""Validated shapes for everything the local LLM produces.

Small local models are not reliable JSON emitters: they return a string where an
object was asked for, wrap a list in a dict, invent a priority value, or repeat
themselves. Rather than fail the whole meeting over that, every model here
coerces what it reasonably can and drops what it cannot, so a slightly wonky
reply still yields usable notes.
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

PRIORITIES = ("high", "medium", "low")
MAX_ITEMS = 40
MAX_TEXT = 4000


def _clean(value: Any, limit: int = MAX_TEXT) -> str:
    """Coerce anything to a trimmed, length-capped string."""
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        value = " ".join(str(item) for item in value)
    elif isinstance(value, dict):
        value = " ".join(f"{k}: {v}" for k, v in value.items())
    return str(value).strip()[:limit]


def _as_string_list(value: Any, limit: int = MAX_ITEMS) -> list[str]:
    """Coerce a model's idea of 'a list of strings' into one."""
    if value is None:
        return []
    if isinstance(value, str):
        items = [line.strip(" -*\t") for line in value.splitlines()]
    elif isinstance(value, dict):
        items = [_clean(item) for item in value.values()]
    elif isinstance(value, (list, tuple)):
        items = []
        for item in value:
            if isinstance(item, dict):
                # e.g. [{"point": "..."}] instead of ["..."]
                items.append(_clean(next(iter(item.values()), "")) if item else "")
            else:
                items.append(_clean(item))
    else:
        items = [_clean(value)]

    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        item = item.strip()
        if not item:
            continue
        key = item.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(item[:MAX_TEXT])
        if len(out) >= limit:
            break
    return out


_HIGH_WORDS = ("high", "urgent", "critical", "blocker", "asap", "p0", "p1")
_LOW_WORDS = ("low", "minor", "nice to have", "optional", "later", "p3", "p4")


def _normalise_priority(value: Any) -> str:
    """Map whatever the model said to one of high / medium / low."""
    text = re.sub(r"[^a-z0-9 ]+", " ", _clean(value, 64).lower())
    words = set(text.split())
    if any(word in text for word in _HIGH_WORDS):
        return "high"
    if any(word in text for word in _LOW_WORDS):
        return "low"
    if words & {"1"}:
        return "high"
    if words & {"3", "4", "5"}:
        return "low"
    return "medium"


class Base(BaseModel):
    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)


class ActionItem(Base):
    """A to-do extracted from the meeting."""

    task: str = ""
    owner: str = "Unassigned"
    due: str = ""
    priority: str = "medium"
    context: str = ""

    @field_validator("task", "owner", "due", "context", mode="before")
    @classmethod
    def _strings(cls, value: Any) -> str:
        return _clean(value, 1000)

    @field_validator("owner", mode="after")
    @classmethod
    def _owner_default(cls, value: str) -> str:
        return value or "Unassigned"

    @field_validator("priority", mode="before")
    @classmethod
    def _priority(cls, value: Any) -> str:
        return _normalise_priority(value)


class PriorityPoint(Base):
    """A key discussion point with a priority ranking."""

    point: str = ""
    priority: str = "medium"
    reason: str = ""

    @field_validator("point", "reason", mode="before")
    @classmethod
    def _strings(cls, value: Any) -> str:
        return _clean(value, 1500)

    @field_validator("priority", mode="before")
    @classmethod
    def _priority(cls, value: Any) -> str:
        return _normalise_priority(value)


class SpeakerSummary(Base):
    """What one participant contributed."""

    speaker: str = ""
    summary: str = ""
    key_points: list[str] = Field(default_factory=list)

    @field_validator("speaker", "summary", mode="before")
    @classmethod
    def _strings(cls, value: Any) -> str:
        return _clean(value, 3000)

    @field_validator("key_points", mode="before")
    @classmethod
    def _points(cls, value: Any) -> list[str]:
        return _as_string_list(value, 15)


class DraftMessage(Base):
    """A ready-to-send follow-up."""

    kind: str = "email"
    audience: str = "The meeting participants"
    subject: str = ""
    body: str = ""

    @field_validator("audience", "subject", mode="before")
    @classmethod
    def _short(cls, value: Any) -> str:
        return _clean(value, 300)

    @field_validator("body", mode="before")
    @classmethod
    def _body(cls, value: Any) -> str:
        return _clean(value, 8000)

    @field_validator("kind", mode="before")
    @classmethod
    def _kind(cls, value: Any) -> str:
        text = _clean(value, 32).lower()
        if any(word in text for word in ("chat", "teams", "slack", "message", "im")):
            return "chat"
        return "email"


class ChunkNotes(Base):
    """The map step: what one slice of the transcript contained."""

    key_points: list[str] = Field(default_factory=list)
    decisions: list[str] = Field(default_factory=list)
    questions: list[str] = Field(default_factory=list)
    action_items: list[ActionItem] = Field(default_factory=list)

    @field_validator("key_points", "decisions", "questions", mode="before")
    @classmethod
    def _lists(cls, value: Any) -> list[str]:
        return _as_string_list(value)

    @field_validator("action_items", mode="before")
    @classmethod
    def _actions(cls, value: Any) -> list[Any]:
        return _coerce_object_list(value, "task")


class MeetingIntelligence(Base):
    """The reduce step: everything the dashboard shows for a meeting."""

    title: str = ""
    summary: str = ""
    bullets: list[str] = Field(default_factory=list)
    action_items: list[ActionItem] = Field(default_factory=list)
    priorities: list[PriorityPoint] = Field(default_factory=list)
    speakers: list[SpeakerSummary] = Field(default_factory=list)
    drafts: list[DraftMessage] = Field(default_factory=list)

    @field_validator("title", mode="before")
    @classmethod
    def _title(cls, value: Any) -> str:
        return _clean(value, 200)

    @field_validator("summary", mode="before")
    @classmethod
    def _summary(cls, value: Any) -> str:
        return _clean(value, 8000)

    @field_validator("bullets", mode="before")
    @classmethod
    def _bullets(cls, value: Any) -> list[str]:
        return _as_string_list(value, 25)

    @field_validator("action_items", mode="before")
    @classmethod
    def _actions(cls, value: Any) -> list[Any]:
        return _coerce_object_list(value, "task")

    @field_validator("priorities", mode="before")
    @classmethod
    def _priorities(cls, value: Any) -> list[Any]:
        return _coerce_object_list(value, "point")

    @field_validator("speakers", mode="before")
    @classmethod
    def _speakers(cls, value: Any) -> list[Any]:
        return _coerce_object_list(value, "speaker")

    @field_validator("drafts", mode="before")
    @classmethod
    def _drafts(cls, value: Any) -> list[Any]:
        return _coerce_object_list(value, "body")

    def is_empty(self) -> bool:
        return not (self.summary or self.bullets or self.action_items
                    or self.priorities or self.drafts)


def _coerce_object_list(value: Any, string_key: str) -> list[Any]:
    """Accept a list of objects, a list of strings, or a dict of objects."""
    if value is None:
        return []
    if isinstance(value, dict):
        value = list(value.values())
    if isinstance(value, str):
        value = _as_string_list(value)
    if not isinstance(value, (list, tuple)):
        return []

    out: list[Any] = []
    for item in value:
        if isinstance(item, dict):
            out.append(item)
        elif isinstance(item, str) and item.strip():
            out.append({string_key: item.strip()})
        if len(out) >= MAX_ITEMS:
            break
    return out
