"""Transcript data types shared by the transcribe, diarize and LLM stages."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

UNKNOWN_SPEAKER = "Unknown"


def format_timestamp(seconds: float) -> str:
    """Render seconds as H:MM:SS (or M:SS under an hour)."""
    seconds = max(0.0, float(seconds))
    hours, remainder = divmod(int(seconds), 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


@dataclass
class Word:
    """One word with its timing, used to split segments at speaker changes."""

    start: float
    end: float
    text: str
    probability: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Word":
        return cls(
            start=float(data.get("start", 0.0)),
            end=float(data.get("end", 0.0)),
            text=str(data.get("text", "")),
            probability=float(data.get("probability", 1.0)),
        )


@dataclass
class TranscriptSegment:
    """A contiguous run of speech, optionally attributed to a speaker."""

    start: float
    end: float
    text: str
    speaker: str | None = None
    words: list[Word] = field(default_factory=list)
    confidence: float | None = None

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def to_dict(self, include_words: bool = True) -> dict[str, Any]:
        data: dict[str, Any] = {
            "start": self.start,
            "end": self.end,
            "text": self.text,
            "speaker": self.speaker,
            "confidence": self.confidence,
        }
        if include_words:
            data["words"] = [word.to_dict() for word in self.words]
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TranscriptSegment":
        return cls(
            start=float(data.get("start", 0.0)),
            end=float(data.get("end", 0.0)),
            text=str(data.get("text", "")),
            speaker=data.get("speaker"),
            words=[Word.from_dict(word) for word in data.get("words") or []],
            confidence=(float(data["confidence"])
                        if data.get("confidence") is not None else None),
        )


@dataclass
class TranscriptResult:
    """A whole transcript plus what the transcriber noticed about it."""

    segments: list[TranscriptSegment] = field(default_factory=list)
    language: str = "en"
    duration_seconds: float = 0.0
    model: str = ""

    @property
    def text(self) -> str:
        return "\n".join(segment.text for segment in self.segments if segment.text)

    def speakers(self) -> list[str]:
        """Distinct speaker labels, in the order they first speak."""
        seen: list[str] = []
        for segment in self.segments:
            if segment.speaker and segment.speaker not in seen:
                seen.append(segment.speaker)
        return seen

    def to_dict(self) -> dict[str, Any]:
        return {
            "segments": [segment.to_dict() for segment in self.segments],
            "language": self.language,
            "duration_seconds": self.duration_seconds,
            "model": self.model,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TranscriptResult":
        return cls(
            segments=[TranscriptSegment.from_dict(item)
                      for item in data.get("segments") or []],
            language=str(data.get("language", "en")),
            duration_seconds=float(data.get("duration_seconds", 0.0)),
            model=str(data.get("model", "")),
        )


def render_transcript(segments: Iterable[TranscriptSegment],
                      with_speakers: bool = True,
                      speaker_names: dict[str, str] | None = None) -> str:
    """Render segments as the plain timestamped text the LLM and UI both use."""
    names = speaker_names or {}
    lines: list[str] = []
    for segment in segments:
        text = segment.text.strip()
        if not text:
            continue
        stamp = format_timestamp(segment.start)
        if with_speakers:
            label = segment.speaker or UNKNOWN_SPEAKER
            lines.append(f"[{stamp}] {names.get(label, label)}: {text}")
        else:
            lines.append(f"[{stamp}] {text}")
    return "\n".join(lines)
