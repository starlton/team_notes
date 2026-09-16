"""Splitting a long transcript into pieces the local model can actually hold.

A one-hour meeting is roughly 9,000 words, which overflows the 8k-token context
we give Ollama by default. Rather than truncate (and silently lose the second
half of the meeting), long transcripts are processed map-reduce style: chunk,
summarise each chunk, then combine.

Chunks are only ever cut at a speaker turn boundary, and each one repeats the
tail of the previous chunk so an action item mentioned across a boundary is not
lost.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from app.transcribe.models import TranscriptSegment, format_timestamp

# Rough but reliable across the models we support: English prose averages a
# little under 4 characters per token. Deliberately conservative.
CHARS_PER_TOKEN = 3.6
# Tokens held back from the context window for the prompt and the model's reply.
RESERVED_TOKENS = 2400
MIN_CHUNK_CHARS = 2000
OVERLAP_SEGMENTS = 2


def estimate_tokens(text: str) -> int:
    """Approximate token count for a piece of text."""
    return int(len(text) / CHARS_PER_TOKEN) + 1


def chunk_budget_chars(num_ctx: int, reserved_tokens: int = RESERVED_TOKENS) -> int:
    """How many characters of transcript fit in one request."""
    usable_tokens = max(512, int(num_ctx) - int(reserved_tokens))
    return max(MIN_CHUNK_CHARS, int(usable_tokens * CHARS_PER_TOKEN))


@dataclass
class TranscriptChunk:
    """One slice of the transcript, already rendered for the prompt."""

    index: int
    total: int
    text: str
    start: float
    end: float

    @property
    def label(self) -> str:
        return (f"part {self.index + 1} of {self.total} "
                f"({format_timestamp(self.start)}-{format_timestamp(self.end)})")


def _render_line(segment: TranscriptSegment,
                 speaker_names: dict[str, str] | None) -> str:
    names = speaker_names or {}
    label = segment.speaker or "Unknown"
    return f"[{format_timestamp(segment.start)}] {names.get(label, label)}: {segment.text.strip()}"


def chunk_transcript(segments: Sequence[TranscriptSegment], budget_chars: int,
                     speaker_names: dict[str, str] | None = None,
                     overlap_segments: int = OVERLAP_SEGMENTS
                     ) -> list[TranscriptChunk]:
    """Split segments into chunks no larger than `budget_chars`."""
    usable = [segment for segment in segments if segment.text.strip()]
    if not usable:
        return []

    budget = max(MIN_CHUNK_CHARS, int(budget_chars))
    overlap_segments = max(0, int(overlap_segments))

    groups: list[list[TranscriptSegment]] = []
    current: list[TranscriptSegment] = []
    current_chars = 0

    for segment in usable:
        line = _render_line(segment, speaker_names)
        line_length = len(line) + 1
        if current and current_chars + line_length > budget:
            groups.append(current)
            # Carry the tail forward so context is not cut mid-thought.
            current = current[-overlap_segments:] if overlap_segments else []
            current_chars = sum(len(_render_line(s, speaker_names)) + 1
                                for s in current)
        current.append(segment)
        current_chars += line_length

    if current:
        groups.append(current)

    total = len(groups)
    return [
        TranscriptChunk(
            index=index,
            total=total,
            text="\n".join(_render_line(segment, speaker_names) for segment in group),
            start=group[0].start,
            end=group[-1].end,
        )
        for index, group in enumerate(groups)
    ]
