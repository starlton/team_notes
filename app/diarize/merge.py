"""Aligning diarization output with transcript timings.

Diarization and transcription run independently and disagree about boundaries,
so this is where the two are reconciled. The approach, in order:

1. Attribute each *word* to the speaker turn it overlaps most. Words are the
   finest timing Whisper gives us, and using them means a segment where two
   people talk over each other gets split rather than handed wholesale to one.
2. Smooth the result. A single word attributed to a different speaker between
   two runs of the same speaker is almost always a boundary artefact, not a
   real interjection, so it is absorbed.
3. Rebuild segments from the smoothed runs, then merge adjacent same-speaker
   segments so the transcript reads as turns rather than fragments.

All of it is pure functions over plain data, so it can be tested without
pyannote, torch or a GPU anywhere in sight.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from app.transcribe.models import TranscriptSegment, Word

# A word this far from any speaker turn is left unattributed.
MAX_ATTRIBUTION_GAP_SECONDS = 2.0
# Runs shorter than this are candidates for smoothing away.
SMOOTHING_MAX_WORDS = 2
SMOOTHING_MAX_SECONDS = 0.8
# Adjacent same-speaker segments closer than this are joined.
MERGE_MAX_GAP_SECONDS = 1.0


@dataclass(frozen=True)
class SpeakerTurn:
    """A stretch of audio that diarization assigned to one speaker."""

    start: float
    end: float
    speaker: str

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    @property
    def centre(self) -> float:
        return (self.start + self.end) / 2.0


def overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    """Length of the intersection of two intervals (0 if they do not touch)."""
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def best_speaker_for_interval(start: float, end: float,
                              turns: Sequence[SpeakerTurn]) -> str | None:
    """The speaker whose turns overlap [start, end] most, else the nearest one.

    Falling back to "nearest" matters because Whisper's word timings and
    pyannote's boundaries routinely differ by a few tens of milliseconds; a
    strict overlap test would leave short words at a turn edge unattributed.
    """
    if not turns:
        return None

    totals: dict[str, float] = {}
    for turn in turns:
        shared = overlap(start, end, turn.start, turn.end)
        if shared > 0:
            totals[turn.speaker] = totals.get(turn.speaker, 0.0) + shared

    if totals:
        # max() over (duration, speaker) keeps ties deterministic.
        return max(totals.items(), key=lambda item: (item[1], item[0]))[0]

    centre = (start + end) / 2.0
    nearest = min(turns, key=lambda turn: _distance_to_turn(centre, turn))
    if _distance_to_turn(centre, nearest) <= MAX_ATTRIBUTION_GAP_SECONDS:
        return nearest.speaker
    return None


def _distance_to_turn(point: float, turn: SpeakerTurn) -> float:
    if turn.start <= point <= turn.end:
        return 0.0
    return turn.start - point if point < turn.start else point - turn.end


def _smooth_runs(labels: list[str | None], words: Sequence[Word]) -> list[str | None]:
    """Absorb very short speaker runs that sit between two identical runs."""
    if len(labels) < 3:
        return labels

    smoothed = list(labels)
    index = 0
    while index < len(smoothed):
        run_end = index
        while run_end + 1 < len(smoothed) and smoothed[run_end + 1] == smoothed[index]:
            run_end += 1

        is_interior = index > 0 and run_end < len(smoothed) - 1
        if is_interior:
            before = smoothed[index - 1]
            after = smoothed[run_end + 1]
            run_words = run_end - index + 1
            run_seconds = words[run_end].end - words[index].start
            if (before is not None and before == after
                    and run_words <= SMOOTHING_MAX_WORDS
                    and run_seconds <= SMOOTHING_MAX_SECONDS):
                for position in range(index, run_end + 1):
                    smoothed[position] = before

        index = run_end + 1
    return smoothed


def _split_segment_by_words(segment: TranscriptSegment,
                            turns: Sequence[SpeakerTurn]) -> list[TranscriptSegment]:
    """Split one segment wherever its words change speaker."""
    words = segment.words
    labels = [best_speaker_for_interval(word.start, word.end, turns) for word in words]
    labels = _smooth_runs(labels, words)

    # If the whole segment belongs to one speaker there is nothing to split, so
    # keep the original text rather than rebuilding it from the word list.
    # Whisper's word timings occasionally omit a word (a missing timestamp, a
    # dropped token), and rebuilding would silently lose it from the transcript.
    if len({label for label in labels}) <= 1:
        return [TranscriptSegment(
            start=segment.start, end=segment.end, text=segment.text,
            speaker=labels[0] if labels else None, words=list(words),
            confidence=segment.confidence,
        )]

    pieces: list[TranscriptSegment] = []
    run_start = 0
    for index in range(1, len(words) + 1):
        if index < len(words) and labels[index] == labels[run_start]:
            continue
        run_words = words[run_start:index]
        text = " ".join(word.text for word in run_words if word.text).strip()
        if text:
            pieces.append(TranscriptSegment(
                start=run_words[0].start,
                end=run_words[-1].end,
                text=text,
                speaker=labels[run_start],
                words=list(run_words),
                confidence=segment.confidence,
            ))
        run_start = index

    return pieces


def assign_speakers(segments: Iterable[TranscriptSegment],
                    turns: Sequence[SpeakerTurn]) -> list[TranscriptSegment]:
    """Attribute transcript segments to speakers, splitting where needed."""
    turns = sorted(turns, key=lambda turn: (turn.start, turn.end))
    result: list[TranscriptSegment] = []

    for segment in segments:
        if not turns:
            result.append(segment)
            continue
        if segment.words:
            result.extend(_split_segment_by_words(segment, turns))
        else:
            speaker = best_speaker_for_interval(segment.start, segment.end, turns)
            result.append(TranscriptSegment(
                start=segment.start, end=segment.end, text=segment.text,
                speaker=speaker, words=[], confidence=segment.confidence,
            ))

    return result


def merge_adjacent(segments: Sequence[TranscriptSegment],
                   max_gap: float = MERGE_MAX_GAP_SECONDS
                   ) -> list[TranscriptSegment]:
    """Join consecutive segments from the same speaker into readable turns."""
    merged: list[TranscriptSegment] = []
    for segment in segments:
        if (merged and merged[-1].speaker == segment.speaker
                and segment.start - merged[-1].end <= max_gap):
            previous = merged[-1]
            joined_text = f"{previous.text.rstrip()} {segment.text.lstrip()}".strip()
            merged[-1] = TranscriptSegment(
                start=previous.start,
                end=max(previous.end, segment.end),
                text=joined_text,
                speaker=previous.speaker,
                words=previous.words + segment.words,
                confidence=_average_confidence(previous, segment),
            )
        else:
            merged.append(segment)
    return merged


def _average_confidence(first: TranscriptSegment,
                        second: TranscriptSegment) -> float | None:
    values = [c for c in (first.confidence, second.confidence) if c is not None]
    if not values:
        return None
    return round(sum(values) / len(values), 4)


def relabel_speakers(segments: Sequence[TranscriptSegment],
                     template: str = "Speaker {n}") -> tuple[list[TranscriptSegment],
                                                             dict[str, str]]:
    """Rename raw labels (SPEAKER_01...) to 'Speaker 1', in order of first speech.

    Returns the relabelled segments and the raw -> friendly mapping, so the
    dashboard's rename step can still trace a label back to diarization output.
    """
    mapping: dict[str, str] = {}
    for segment in segments:
        raw = segment.speaker
        if raw and raw not in mapping:
            mapping[raw] = template.format(n=len(mapping) + 1)

    relabelled = [
        TranscriptSegment(
            start=segment.start, end=segment.end, text=segment.text,
            speaker=mapping.get(segment.speaker) if segment.speaker else None,
            words=segment.words, confidence=segment.confidence,
        )
        for segment in segments
    ]
    return relabelled, mapping


def speaker_statistics(segments: Sequence[TranscriptSegment]) -> list[dict[str, object]]:
    """Per-speaker talk time and word counts, ordered by most talkative."""
    stats: dict[str, dict[str, float]] = {}
    for segment in segments:
        label = segment.speaker or "Unknown"
        entry = stats.setdefault(label, {"seconds": 0.0, "words": 0.0, "turns": 0.0})
        entry["seconds"] += segment.duration
        entry["words"] += len(segment.words) or len(segment.text.split())
        entry["turns"] += 1

    total_seconds = sum(entry["seconds"] for entry in stats.values()) or 1.0
    rows = [
        {
            "speaker": label,
            "seconds": round(entry["seconds"], 2),
            "words": int(entry["words"]),
            "turns": int(entry["turns"]),
            "share": round(entry["seconds"] / total_seconds, 4),
        }
        for label, entry in stats.items()
    ]
    rows.sort(key=lambda row: float(row["seconds"]), reverse=True)
    return rows


def apply_speaker_names(segments: Sequence[TranscriptSegment],
                        names: dict[str, str]) -> list[TranscriptSegment]:
    """Replace speaker labels using a label -> real name mapping."""
    if not names:
        return list(segments)
    return [
        TranscriptSegment(
            start=segment.start, end=segment.end, text=segment.text,
            speaker=names.get(segment.speaker or "", segment.speaker),
            words=segment.words, confidence=segment.confidence,
        )
        for segment in segments
    ]
