"""Turning a labelled transcript into meeting notes.

Short meetings go straight to the reduce prompts. Long ones are chunked, each
chunk is summarised into structured notes (the map step), the notes are merged,
and the reduce prompts run over the merged notes instead of the raw transcript.

Each reduce step is isolated: if the model fumbles the drafts, the summary and
to-dos are still produced and the failure is recorded as a warning rather than
losing the whole meeting's notes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

from app.errors import OllamaError
from app.intelligence import prompts
from app.intelligence.chunking import chunk_budget_chars, chunk_transcript
from app.intelligence.ollama_client import OllamaClient
from app.intelligence.schemas import (ChunkNotes, DraftMessage, MeetingIntelligence,
                                      SpeakerSummary)
from app.logging_setup import get_logger
from app.transcribe.models import TranscriptSegment

log = get_logger(__name__)

ProgressCallback = Callable[[float, str], None]

# How much of each speaker's words to send for the per-speaker breakdown.
SPEAKER_BLOCK_BUDGET = 9000
# Chunk notes are merged this many at a time to stay inside the context window.
MERGE_BATCH_SIZE = 6


@dataclass
class AnalysisOutcome:
    """The notes, plus anything that went wrong while producing them."""

    intelligence: MeetingIntelligence
    warnings: list[str] = field(default_factory=list)
    chunks_processed: int = 0


def build_speaker_blocks(segments: Sequence[TranscriptSegment],
                         budget_chars: int = SPEAKER_BLOCK_BUDGET) -> str:
    """Group the transcript by speaker, fairly sharing a character budget."""
    grouped: dict[str, list[str]] = {}
    for segment in segments:
        text = segment.text.strip()
        if not text:
            continue
        grouped.setdefault(segment.speaker or "Unknown", []).append(text)

    if not grouped:
        return ""

    per_speaker = max(500, budget_chars // len(grouped))
    blocks: list[str] = []
    for speaker, lines in grouped.items():
        body = " ".join(lines)
        if len(body) > per_speaker:
            # Keep the start and the end: openings set context, closings hold
            # the commitments.
            head = body[: per_speaker // 2].rstrip()
            tail = body[-(per_speaker // 2):].lstrip()
            body = f"{head} [...] {tail}"
        blocks.append(f"### {speaker}\n{body}")
    return "\n\n".join(blocks)


def _render_notes(notes: ChunkNotes) -> str:
    """Render structured notes back into text for the next prompt."""
    parts: list[str] = []
    if notes.key_points:
        parts.append("Key points:\n" + "\n".join(f"- {p}" for p in notes.key_points))
    if notes.decisions:
        parts.append("Decisions:\n" + "\n".join(f"- {d}" for d in notes.decisions))
    if notes.questions:
        parts.append("Open questions:\n" + "\n".join(f"- {q}" for q in notes.questions))
    if notes.action_items:
        lines = [
            f"- {item.task} (owner: {item.owner}"
            + (f", due: {item.due}" if item.due else "")
            + f", priority: {item.priority})"
            for item in notes.action_items
        ]
        parts.append("Action items:\n" + "\n".join(lines))
    return "\n\n".join(parts)


def _combine_notes(batch: Sequence[ChunkNotes]) -> ChunkNotes:
    """Concatenate notes locally, de-duplicating case-insensitively."""
    combined = ChunkNotes()
    for attr in ("key_points", "decisions", "questions"):
        seen: set[str] = set()
        merged: list[str] = []
        for notes in batch:
            for item in getattr(notes, attr):
                key = item.casefold()
                if key not in seen:
                    seen.add(key)
                    merged.append(item)
        setattr(combined, attr, merged)

    seen_tasks: set[str] = set()
    for notes in batch:
        for item in notes.action_items:
            key = item.task.casefold()
            if key and key not in seen_tasks:
                seen_tasks.add(key)
                combined.action_items.append(item)
    return combined


class IntelligenceEngine:
    """Runs the map-reduce analysis over a transcript."""

    def __init__(self, client: OllamaClient) -> None:
        self.client = client

    # --- public API -------------------------------------------------------

    def analyse(self, segments: Sequence[TranscriptSegment],
                speaker_names: dict[str, str] | None = None,
                progress: ProgressCallback | None = None) -> AnalysisOutcome:
        """Produce summary, bullets, to-dos, priorities, speakers and drafts."""
        warnings: list[str] = []
        names = speaker_names or {}
        usable = [segment for segment in segments if segment.text.strip()]
        if not usable:
            return AnalysisOutcome(
                intelligence=MeetingIntelligence(
                    summary="The recording contained no speech to summarise."),
                warnings=["Transcript was empty."],
            )

        self.client.ensure_ready()

        participants = []
        for segment in usable:
            label = names.get(segment.speaker or "", segment.speaker or "")
            if label and label not in participants:
                participants.append(label)

        budget = chunk_budget_chars(self.client.num_ctx)
        chunks = chunk_transcript(usable, budget, names)
        _report(progress, 0.05, f"Analysing {len(chunks)} transcript part(s)")

        if len(chunks) <= 1:
            body = chunks[0].text if chunks else ""
        else:
            notes, map_warnings = self._map_chunks(chunks, progress)
            warnings.extend(map_warnings)
            body = _render_notes(notes) or (chunks[0].text if chunks else "")

        result = MeetingIntelligence()

        # --- overview ----------------------------------------------------
        _report(progress, 0.55, "Writing the summary")
        try:
            system, user = prompts.overview_prompt(body, participants)
            overview = MeetingIntelligence.model_validate(
                self.client.chat_json(system, user, max_tokens=1200))
            result.title = overview.title
            result.summary = overview.summary
            result.bullets = overview.bullets
            result.priorities = overview.priorities
        except OllamaError as exc:
            warnings.append(f"Summary step failed: {exc.message}")
            log.warning("Overview step failed: %s", exc.message)

        # --- action items -------------------------------------------------
        _report(progress, 0.70, "Pulling out action items")
        try:
            system, user = prompts.actions_prompt(body, participants)
            actions = MeetingIntelligence.model_validate(
                self.client.chat_json(system, user, max_tokens=1200))
            result.action_items = actions.action_items
        except OllamaError as exc:
            warnings.append(f"Action item step failed: {exc.message}")
            log.warning("Actions step failed: %s", exc.message)

        # --- per-speaker breakdown ---------------------------------------
        _report(progress, 0.82, "Summarising each speaker")
        if len(participants) > 1:
            try:
                blocks = build_speaker_blocks(
                    [_renamed(segment, names) for segment in usable])
                system, user = prompts.speakers_prompt(blocks, participants)
                speakers = MeetingIntelligence.model_validate(
                    self.client.chat_json(system, user, max_tokens=1400))
                result.speakers = speakers.speakers
            except OllamaError as exc:
                warnings.append(f"Per-speaker step failed: {exc.message}")
                log.warning("Speakers step failed: %s", exc.message)
        elif participants:
            result.speakers = [SpeakerSummary(speaker=participants[0],
                                              summary=result.summary)]

        # --- drafts --------------------------------------------------------
        _report(progress, 0.92, "Drafting follow-up messages")
        try:
            system, user = prompts.drafts_prompt(
                result.summary, result.bullets,
                [f"{item.task} ({item.owner})" for item in result.action_items],
                participants,
            )
            drafts = MeetingIntelligence.model_validate(
                self.client.chat_json(system, user, temperature=0.4, max_tokens=1400))
            result.drafts = drafts.drafts
        except OllamaError as exc:
            warnings.append(f"Draft message step failed: {exc.message}")
            log.warning("Drafts step failed: %s", exc.message)

        if result.is_empty():
            raise OllamaError(
                "The local model produced no usable notes for this meeting.",
                "Check that Ollama has enough free RAM, or try the lighter model: "
                "ollama pull llama3.2:3b, then set OLLAMA_MODEL=llama3.2:3b",
            )

        _report(progress, 1.0, "Notes ready")
        return AnalysisOutcome(intelligence=result, warnings=warnings,
                               chunks_processed=len(chunks))

    # --- map step ---------------------------------------------------------

    def _map_chunks(self, chunks: Sequence, progress: ProgressCallback | None
                    ) -> tuple[ChunkNotes, list[str]]:
        """Summarise each chunk, then merge the notes."""
        warnings: list[str] = []
        per_chunk: list[ChunkNotes] = []

        for chunk in chunks:
            fraction = 0.05 + 0.45 * ((chunk.index + 1) / max(1, len(chunks)))
            _report(progress, fraction, f"Reading {chunk.label}")
            try:
                system, user = prompts.chunk_prompt(chunk.text, chunk.label)
                per_chunk.append(ChunkNotes.model_validate(
                    self.client.chat_json(system, user, max_tokens=1200)))
            except OllamaError as exc:
                warnings.append(f"Could not read {chunk.label}: {exc.message}")
                log.warning("Chunk %d failed: %s", chunk.index, exc.message)

        if not per_chunk:
            return ChunkNotes(), warnings

        merged = _combine_notes(per_chunk)
        # Only ask the model to merge when local de-duplication left a lot; it
        # catches near-duplicates that differ in wording.
        if len(per_chunk) > 1 and len(merged.key_points) > MERGE_BATCH_SIZE:
            try:
                system, user = prompts.merge_notes_prompt(_render_notes(merged))
                merged = ChunkNotes.model_validate(
                    self.client.chat_json(system, user, max_tokens=1600))
            except OllamaError as exc:
                warnings.append(f"Note consolidation failed: {exc.message}")
                log.warning("Merge step failed: %s", exc.message)

        return merged, warnings


def _renamed(segment: TranscriptSegment,
             names: dict[str, str]) -> TranscriptSegment:
    label = segment.speaker or "Unknown"
    return TranscriptSegment(
        start=segment.start, end=segment.end, text=segment.text,
        speaker=names.get(label, label), words=segment.words,
        confidence=segment.confidence,
    )


def _report(progress: ProgressCallback | None, fraction: float, message: str) -> None:
    if progress:
        progress(max(0.0, min(1.0, fraction)), message)


def fallback_drafts(summary: str, action_items: Sequence) -> list[DraftMessage]:
    """A plain recap built without the model, used when drafting fails."""
    lines = [f"- {item.task} ({item.owner})" for item in action_items]
    body = (f"Hi all,\n\nThanks for the time today.\n\n{summary}\n\n"
            + ("Action items:\n" + "\n".join(lines) + "\n\n" if lines else "")
            + "Shout if I've missed anything.\n\n[Your name]")
    return [DraftMessage(kind="email", audience="The meeting participants",
                         subject="Meeting recap", body=body)]
