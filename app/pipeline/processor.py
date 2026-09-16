"""The post-meeting processing pass.

This is the "after" half of the live/after split. Nothing here runs while the
meeting is in progress, because all of it is far too heavy for a CPU-only
machine to do alongside capture.

Order: transcribe -> diarize -> merge -> per-speaker stats -> LLM notes.

Each stage degrades rather than aborts where it sensibly can. No Hugging Face
token means an unattributed transcript rather than no transcript. Ollama being
down means a stored transcript with a clear error rather than a lost meeting.
Only transcription is truly required, because everything downstream needs it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from app.db.repository import (Repository, STATUS_COMPLETE, STATUS_FAILED,
                               STATUS_PROCESSING)
from app.diarize.merge import (apply_speaker_names, assign_speakers, merge_adjacent,
                               relabel_speakers, speaker_statistics)
from app.diarize.pyannote_runner import Diarizer
from app.errors import TeamsNotesError, TranscriptionError
from app.intelligence.ollama_client import OllamaClient
from app.intelligence.pipeline import IntelligenceEngine, fallback_drafts
from app.logging_setup import get_logger
from app.transcribe.models import TranscriptResult, TranscriptSegment
from app.transcribe.whisper import Transcriber
from config import Settings

log = get_logger(__name__)

ProgressCallback = Callable[[str, float], None]

# Stage weights, so the dashboard's progress bar moves at a believable pace.
STAGE_WEIGHTS = {
    "transcribe": (0.05, 0.50),
    "diarize": (0.50, 0.75),
    "analyse": (0.75, 0.99),
}


@dataclass
class ProcessingOutcome:
    """What the pass produced."""

    meeting_id: int
    segments: list[TranscriptSegment] = field(default_factory=list)
    speakers: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    failed: bool = False
    error: str = ""


class MeetingProcessor:
    """Runs a recorded meeting through the whole post-meeting pipeline."""

    def __init__(self, settings: Settings, repository: Repository) -> None:
        self.settings = settings
        self.repo = repository

    # --- component factories (overridable in tests) -----------------------

    def make_transcriber(self) -> Transcriber:
        return Transcriber(
            model_size=self.settings.whisper_model,
            compute_type=self.settings.whisper_compute_type,
            cpu_threads=self.settings.whisper_cpu_threads,
            download_root=self.settings.data_dir / "models",
        )

    def make_diarizer(self) -> Diarizer:
        return Diarizer(
            model=self.settings.pyannote_model,
            hf_token=self.settings.hf_token.get(),
            min_speakers=self.settings.diarization_min_speakers,
            max_speakers=self.settings.diarization_max_speakers,
            cache_dir=self.settings.data_dir / "models",
        )

    def make_engine(self) -> IntelligenceEngine:
        return IntelligenceEngine(OllamaClient(
            host=self.settings.ollama_host,
            model=self.settings.ollama_model,
            timeout_seconds=self.settings.ollama_timeout_seconds,
            num_ctx=self.settings.ollama_num_ctx,
        ))

    # --- the pass ---------------------------------------------------------

    def process(self, meeting_id: int,
                progress: ProgressCallback | None = None) -> ProcessingOutcome:
        """Process one meeting end to end, recording progress as it goes."""
        meeting = self.repo.get_meeting(meeting_id)
        audio_path = Path(str(meeting.get("audio_path") or ""))
        warnings: list[str] = list(meeting.get("warnings") or [])

        def report(stage: str, fraction: float, message: str) -> None:
            low, high = STAGE_WEIGHTS.get(stage, (0.0, 1.0))
            overall = low + (high - low) * max(0.0, min(1.0, fraction))
            self.repo.set_progress(meeting_id, message, overall)
            if progress:
                progress(message, overall)

        if not audio_path.is_file():
            error = "The recording's audio file is missing, so it cannot be processed."
            self.repo.update_meeting(meeting_id, status=STATUS_FAILED, error=error,
                                     stage="", progress=0)
            return ProcessingOutcome(meeting_id=meeting_id, failed=True, error=error)

        self.repo.update_meeting(meeting_id, status=STATUS_PROCESSING, error="",
                                 stage="Starting", progress=0.02)

        try:
            transcript = self._transcribe(audio_path, report)
        except TeamsNotesError as exc:
            log.error("Transcription failed for meeting %d: %s", meeting_id, exc)
            self.repo.update_meeting(meeting_id, status=STATUS_FAILED,
                                     error=str(exc), stage="", progress=0)
            return ProcessingOutcome(meeting_id=meeting_id, failed=True, error=str(exc))

        segments = transcript.segments
        if not segments:
            warnings.append("No speech was detected in the recording.")

        # --- diarization (optional) ---------------------------------------
        segments, _label_map, diarize_warnings = self._diarize(audio_path, segments,
                                                               report)
        warnings.extend(diarize_warnings)

        # --- persist the transcript early ---------------------------------
        # Saving before the LLM step means a failure there still leaves the user
        # with a usable transcript rather than nothing.
        self.repo.replace_transcript(meeting_id, segments)
        stats = speaker_statistics(segments)
        self.repo.replace_speakers(meeting_id, [
            {
                "label": row["speaker"], "display_name": row["speaker"],
                "talk_seconds": row["seconds"], "word_count": row["words"],
                "turn_count": row["turns"], "share": row["share"],
            }
            for row in stats
        ])
        if not meeting.get("title"):
            self.repo.update_meeting(meeting_id, title=_default_title(meeting))

        # --- LLM notes -----------------------------------------------------
        report("analyse", 0.0, "Generating notes with the local model")
        try:
            outcome = self.make_engine().analyse(
                segments,
                speaker_names=self.repo.speaker_name_map(meeting_id),
                progress=lambda fraction, message: report("analyse", fraction, message),
            )
        except TeamsNotesError as exc:
            warnings.append(f"Notes were not generated: {exc}")
            log.error("Analysis failed for meeting %d: %s", meeting_id, exc)
            self.repo.update_meeting(meeting_id, status=STATUS_FAILED, error=str(exc),
                                     stage="", progress=0, warnings=warnings)
            return ProcessingOutcome(meeting_id=meeting_id, segments=segments,
                                     speakers=stats, warnings=warnings,
                                     failed=True, error=str(exc))

        intelligence = outcome.intelligence
        warnings.extend(outcome.warnings)
        if not intelligence.drafts:
            intelligence.drafts = fallback_drafts(intelligence.summary,
                                                  intelligence.action_items)
            warnings.append("Follow-up drafts fell back to a plain recap.")

        self.repo.save_intelligence(meeting_id, intelligence,
                                    self.settings.ollama_model)
        self._merge_speaker_summaries(meeting_id, intelligence, stats)

        if intelligence.title and not meeting.get("title"):
            self.repo.update_meeting(meeting_id, title=intelligence.title)

        self.repo.update_meeting(meeting_id, status=STATUS_COMPLETE, stage="Done",
                                 progress=1.0, error="", warnings=warnings)
        if progress:
            progress("Done", 1.0)
        log.info("Meeting %d processed: %d segments, %d speakers, %d action items",
                 meeting_id, len(segments), len(stats), len(intelligence.action_items))
        return ProcessingOutcome(meeting_id=meeting_id, segments=segments,
                                 speakers=stats, warnings=warnings)

    # --- stages -----------------------------------------------------------

    def _transcribe(self, audio_path: Path,
                    report: Callable[[str, float, str], None]) -> TranscriptResult:
        transcriber = self.make_transcriber()
        result = transcriber.transcribe(
            audio_path,
            progress=lambda fraction, message: report("transcribe", fraction, message),
        )
        if result is None:  # pragma: no cover - defensive
            raise TranscriptionError("The transcriber returned nothing.", "")
        return result

    def _diarize(self, audio_path: Path, segments: list[TranscriptSegment],
                 report: Callable[[str, float, str], None]
                 ) -> tuple[list[TranscriptSegment], dict[str, str], list[str]]:
        """Attribute segments to speakers, or leave them unattributed."""
        warnings: list[str] = []
        if not segments:
            return segments, {}, warnings

        if not self.settings.diarization_enabled:
            return segments, {}, warnings

        diarizer = self.make_diarizer()
        if not diarizer.available:
            warnings.append(
                "Speakers were not identified because no Hugging Face token is set. "
                "See the README for the one-time setup."
            )
            return segments, {}, warnings

        report("diarize", 0.0, "Identifying speakers")
        try:
            turns = diarizer.diarize(
                audio_path,
                progress=lambda fraction, message: report("diarize", fraction, message),
            )
        except TeamsNotesError as exc:
            warnings.append(f"Speakers were not identified: {exc}")
            log.warning("Diarization failed: %s", exc)
            return segments, {}, warnings

        if not turns:
            warnings.append("Diarization found no speaker turns.")
            return segments, {}, warnings

        attributed = merge_adjacent(assign_speakers(segments, turns))
        relabelled, label_map = relabel_speakers(attributed)
        report("diarize", 1.0, f"Found {len(label_map)} speaker(s)")
        return relabelled, label_map, warnings

    def _merge_speaker_summaries(self, meeting_id: int, intelligence,
                                 stats: list[dict],
                                 names: dict[str, str] | None = None) -> None:
        """Attach the LLM's per-speaker prose to the computed talk-time stats.

        `names` maps diarization label -> the display name the LLM was given.
        Stats are always keyed by the original label so that renaming stays
        reversible: the row's key never becomes the person's name.
        """
        names = names or {}
        by_name = {entry.speaker.strip().casefold(): entry
                   for entry in intelligence.speakers if entry.speaker}
        rows = []
        for row in stats:
            label = str(row["speaker"])
            display = names.get(label, label)
            match = by_name.get(display.casefold()) or by_name.get(label.casefold())
            rows.append({
                "label": label,
                "display_name": display,
                "talk_seconds": row["seconds"],
                "word_count": row["words"],
                "turn_count": row["turns"],
                "share": row["share"],
                "summary": match.summary if match else "",
                "key_points": match.key_points if match else [],
            })
        self.repo.replace_speakers(meeting_id, rows)

    # --- re-runs ----------------------------------------------------------

    def regenerate_notes(self, meeting_id: int,
                         progress: ProgressCallback | None = None) -> ProcessingOutcome:
        """Re-run only the LLM step, e.g. after renaming speakers."""
        segments = self.repo.get_transcript(meeting_id)
        if not segments:
            return ProcessingOutcome(meeting_id=meeting_id, failed=True,
                                     error="This meeting has no transcript to work from.")

        self.repo.update_meeting(meeting_id, status=STATUS_PROCESSING, error="",
                                 stage="Regenerating notes", progress=0.1)
        names = self.repo.speaker_name_map(meeting_id)
        named_segments = apply_speaker_names(segments, names)

        def report(fraction: float, message: str) -> None:
            self.repo.set_progress(meeting_id, message, 0.1 + 0.89 * fraction)
            if progress:
                progress(message, 0.1 + 0.89 * fraction)

        try:
            outcome = self.make_engine().analyse(named_segments, progress=report)
        except TeamsNotesError as exc:
            self.repo.update_meeting(meeting_id, status=STATUS_FAILED, error=str(exc),
                                     stage="", progress=0)
            return ProcessingOutcome(meeting_id=meeting_id, failed=True, error=str(exc))

        intelligence = outcome.intelligence
        if not intelligence.drafts:
            intelligence.drafts = fallback_drafts(intelligence.summary,
                                                  intelligence.action_items)

        self.repo.save_intelligence(meeting_id, intelligence, self.settings.ollama_model)
        # Stats come from the original labels so the label -> name mapping the
        # user set in the dashboard survives a regenerate.
        stats = speaker_statistics(segments)
        self._merge_speaker_summaries(meeting_id, intelligence, stats, names)
        self.repo.update_meeting(meeting_id, status=STATUS_COMPLETE, stage="Done",
                                 progress=1.0, error="", warnings=outcome.warnings)
        return ProcessingOutcome(meeting_id=meeting_id, segments=segments,
                                 speakers=stats, warnings=outcome.warnings)


def _default_title(meeting: dict) -> str:
    started = str(meeting.get("started_at") or "")
    return f"Meeting {started[:16].replace('T', ' ')}".strip() or "Meeting"
