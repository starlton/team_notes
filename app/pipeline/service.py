"""The one object the CLI, the tray and the web API all drive.

Keeping recording state in a single place means the tray menu, the dashboard
and the auto-detector can never disagree about whether a recording is running,
and the consent rules are enforced once rather than in three places.
"""

from __future__ import annotations

import shutil
import threading
from pathlib import Path
from typing import Any

from app.capture.recorder import DualRecorder
from app.db.database import Database, get_database
from app.db.repository import (Repository, STATUS_COMPLETE, STATUS_DISCARDED,
                               STATUS_FAILED, STATUS_RECORDED, utc_now)
from app.errors import CaptureError, NotFoundError, TeamsNotesError
from app.intelligence.ollama_client import OllamaClient
from app.logging_setup import get_logger
from app.paths import meeting_audio_dir
from app.pipeline.jobs import JobQueue
from app.pipeline.processor import MeetingProcessor
from config import Settings

log = get_logger(__name__)

# app_settings keys
CONSENT_ACKNOWLEDGED = "consent_acknowledged"
CONSENT_REMINDER = "consent_reminder_enabled"
AUTO_DETECT = "auto_detect_enabled"

CONSENT_NOTICE = (
    "This app records meeting audio on this computer. Recording a conversation "
    "may require telling the other participants, and may be covered by your "
    "employer's policy. You are responsible for getting whatever consent is "
    "needed before you record. Nothing recorded here leaves this machine."
)


class MeetingService:
    """Recording state, the processing queue, and everything the UI asks about."""

    def __init__(self, settings: Settings, database: Database | None = None) -> None:
        self.settings = settings
        settings.ensure_dirs()
        self.db = database or get_database(settings.db_path)
        self.db.initialise()
        self.repo = Repository(self.db)
        self.jobs = JobQueue()
        self.processor = MeetingProcessor(settings, self.repo)

        self._lock = threading.RLock()
        self._recorder: DualRecorder | None = None
        self._meeting_id: int | None = None
        self._started = False

    # --- lifecycle --------------------------------------------------------

    def start(self) -> None:
        """Start background services and clean up after any previous crash."""
        with self._lock:
            if self._started:
                return
            recovered = self.repo.recover_interrupted()
            if recovered:
                log.warning("Marked %d interrupted meeting(s) as failed", recovered)
            self.jobs.start()
            self._started = True
            self._apply_retention()

    def shutdown(self) -> None:
        """Stop cleanly, finishing any recording rather than losing it."""
        with self._lock:
            if self._recorder is not None and self._recorder.is_recording:
                log.info("Stopping the active recording before shutdown")
                try:
                    self.stop_recording(process=False)
                except TeamsNotesError:
                    log.exception("Could not stop the recording cleanly")
            self.jobs.stop()
            self._started = False

    # --- consent ----------------------------------------------------------

    @property
    def consent_acknowledged(self) -> bool:
        return self.repo.get_bool_setting(CONSENT_ACKNOWLEDGED, False)

    def acknowledge_consent(self, acknowledged: bool = True) -> None:
        self.repo.set_bool_setting(CONSENT_ACKNOWLEDGED, acknowledged)
        log.info("Consent notice acknowledged: %s", acknowledged)

    @property
    def consent_reminder_enabled(self) -> bool:
        return self.repo.get_bool_setting(CONSENT_REMINDER, True)

    def set_consent_reminder(self, enabled: bool) -> None:
        self.repo.set_bool_setting(CONSENT_REMINDER, enabled)

    def consent_state(self) -> dict[str, Any]:
        return {
            "notice": CONSENT_NOTICE,
            "acknowledged": self.consent_acknowledged,
            "reminder_enabled": self.consent_reminder_enabled,
        }

    # --- auto-detect toggle ------------------------------------------------

    @property
    def auto_detect_enabled(self) -> bool:
        return self.repo.get_bool_setting(AUTO_DETECT, self.settings.auto_detect_enabled)

    def set_auto_detect(self, enabled: bool) -> None:
        self.repo.set_bool_setting(AUTO_DETECT, enabled)
        log.info("Automatic meeting detection: %s", "on" if enabled else "off")

    # --- recording --------------------------------------------------------

    @property
    def is_recording(self) -> bool:
        with self._lock:
            return self._recorder is not None and self._recorder.is_recording

    @property
    def current_meeting_id(self) -> int | None:
        with self._lock:
            return self._meeting_id

    def start_recording(self, title: str = "", source: str = "manual",
                        participants_informed: bool = False) -> int:
        """Begin recording. Returns the new meeting id."""
        with self._lock:
            if self.is_recording:
                raise CaptureError(
                    "A recording is already in progress.",
                    "Stop it before starting another one.",
                )
            if not self.consent_acknowledged:
                raise CaptureError(
                    "The recording notice has not been acknowledged yet.",
                    "Open the dashboard and confirm the notice once, then record.",
                )

            meeting_id = self.repo.create_meeting(
                title=title, source=source,
                participants_informed=participants_informed)
            output_dir = meeting_audio_dir(self.settings.audio_dir, meeting_id)

            recorder = DualRecorder(
                output_dir=output_dir,
                target_sample_rate=self.settings.capture_sample_rate,
                mic_gain=self.settings.mic_gain,
                loopback_gain=self.settings.loopback_gain,
                mic_device_name=self.settings.mic_device_name,
                loopback_device_name=self.settings.loopback_device_name,
                keep_raw_tracks=self.settings.keep_raw_tracks,
            )
            try:
                recorder.start()
            except TeamsNotesError as exc:
                self.repo.update_meeting(meeting_id, status=STATUS_FAILED,
                                         error=str(exc), ended_at=utc_now())
                raise

            self._recorder = recorder
            self._meeting_id = meeting_id
            log.info("Recording started for meeting %d (%s)", meeting_id, source)
            return meeting_id

    def stop_recording(self, process: bool = True,
                       min_seconds: float = 0.0) -> dict[str, Any]:
        """Stop recording, save the audio, and optionally queue processing."""
        with self._lock:
            recorder = self._recorder
            meeting_id = self._meeting_id
            if recorder is None or meeting_id is None or not recorder.is_recording:
                raise CaptureError("No recording is in progress.", "")

            try:
                result = recorder.stop()
            except TeamsNotesError as exc:
                self.repo.update_meeting(meeting_id, status=STATUS_FAILED,
                                         error=str(exc), ended_at=utc_now())
                self._recorder = None
                self._meeting_id = None
                raise
            finally:
                self._recorder = None
                self._meeting_id = None

            if min_seconds and result.duration_seconds < min_seconds:
                log.info("Discarding meeting %d: only %.1fs long",
                         meeting_id, result.duration_seconds)
                self.discard_meeting(meeting_id)
                return {"meeting_id": meeting_id, "discarded": True,
                        "duration_seconds": result.duration_seconds,
                        "queued": False, "warnings": result.warnings}

            self.repo.update_meeting(
                meeting_id,
                status=STATUS_RECORDED,
                ended_at=utc_now(),
                duration_seconds=result.duration_seconds,
                audio_path=str(result.mixed_path),
                warnings=result.warnings,
                stage="Recorded",
                progress=0.0,
            )

        queued = self.queue_processing(meeting_id) if process else False
        return {"meeting_id": meeting_id, "discarded": False,
                "duration_seconds": result.duration_seconds,
                "audio_path": str(result.mixed_path), "queued": queued,
                "warnings": result.warnings}

    def discard_recording(self) -> int | None:
        """Abort the current recording and throw its audio away."""
        with self._lock:
            recorder = self._recorder
            meeting_id = self._meeting_id
            if recorder is None or meeting_id is None:
                return None
            recorder.abort()
            self._recorder = None
            self._meeting_id = None
        self.discard_meeting(meeting_id)
        return meeting_id

    def discard_meeting(self, meeting_id: int) -> None:
        """Mark a meeting discarded and remove its audio."""
        self._delete_audio(meeting_id)
        self.repo.update_meeting(meeting_id, status=STATUS_DISCARDED,
                                 ended_at=utc_now(), audio_path="", stage="",
                                 progress=0.0)

    def delete_meeting(self, meeting_id: int) -> None:
        """Remove a meeting and everything belonging to it, audio included."""
        self.repo.get_meeting(meeting_id)  # raises NotFoundError if absent
        self._delete_audio(meeting_id)
        self.repo.delete_meeting(meeting_id)
        log.info("Deleted meeting %d", meeting_id)

    def _delete_audio(self, meeting_id: int) -> None:
        directory = self.settings.audio_dir / f"meeting-{int(meeting_id):06d}"
        try:
            if directory.is_dir():
                shutil.rmtree(directory, ignore_errors=True)
        except OSError:  # pragma: no cover - file locked by another process
            log.warning("Could not remove audio for meeting %d", meeting_id)

    # --- processing -------------------------------------------------------

    def queue_processing(self, meeting_id: int) -> bool:
        """Queue the post-meeting pass for a recorded meeting."""
        meeting = self.repo.get_meeting(meeting_id)
        if not meeting.get("audio_path"):
            raise NotFoundError(
                f"Meeting {meeting_id} has no audio to process.",
                "It may have been discarded or its audio deleted.",
            )
        self.jobs.start()
        return self.jobs.submit(
            key=f"process-{meeting_id}",
            description=f"Processing meeting {meeting_id}",
            run=lambda: self.processor.process(meeting_id),
        )

    def queue_regenerate(self, meeting_id: int) -> bool:
        """Queue a re-run of only the LLM step, e.g. after renaming speakers."""
        self.repo.get_meeting(meeting_id)
        self.jobs.start()
        return self.jobs.submit(
            key=f"regenerate-{meeting_id}",
            description=f"Regenerating notes for meeting {meeting_id}",
            run=lambda: self.processor.regenerate_notes(meeting_id),
        )

    # --- status -----------------------------------------------------------

    def status(self) -> dict[str, Any]:
        """Everything the dashboard's header and the tray tooltip need."""
        with self._lock:
            recorder = self._recorder
            meeting_id = self._meeting_id
            recording = recorder is not None and recorder.is_recording
            elapsed = recorder.elapsed_seconds if recorder else 0.0
            levels = recorder.levels() if recorder else {}

        return {
            "recording": recording,
            "meeting_id": meeting_id,
            "elapsed_seconds": round(elapsed, 1),
            "levels": {key: (None if value == float("-inf") else round(value, 1))
                       for key, value in levels.items()},
            "jobs": self.jobs.status(),
            "consent": self.consent_state(),
            "auto_detect_enabled": self.auto_detect_enabled,
        }

    def health(self) -> dict[str, Any]:
        """Report which optional pieces are ready, without raising."""
        ollama = OllamaClient(
            host=self.settings.ollama_host, model=self.settings.ollama_model,
            timeout_seconds=self.settings.ollama_timeout_seconds,
            num_ctx=self.settings.ollama_num_ctx,
        ).health()

        diarization: dict[str, Any]
        if not self.settings.diarization_enabled:
            diarization = {"ok": False, "error": "Diarization is turned off in .env.",
                           "remedy": "Set DIARIZATION_ENABLED=true to label speakers."}
        elif not self.settings.hf_token:
            diarization = {
                "ok": False,
                "error": "No Hugging Face token is set, so speakers cannot be labelled.",
                "remedy": "Add HF_TOKEN to your .env file. See the README.",
            }
        else:
            diarization = {"ok": True, "error": "", "remedy": ""}

        return {
            "ollama": ollama,
            "diarization": diarization,
            "whisper": {"model": self.settings.whisper_model,
                        "compute_type": self.settings.whisper_compute_type},
            "storage": {"data_dir": str(self.settings.data_dir),
                        "meetings": self.repo.count_meetings()},
            "consent": self.consent_state(),
        }

    # --- maintenance ------------------------------------------------------

    def _apply_retention(self) -> None:
        """Delete audio past the retention window, keeping the notes."""
        days = self.settings.audio_retention_days
        if days <= 0:
            return
        for meeting in self.repo.meetings_with_audio_older_than(days):
            meeting_id = int(meeting["id"])
            if meeting.get("status") not in (STATUS_COMPLETE, STATUS_DISCARDED):
                continue
            self._delete_audio(meeting_id)
            self.repo.clear_audio_path(meeting_id)
            log.info("Removed audio for meeting %d (retention: %d days)",
                     meeting_id, days)

    def audio_path_for(self, meeting_id: int) -> Path | None:
        """The meeting's mixed WAV, if it still exists inside our data dir."""
        from app.paths import resolve_within

        meeting = self.repo.get_meeting(meeting_id)
        raw = str(meeting.get("audio_path") or "")
        if not raw:
            return None
        try:
            path = resolve_within(self.settings.audio_dir, raw)
        except TeamsNotesError:
            log.warning("Meeting %d has an audio path outside the data directory",
                        meeting_id)
            return None
        return path if path.is_file() else None
