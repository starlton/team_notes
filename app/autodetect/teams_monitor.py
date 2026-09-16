"""Watching Windows for a Teams call.

The probe asks the Windows audio session API which processes currently hold an
active audio session. Teams holds one while a call is up and releases it when
the call ends, which is the closest thing to a local "am I in a meeting?" signal
that does not need the Microsoft Graph API, an admin consent flow, or a paid
tier.

Probing is separated from deciding (see state_machine.py) so the timing rules
can be tested without Windows.
"""

from __future__ import annotations

import sys
import threading
import time
from dataclasses import dataclass
from typing import Callable, Sequence

from app.autodetect.state_machine import Action, MeetingDetector
from app.errors import TeamsNotesError
from app.logging_setup import get_logger

log = get_logger(__name__)

POLL_INTERVAL_SECONDS = 2.0
# Windows AudioSessionState: 0 inactive, 1 active, 2 expired.
AUDIO_SESSION_ACTIVE = 1
# Cooldown after a recording stops, so the tail of a call cannot immediately
# re-arm the detector.
RESTART_COOLDOWN_SECONDS = 20.0


@dataclass
class Probe:
    """One observation of the system."""

    teams_running: bool = False
    audio_active: bool = False
    detail: str = ""


class TeamsAudioProbe:
    """Reports whether a Teams process currently has active audio."""

    def __init__(self, process_names: Sequence[str]) -> None:
        self.process_names = {name.casefold() for name in process_names if name}
        self._unavailable_reason = ""
        self._warned = False

    @property
    def unavailable_reason(self) -> str:
        return self._unavailable_reason

    def available(self) -> bool:
        """Whether the probe can run at all on this machine."""
        if sys.platform != "win32":
            self._unavailable_reason = (
                "Automatic meeting detection needs the Windows audio session API."
            )
            return False
        try:
            from pycaw.pycaw import AudioUtilities  # noqa: F401
        except ImportError:
            self._unavailable_reason = (
                "Automatic detection needs the 'pycaw' package. "
                "Install it with: pip install pycaw"
            )
            return False
        self._unavailable_reason = ""
        return True

    def probe(self) -> Probe:
        """Take one reading. Never raises; a failed probe reads as 'no call'."""
        if not self.available():
            return Probe(detail=self._unavailable_reason)

        try:
            from pycaw.pycaw import AudioUtilities

            sessions = AudioUtilities.GetAllSessions()
        except Exception as exc:  # pragma: no cover - COM is unpredictable
            if not self._warned:
                log.warning("Could not read Windows audio sessions: %s", exc)
                self._warned = True
            return Probe(detail=f"audio session query failed: {exc}")

        teams_running = False
        audio_active = False
        for session in sessions:
            process = getattr(session, "Process", None)
            if process is None:
                continue
            try:
                name = str(process.name()).casefold()
            except Exception:  # pragma: no cover - process died mid-enumeration
                continue
            if name not in self.process_names:
                continue
            teams_running = True
            try:
                if int(session.State) == AUDIO_SESSION_ACTIVE:
                    audio_active = True
                    break
            except Exception:  # pragma: no cover
                continue

        return Probe(teams_running=teams_running, audio_active=audio_active,
                     detail="")


class TeamsMonitor:
    """Polls the probe and starts/stops recordings through the service."""

    def __init__(self, service, probe: TeamsAudioProbe | None = None,
                 poll_interval: float = POLL_INTERVAL_SECONDS,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.service = service
        settings = service.settings
        self.probe = probe or TeamsAudioProbe(settings.teams_process_names)
        self.poll_interval = poll_interval
        self._clock = clock
        self.detector = MeetingDetector(
            start_delay=settings.auto_start_after_seconds,
            stop_delay=settings.auto_stop_after_seconds,
        )
        self.min_meeting_seconds = settings.auto_min_meeting_seconds

        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        # -inf, not 0: on a freshly booted machine time.monotonic() is small,
        # and a 0 here would put the very first call inside the cooldown.
        self._last_stop_at = float("-inf")
        self._auto_meeting_id: int | None = None
        self.last_probe = Probe()

    # --- lifecycle --------------------------------------------------------

    def start(self) -> bool:
        """Start the polling thread. Returns False if the probe cannot run."""
        if not self.probe.available():
            log.info("Automatic meeting detection is off: %s",
                     self.probe.unavailable_reason)
            return False
        if self._thread is not None and self._thread.is_alive():
            return True
        self._stop.clear()
        self.detector.reset(self._clock())
        self._thread = threading.Thread(target=self._loop, name="teams-monitor",
                                        daemon=True)
        self._thread.start()
        log.info("Watching for Teams calls (start after %.0fs, stop after %.0fs)",
                 self.detector.start_delay, self.detector.stop_delay)
        return True

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=timeout)
        self._thread = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # --- polling ----------------------------------------------------------

    def _loop(self) -> None:
        while not self._stop.wait(self.poll_interval):
            try:
                self.tick()
            except Exception:  # noqa: BLE001 - the watcher must never die
                log.exception("Meeting detection tick failed")

    def tick(self) -> Action:
        """One poll cycle. Exposed separately so tests can drive it directly."""
        now = self._clock()

        if not self.service.auto_detect_enabled:
            if self._auto_meeting_id is not None:
                self._finish_recording(now)
            return Action.NONE

        probe = self.probe.probe()
        self.last_probe = probe

        # A recording the user started by hand is theirs to stop.
        if self.service.is_recording and self._auto_meeting_id is None:
            self.detector.externally_started(now)
            return Action.NONE
        if not self.service.is_recording and self._auto_meeting_id is not None:
            # Stopped from the tray or the dashboard while we were recording.
            self._auto_meeting_id = None
            self.detector.externally_stopped(now)
            return Action.NONE

        action = self.detector.update(probe.audio_active, now)

        if action is Action.START:
            if now - self._last_stop_at < RESTART_COOLDOWN_SECONDS:
                log.debug("Ignoring a call start inside the restart cooldown")
                self.detector.externally_stopped(now)
                return Action.NONE
            self._start_recording(now)
        elif action is Action.STOP:
            self._finish_recording(now)
        return action

    # --- actions ----------------------------------------------------------

    def _start_recording(self, now: float) -> None:
        try:
            meeting_id = self.service.start_recording(source="auto")
        except TeamsNotesError as exc:
            # Most likely the consent notice has not been acknowledged yet.
            log.warning("Automatic recording did not start: %s", exc)
            self.detector.externally_stopped(now)
            return
        self._auto_meeting_id = meeting_id
        log.info("Teams call detected; recording meeting %d", meeting_id)

    def _finish_recording(self, now: float) -> None:
        meeting_id = self._auto_meeting_id
        self._auto_meeting_id = None
        self._last_stop_at = now
        self.detector.externally_stopped(now)
        if meeting_id is None or not self.service.is_recording:
            return
        try:
            result = self.service.stop_recording(
                process=True, min_seconds=self.min_meeting_seconds)
        except TeamsNotesError as exc:
            log.warning("Automatic recording did not stop cleanly: %s", exc)
            return
        if result.get("discarded"):
            log.info("Call was too short to keep (%.0fs); discarded",
                     result.get("duration_seconds", 0.0))
        else:
            log.info("Teams call ended; processing meeting %d", meeting_id)

    def status(self) -> dict[str, object]:
        return {
            "running": self.running,
            "available": self.probe.available(),
            "unavailable_reason": self.probe.unavailable_reason,
            "state": self.detector.state.value,
            "teams_running": self.last_probe.teams_running,
            "audio_active": self.last_probe.audio_active,
            "auto_meeting_id": self._auto_meeting_id,
        }
