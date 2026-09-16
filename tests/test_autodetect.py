"""Hands-free detection: the timing rules and the monitor that drives them."""

from __future__ import annotations

import pytest

from app.autodetect.state_machine import Action, MeetingDetector, State
from app.autodetect.teams_monitor import Probe, TeamsMonitor
from app.errors import CaptureError


# --- the state machine ------------------------------------------------------

def test_a_short_chime_never_starts_a_recording():
    """A notification sound makes Teams audio active for a second or two."""
    detector = MeetingDetector(start_delay=10, stop_delay=45)
    detector.reset(0)
    assert detector.update(True, 1) is Action.NONE
    assert detector.update(True, 4) is Action.NONE
    assert detector.update(False, 5) is Action.NONE
    assert detector.state is State.IDLE


def test_sustained_audio_starts_a_recording():
    detector = MeetingDetector(start_delay=10, stop_delay=45)
    detector.reset(0)
    detector.update(True, 1)
    assert detector.update(True, 5) is Action.NONE
    assert detector.update(True, 12) is Action.START
    assert detector.state is State.RECORDING


def test_a_start_only_fires_once():
    detector = MeetingDetector(start_delay=10, stop_delay=45)
    detector.reset(0)
    detector.update(True, 1)
    detector.update(True, 12)
    assert detector.update(True, 20) is Action.NONE


def test_a_brief_dropout_does_not_end_the_meeting():
    detector = MeetingDetector(start_delay=10, stop_delay=45)
    detector.reset(0)
    detector.update(True, 1)
    detector.update(True, 12)
    assert detector.update(False, 30) is Action.NONE      # someone muted
    assert detector.state is State.DISARMING
    assert detector.update(True, 40) is Action.NONE       # they spoke again
    assert detector.state is State.RECORDING


def test_sustained_silence_ends_the_meeting():
    detector = MeetingDetector(start_delay=10, stop_delay=45)
    detector.reset(0)
    detector.update(True, 1)
    detector.update(True, 12)
    detector.update(False, 100)
    assert detector.update(False, 120) is Action.NONE
    assert detector.update(False, 146) is Action.STOP
    assert detector.state is State.IDLE


def test_recording_seconds_tracks_the_current_meeting():
    detector = MeetingDetector(start_delay=10, stop_delay=45)
    detector.reset(0)
    detector.update(True, 1)
    detector.update(True, 12)
    assert detector.recording_seconds(72) == pytest.approx(60.0)
    assert MeetingDetector().recording_seconds(100) == 0.0


def test_a_manual_recording_is_adopted_not_restarted():
    detector = MeetingDetector()
    detector.externally_started(10)
    assert detector.state is State.RECORDING
    detector.externally_stopped(20)
    assert detector.state is State.IDLE


# --- the monitor ------------------------------------------------------------

class FakeProbe:
    def __init__(self, available: bool = True) -> None:
        self.audio_active = False
        self.teams_running = True
        self._available = available
        self.unavailable_reason = "" if available else "not Windows"

    def available(self) -> bool:
        return self._available

    def probe(self) -> Probe:
        return Probe(teams_running=self.teams_running,
                     audio_active=self.audio_active)


class FakeService:
    def __init__(self, settings) -> None:
        self.settings = settings
        self.auto_detect_enabled = True
        self.is_recording = False
        self.started: list[str] = []
        self.stopped: list[dict] = []
        self.start_error: Exception | None = None
        self._next_id = 1

    def start_recording(self, title: str = "", source: str = "manual",
                        participants_informed: bool = False) -> int:
        if self.start_error:
            raise self.start_error
        self.started.append(source)
        self.is_recording = True
        meeting_id = self._next_id
        self._next_id += 1
        return meeting_id

    def stop_recording(self, process: bool = True, min_seconds: float = 0.0) -> dict:
        self.is_recording = False
        duration = 300.0
        record = {"meeting_id": self._next_id - 1, "queued": process,
                  "duration_seconds": duration,
                  "discarded": bool(min_seconds and duration < min_seconds)}
        self.stopped.append(record)
        return record


@pytest.fixture
def monitor(settings):
    clock = {"now": 0.0}
    service = FakeService(settings)
    probe = FakeProbe()
    mon = TeamsMonitor(service, probe=probe, poll_interval=0.01,
                       clock=lambda: clock["now"])
    mon.detector.reset(0.0)
    return mon, service, probe, clock


def advance(monitor_tuple, seconds: float, audio: bool) -> Action:
    mon, _service, probe, clock = monitor_tuple
    clock["now"] += seconds
    probe.audio_active = audio
    return mon.tick()


def test_a_call_starts_and_stops_a_recording(monitor):
    mon, service, _probe, _clock = monitor
    advance(monitor, 2, True)
    assert advance(monitor, 12, True) is Action.START
    assert service.started == ["auto"]
    assert service.is_recording

    advance(monitor, 10, False)
    assert advance(monitor, 50, False) is Action.STOP
    assert service.stopped[0]["queued"] is True
    assert not service.is_recording


def test_a_ringtone_is_ignored(monitor):
    _mon, service, _probe, _clock = monitor
    advance(monitor, 2, True)
    advance(monitor, 3, False)
    advance(monitor, 3, True)
    assert service.started == []


def test_nothing_happens_when_auto_detect_is_off(monitor):
    _mon, service, _probe, _clock = monitor
    service.auto_detect_enabled = False
    advance(monitor, 2, True)
    advance(monitor, 30, True)
    assert service.started == []


def test_turning_auto_detect_off_mid_call_stops_the_recording(monitor):
    mon, service, _probe, _clock = monitor
    advance(monitor, 2, True)
    advance(monitor, 12, True)
    assert service.is_recording

    service.auto_detect_enabled = False
    advance(monitor, 2, True)
    assert not service.is_recording
    assert service.stopped


def test_a_manual_recording_is_left_alone(monitor):
    mon, service, _probe, _clock = monitor
    service.is_recording = True          # started from the tray
    advance(monitor, 2, True)
    advance(monitor, 60, False)
    assert service.stopped == []         # the monitor must not stop it
    assert mon.detector.state is State.RECORDING


def test_stopping_from_the_dashboard_mid_call_is_noticed(monitor):
    mon, service, _probe, _clock = monitor
    advance(monitor, 2, True)
    advance(monitor, 12, True)
    service.is_recording = False         # stopped elsewhere
    advance(monitor, 2, True)
    assert mon._auto_meeting_id is None
    assert mon.detector.state is State.IDLE


def test_a_short_call_is_discarded(monitor):
    mon, service, _probe, _clock = monitor
    mon.min_meeting_seconds = 600.0      # longer than the fake's 300s
    advance(monitor, 2, True)
    advance(monitor, 12, True)
    advance(monitor, 10, False)
    advance(monitor, 50, False)
    assert service.stopped[0]["discarded"] is True


def test_a_new_call_inside_the_cooldown_is_ignored(monitor):
    """The tail of a call must not immediately re-arm the detector."""
    mon, service, _probe, _clock = monitor
    advance(monitor, 2, True)
    advance(monitor, 12, True)
    advance(monitor, 10, False)
    advance(monitor, 50, False)
    assert len(service.started) == 1

    advance(monitor, 1, True)
    advance(monitor, 11, True)           # within RESTART_COOLDOWN_SECONDS
    assert len(service.started) == 1


def test_a_refused_start_does_not_wedge_the_detector(monitor):
    """If consent has not been acknowledged, recording is refused each time."""
    mon, service, _probe, _clock = monitor
    service.start_error = CaptureError("notice not acknowledged", "")
    advance(monitor, 2, True)
    advance(monitor, 12, True)
    assert service.started == []
    assert mon.detector.state is State.IDLE


def test_an_unavailable_probe_does_not_start_the_monitor(settings):
    service = FakeService(settings)
    mon = TeamsMonitor(service, probe=FakeProbe(available=False))
    assert mon.start() is False
    assert mon.running is False


def test_status_is_reportable(monitor):
    mon, _service, _probe, _clock = monitor
    advance(monitor, 2, True)
    status = mon.status()
    assert status["available"] is True
    assert status["state"] == "arming"
    assert status["audio_active"] is True
