"""Deciding when a Teams call has started and ended.

The signal available locally is "Teams currently has an active audio session".
It is noisy: an incoming-call ringtone, a notification chime, or Teams briefly
opening the device all look identical to a call starting, and audio can drop out
for a second mid-call without the call ending.

So the raw signal is never acted on directly. It has to hold steady for a while
in each direction before anything happens:

    idle --(audio active for START_DELAY)--> recording
    recording --(audio quiet for STOP_DELAY)--> idle

Keeping this as a pure state machine means all of that timing behaviour is
testable without Teams, Windows, or a sound card.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class State(str, Enum):
    IDLE = "idle"
    ARMING = "arming"          # audio started; waiting to be sure it is a call
    RECORDING = "recording"
    DISARMING = "disarming"    # audio stopped; waiting to be sure the call ended


class Action(str, Enum):
    NONE = "none"
    START = "start"
    STOP = "stop"


@dataclass
class MeetingDetector:
    """Turns a noisy "Teams audio is active" signal into start/stop decisions."""

    start_delay: float = 10.0
    stop_delay: float = 45.0
    state: State = State.IDLE
    _changed_at: float = 0.0
    _recording_since: float = 0.0

    def reset(self, now: float = 0.0) -> None:
        self.state = State.IDLE
        self._changed_at = now
        self._recording_since = 0.0

    def recording_seconds(self, now: float) -> float:
        if self.state not in (State.RECORDING, State.DISARMING):
            return 0.0
        return max(0.0, now - self._recording_since)

    def externally_started(self, now: float) -> None:
        """Tell the detector a recording began some other way (tray, dashboard)."""
        self.state = State.RECORDING
        self._changed_at = now
        self._recording_since = now

    def externally_stopped(self, now: float) -> None:
        """Tell the detector the recording it was tracking has ended."""
        self.state = State.IDLE
        self._changed_at = now
        self._recording_since = 0.0

    def update(self, audio_active: bool, now: float) -> Action:
        """Feed one observation, and get back what to do about it."""
        if self.state is State.IDLE:
            if audio_active:
                self.state = State.ARMING
                self._changed_at = now
            return Action.NONE

        if self.state is State.ARMING:
            if not audio_active:
                # A chime or a ringtone, not a call.
                self.state = State.IDLE
                self._changed_at = now
                return Action.NONE
            if now - self._changed_at >= self.start_delay:
                self.state = State.RECORDING
                self._changed_at = now
                self._recording_since = now
                return Action.START
            return Action.NONE

        if self.state is State.RECORDING:
            if not audio_active:
                self.state = State.DISARMING
                self._changed_at = now
            return Action.NONE

        # DISARMING
        if audio_active:
            # Just a gap in the call; carry on recording.
            self.state = State.RECORDING
            self._changed_at = now
            return Action.NONE
        if now - self._changed_at >= self.stop_delay:
            self.state = State.IDLE
            self._changed_at = now
            self._recording_since = 0.0
            return Action.STOP
        return Action.NONE
