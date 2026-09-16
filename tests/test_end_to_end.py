"""One test that walks the whole app, phase by phase.

Capture, transcription, diarization and the LLM are faked -- those are the four
places that need Windows, a 5GB model or a running Ollama -- but everything
between them is the real code: mixing, the merge logic, the database, the job
queue, the service, the detector and the HTTP API.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.autodetect.state_machine import Action
from app.autodetect.teams_monitor import TeamsMonitor
from app.capture.recorder import mix_existing_tracks
from app.capture.wav_io import read_wav_mono, write_wav
from app.db.repository import STATUS_COMPLETE, STATUS_RECORDED
from app.diarize.merge import SpeakerTurn
from app.pipeline.processor import MeetingProcessor
from app.transcribe.models import TranscriptSegment, Word
from app.web.security import CSRF_HEADER, CSRF_VALUE
from app.web.server import create_app
from tests.conftest import FakeDiarizer, FakeEngine, FakeTranscriber, tone

BASE_URL = "http://127.0.0.1:8765"
WRITE = {CSRF_HEADER: CSRF_VALUE}


def words(*items):
    return [Word(start, end, text) for start, end, text in items]


TRANSCRIPT = [
    TranscriptSegment(0.0, 4.0, "Shall we ship the release on Friday?",
                      words=words((0.0, 1.0, "Shall"), (1.0, 2.0, "we"),
                                  (2.0, 3.0, "ship"), (3.0, 4.0, "Friday?"))),
    TranscriptSegment(4.5, 9.0, "Yes, and I will send the notes.",
                      words=words((4.5, 5.5, "Yes,"), (5.5, 6.5, "I"),
                                  (6.5, 7.5, "will"), (7.5, 9.0, "send"))),
    TranscriptSegment(9.5, 13.0, "Great, that works for me.",
                      words=words((9.5, 10.5, "Great,"), (10.5, 11.5, "that"),
                                  (11.5, 13.0, "works"))),
]

TURNS = [
    SpeakerTurn(0.0, 4.2, "SPEAKER_01"),
    SpeakerTurn(4.2, 9.2, "SPEAKER_00"),
    SpeakerTurn(9.2, 13.0, "SPEAKER_01"),
]


def test_phase_1_two_tracks_mix_into_one_audible_wav(tmp_path: Path):
    """The Windows gotcha: loopback holds the others, the mic holds you."""
    others = write_wav(tmp_path / "loopback.wav", tone(3.0, 300.0, 48000), 48000)
    me = write_wav(tmp_path / "microphone.wav",
                   tone(3.0, 900.0, 44100, amplitude=0.05), 44100)

    mixed_path = mix_existing_tracks(others, me, tmp_path / "mixed.wav",
                                     sample_rate=16000, mic_offset=0.25)
    mixed, rate = read_wav_mono(mixed_path)

    assert rate == 16000
    assert len(mixed) == pytest.approx(16000 * 3.25, rel=0.02)

    # Both voices must be present in the result, not just the louder one.
    spectrum = np.abs(np.fft.rfft(mixed))
    freqs = np.fft.rfftfreq(len(mixed), 1 / rate)
    others_energy = spectrum[np.argmin(np.abs(freqs - 300))]
    my_energy = spectrum[np.argmin(np.abs(freqs - 900))]
    assert others_energy > 0
    assert my_energy > others_energy * 0.05
    assert float(np.max(np.abs(mixed))) <= 1.0


def test_phase_1_mixing_survives_a_missing_microphone(tmp_path: Path):
    others = write_wav(tmp_path / "loopback.wav", tone(1.0), 16000)
    mixed = mix_existing_tracks(others, None, tmp_path / "mixed.wav")
    assert read_wav_mono(mixed)[0].size > 0


def test_phases_2_to_6_end_to_end(service, tmp_path):
    settings = service.settings
    repo = service.repo

    # --- phase 1: a recording lands on disk -------------------------------
    meeting_id = repo.create_meeting("Weekly release sync",
                                     participants_informed=True)
    audio = write_wav(settings.audio_dir / f"meeting-{meeting_id:06d}" / "mixed.wav",
                      tone(13.0), 16000)
    repo.update_meeting(meeting_id, status=STATUS_RECORDED, audio_path=str(audio),
                        duration_seconds=13.0)

    # --- phases 2-4: the processing pass ----------------------------------
    processor = MeetingProcessor(settings, repo)
    processor.make_transcriber = lambda: FakeTranscriber(segments=list(TRANSCRIPT))
    processor.make_diarizer = lambda: FakeDiarizer(turns=list(TURNS))
    engine = FakeEngine()
    processor.make_engine = lambda: engine
    outcome = processor.process(meeting_id)

    assert not outcome.failed
    assert repo.get_meeting(meeting_id)["status"] == STATUS_COMPLETE

    # Phase 3: two speakers, and the person who spoke first is Speaker 1.
    transcript = repo.get_transcript(meeting_id)
    assert {segment.speaker for segment in transcript} == {"Speaker 1", "Speaker 2"}
    assert transcript[0].speaker == "Speaker 1"
    # Their two turns were merged back together by talk order, not split.
    assert len(transcript) == 3

    # Phase 4: notes exist and are attributed.
    assert repo.get_summary(meeting_id)["summary"]
    assert repo.get_action_items(meeting_id)
    assert repo.get_priorities(meeting_id)
    assert repo.get_drafts(meeting_id)

    # --- phase 5: the dashboard -------------------------------------------
    app = create_app(service)
    with TestClient(app, base_url=BASE_URL) as client:
        client.headers.update({"X-Auth-Token": app.state.dashboard_token})

        listing = client.get("/api/meetings").json()
        assert listing["meetings"][0]["participants_informed"] is True

        detail = client.get(f"/api/meetings/{meeting_id}").json()
        assert len(detail["transcript"]) == 3
        assert len(detail["speakers"]) == 2

        # The rename step, then regenerating notes with the real names.
        renamed = client.post(f"/api/meetings/{meeting_id}/speakers",
                              json={"names": {"Speaker 1": "Ada", "Speaker 2": "Grace"}},
                              headers=WRITE)
        assert renamed.json()["updated"] == 2

        after = client.get(f"/api/meetings/{meeting_id}").json()
        assert [line["speaker"] for line in after["transcript"]] == \
            ["Ada", "Grace", "Ada"]

        text = client.get(f"/api/meetings/{meeting_id}/transcript.txt").text
        assert "Ada: Shall we ship the release on Friday?" in text
        assert client.get(f"/api/meetings/{meeting_id}/audio").status_code == 200

    # Regenerating must hand the LLM the real names, and keep the mapping.
    processor.regenerate_notes(meeting_id)
    assert {segment.speaker for segment in engine.calls[-1][0]} == {"Ada", "Grace"}
    assert repo.speaker_name_map(meeting_id) == {"Speaker 1": "Ada",
                                                 "Speaker 2": "Grace"}

    # --- phase 6: hands-free ----------------------------------------------
    clock = {"now": 1000.0}
    started: list[str] = []
    stopped: list[bool] = []

    class Harness:
        settings = service.settings
        auto_detect_enabled = True
        is_recording = False

        def start_recording(self, title="", source="manual",
                            participants_informed=False):
            started.append(source)
            self.is_recording = True
            return 99

        def stop_recording(self, process=True, min_seconds=0.0):
            self.is_recording = False
            stopped.append(process)
            return {"meeting_id": 99, "discarded": False, "duration_seconds": 600.0}

    class Probe:
        active = False
        unavailable_reason = ""

        def available(self):
            return True

        def probe(self):
            from app.autodetect.teams_monitor import Probe as Reading

            return Reading(teams_running=True, audio_active=self.active)

    probe = Probe()
    monitor = TeamsMonitor(Harness(), probe=probe, clock=lambda: clock["now"])
    monitor.detector.reset(clock["now"])

    def tick(seconds, audio):
        clock["now"] += seconds
        probe.active = audio
        return monitor.tick()

    tick(2, True)
    assert tick(12, True) is Action.START       # the call is up
    tick(300, True)                             # the meeting runs
    tick(5, False)                              # everyone stops talking
    assert tick(50, False) is Action.STOP       # the call ended

    assert started == ["auto"]
    assert stopped == [True]                    # processing was queued
