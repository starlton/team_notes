"""Transcript types, and the error paths of the two platform-bound wrappers."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from app.capture.devices import AudioHost, DeviceInfo, import_pyaudio
from app.errors import DependencyMissingError, TranscriptionError
from app.transcribe.models import (TranscriptResult, TranscriptSegment, Word,
                                   format_timestamp, render_transcript)
from app.transcribe.whisper import Transcriber, _segment_confidence


# --- transcript types -------------------------------------------------------

@pytest.mark.parametrize("seconds,expected", [
    (0, "0:00"), (9, "0:09"), (65, "1:05"), (600, "10:00"),
    (3600, "1:00:00"), (3725, "1:02:05"), (-5, "0:00"),
])
def test_timestamp_formatting(seconds, expected):
    assert format_timestamp(seconds) == expected


def test_segment_round_trips_through_a_dict():
    segment = TranscriptSegment(1.0, 3.5, "Hello", speaker="Speaker 1",
                                words=[Word(1.0, 1.5, "Hello", 0.9)],
                                confidence=0.8)
    restored = TranscriptSegment.from_dict(segment.to_dict())
    assert restored.text == "Hello"
    assert restored.speaker == "Speaker 1"
    assert restored.words[0].probability == 0.9
    assert restored.confidence == 0.8
    assert restored.duration == 2.5


def test_segment_from_a_sparse_dict_uses_defaults():
    segment = TranscriptSegment.from_dict({"text": "hi"})
    assert segment.start == 0.0 and segment.confidence is None and segment.words == []


def test_result_round_trips_and_lists_speakers_in_order():
    result = TranscriptResult(segments=[
        TranscriptSegment(0, 1, "a", speaker="Speaker 2"),
        TranscriptSegment(1, 2, "b", speaker="Speaker 1"),
        TranscriptSegment(2, 3, "c", speaker="Speaker 2"),
    ], duration_seconds=3.0, model="small.en")
    assert result.speakers() == ["Speaker 2", "Speaker 1"]
    assert result.text == "a\nb\nc"
    restored = TranscriptResult.from_dict(result.to_dict())
    assert restored.model == "small.en"
    assert len(restored.segments) == 3


def test_rendering_with_and_without_speakers():
    segments = [TranscriptSegment(0, 2, "Hello", speaker="Speaker 1"),
                TranscriptSegment(65, 70, "Bye", speaker="Speaker 2")]
    assert render_transcript(segments) == "[0:00] Speaker 1: Hello\n[1:05] Speaker 2: Bye"
    assert render_transcript(segments, with_speakers=False) == "[0:00] Hello\n[1:05] Bye"
    assert "Ada" in render_transcript(segments, speaker_names={"Speaker 1": "Ada"})


def test_rendering_skips_blank_lines_and_labels_unattributed_speech():
    rendered = render_transcript([TranscriptSegment(0, 1, "   "),
                                  TranscriptSegment(1, 2, "real")])
    assert rendered == "[0:01] Unknown: real"


# --- whisper wrapper --------------------------------------------------------

def test_transcribing_a_missing_file_explains_itself(tmp_path: Path):
    with pytest.raises(TranscriptionError) as excinfo:
        Transcriber().transcribe(tmp_path / "nope.wav")
    assert "not found" in excinfo.value.message.lower()


def test_segment_confidence_converts_log_probability():
    class Raw:
        avg_logprob = -0.2

    assert _segment_confidence(Raw()) == pytest.approx(0.8187, abs=1e-3)


def test_segment_confidence_is_none_when_whisper_gives_none():
    class Raw:
        avg_logprob = None

    assert _segment_confidence(Raw()) is None
    assert _segment_confidence(object()) is None


def test_segment_confidence_is_clamped():
    class Raw:
        avg_logprob = 5.0        # exp() would exceed 1.0

    assert _segment_confidence(Raw()) == 1.0


def test_loading_a_model_without_faster_whisper_says_what_to_install(monkeypatch):
    monkeypatch.setitem(sys.modules, "faster_whisper", None)
    with pytest.raises((DependencyMissingError, TranscriptionError)) as excinfo:
        Transcriber(model_size="definitely-not-a-model").load()
    assert "pip install" in str(excinfo.value) or "Whisper model" in str(excinfo.value)


# --- audio devices ----------------------------------------------------------

def test_device_info_describes_itself():
    loopback = DeviceInfo(1, "Speakers", 2, 48000, True)
    microphone = DeviceInfo(2, "Headset", 1, 44100, False)
    assert "loopback" in loopback.describe() and "48000Hz" in loopback.describe()
    assert "input" in microphone.describe()


@pytest.mark.skipif(sys.platform == "win32", reason="pyaudiowpatch exists on Windows")
def test_importing_pyaudio_off_windows_says_why(monkeypatch):
    with pytest.raises(DependencyMissingError) as excinfo:
        import_pyaudio()
    assert "Windows" in str(excinfo.value)


@pytest.mark.skipif(sys.platform == "win32", reason="pyaudiowpatch exists on Windows")
def test_creating_an_audio_host_off_windows_fails_cleanly():
    with pytest.raises(DependencyMissingError):
        AudioHost()


def test_a_closed_audio_host_refuses_to_hand_out_a_handle(monkeypatch):
    """Using the host outside its context manager must not crash obscurely."""
    from app.errors import AudioDeviceError

    monkeypatch.setattr("app.capture.devices.import_pyaudio", lambda: object())
    host = AudioHost()
    with pytest.raises(AudioDeviceError):
        _ = host.pa
    host.close()      # safe even though it was never opened
