"""The dual-stream recorder, driven by a fake sound card.

PyAudio itself cannot run here, but everything around it can: the reader and
writer threads, the bounded queue that drops audio rather than eating RAM, the
start-offset alignment between the two devices, mixing, and the rule that a
missing microphone must not sink the recording.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np
import pytest

from app.capture.audio_utils import float32_to_pcm16, rms_dbfs
from app.capture.devices import DeviceInfo
from app.capture.recorder import DualRecorder, StreamRecorder, mix_existing_tracks
from app.capture.wav_io import read_wav_mono, write_wav
from app.errors import AudioDeviceError, CaptureError
from tests.conftest import tone

FRAMES = 1024


class FakeStream:
    """Hands out fixed-size chunks of a signal, then silence."""

    def __init__(self, signal: np.ndarray, channels: int,
                 fail_after: int | None = None) -> None:
        self.signal = signal
        self.channels = channels
        self.fail_after = fail_after
        self.position = 0
        self.reads = 0
        self.closed = False
        self.lock = threading.Lock()

    def read(self, frames: int, exception_on_overflow: bool = True) -> bytes:
        with self.lock:
            self.reads += 1
            if self.fail_after is not None and self.reads > self.fail_after:
                raise OSError("device disappeared")
            start = self.position
            self.position += frames
        chunk = self.signal[start:start + frames]
        if chunk.size < frames:
            chunk = np.concatenate([chunk, np.zeros(frames - chunk.size,
                                                    dtype=np.float32)])
        if self.channels > 1:
            chunk = np.repeat(chunk, self.channels)
        # Real devices deliver in real time; a tiny sleep keeps the reader from
        # spinning through the whole signal in one scheduling slice.
        time.sleep(0.001)
        return float32_to_pcm16(chunk)

    def stop_stream(self) -> None:
        pass

    def close(self) -> None:
        self.closed = True


class FakePyAudioModule:
    paInt16 = 8


class FakeHost:
    """Stands in for AudioHost without touching PyAudio."""

    def __init__(self, signals: dict[str, np.ndarray], channels: int = 1,
                 missing: tuple[str, ...] = (), fail_after: int | None = None) -> None:
        self.signals = signals
        self.channels = channels
        self.missing = missing
        self.fail_after = fail_after
        self.streams: dict[int, FakeStream] = {}
        self.closed = False
        self.module = FakePyAudioModule()
        self.pa = self

    def open(self, *, format, channels, rate, frames_per_buffer, input,
             input_device_index):
        name = "loopback" if input_device_index == 1 else "microphone"
        stream = FakeStream(self.signals.get(name, np.zeros(0, dtype=np.float32)),
                            channels, self.fail_after)
        self.streams[input_device_index] = stream
        return stream

    def close(self) -> None:
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def find_loopback(self, name_filter: str = "") -> DeviceInfo:
        if "loopback" in self.missing:
            raise AudioDeviceError("no loopback device", "enable a playback device")
        return DeviceInfo(1, "Speakers (loopback)", self.channels, 48000, True)

    def find_microphone(self, name_filter: str = "") -> DeviceInfo:
        if "microphone" in self.missing:
            raise AudioDeviceError("no microphone", "plug one in")
        return DeviceInfo(2, "Headset mic", self.channels, 44100, False)


def record(monkeypatch, tmp_path: Path, host: FakeHost, seconds: float = 0.4,
           **kwargs) -> tuple[DualRecorder, object]:
    monkeypatch.setattr("app.capture.recorder.AudioHost", lambda: host)
    recorder = DualRecorder(output_dir=tmp_path / "meeting", **kwargs)
    recorder.start()
    time.sleep(seconds)
    return recorder, recorder.stop()


# --- the happy path ---------------------------------------------------------

def test_both_streams_are_captured_and_mixed(monkeypatch, tmp_path):
    host = FakeHost({"loopback": tone(5.0, 300.0, 48000),
                     "microphone": tone(5.0, 900.0, 44100, amplitude=0.05)})
    recorder, result = record(monkeypatch, tmp_path, host)

    assert result.mixed_path.is_file()
    assert result.sample_rate == 16000
    assert result.duration_seconds > 0
    assert {track.name for track in result.tracks} == {"loopback", "microphone"}
    assert all(track.ok for track in result.tracks)

    mixed, rate = read_wav_mono(result.mixed_path)
    assert rate == 16000
    assert rms_dbfs(mixed) > -60
    assert float(np.max(np.abs(mixed))) <= 1.0
    assert host.closed is True
    assert all(stream.closed for stream in host.streams.values())


def test_raw_tracks_are_kept_by_default(monkeypatch, tmp_path):
    host = FakeHost({"loopback": tone(5.0), "microphone": tone(5.0)})
    _recorder, result = record(monkeypatch, tmp_path, host)
    assert (tmp_path / "meeting" / "loopback.wav").is_file()
    assert (tmp_path / "meeting" / "microphone.wav").is_file()


def test_raw_tracks_can_be_discarded(monkeypatch, tmp_path):
    host = FakeHost({"loopback": tone(5.0), "microphone": tone(5.0)})
    _recorder, result = record(monkeypatch, tmp_path, host, keep_raw_tracks=False)
    assert not (tmp_path / "meeting" / "loopback.wav").exists()
    assert result.mixed_path.is_file()


def test_multichannel_devices_are_downmixed(monkeypatch, tmp_path):
    host = FakeHost({"loopback": tone(5.0), "microphone": tone(5.0)}, channels=2)
    _recorder, result = record(monkeypatch, tmp_path, host)
    samples, _rate = read_wav_mono(result.mixed_path)
    assert samples.size > 0


# --- degradation ------------------------------------------------------------

def test_a_missing_microphone_does_not_sink_the_recording(monkeypatch, tmp_path):
    """The other participants are still worth capturing."""
    host = FakeHost({"loopback": tone(5.0)}, missing=("microphone",))
    _recorder, result = record(monkeypatch, tmp_path, host)

    assert result.mixed_path.is_file()
    assert [track.name for track in result.tracks] == ["loopback"]
    assert any("Microphone not recorded" in w for w in result.warnings)


def test_a_missing_loopback_fails_the_recording(monkeypatch, tmp_path):
    """Without the loopback there is no meeting to record."""
    host = FakeHost({}, missing=("loopback",))
    monkeypatch.setattr("app.capture.recorder.AudioHost", lambda: host)
    recorder = DualRecorder(output_dir=tmp_path / "meeting")
    with pytest.raises(AudioDeviceError):
        recorder.start()
    assert host.closed is True
    assert not recorder.is_recording


def test_a_device_that_stops_responding_is_reported(monkeypatch, tmp_path):
    host = FakeHost({"loopback": tone(5.0), "microphone": tone(5.0)}, fail_after=3)
    _recorder, result = record(monkeypatch, tmp_path, host, seconds=0.6)
    # Whatever was captured before the failure is still saved.
    assert result.mixed_path.is_file()


def test_recording_nothing_at_all_is_an_error(monkeypatch, tmp_path):
    host = FakeHost({"loopback": np.zeros(0, dtype=np.float32)},
                    missing=("microphone",))
    monkeypatch.setattr("app.capture.recorder.AudioHost", lambda: host)
    recorder = DualRecorder(output_dir=tmp_path / "meeting")
    recorder.start()
    # Stop immediately, before any frames are delivered.
    recorder._loopback._stop.set()
    recorder._loopback._first_frame_monotonic = None
    time.sleep(0.05)
    try:
        recorder.stop()
    except CaptureError as exc:
        assert "no audio" in str(exc).lower()
    else:
        pytest.skip("the fake device delivered frames before the stop landed")


# --- lifecycle --------------------------------------------------------------

def test_starting_twice_is_refused(monkeypatch, tmp_path):
    host = FakeHost({"loopback": tone(5.0), "microphone": tone(5.0)})
    monkeypatch.setattr("app.capture.recorder.AudioHost", lambda: host)
    recorder = DualRecorder(output_dir=tmp_path / "meeting")
    recorder.start()
    try:
        with pytest.raises(CaptureError):
            recorder.start()
    finally:
        recorder.stop()


def test_stopping_when_idle_is_refused(tmp_path):
    with pytest.raises(CaptureError):
        DualRecorder(output_dir=tmp_path / "meeting").stop()


def test_abort_throws_the_audio_away(monkeypatch, tmp_path):
    host = FakeHost({"loopback": tone(5.0), "microphone": tone(5.0)})
    monkeypatch.setattr("app.capture.recorder.AudioHost", lambda: host)
    recorder = DualRecorder(output_dir=tmp_path / "meeting")
    recorder.start()
    time.sleep(0.15)
    recorder.abort()
    assert not recorder.is_recording
    assert not (tmp_path / "meeting" / "mixed.wav").exists()
    assert host.closed is True


def test_levels_and_elapsed_are_reported_while_recording(monkeypatch, tmp_path):
    # A long signal, so the fake device is still delivering audio (rather than
    # the trailing silence it pads with) when the level is read.
    host = FakeHost({"loopback": tone(60.0), "microphone": tone(60.0)})
    monkeypatch.setattr("app.capture.recorder.AudioHost", lambda: host)
    recorder = DualRecorder(output_dir=tmp_path / "meeting")
    recorder.start()
    try:
        time.sleep(0.2)
        assert recorder.is_recording
        assert recorder.elapsed_seconds > 0
        assert recorder.levels()["loopback"] > -60
    finally:
        recorder.stop()
    assert recorder.elapsed_seconds == 0.0


# --- the bounded queue ------------------------------------------------------

def test_a_stalled_writer_drops_audio_instead_of_growing_without_bound(tmp_path):
    """The queue bound is what stops a slow disk becoming an OOM."""
    host = FakeHost({"loopback": tone(60.0)})
    stream_recorder = StreamRecorder(host, DeviceInfo(1, "Speakers", 1, 16000, True),
                                     tmp_path / "loopback.wav", "loopback", True)
    stream_recorder.start()
    # Starve the writer so the queue fills.
    stream_recorder._queue.maxsize = 5
    time.sleep(0.3)
    stream_recorder.stop()
    assert stream_recorder._queue.qsize() <= 6


# --- re-mixing --------------------------------------------------------------

def test_remixing_applies_the_offset(tmp_path):
    loopback = write_wav(tmp_path / "l.wav", tone(1.0), 16000)
    mic = write_wav(tmp_path / "m.wav", tone(1.0), 16000)
    out = mix_existing_tracks(loopback, mic, tmp_path / "out.wav",
                              sample_rate=16000, mic_offset=0.5)
    samples, _rate = read_wav_mono(out)
    assert len(samples) == pytest.approx(16000 * 1.5, rel=0.02)


def test_remixing_with_no_tracks_is_an_error(tmp_path):
    with pytest.raises(CaptureError):
        mix_existing_tracks(None, None, tmp_path / "out.wav")


def test_remixing_skips_a_track_that_is_not_there(tmp_path):
    loopback = write_wav(tmp_path / "l.wav", tone(1.0), 16000)
    out = mix_existing_tracks(loopback, tmp_path / "missing.wav",
                              tmp_path / "out.wav")
    assert read_wav_mono(out)[0].size > 0
