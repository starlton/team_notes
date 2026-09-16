"""Recording the meeting: loopback + microphone, mixed into one WAV.

The Windows gotcha this is built around: WASAPI loopback captures what Windows
plays to the speakers, which is everyone else on the call. It does *not*
capture the local microphone, because your own voice is sent out rather than
played back to you. So two independent streams are recorded and mixed.

Each device gets a reader thread that does nothing but pull frames and hand
them to a bounded queue, plus a writer thread that converts and writes to disk.
Splitting them keeps slow disk I/O out of the audio callback path, and the
bound means a stalled disk drops audio instead of exhausting RAM.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from app.capture.audio_utils import Track, mix_tracks, pcm_bytes_to_float32, rms_dbfs, to_mono
from app.capture.devices import AudioHost, DeviceInfo
from app.capture.wav_io import WavWriter, read_wav_mono, write_wav
from app.errors import AudioDeviceError, CaptureError
from app.logging_setup import get_logger

log = get_logger(__name__)

FRAMES_PER_BUFFER = 1024
# ~30 seconds of 48 kHz stereo buffers. Enough to ride out a slow disk, small
# enough that a wedged writer cannot eat the machine's memory.
QUEUE_MAX_CHUNKS = 1400


@dataclass
class TrackResult:
    """One finished stream."""

    name: str
    path: Path | None
    sample_rate: int
    duration_seconds: float
    start_offset_seconds: float
    dropped_chunks: int = 0
    error: str = ""

    @property
    def ok(self) -> bool:
        return self.path is not None and self.duration_seconds > 0.0


@dataclass
class RecordingResult:
    """Everything a finished recording produced."""

    mixed_path: Path
    duration_seconds: float
    sample_rate: int
    tracks: list[TrackResult] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def track(self, name: str) -> TrackResult | None:
        return next((t for t in self.tracks if t.name == name), None)


class StreamRecorder:
    """Records a single device to its own mono WAV file."""

    def __init__(self, host: AudioHost, device: DeviceInfo, output_path: Path,
                 name: str, loopback: bool) -> None:
        self._host = host
        self._device = device
        self._output_path = output_path
        self.name = name
        self._loopback = loopback

        self._stop = threading.Event()
        self._reader: threading.Thread | None = None
        self._writer_thread: threading.Thread | None = None
        self._queue: queue.Queue[bytes | None] = queue.Queue(maxsize=QUEUE_MAX_CHUNKS)
        self._stream: Any = None
        self._wav: WavWriter | None = None

        self._first_frame_monotonic: float | None = None
        self._dropped = 0
        self._error: str = ""
        self._level_dbfs: float = float("-inf")
        self._sample_width = 2

    # --- lifecycle --------------------------------------------------------

    def start(self) -> None:
        pa = self._host.pa
        module = self._host.module
        try:
            self._stream = pa.open(
                format=module.paInt16,
                channels=self._device.channels,
                rate=self._device.sample_rate,
                frames_per_buffer=FRAMES_PER_BUFFER,
                input=True,
                input_device_index=self._device.index,
            )
        except Exception as exc:
            raise AudioDeviceError(
                f"Could not open {'loopback' if self._loopback else 'microphone'} "
                f"device {self._device.name!r}: {exc}",
                "Another app may have exclusive access to it. Close it and retry.",
            ) from exc

        self._wav = WavWriter(self._output_path, self._device.sample_rate)
        self._wav.open()

        self._reader = threading.Thread(target=self._read_loop,
                                        name=f"capture-read-{self.name}", daemon=True)
        self._writer_thread = threading.Thread(target=self._write_loop,
                                               name=f"capture-write-{self.name}",
                                               daemon=True)
        self._reader.start()
        self._writer_thread.start()
        log.info("Recording %s from %s", self.name, self._device.describe())

    def stop(self, timeout: float = 10.0) -> TrackResult:
        self._stop.set()
        if self._reader is not None:
            self._reader.join(timeout=timeout)
        # A sentinel wakes the writer even if the queue is empty.
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        if self._writer_thread is not None:
            self._writer_thread.join(timeout=timeout)

        if self._stream is not None:
            try:
                self._stream.stop_stream()
                self._stream.close()
            except Exception:  # pragma: no cover - driver teardown
                log.debug("Closing %s stream raised", self.name, exc_info=True)
            self._stream = None

        path = self._wav.close() if self._wav is not None else None
        duration = self._wav.duration_seconds if self._wav is not None else 0.0
        return TrackResult(
            name=self.name,
            path=path,
            sample_rate=self._device.sample_rate,
            duration_seconds=duration,
            start_offset_seconds=0.0,  # filled in by the caller once both are known
            dropped_chunks=self._dropped,
            error=self._error,
        )

    # --- observation ------------------------------------------------------

    @property
    def first_frame_monotonic(self) -> float | None:
        return self._first_frame_monotonic

    @property
    def level_dbfs(self) -> float:
        return self._level_dbfs

    @property
    def error(self) -> str:
        return self._error

    # --- threads ----------------------------------------------------------

    def _read_loop(self) -> None:
        consecutive_failures = 0
        while not self._stop.is_set():
            try:
                data = self._stream.read(FRAMES_PER_BUFFER, exception_on_overflow=False)
                consecutive_failures = 0
            except Exception as exc:  # pragma: no cover - driver specific
                consecutive_failures += 1
                if consecutive_failures >= 10:
                    self._error = f"audio device stopped responding: {exc}"
                    log.error("%s: %s", self.name, self._error)
                    break
                time.sleep(0.05)
                continue

            if self._first_frame_monotonic is None:
                self._first_frame_monotonic = time.monotonic()

            try:
                self._queue.put_nowait(data)
            except queue.Full:
                # Drop the newest chunk rather than block the device read.
                self._dropped += 1
                if self._dropped in (1, 10, 100) or self._dropped % 500 == 0:
                    log.warning("%s: dropped %d audio chunks (disk too slow?)",
                                self.name, self._dropped)

    def _write_loop(self) -> None:
        while True:
            try:
                data = self._queue.get(timeout=0.25)
            except queue.Empty:
                if self._stop.is_set() and (self._reader is None
                                            or not self._reader.is_alive()):
                    return
                continue

            if data is None:
                return
            try:
                samples = to_mono(
                    pcm_bytes_to_float32(data, self._sample_width),
                    self._device.channels,
                )
                self._level_dbfs = rms_dbfs(samples)
                if self._wav is not None:
                    self._wav.write(samples)
            except Exception as exc:  # pragma: no cover - disk full etc.
                self._error = f"could not write audio: {exc}"
                log.error("%s: %s", self.name, self._error)
                return


class DualRecorder:
    """Records loopback + microphone together and mixes them on stop."""

    MIXED_FILENAME = "mixed.wav"
    LOOPBACK_FILENAME = "loopback.wav"
    MIC_FILENAME = "microphone.wav"

    def __init__(self, output_dir: Path, *, target_sample_rate: int = 16000,
                 mic_gain: float = 1.0, loopback_gain: float = 1.0,
                 mic_device_name: str = "", loopback_device_name: str = "",
                 keep_raw_tracks: bool = True) -> None:
        self.output_dir = Path(output_dir)
        self.target_sample_rate = int(target_sample_rate)
        self.mic_gain = float(mic_gain)
        self.loopback_gain = float(loopback_gain)
        self.mic_device_name = mic_device_name
        self.loopback_device_name = loopback_device_name
        self.keep_raw_tracks = keep_raw_tracks

        self._host: AudioHost | None = None
        self._loopback: StreamRecorder | None = None
        self._mic: StreamRecorder | None = None
        self._started_monotonic: float | None = None
        self._warnings: list[str] = []
        self._lock = threading.Lock()
        self._running = False

    # --- lifecycle --------------------------------------------------------

    @property
    def is_recording(self) -> bool:
        return self._running

    @property
    def elapsed_seconds(self) -> float:
        if self._started_monotonic is None:
            return 0.0
        return max(0.0, time.monotonic() - self._started_monotonic)

    def levels(self) -> dict[str, float]:
        """Current RMS level per track, for the dashboard's meter."""
        return {
            "loopback": self._loopback.level_dbfs if self._loopback else float("-inf"),
            "microphone": self._mic.level_dbfs if self._mic else float("-inf"),
        }

    def start(self) -> None:
        with self._lock:
            if self._running:
                raise CaptureError("A recording is already running.", "")

            self.output_dir.mkdir(parents=True, exist_ok=True)
            self._warnings = []
            host = AudioHost()
            host.__enter__()
            self._host = host

            try:
                loopback_device = host.find_loopback(self.loopback_device_name)
                self._loopback = StreamRecorder(
                    host, loopback_device,
                    self.output_dir / self.LOOPBACK_FILENAME,
                    name="loopback", loopback=True,
                )
                self._loopback.start()
            except Exception:
                self._teardown_host()
                raise

            # A missing microphone should not sink the recording: the other
            # participants are still worth capturing.
            try:
                mic_device = host.find_microphone(self.mic_device_name)
                self._mic = StreamRecorder(
                    host, mic_device,
                    self.output_dir / self.MIC_FILENAME,
                    name="microphone", loopback=False,
                )
                self._mic.start()
            except Exception as exc:
                self._mic = None
                message = f"Microphone not recorded: {exc}"
                self._warnings.append(message)
                log.warning(message)

            self._started_monotonic = time.monotonic()
            self._running = True

    def stop(self) -> RecordingResult:
        with self._lock:
            if not self._running:
                raise CaptureError("No recording is running.", "")
            self._running = False

            loopback_start = self._loopback.first_frame_monotonic if self._loopback else None
            mic_start = self._mic.first_frame_monotonic if self._mic else None

            tracks: list[TrackResult] = []
            if self._loopback is not None:
                tracks.append(self._loopback.stop())
            if self._mic is not None:
                tracks.append(self._mic.stop())
            self._teardown_host()

            # Align the two streams to whichever actually delivered audio first.
            starts = [t for t in (loopback_start, mic_start) if t is not None]
            origin = min(starts) if starts else 0.0
            offsets = {
                "loopback": (loopback_start - origin) if loopback_start else 0.0,
                "microphone": (mic_start - origin) if mic_start else 0.0,
            }
            for track in tracks:
                track.start_offset_seconds = offsets.get(track.name, 0.0)
                if track.error:
                    self._warnings.append(f"{track.name}: {track.error}")
                if track.dropped_chunks:
                    self._warnings.append(
                        f"{track.name}: {track.dropped_chunks} audio chunks were dropped"
                    )

            result = self._mix(tracks)
            self._loopback = None
            self._mic = None
            self._started_monotonic = None
            return result

    def _teardown_host(self) -> None:
        if self._host is not None:
            self._host.close()
            self._host = None

    # --- mixing -----------------------------------------------------------

    def _mix(self, tracks: list[TrackResult]) -> RecordingResult:
        gains = {"loopback": self.loopback_gain, "microphone": self.mic_gain}
        to_mix: list[Track] = []

        for track in tracks:
            if not track.ok or track.path is None:
                continue
            try:
                samples, rate = read_wav_mono(track.path)
            except Exception as exc:
                self._warnings.append(f"Could not read {track.name} track: {exc}")
                log.warning("Could not read %s track", track.name, exc_info=True)
                continue
            to_mix.append(Track(
                samples=samples,
                sample_rate=rate,
                gain=gains.get(track.name, 1.0),
                start_offset_seconds=track.start_offset_seconds,
                name=track.name,
            ))

        if not to_mix:
            raise CaptureError(
                "The recording produced no audio.",
                "Check that Windows is playing meeting audio through the device the "
                "app is capturing, and that the app has microphone permission.",
            )

        mixed = mix_tracks(to_mix, self.target_sample_rate)
        mixed_path = write_wav(self.output_dir / self.MIXED_FILENAME, mixed,
                               self.target_sample_rate)

        if not self.keep_raw_tracks:
            for track in tracks:
                if track.path is not None:
                    try:
                        track.path.unlink(missing_ok=True)
                        track.path = None
                    except OSError:  # pragma: no cover - file locked
                        log.debug("Could not remove raw track %s", track.name)

        duration = len(mixed) / self.target_sample_rate if self.target_sample_rate else 0.0
        log.info("Recorded %.1fs of audio to %s", duration, mixed_path)
        return RecordingResult(
            mixed_path=mixed_path,
            duration_seconds=duration,
            sample_rate=self.target_sample_rate,
            tracks=tracks,
            warnings=list(self._warnings),
        )

    def abort(self) -> None:
        """Stop without mixing. Used when a recording is being discarded."""
        with self._lock:
            self._running = False
            for recorder in (self._loopback, self._mic):
                if recorder is not None:
                    try:
                        recorder.stop()
                    except Exception:  # pragma: no cover
                        log.debug("Abort teardown raised", exc_info=True)
            self._teardown_host()
            self._loopback = None
            self._mic = None
            self._started_monotonic = None


def mix_existing_tracks(loopback_path: Path | None, mic_path: Path | None,
                        output_path: Path, sample_rate: int = 16000,
                        loopback_gain: float = 1.0, mic_gain: float = 1.0,
                        loopback_offset: float = 0.0,
                        mic_offset: float = 0.0) -> Path:
    """Re-mix two already-recorded tracks. Useful for retries and for tests."""
    tracks: list[Track] = []
    for path, gain, offset, name in (
        (loopback_path, loopback_gain, loopback_offset, "loopback"),
        (mic_path, mic_gain, mic_offset, "microphone"),
    ):
        if path is None or not Path(path).is_file():
            continue
        samples, rate = read_wav_mono(Path(path))
        tracks.append(Track(samples=samples, sample_rate=rate, gain=gain,
                            start_offset_seconds=offset, name=name))
    if not tracks:
        raise CaptureError("No track files to mix.", "")
    mixed = mix_tracks(tracks, sample_rate)
    return write_wav(Path(output_path), mixed, sample_rate)
