"""Reading and writing 16-bit PCM WAV files.

Writes go to a `.part` file that is renamed into place only once the header has
been finalised, so a crash mid-recording can never leave a half-written file
that later stages mistake for a complete one.
"""

from __future__ import annotations

import os
import wave
from pathlib import Path

import numpy as np

from app.capture.audio_utils import float32_to_pcm16, pcm_bytes_to_float32, to_mono
from app.errors import StorageError

SAMPLE_WIDTH = 2  # 16-bit
# Guard against a runaway recording filling the disk: 24h of 16 kHz mono PCM.
MAX_FRAMES = 16000 * 60 * 60 * 24


class WavWriter:
    """Incrementally write mono 16-bit PCM to a WAV file."""

    def __init__(self, path: Path, sample_rate: int) -> None:
        self.path = Path(path)
        self.sample_rate = int(sample_rate)
        self._temp_path = self.path.with_suffix(self.path.suffix + ".part")
        self._handle: wave.Wave_write | None = None
        self._frames_written = 0
        self._closed = False

    def __enter__(self) -> "WavWriter":
        self.open()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def open(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = wave.open(str(self._temp_path), "wb")
        handle.setnchannels(1)
        handle.setsampwidth(SAMPLE_WIDTH)
        handle.setframerate(self.sample_rate)
        self._handle = handle

    @property
    def frames_written(self) -> int:
        return self._frames_written

    @property
    def duration_seconds(self) -> float:
        return self._frames_written / self.sample_rate if self.sample_rate else 0.0

    def write(self, samples: np.ndarray) -> None:
        """Append float32 mono samples. Silently stops at MAX_FRAMES."""
        if self._handle is None:
            raise StorageError("WAV writer used before open().", "")
        if samples.size == 0 or self._frames_written >= MAX_FRAMES:
            return
        remaining = MAX_FRAMES - self._frames_written
        if samples.size > remaining:
            samples = samples[:remaining]
        self._handle.writeframes(float32_to_pcm16(samples))
        self._frames_written += int(samples.size)

    def close(self) -> Path | None:
        """Finalise the header and move the file into place.

        Returns the final path, or None if nothing was recorded.
        """
        if self._closed:
            return self.path if self.path.exists() else None
        self._closed = True

        if self._handle is not None:
            try:
                self._handle.close()
            finally:
                self._handle = None

        if self._frames_written == 0:
            self._temp_path.unlink(missing_ok=True)
            return None

        os.replace(self._temp_path, self.path)
        return self.path


def write_wav(path: Path, samples: np.ndarray, sample_rate: int) -> Path:
    """Write a whole float32 mono signal to a WAV file in one go."""
    with WavWriter(path, sample_rate) as writer:
        writer.write(np.asarray(samples, dtype=np.float32))
    written = writer.close()
    if written is None:
        raise StorageError(f"Refusing to write an empty WAV file to {path.name}.", "")
    return written


def read_wav_mono(path: Path) -> tuple[np.ndarray, int]:
    """Read any PCM WAV file as mono float32, returning (samples, sample_rate)."""
    path = Path(path)
    if not path.is_file():
        raise StorageError(
            f"Audio file not found: {path.name}",
            "The recording may have been moved or deleted.",
        )
    try:
        with wave.open(str(path), "rb") as handle:
            channels = handle.getnchannels()
            width = handle.getsampwidth()
            rate = handle.getframerate()
            raw = handle.readframes(handle.getnframes())
    except (wave.Error, EOFError) as exc:
        raise StorageError(
            f"{path.name} is not a readable WAV file ({exc or 'file is truncated'}).",
            "Re-record the meeting, or convert the file with ffmpeg first.",
        ) from exc

    samples = pcm_bytes_to_float32(raw, width)
    return to_mono(samples, channels), rate


def wav_duration_seconds(path: Path) -> float:
    """Duration of a WAV file without loading its samples."""
    try:
        with wave.open(str(path), "rb") as handle:
            rate = handle.getframerate()
            return handle.getnframes() / rate if rate else 0.0
    except (wave.Error, EOFError, OSError):
        return 0.0
