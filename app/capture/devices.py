"""Discovering WASAPI loopback and microphone devices on Windows.

`pyaudiowpatch` is a fork of PyAudio that exposes WASAPI loopback, which plain
PyAudio does not. It is Windows-only, so every import is deferred and produces a
readable message elsewhere.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any, Iterator

from app.errors import AudioDeviceError, DependencyMissingError
from app.logging_setup import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class DeviceInfo:
    """A sound device we can open for input."""

    index: int
    name: str
    channels: int
    sample_rate: int
    is_loopback: bool

    def describe(self) -> str:
        kind = "loopback" if self.is_loopback else "input"
        return f"[{self.index}] {self.name} ({kind}, {self.channels}ch, {self.sample_rate}Hz)"


def import_pyaudio() -> Any:
    """Import pyaudiowpatch, with a message that says what to install."""
    try:
        import pyaudiowpatch  # type: ignore[import-not-found]
    except ImportError as exc:
        if sys.platform != "win32":
            raise DependencyMissingError(
                "Audio capture needs WASAPI loopback, which only exists on Windows.",
                "Run the capture side of this app on Windows 10/11.",
            ) from exc
        raise DependencyMissingError(
            "The 'pyaudiowpatch' package is not installed.",
            "Activate the venv and run: pip install pyaudiowpatch",
        ) from exc
    return pyaudiowpatch


class AudioHost:
    """Owns the PyAudio instance. Always use as a context manager."""

    def __init__(self) -> None:
        self._pyaudio_module = import_pyaudio()
        self._pa: Any | None = None

    def __enter__(self) -> "AudioHost":
        self._pa = self._pyaudio_module.PyAudio()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        if self._pa is not None:
            try:
                self._pa.terminate()
            except Exception:  # pragma: no cover - driver teardown is best effort
                log.debug("PyAudio terminate raised during shutdown", exc_info=True)
            self._pa = None

    @property
    def pa(self) -> Any:
        if self._pa is None:
            raise AudioDeviceError("Audio host is not open.", "")
        return self._pa

    @property
    def module(self) -> Any:
        return self._pyaudio_module

    # --- discovery --------------------------------------------------------

    def iter_devices(self) -> Iterator[DeviceInfo]:
        for index in range(self.pa.get_device_count()):
            try:
                raw = self.pa.get_device_info_by_index(index)
            except Exception:  # pragma: no cover - flaky drivers
                continue
            if int(raw.get("maxInputChannels", 0)) < 1:
                continue
            yield DeviceInfo(
                index=int(raw["index"]),
                name=str(raw.get("name", "unknown")),
                channels=int(raw.get("maxInputChannels", 1)),
                sample_rate=int(raw.get("defaultSampleRate", 48000) or 48000),
                is_loopback=bool(raw.get("isLoopbackDevice", False)),
            )

    def list_devices(self) -> list[DeviceInfo]:
        return list(self.iter_devices())

    def find_loopback(self, name_filter: str = "") -> DeviceInfo:
        """Find the WASAPI loopback device that mirrors the default speakers."""
        candidates = [dev for dev in self.iter_devices() if dev.is_loopback]
        if not candidates:
            raise AudioDeviceError(
                "No WASAPI loopback device was found.",
                "Check that a playback device is enabled in Windows Sound settings, "
                "then restart the app.",
            )

        if name_filter:
            wanted = name_filter.casefold()
            matched = [dev for dev in candidates if wanted in dev.name.casefold()]
            if not matched:
                available = ", ".join(dev.name for dev in candidates)
                raise AudioDeviceError(
                    f"No loopback device matches LOOPBACK_DEVICE_NAME={name_filter!r}.",
                    f"Available loopback devices: {available}",
                )
            return matched[0]

        # Prefer the loopback that belongs to the current default speakers.
        try:
            default_speakers = self.pa.get_default_output_device_info()
            default_name = str(default_speakers.get("name", "")).casefold()
            for dev in candidates:
                if default_name and default_name in dev.name.casefold():
                    return dev
        except Exception:  # pragma: no cover - no default output configured
            log.debug("Could not read the default output device", exc_info=True)

        return candidates[0]

    def find_microphone(self, name_filter: str = "") -> DeviceInfo:
        """Find the microphone to record the local user with."""
        candidates = [dev for dev in self.iter_devices() if not dev.is_loopback]
        if not candidates:
            raise AudioDeviceError(
                "No microphone input device was found.",
                "Plug in or enable a microphone in Windows Sound settings.",
            )

        if name_filter:
            wanted = name_filter.casefold()
            matched = [dev for dev in candidates if wanted in dev.name.casefold()]
            if not matched:
                available = ", ".join(dev.name for dev in candidates)
                raise AudioDeviceError(
                    f"No microphone matches MIC_DEVICE_NAME={name_filter!r}.",
                    f"Available input devices: {available}",
                )
            return matched[0]

        try:
            raw = self.pa.get_default_input_device_info()
            return DeviceInfo(
                index=int(raw["index"]),
                name=str(raw.get("name", "default microphone")),
                channels=int(raw.get("maxInputChannels", 1)),
                sample_rate=int(raw.get("defaultSampleRate", 48000) or 48000),
                is_loopback=False,
            )
        except Exception:  # pragma: no cover - no default input configured
            return candidates[0]
