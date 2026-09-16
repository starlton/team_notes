"""faster-whisper wrapper.

Settings are deliberately tuned for CPU:

* `int8` quantisation and `small.en` keep this comfortably real-time.
* `beam_size=1` (greedy) is roughly twice as fast as the default beam search and
  costs very little accuracy on clear meeting audio.
* `condition_on_previous_text=False` because the failure mode it prevents --
  Whisper latching onto its own output and repeating a phrase for minutes -- is
  far worse on long meeting recordings than the small context it gives up.
* The VAD filter skips silence, which on a typical call is most of the audio.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Callable

from app.errors import DependencyMissingError, TranscriptionError
from app.logging_setup import get_logger
from app.transcribe.models import TranscriptResult, TranscriptSegment, Word

log = get_logger(__name__)

ProgressCallback = Callable[[float, str], None]

_MODEL_CACHE: dict[tuple[str, str, int], Any] = {}
_CACHE_LOCK = threading.Lock()


def _import_faster_whisper() -> Any:
    try:
        from faster_whisper import WhisperModel  # type: ignore[import-not-found]
    except ImportError as exc:
        raise DependencyMissingError(
            "The 'faster-whisper' package is not installed.",
            "Activate the venv and run: pip install faster-whisper",
        ) from exc
    return WhisperModel


class Transcriber:
    """Loads a Whisper model once and transcribes WAV files with it."""

    def __init__(self, model_size: str = "small.en", compute_type: str = "int8",
                 cpu_threads: int = 0, download_root: Path | None = None) -> None:
        self.model_size = model_size
        self.compute_type = compute_type
        self.cpu_threads = max(0, int(cpu_threads))
        self.download_root = Path(download_root) if download_root else None

    def _cache_key(self) -> tuple[str, str, int]:
        return (self.model_size, self.compute_type, self.cpu_threads)

    def load(self) -> Any:
        """Load (and cache) the model. The first call downloads it."""
        key = self._cache_key()
        with _CACHE_LOCK:
            cached = _MODEL_CACHE.get(key)
            if cached is not None:
                return cached

            WhisperModel = _import_faster_whisper()
            log.info("Loading Whisper model %s (%s)", self.model_size, self.compute_type)
            kwargs: dict[str, Any] = {
                "device": "cpu",
                "compute_type": self.compute_type,
            }
            if self.cpu_threads:
                kwargs["cpu_threads"] = self.cpu_threads
            if self.download_root:
                self.download_root.mkdir(parents=True, exist_ok=True)
                kwargs["download_root"] = str(self.download_root)

            try:
                model = WhisperModel(self.model_size, **kwargs)
            except Exception as exc:
                raise TranscriptionError(
                    f"Could not load the Whisper model {self.model_size!r}: {exc}",
                    "The first run downloads it, so check your internet connection, "
                    "or set WHISPER_MODEL to a model you already have.",
                ) from exc

            _MODEL_CACHE[key] = model
            return model

    def transcribe(self, audio_path: Path,
                   progress: ProgressCallback | None = None) -> TranscriptResult:
        """Transcribe a WAV file into timestamped segments with word timings."""
        audio_path = Path(audio_path)
        if not audio_path.is_file():
            raise TranscriptionError(
                f"Audio file not found: {audio_path.name}",
                "The recording may have been deleted.",
            )

        model = self.load()
        if progress:
            progress(0.0, "Transcribing audio")

        try:
            raw_segments, info = model.transcribe(
                str(audio_path),
                beam_size=1,
                language="en" if self.model_size.endswith(".en") else None,
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 500},
                word_timestamps=True,
                condition_on_previous_text=False,
            )
        except Exception as exc:
            raise TranscriptionError(
                f"Whisper failed while transcribing {audio_path.name}: {exc}",
                "Check that the WAV file plays back correctly.",
            ) from exc

        total = float(getattr(info, "duration", 0.0) or 0.0)
        segments: list[TranscriptSegment] = []

        # faster-whisper yields lazily; this loop is where the work happens.
        try:
            for raw in raw_segments:
                text = (raw.text or "").strip()
                if not text:
                    continue
                words = [
                    Word(start=float(w.start), end=float(w.end),
                         text=(w.word or "").strip(),
                         probability=float(getattr(w, "probability", 1.0) or 1.0))
                    for w in (getattr(raw, "words", None) or [])
                    if w.start is not None and w.end is not None
                ]
                segments.append(TranscriptSegment(
                    start=float(raw.start),
                    end=float(raw.end),
                    text=text,
                    words=words,
                    confidence=_segment_confidence(raw),
                ))
                if progress and total > 0:
                    progress(min(0.99, float(raw.end) / total), "Transcribing audio")
        except Exception as exc:
            raise TranscriptionError(
                f"Whisper stopped partway through {audio_path.name}: {exc}",
                "The audio file may be truncated or corrupt.",
            ) from exc

        if progress:
            progress(1.0, "Transcription complete")

        log.info("Transcribed %s into %d segments", audio_path.name, len(segments))
        return TranscriptResult(
            segments=segments,
            language=str(getattr(info, "language", "en") or "en"),
            duration_seconds=total,
            model=self.model_size,
        )


def _segment_confidence(raw: Any) -> float | None:
    """Turn Whisper's average log-probability into a 0..1 confidence."""
    logprob = getattr(raw, "avg_logprob", None)
    if logprob is None:
        return None
    try:
        import math

        return round(min(1.0, max(0.0, math.exp(float(logprob)))), 4)
    except (TypeError, ValueError, OverflowError):  # pragma: no cover
        return None
