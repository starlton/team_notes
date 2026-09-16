"""pyannote.audio wrapper.

Diarization is the heaviest part of the pipeline, which is exactly why it runs
after the meeting rather than during it. On a CPU-only laptop expect roughly
0.3-1.0x real time.

pyannote's models are gated on Hugging Face: the user needs a free account, a
read token, and a one-time click-through of the model terms. Every way that can
go wrong is turned into a message that names the fix.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any, Callable

from app.diarize.merge import SpeakerTurn
from app.errors import DependencyMissingError, DiarizationError
from app.logging_setup import get_logger

log = get_logger(__name__)

ProgressCallback = Callable[[float, str], None]

_PIPELINE_CACHE: dict[str, Any] = {}
_CACHE_LOCK = threading.Lock()

_TERMS_URLS = (
    "https://huggingface.co/pyannote/speaker-diarization-3.1",
    "https://huggingface.co/pyannote/segmentation-3.0",
)


class Diarizer:
    """Runs pyannote's speaker-diarization pipeline over a WAV file."""

    def __init__(self, model: str = "pyannote/speaker-diarization-3.1",
                 hf_token: str = "", min_speakers: int | None = None,
                 max_speakers: int | None = None,
                 cache_dir: Path | None = None) -> None:
        self.model = model
        self._hf_token = hf_token or ""
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers
        self.cache_dir = Path(cache_dir) if cache_dir else None

    @property
    def available(self) -> bool:
        return bool(self._hf_token)

    def _load_pipeline(self) -> Any:
        if not self._hf_token:
            raise DiarizationError(
                "No Hugging Face token is configured, so speakers cannot be labelled.",
                "Add HF_TOKEN=... to your .env file (see README), or set "
                "DIARIZATION_ENABLED=false to skip speaker labelling.",
            )

        with _CACHE_LOCK:
            cached = _PIPELINE_CACHE.get(self.model)
            if cached is not None:
                return cached

            try:
                import torch  # type: ignore[import-not-found]
                from pyannote.audio import Pipeline  # type: ignore[import-not-found]
            except ImportError as exc:
                raise DependencyMissingError(
                    "The 'pyannote.audio' package is not installed.",
                    "Activate the venv and run: pip install pyannote.audio",
                ) from exc

            if self.cache_dir:
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                os.environ.setdefault("HF_HOME", str(self.cache_dir))

            log.info("Loading diarization pipeline %s", self.model)
            try:
                pipeline = Pipeline.from_pretrained(self.model,
                                                    use_auth_token=self._hf_token)
            except Exception as exc:
                raise DiarizationError(
                    f"Could not load the diarization model {self.model!r}: "
                    f"{_describe_load_failure(exc)}",
                    "Sign in to Hugging Face and accept the model terms at "
                    + " and ".join(_TERMS_URLS)
                    + ", then check HF_TOKEN in your .env file.",
                ) from exc

            if pipeline is None:
                raise DiarizationError(
                    "Hugging Face returned no pipeline, which almost always means "
                    "the model terms have not been accepted for this account.",
                    "Accept the terms at " + " and ".join(_TERMS_URLS)
                    + ", then try again.",
                )

            try:
                pipeline.to(torch.device("cpu"))
                # Leave a core free so the machine stays usable during processing.
                threads = max(1, (os.cpu_count() or 2) - 1)
                torch.set_num_threads(threads)
            except Exception:  # pragma: no cover - torch build dependent
                log.debug("Could not pin pyannote to CPU explicitly", exc_info=True)

            _PIPELINE_CACHE[self.model] = pipeline
            return pipeline

    def diarize(self, audio_path: Path,
                progress: ProgressCallback | None = None) -> list[SpeakerTurn]:
        """Return speaker turns for a WAV file, ordered by start time."""
        audio_path = Path(audio_path)
        if not audio_path.is_file():
            raise DiarizationError(
                f"Audio file not found: {audio_path.name}", "",
            )

        pipeline = self._load_pipeline()
        if progress:
            progress(0.0, "Identifying speakers")

        kwargs: dict[str, Any] = {}
        if self.min_speakers:
            kwargs["min_speakers"] = int(self.min_speakers)
        if self.max_speakers:
            kwargs["max_speakers"] = int(self.max_speakers)

        try:
            annotation = pipeline(str(audio_path), **kwargs)
        except Exception as exc:
            raise DiarizationError(
                f"Speaker diarization failed on {audio_path.name}: {exc}",
                "If this repeats, set DIARIZATION_ENABLED=false to skip speaker "
                "labelling; the rest of the pipeline works without it.",
            ) from exc

        turns = [
            SpeakerTurn(start=float(segment.start), end=float(segment.end),
                        speaker=str(label))
            for segment, _, label in annotation.itertracks(yield_label=True)
            if float(segment.end) > float(segment.start)
        ]
        turns.sort(key=lambda turn: (turn.start, turn.end))

        if progress:
            progress(1.0, "Speakers identified")
        log.info("Diarization found %d turns across %d speakers",
                 len(turns), len({turn.speaker for turn in turns}))
        return turns


def _describe_load_failure(exc: Exception) -> str:
    """Turn a Hugging Face error into something a human can act on."""
    text = str(exc)
    lowered = text.lower()
    if "401" in text or "unauthorized" in lowered or "invalid" in lowered:
        return "Hugging Face rejected the token (401)"
    if "403" in text or "gated" in lowered or "awaiting" in lowered:
        return "the model is gated and this account has not accepted its terms (403)"
    if "connection" in lowered or "timed out" in lowered or "network" in lowered:
        return "the download could not reach Hugging Face"
    return text
