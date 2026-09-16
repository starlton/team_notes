"""Configuration for Teams Notes.

Everything is read from environment variables, optionally populated from a
`.env` file in the project root. Values are validated once at import of
`get_settings()` so a typo fails loudly at startup instead of halfway through a
meeting.

Secrets (currently only the Hugging Face token) are wrapped in `Secret` so that
they cannot be printed, logged or serialised by accident.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}


class ConfigError(RuntimeError):
    """Raised when the configuration cannot be understood."""


class Secret:
    """A string that refuses to reveal itself in logs, reprs or tracebacks."""

    __slots__ = ("_value",)

    def __init__(self, value: str = "") -> None:
        self._value = value or ""

    def get(self) -> str:
        return self._value

    def __bool__(self) -> bool:
        return bool(self._value)

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return "Secret('***')" if self._value else "Secret(empty)"

    __str__ = __repr__


def _load_dotenv(path: Path) -> None:
    """Populate os.environ from a .env file without overriding real env vars."""
    if not path.is_file():
        return
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _get(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _get_bool(name: str, default: bool) -> bool:
    raw = _get(name).lower()
    if not raw:
        return default
    if raw in _TRUE:
        return True
    if raw in _FALSE:
        return False
    raise ConfigError(f"{name} must be true or false, got {raw!r}")


def _get_int(name: str, default: int, *, minimum: int | None = None,
             maximum: int | None = None) -> int:
    raw = _get(name)
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a whole number, got {raw!r}") from exc
    if minimum is not None and value < minimum:
        raise ConfigError(f"{name} must be >= {minimum}, got {value}")
    if maximum is not None and value > maximum:
        raise ConfigError(f"{name} must be <= {maximum}, got {value}")
    return value


def _get_float(name: str, default: float, *, minimum: float | None = None,
               maximum: float | None = None) -> float:
    raw = _get(name)
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a number, got {raw!r}") from exc
    if minimum is not None and value < minimum:
        raise ConfigError(f"{name} must be >= {minimum}, got {value}")
    if maximum is not None and value > maximum:
        raise ConfigError(f"{name} must be <= {maximum}, got {value}")
    return value


def _get_opt_int(name: str, *, minimum: int | None = None) -> Optional[int]:
    raw = _get(name)
    if not raw:
        return None
    return _get_int(name, 0, minimum=minimum)


def _get_list(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    raw = _get(name)
    if not raw:
        return default
    return tuple(part.strip() for part in raw.split(",") if part.strip())


@dataclass(frozen=True)
class Settings:
    """Immutable snapshot of the app configuration."""

    # Paths
    data_dir: Path = DEFAULT_DATA_DIR
    # Secrets
    hf_token: Secret = field(default_factory=Secret, repr=False)
    # Transcription
    whisper_model: str = "small.en"
    whisper_compute_type: str = "int8"
    whisper_cpu_threads: int = 0
    # Diarization
    diarization_enabled: bool = True
    pyannote_model: str = "pyannote/speaker-diarization-3.1"
    diarization_min_speakers: Optional[int] = None
    diarization_max_speakers: Optional[int] = None
    # LLM
    ollama_host: str = "http://127.0.0.1:11434"
    ollama_model: str = "qwen2.5:7b-instruct"
    ollama_timeout_seconds: int = 600
    ollama_num_ctx: int = 8192
    # Audio
    capture_sample_rate: int = 16000
    mic_gain: float = 1.0
    loopback_gain: float = 1.0
    mic_device_name: str = ""
    loopback_device_name: str = ""
    keep_raw_tracks: bool = True
    # Web
    web_host: str = "127.0.0.1"
    web_port: int = 8765
    open_browser_on_start: bool = False
    # Auto-detect
    auto_detect_enabled: bool = True
    teams_process_names: tuple[str, ...] = ("Teams.exe", "ms-teams.exe")
    auto_start_after_seconds: float = 10.0
    auto_stop_after_seconds: float = 45.0
    auto_min_meeting_seconds: float = 60.0
    # Retention
    audio_retention_days: int = 0
    # Logging
    log_level: str = "INFO"

    # --- derived paths ----------------------------------------------------
    @property
    def db_path(self) -> Path:
        return self.data_dir / "teams_notes.db"

    @property
    def audio_dir(self) -> Path:
        return self.data_dir / "audio"

    @property
    def log_dir(self) -> Path:
        return self.data_dir / "logs"

    @property
    def runtime_dir(self) -> Path:
        return self.data_dir / "runtime"

    @property
    def dashboard_url(self) -> str:
        return f"http://{self.web_host}:{self.web_port}/"

    def ensure_dirs(self) -> None:
        for path in (self.data_dir, self.audio_dir, self.log_dir, self.runtime_dir):
            path.mkdir(parents=True, exist_ok=True)
        _harden_directory(self.runtime_dir)


def _harden_directory(path: Path) -> None:
    """Best-effort: make a directory readable by the owner only."""
    try:
        if os.name == "posix":
            path.chmod(0o700)
    except OSError:  # pragma: no cover - platform dependent
        pass


def load_settings(env_file: Path | None = None,
                  data_dir: Path | None = None) -> Settings:
    """Build a Settings object from the environment (and optional .env file)."""
    _load_dotenv(env_file if env_file is not None else PROJECT_ROOT / ".env")

    log_level = _get("LOG_LEVEL", "INFO").upper()
    if log_level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
        raise ConfigError(f"LOG_LEVEL must be a Python log level, got {log_level!r}")

    compute_type = _get("WHISPER_COMPUTE_TYPE", "int8")
    if compute_type not in {"int8", "int8_float32", "float32", "int16"}:
        raise ConfigError(
            "WHISPER_COMPUTE_TYPE must be one of int8, int8_float32, int16, float32"
        )

    web_host = _get("WEB_HOST", "127.0.0.1")
    if web_host not in {"127.0.0.1", "localhost", "::1"}:
        # The dashboard has no authentication story beyond a local token, and
        # the app records private conversations. Refuse to listen off-machine.
        raise ConfigError(
            "WEB_HOST must stay on loopback (127.0.0.1, localhost or ::1). "
            "The dashboard is deliberately not safe to expose to a network."
        )

    min_speakers = _get_opt_int("DIARIZATION_MIN_SPEAKERS", minimum=1)
    max_speakers = _get_opt_int("DIARIZATION_MAX_SPEAKERS", minimum=1)
    if min_speakers and max_speakers and min_speakers > max_speakers:
        raise ConfigError(
            "DIARIZATION_MIN_SPEAKERS cannot be greater than DIARIZATION_MAX_SPEAKERS"
        )

    ollama_host = _get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
    if not ollama_host.startswith(("http://", "https://")):
        raise ConfigError("OLLAMA_HOST must start with http:// or https://")

    resolved_data_dir = (data_dir or Path(_get("DATA_DIR") or DEFAULT_DATA_DIR)).resolve()

    return Settings(
        data_dir=resolved_data_dir,
        hf_token=Secret(_get("HF_TOKEN")),
        whisper_model=_get("WHISPER_MODEL", "small.en"),
        whisper_compute_type=compute_type,
        whisper_cpu_threads=_get_int("WHISPER_CPU_THREADS", 0, minimum=0, maximum=128),
        diarization_enabled=_get_bool("DIARIZATION_ENABLED", True),
        pyannote_model=_get("PYANNOTE_MODEL", "pyannote/speaker-diarization-3.1"),
        diarization_min_speakers=min_speakers,
        diarization_max_speakers=max_speakers,
        ollama_host=ollama_host,
        ollama_model=_get("OLLAMA_MODEL", "qwen2.5:7b-instruct"),
        ollama_timeout_seconds=_get_int("OLLAMA_TIMEOUT_SECONDS", 600, minimum=10,
                                        maximum=7200),
        ollama_num_ctx=_get_int("OLLAMA_NUM_CTX", 8192, minimum=1024, maximum=131072),
        capture_sample_rate=_get_int("CAPTURE_SAMPLE_RATE", 16000, minimum=8000,
                                     maximum=48000),
        mic_gain=_get_float("MIC_GAIN", 1.0, minimum=0.0, maximum=10.0),
        loopback_gain=_get_float("LOOPBACK_GAIN", 1.0, minimum=0.0, maximum=10.0),
        mic_device_name=_get("MIC_DEVICE_NAME"),
        loopback_device_name=_get("LOOPBACK_DEVICE_NAME"),
        keep_raw_tracks=_get_bool("KEEP_RAW_TRACKS", True),
        web_host=web_host,
        web_port=_get_int("WEB_PORT", 8765, minimum=1024, maximum=65535),
        open_browser_on_start=_get_bool("OPEN_BROWSER_ON_START", False),
        auto_detect_enabled=_get_bool("AUTO_DETECT_ENABLED", True),
        teams_process_names=_get_list("TEAMS_PROCESS_NAMES",
                                      ("Teams.exe", "ms-teams.exe")),
        auto_start_after_seconds=_get_float("AUTO_START_AFTER_SECONDS", 10.0,
                                            minimum=0.0, maximum=600.0),
        auto_stop_after_seconds=_get_float("AUTO_STOP_AFTER_SECONDS", 45.0,
                                           minimum=5.0, maximum=3600.0),
        auto_min_meeting_seconds=_get_float("AUTO_MIN_MEETING_SECONDS", 60.0,
                                            minimum=0.0, maximum=3600.0),
        audio_retention_days=_get_int("AUDIO_RETENTION_DAYS", 0, minimum=0,
                                      maximum=3650),
        log_level=log_level,
    )


_CACHED: Settings | None = None


def get_settings(refresh: bool = False) -> Settings:
    """Return the process-wide Settings, loading them on first use."""
    global _CACHED
    if _CACHED is None or refresh:
        _CACHED = load_settings()
    return _CACHED
