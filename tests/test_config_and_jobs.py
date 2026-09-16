"""Configuration validation, credential redaction, and the job queue."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import pytest

from app.logging_setup import RedactingFilter
from app.pipeline.jobs import JobQueue
from config import ConfigError, Secret, load_settings

NOWHERE = Path("/nonexistent/.env")


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Start each test from a known-empty environment."""
    for key in list(__import__("os").environ):
        if key.split("_")[0] in {"WHISPER", "OLLAMA", "DIARIZATION", "WEB", "AUTO",
                                 "CAPTURE", "MIC", "LOOPBACK", "HF", "LOG", "AUDIO",
                                 "PYANNOTE", "TEAMS", "KEEP", "DATA"}:
            monkeypatch.delenv(key, raising=False)


def test_defaults_are_sane(tmp_path):
    settings = load_settings(env_file=NOWHERE, data_dir=tmp_path)
    assert settings.whisper_model == "small.en"
    assert settings.whisper_compute_type == "int8"
    assert settings.ollama_model == "qwen2.5:7b-instruct"
    assert settings.web_host == "127.0.0.1"
    assert settings.db_path == tmp_path / "teams_notes.db"


def test_env_overrides_are_applied(monkeypatch, tmp_path):
    monkeypatch.setenv("WHISPER_MODEL", "medium.en")
    monkeypatch.setenv("OLLAMA_MODEL", "llama3.2:3b")
    monkeypatch.setenv("KEEP_RAW_TRACKS", "false")
    monkeypatch.setenv("TEAMS_PROCESS_NAMES", "Teams.exe, ms-teams.exe , other.exe")
    settings = load_settings(env_file=NOWHERE, data_dir=tmp_path)
    assert settings.whisper_model == "medium.en"
    assert settings.ollama_model == "llama3.2:3b"
    assert settings.keep_raw_tracks is False
    assert settings.teams_process_names == ("Teams.exe", "ms-teams.exe", "other.exe")


def test_the_dashboard_refuses_to_listen_off_machine(monkeypatch, tmp_path):
    """Binding to 0.0.0.0 would expose private recordings to the network."""
    monkeypatch.setenv("WEB_HOST", "0.0.0.0")
    with pytest.raises(ConfigError) as excinfo:
        load_settings(env_file=NOWHERE, data_dir=tmp_path)
    assert "loopback" in str(excinfo.value)


@pytest.mark.parametrize("key,value", [
    ("WEB_PORT", "not-a-number"),
    ("WEB_PORT", "80"),                      # below the unprivileged range
    ("WEB_PORT", "99999"),
    ("LOG_LEVEL", "CHATTY"),
    ("WHISPER_COMPUTE_TYPE", "float64"),
    ("DIARIZATION_ENABLED", "maybe"),
    ("OLLAMA_HOST", "127.0.0.1:11434"),      # no scheme
    ("MIC_GAIN", "99"),
])
def test_bad_configuration_fails_loudly(monkeypatch, tmp_path, key, value):
    monkeypatch.setenv(key, value)
    with pytest.raises(ConfigError):
        load_settings(env_file=NOWHERE, data_dir=tmp_path)


def test_contradictory_speaker_bounds_are_rejected(monkeypatch, tmp_path):
    monkeypatch.setenv("DIARIZATION_MIN_SPEAKERS", "5")
    monkeypatch.setenv("DIARIZATION_MAX_SPEAKERS", "2")
    with pytest.raises(ConfigError):
        load_settings(env_file=NOWHERE, data_dir=tmp_path)


def test_a_dotenv_file_is_read(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text('WHISPER_MODEL="medium.en"\n# a comment\n\nMIC_GAIN=1.5\n')
    settings = load_settings(env_file=env_file, data_dir=tmp_path)
    assert settings.whisper_model == "medium.en"
    assert settings.mic_gain == 1.5


def test_real_environment_beats_the_dotenv_file(tmp_path, monkeypatch):
    monkeypatch.setenv("WHISPER_MODEL", "small.en")
    (tmp_path / ".env").write_text("WHISPER_MODEL=medium.en\n")
    assert load_settings(env_file=tmp_path / ".env",
                         data_dir=tmp_path).whisper_model == "small.en"


def test_ensure_dirs_creates_everything(tmp_path):
    settings = load_settings(env_file=NOWHERE, data_dir=tmp_path / "fresh")
    settings.ensure_dirs()
    for path in (settings.data_dir, settings.audio_dir, settings.log_dir,
                 settings.runtime_dir):
        assert path.is_dir()


# --- secrets ----------------------------------------------------------------

def test_a_secret_never_reveals_itself():
    secret = Secret("hf_abcdefghijklmnop")
    assert "hf_" not in repr(secret)
    assert "hf_" not in str(secret)
    assert "hf_" not in f"{secret}"
    assert secret.get() == "hf_abcdefghijklmnop"
    assert bool(secret) is True
    assert bool(Secret("")) is False


def test_settings_repr_does_not_leak_the_token(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_supersecrettoken123456")
    settings = load_settings(env_file=NOWHERE, data_dir=tmp_path)
    assert "hf_supersecrettoken123456" not in repr(settings)


@pytest.mark.parametrize("message", [
    "failed with token hf_abcdefghijklmnopqrstuvwx",
    "Authorization: Bearer abc123def456",
    "token=sk-secret-value",
])
def test_logging_redacts_credentials(message):
    record = logging.LogRecord("t", logging.ERROR, __file__, 1, message, (), None)
    RedactingFilter().filter(record)
    assert "[redacted]" in record.msg
    assert "abcdefghijklmnopqrstuvwx" not in record.msg
    assert "abc123def456" not in record.msg
    assert "sk-secret-value" not in record.msg


def test_logging_leaves_ordinary_messages_alone():
    record = logging.LogRecord("t", logging.INFO, __file__, 1,
                              "Recorded 120.0s of audio", (), None)
    RedactingFilter().filter(record)
    assert record.msg == "Recorded 120.0s of audio"


# --- the job queue ----------------------------------------------------------

def test_jobs_run_in_order():
    queue = JobQueue()
    queue.start()
    done: list[int] = []
    queue.submit("a", "first", lambda: done.append(1))
    queue.submit("b", "second", lambda: done.append(2))
    _wait_until(lambda: len(done) == 2)
    queue.stop()
    assert done == [1, 2]


def test_a_duplicate_key_is_not_queued_twice():
    queue = JobQueue()
    runs: list[int] = []
    assert queue.submit("same", "job", lambda: (time.sleep(0.2), runs.append(1))) is True
    assert queue.submit("same", "job", lambda: runs.append(2)) is False
    _wait_until(lambda: not queue.is_busy())
    queue.stop()
    assert runs == [1]


def test_a_failing_job_does_not_kill_the_worker():
    queue = JobQueue()
    done: list[int] = []
    queue.submit("bad", "explodes", lambda: 1 / 0)
    _wait_until(lambda: queue.job_state("bad") == "failed")
    queue.submit("good", "works", lambda: done.append(1))
    _wait_until(lambda: done == [1])
    queue.stop()
    assert queue.job_state("good") == "done"
    assert "division by zero" in queue.status()["recent"][1]["error"]


def test_job_state_of_something_unknown():
    assert JobQueue().job_state("never-submitted") == "unknown"


def test_stopping_an_idle_queue_is_safe():
    queue = JobQueue()
    queue.start()
    queue.stop()
    queue.stop()


def _wait_until(predicate, timeout: float = 5.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError("condition was not met in time")
