"""WAV I/O and the path sandbox."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.capture.wav_io import WavWriter, read_wav_mono, wav_duration_seconds, write_wav
from app.errors import StorageError
from app.paths import meeting_audio_dir, resolve_within, slugify, unique_path
from tests.conftest import tone


def test_write_then_read_roundtrip(tmp_path: Path):
    signal = tone(1.0)
    path = write_wav(tmp_path / "a.wav", signal, 16000)
    restored, rate = read_wav_mono(path)
    assert rate == 16000
    assert np.allclose(signal, restored, atol=1e-4)


def test_writer_only_publishes_the_file_on_close(tmp_path: Path):
    """A crash mid-recording must not leave a file later stages will trust."""
    target = tmp_path / "partial.wav"
    writer = WavWriter(target, 16000)
    writer.open()
    writer.write(tone(0.5))
    assert not target.exists()          # still a .part file
    assert writer.close() == target
    assert target.exists()


def test_writer_discards_an_empty_recording(tmp_path: Path):
    writer = WavWriter(tmp_path / "empty.wav", 16000)
    writer.open()
    assert writer.close() is None
    assert not (tmp_path / "empty.wav").exists()
    assert not (tmp_path / "empty.wav.part").exists()


def test_writer_tracks_duration(tmp_path: Path):
    with WavWriter(tmp_path / "d.wav", 16000) as writer:
        writer.write(tone(1.5))
        assert writer.duration_seconds == pytest.approx(1.5, abs=0.01)


def test_reading_a_missing_file_explains_itself(tmp_path: Path):
    with pytest.raises(StorageError) as excinfo:
        read_wav_mono(tmp_path / "nope.wav")
    assert "not found" in str(excinfo.value).lower()


def test_reading_a_non_wav_file_explains_itself(tmp_path: Path):
    bogus = tmp_path / "bogus.wav"
    bogus.write_bytes(b"this is not a wav file at all")
    with pytest.raises(StorageError) as excinfo:
        read_wav_mono(bogus)
    assert "ffmpeg" in str(excinfo.value)


def test_duration_of_an_unreadable_file_is_zero(tmp_path: Path):
    bad = tmp_path / "bad.wav"
    bad.write_bytes(b"nope")
    assert wav_duration_seconds(bad) == 0.0


def test_refusing_to_write_an_empty_wav(tmp_path: Path):
    with pytest.raises(StorageError):
        write_wav(tmp_path / "x.wav", np.zeros(0, dtype=np.float32), 16000)


# --- the path sandbox -------------------------------------------------------

@pytest.mark.parametrize("attack", [
    "../../etc/passwd",
    "..\\..\\Windows\\System32\\config\\SAM",
    "subdir/../../../outside.txt",
])
def test_resolve_within_blocks_traversal(tmp_path: Path, attack: str):
    root = tmp_path / "audio"
    root.mkdir()
    with pytest.raises(StorageError):
        resolve_within(root, attack)


def test_resolve_within_blocks_an_absolute_path_elsewhere(tmp_path: Path):
    root = tmp_path / "audio"
    root.mkdir()
    with pytest.raises(StorageError):
        resolve_within(root, tmp_path / "elsewhere" / "file.wav")


def test_resolve_within_allows_paths_inside(tmp_path: Path):
    root = tmp_path / "audio"
    (root / "meeting-000001").mkdir(parents=True)
    resolved = resolve_within(root, root / "meeting-000001" / "mixed.wav")
    assert str(resolved).startswith(str(root.resolve()))


def test_meeting_audio_dir_is_predictable(tmp_path: Path):
    assert meeting_audio_dir(tmp_path, 7).name == "meeting-000007"


def test_meeting_audio_dir_rejects_a_bad_id(tmp_path: Path):
    with pytest.raises(StorageError):
        meeting_audio_dir(tmp_path, 0)


@pytest.mark.parametrize("raw,expected", [
    ("Weekly Sync: Q3", "Weekly-Sync-Q3"),
    ("../../etc/passwd", "etc-passwd"),
    ("", "meeting"),
    ("!!!", "meeting"),
])
def test_slugify(raw: str, expected: str):
    assert slugify(raw) == expected


def test_unique_path_avoids_overwriting(tmp_path: Path):
    first = tmp_path / "mixed.wav"
    first.write_bytes(b"x")
    assert unique_path(first).name == "mixed-1.wav"
