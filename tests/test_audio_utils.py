"""Tests for the pure audio maths."""

from __future__ import annotations

import numpy as np
import pytest

from app.capture.audio_utils import (PEAK_CEILING, Track, float32_to_pcm16, mix_tracks,
                                     normalise_loudness, pcm_bytes_to_float32, resample,
                                     rms_dbfs, to_mono)
from tests.conftest import tone


def test_pcm16_roundtrip_preserves_signal():
    original = np.array([0.0, 0.5, -0.5, 0.999], dtype=np.float32)
    restored = pcm_bytes_to_float32(float32_to_pcm16(original), 2)
    assert np.allclose(original, restored, atol=1e-4)


def test_pcm16_clips_instead_of_wrapping():
    # The classic bug: +1.0 scaled by 32768 wraps to the most negative value.
    encoded = pcm_bytes_to_float32(float32_to_pcm16(
        np.array([1.5, -1.5], dtype=np.float32)), 2)
    assert encoded[0] > 0.99
    assert encoded[1] < -0.99


def test_pcm_bytes_handles_a_truncated_frame():
    # A device can hand back a buffer that ends mid-sample.
    assert len(pcm_bytes_to_float32(b"\x00\x01\x00", 2)) == 1


def test_pcm_bytes_rejects_unknown_width():
    with pytest.raises(ValueError):
        pcm_bytes_to_float32(b"\x00" * 6, 3)


def test_to_mono_averages_channels():
    interleaved = np.array([1.0, -1.0, 0.5, 0.5], dtype=np.float32)
    assert np.allclose(to_mono(interleaved, 2), [0.0, 0.5])


def test_to_mono_passes_through_single_channel():
    samples = np.array([0.1, 0.2], dtype=np.float32)
    assert np.allclose(to_mono(samples, 1), samples)


def test_resample_changes_length_and_keeps_level():
    signal = tone(1.0, 440.0, 48000)
    out = resample(signal, 48000, 16000)
    assert len(out) == 16000
    assert abs(rms_dbfs(out) - rms_dbfs(signal)) < 1.0


def test_resample_is_a_noop_at_the_same_rate():
    signal = tone(0.2)
    assert resample(signal, 16000, 16000) is signal


def test_downsampling_attenuates_content_above_the_new_nyquist():
    """Without an anti-alias filter this tone would fold back into speech range."""
    above_nyquist = tone(1.0, 15000.0, 48000)   # 15 kHz, well above 8 kHz
    out = resample(above_nyquist, 48000, 16000)
    assert rms_dbfs(out) < rms_dbfs(above_nyquist) - 20


def test_normalise_loudness_lifts_a_quiet_track():
    quiet = tone(1.0, amplitude=0.01)
    assert rms_dbfs(normalise_loudness(quiet)) > rms_dbfs(quiet)


def test_normalise_loudness_leaves_silence_alone():
    silence = np.zeros(1000, dtype=np.float32)
    assert np.array_equal(normalise_loudness(silence), silence)


def test_normalise_loudness_caps_the_gain():
    # Near-silence must not be amplified without limit.
    very_quiet = tone(1.0, amplitude=1e-5)
    boosted = normalise_loudness(very_quiet)
    assert float(np.max(np.abs(boosted))) <= 1e-5 * 8.0 + 1e-9


def test_mix_aligns_tracks_by_their_start_offset():
    mix = mix_tracks([
        Track(tone(1.0), 16000, name="loopback"),
        Track(tone(1.0), 16000, start_offset_seconds=0.5, name="microphone"),
    ], 16000)
    # The later track pushes the total length out by its offset.
    assert len(mix) == 16000 + 8000
    # The first half second holds only the first track.
    assert rms_dbfs(mix[:8000]) < rms_dbfs(mix[8000:16000])


def test_mix_never_clips():
    loud = np.ones(16000, dtype=np.float32) * 0.95
    mix = mix_tracks([Track(loud, 16000), Track(loud, 16000)], 16000, normalise=False)
    assert float(np.max(np.abs(mix))) <= PEAK_CEILING + 1e-6


def test_mix_resamples_tracks_to_a_common_rate():
    mix = mix_tracks([Track(tone(1.0, 440, 48000), 48000),
                      Track(tone(1.0, 440, 16000), 16000)], 16000)
    assert len(mix) == 16000


def test_mix_of_nothing_is_empty():
    assert mix_tracks([], 16000).size == 0
    assert mix_tracks([Track(np.zeros(0, dtype=np.float32), 16000)], 16000).size == 0


def test_mix_keeps_a_quiet_microphone_audible():
    """The whole point of level matching: a quiet mic must survive the mix."""
    loud_meeting = tone(1.0, 300, amplitude=0.8)
    quiet_mic = tone(1.0, 900, amplitude=0.02)
    mix = mix_tracks([Track(loud_meeting, 16000, name="loopback"),
                      Track(quiet_mic, 16000, name="microphone")], 16000)
    spectrum = np.abs(np.fft.rfft(mix))
    freqs = np.fft.rfftfreq(len(mix), 1 / 16000)
    mic_energy = spectrum[np.argmin(np.abs(freqs - 900))]
    meeting_energy = spectrum[np.argmin(np.abs(freqs - 300))]
    # Without normalisation this ratio would be about 1:40.
    assert mic_energy > meeting_energy * 0.1


def test_rms_of_silence_is_negative_infinity():
    assert rms_dbfs(np.zeros(100, dtype=np.float32)) == float("-inf")
    assert rms_dbfs(np.zeros(0, dtype=np.float32)) == float("-inf")
