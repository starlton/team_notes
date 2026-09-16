"""Pure audio maths: format conversion, resampling and mixing.

Nothing in here touches a sound card, which means all of it is unit-testable on
any machine. The capture layer feeds raw device bytes in and gets back a single
16 kHz mono float32 signal ready for Whisper.

Why mix offline instead of live: the loopback device and the microphone are
driven by two independent clocks. Summing their callbacks in real time makes
the result sensitive to buffer jitter on either side, and a single dropped
callback shifts everything after it. Instead each stream is written to its own
track, and the tracks are aligned once at the end using the wall-clock instant
each stream actually delivered its first frame.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Sum of both tracks is scaled below this so the mix never hard-clips.
PEAK_CEILING = 0.97
# Each track is loudness-matched to this level before mixing, so a quiet
# microphone is not buried under loud meeting audio.
TARGET_RMS_DBFS = -20.0
# A track quieter than this is treated as silence and left alone (normalising
# near-silence would just amplify the noise floor).
SILENCE_RMS_DBFS = -60.0
# Loudness matching may not amplify a track by more than this, otherwise room
# noise in a muted stream gets boosted to a roar.
MAX_NORMALISATION_GAIN = 8.0

_DTYPE_FOR_WIDTH = {1: np.uint8, 2: np.int16, 4: np.int32}


def pcm_bytes_to_float32(data: bytes, sample_width: int) -> np.ndarray:
    """Decode interleaved PCM bytes into float32 samples in [-1.0, 1.0]."""
    dtype = _DTYPE_FOR_WIDTH.get(sample_width)
    if dtype is None:
        raise ValueError(f"unsupported sample width: {sample_width} bytes")

    usable = len(data) - (len(data) % sample_width)
    raw = np.frombuffer(data[:usable], dtype=dtype)

    if sample_width == 1:
        # 8-bit PCM is unsigned with a 128 midpoint.
        return (raw.astype(np.float32) - 128.0) / 128.0
    if sample_width == 2:
        return raw.astype(np.float32) / 32768.0
    return raw.astype(np.float32) / 2147483648.0


def float32_to_pcm16(samples: np.ndarray) -> bytes:
    """Encode float32 samples as 16-bit little-endian PCM, clipping safely."""
    clipped = np.clip(samples, -1.0, 1.0)
    # 32767 rather than 32768 so +1.0 does not wrap to the most negative value.
    return (clipped * 32767.0).astype("<i2").tobytes()


def to_mono(samples: np.ndarray, channels: int) -> np.ndarray:
    """Average interleaved multi-channel samples down to a single channel."""
    if channels < 1:
        raise ValueError("channels must be at least 1")
    if channels == 1:
        return samples.astype(np.float32, copy=False)

    usable = len(samples) - (len(samples) % channels)
    if usable == 0:
        return np.zeros(0, dtype=np.float32)
    frames = samples[:usable].reshape(-1, channels)
    return frames.mean(axis=1).astype(np.float32)


def _design_lowpass(cutoff_ratio: float, taps: int = 129) -> np.ndarray:
    """A windowed-sinc low-pass kernel. cutoff_ratio is relative to Nyquist."""
    if taps % 2 == 0:
        taps += 1
    n = np.arange(taps, dtype=np.float64) - (taps - 1) / 2.0
    kernel = np.sinc(n * cutoff_ratio) * cutoff_ratio
    kernel *= np.blackman(taps)
    total = kernel.sum()
    if total:
        kernel /= total
    return kernel.astype(np.float32)


def _convolve_same(signal: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """FFT overlap-add convolution, centred like mode='same'.

    numpy's direct convolve is O(n * taps), which is far too slow for an hour of
    48 kHz audio. Overlap-add keeps it near-linear without pulling in scipy.
    """
    taps = len(kernel)
    if taps <= 1 or len(signal) == 0:
        return signal.astype(np.float32, copy=False)

    block = max(4096, 1 << int(np.ceil(np.log2(max(taps * 8, 4096)))))
    fft_size = block + taps - 1
    kernel_f = np.fft.rfft(kernel, fft_size)

    full = np.zeros(len(signal) + taps - 1, dtype=np.float32)
    for start in range(0, len(signal), block):
        chunk = signal[start:start + block]
        spectrum = np.fft.rfft(chunk, fft_size) * kernel_f
        piece = np.fft.irfft(spectrum, fft_size)[: len(chunk) + taps - 1]
        full[start:start + len(piece)] += piece.astype(np.float32)

    offset = (taps - 1) // 2
    return full[offset:offset + len(signal)]


def resample(samples: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Resample mono float32 audio, low-passing first when downsampling.

    Skipping the anti-alias filter on a 48k -> 16k downsample folds everything
    above 8 kHz back into the speech band as noise, which measurably hurts
    Whisper. scipy's polyphase resampler is used when available because it is
    faster and slightly cleaner; the numpy path is an exact-enough fallback.
    """
    if src_rate <= 0 or dst_rate <= 0:
        raise ValueError("sample rates must be positive")
    samples = np.asarray(samples, dtype=np.float32)
    if src_rate == dst_rate or samples.size == 0:
        return samples

    try:  # pragma: no cover - depends on whether scipy is installed
        from math import gcd

        from scipy.signal import resample_poly

        divisor = gcd(src_rate, dst_rate)
        return resample_poly(samples, dst_rate // divisor,
                             src_rate // divisor).astype(np.float32)
    except ImportError:
        pass

    if dst_rate < src_rate:
        # 0.90 of the destination Nyquist leaves a little transition band.
        kernel = _design_lowpass(0.90 * (dst_rate / src_rate))
        samples = _convolve_same(samples, kernel)

    duration = len(samples) / src_rate
    out_len = int(round(duration * dst_rate))
    if out_len <= 0:
        return np.zeros(0, dtype=np.float32)

    src_positions = np.arange(len(samples), dtype=np.float64)
    dst_positions = np.arange(out_len, dtype=np.float64) * (src_rate / dst_rate)
    return np.interp(dst_positions, src_positions, samples).astype(np.float32)


def rms_dbfs(samples: np.ndarray) -> float:
    """Root-mean-square level of a signal in dBFS. Silence returns -inf."""
    if samples.size == 0:
        return float("-inf")
    mean_square = float(np.mean(np.square(samples, dtype=np.float64)))
    if mean_square <= 0.0:
        return float("-inf")
    return 10.0 * float(np.log10(mean_square))


def normalise_loudness(samples: np.ndarray,
                       target_dbfs: float = TARGET_RMS_DBFS) -> np.ndarray:
    """Bring a track towards a target RMS level, within safe gain limits."""
    level = rms_dbfs(samples)
    if level == float("-inf") or level < SILENCE_RMS_DBFS:
        return samples
    gain = float(10.0 ** ((target_dbfs - level) / 20.0))
    gain = min(gain, MAX_NORMALISATION_GAIN)
    return (samples * gain).astype(np.float32)


@dataclass
class Track:
    """One recorded stream, ready to be mixed.

    `start_offset_seconds` is how long after the earliest track this one began.
    """

    samples: np.ndarray
    sample_rate: int
    gain: float = 1.0
    start_offset_seconds: float = 0.0
    name: str = ""


def mix_tracks(tracks: list[Track], sample_rate: int,
               normalise: bool = True) -> np.ndarray:
    """Align, level-match and sum tracks into one mono signal.

    Tracks are resampled to `sample_rate`, delayed by their start offset, padded
    to a common length and summed. If the sum would clip, the whole mix is
    scaled down by a single factor so relative levels are preserved.
    """
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")

    prepared: list[np.ndarray] = []
    for track in tracks:
        samples = np.asarray(track.samples, dtype=np.float32)
        if samples.size == 0:
            continue
        samples = resample(samples, track.sample_rate, sample_rate)
        if normalise:
            samples = normalise_loudness(samples)
        if track.gain != 1.0:
            samples = (samples * float(track.gain)).astype(np.float32)
        offset = max(0, int(round(max(0.0, track.start_offset_seconds) * sample_rate)))
        if offset:
            samples = np.concatenate([np.zeros(offset, dtype=np.float32), samples])
        prepared.append(samples)

    if not prepared:
        return np.zeros(0, dtype=np.float32)

    length = max(len(part) for part in prepared)
    mix = np.zeros(length, dtype=np.float32)
    for part in prepared:
        mix[: len(part)] += part

    peak = float(np.max(np.abs(mix))) if mix.size else 0.0
    if peak > PEAK_CEILING:
        mix *= PEAK_CEILING / peak
    return mix
