"""
processing.py – Core signal processing functions
=================================================
Audio I/O, degradation, metrics, and plot helpers.

All **numerical interpolation / reconstruction** logic lives in
``backend.interpolation``  (scipy-backed, research-grade).
This module delegates to it via ``reconstruct_signal()``.
"""

import numpy as np
import struct
import wave
import io
import base64

from .interpolation import reconstruct as _interpolation_reconstruct


# ═══════════════════════════════════════════════════════════════════
# Audio I/O
# ═══════════════════════════════════════════════════════════════════

def load_wav_bytes(raw_bytes: bytes) -> tuple[int, np.ndarray]:
    """Read a WAV file from bytes, return (sample_rate, mono_samples)."""
    buf = io.BytesIO(raw_bytes)
    with wave.open(buf, "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        sample_rate = wf.getframerate()
        n_frames = wf.getnframes()
        raw_data = wf.readframes(n_frames)

    # Convert raw bytes to numpy array based on sample width
    if sampwidth == 1:
        # 8-bit unsigned
        samples = np.frombuffer(raw_data, dtype=np.uint8).astype(np.float64) - 128.0
        samples /= 128.0
    elif sampwidth == 2:
        # 16-bit signed (most common)
        samples = np.frombuffer(raw_data, dtype=np.int16).astype(np.float64)
        samples /= 32768.0
    elif sampwidth == 3:
        # 24-bit signed – unpack manually
        n_samples = len(raw_data) // 3
        samples = np.zeros(n_samples, dtype=np.float64)
        for i in range(n_samples):
            b = raw_data[i * 3 : i * 3 + 3]
            val = int.from_bytes(b, byteorder="little", signed=True)
            samples[i] = val / 8388608.0
    elif sampwidth == 4:
        # 32-bit signed
        samples = np.frombuffer(raw_data, dtype=np.int32).astype(np.float64)
        samples /= 2147483648.0
    else:
        raise ValueError(f"Unsupported sample width: {sampwidth}")

    # Convert to mono by averaging channels
    if n_channels > 1:
        samples = samples.reshape(-1, n_channels).mean(axis=1)

    # Limit length: truncate to first 5 seconds for manageability
    max_samples = sample_rate * 5
    if len(samples) > max_samples:
        samples = samples[:max_samples]

    return sample_rate, samples


def signal_to_wav_base64(signal: np.ndarray, sample_rate: int) -> str:
    """Convert a float64 signal [-1, 1] to a base64-encoded 16-bit WAV."""
    # Clip and scale to int16
    clipped = np.clip(signal, -1.0, 1.0)
    int_samples = (clipped * 32767).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(int_samples.tobytes())

    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


# ═══════════════════════════════════════════════════════════════════
# Demo signal generation
# ═══════════════════════════════════════════════════════════════════

def generate_demo_signal(duration: float = 1.0, sr: int = 8000) -> tuple[int, np.ndarray]:
    """
    Generate a composite test signal:
      440 Hz tone + 880 Hz harmonic + amplitude envelope
    Sounds like a brief piano-like note — perfect for demonstrating
    how interpolation preserves harmonic structure.
    """
    t = np.arange(int(sr * duration)) / sr
    # Fundamental + harmonic
    signal = 0.6 * np.sin(2 * np.pi * 440 * t) + 0.3 * np.sin(2 * np.pi * 880 * t)
    # Gentle amplitude envelope (attack + decay)
    envelope = np.minimum(t / 0.05, 1.0) * np.exp(-t * 2.0)
    signal *= envelope
    # Normalize
    peak = np.max(np.abs(signal))
    if peak > 0:
        signal /= peak
    return sr, signal


# ═══════════════════════════════════════════════════════════════════
# Signal Degradation
# ═══════════════════════════════════════════════════════════════════

def degrade_signal(
    original: np.ndarray,
    dropout_pct: float = 20.0,
    noise_level: float = 0.02,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Degrade a signal with random dropouts and Gaussian noise.

    Args:
        original:    clean signal array
        dropout_pct: percentage of samples to zero out (0-80)
        noise_level: std deviation of additive Gaussian noise (0-0.5)

    Returns:
        (spoiled_signal, mask)
        mask[i] = True means sample i is VALID (not dropped)
    """
    n = len(original)
    spoiled = original.copy()

    # Clamp parameters to safe ranges
    dropout_pct = np.clip(dropout_pct, 0.0, 80.0)
    noise_level = np.clip(noise_level, 0.0, 0.5)

    # Create dropout mask
    rng = np.random.default_rng(42)  # reproducible for demo
    n_drop = int(n * dropout_pct / 100.0)
    drop_indices = rng.choice(n, size=n_drop, replace=False)

    mask = np.ones(n, dtype=bool)
    mask[drop_indices] = False
    spoiled[drop_indices] = 0.0  # zero out dropped samples

    # Add Gaussian noise to surviving samples
    if noise_level > 0:
        noise = rng.normal(0, noise_level, size=n)
        spoiled[mask] += noise[mask]

    # Clip to [-1, 1]
    spoiled = np.clip(spoiled, -1.0, 1.0)

    return spoiled, mask


# ═══════════════════════════════════════════════════════════════════
# Numerical Interpolation / Reconstruction
# ═══════════════════════════════════════════════════════════════════
# All interpolation logic is in backend/interpolation.py (SciPy-backed).
# This thin wrapper keeps the public interface unchanged for main.py.
#
# 🚫 Lagrange interpolation is NOT supported.
#    Lagrange interpolation is numerically unstable for dense and noisy
#    signals such as audio.
# ═══════════════════════════════════════════════════════════════════

def reconstruct_signal(
    time_axis: np.ndarray,
    spoiled: np.ndarray,
    mask: np.ndarray,
    method: str = "spline",
) -> np.ndarray:
    """Delegate to ``backend.interpolation.reconstruct``."""
    return _interpolation_reconstruct(time_axis, spoiled, mask, method=method)


# ═══════════════════════════════════════════════════════════════════
# Error Metrics
# ═══════════════════════════════════════════════════════════════════

def compute_metrics(original: np.ndarray, reconstructed: np.ndarray) -> dict:
    """Compute reconstruction quality metrics."""
    diff = original - reconstructed
    finite_diff = diff[np.isfinite(diff)]

    if len(finite_diff) == 0:
        return {"mse": 0.0, "rmse": 0.0, "mae": 0.0, "snr_db": 0.0}

    mse_val = float(np.mean(finite_diff ** 2))
    rmse_val = float(np.sqrt(mse_val))
    mae_val = float(np.mean(np.abs(finite_diff)))

    # Signal-to-Noise Ratio (dB)
    signal_power = float(np.mean(original ** 2))
    noise_power = mse_val
    if noise_power > 0 and signal_power > 0:
        snr_db = float(10 * np.log10(signal_power / noise_power))
    else:
        snr_db = float("inf") if noise_power == 0 else 0.0

    return {
        "mse": round(mse_val, 8),
        "rmse": round(rmse_val, 8),
        "mae": round(mae_val, 8),
        "snr_db": round(snr_db, 2),
    }


# ═══════════════════════════════════════════════════════════════════
# Plot Downsampling (for large signals)
# ═══════════════════════════════════════════════════════════════════

def downsample_for_plot(
    time_axis: np.ndarray, signal: np.ndarray, max_points: int = 8000
) -> tuple[np.ndarray, np.ndarray]:
    """
    Downsample signal for frontend plotting using LTTB-like approach.
    Keeps first and last points, takes evenly spaced samples between.
    """
    n = len(signal)
    if n <= max_points:
        return time_axis.copy(), signal.copy()

    indices = np.linspace(0, n - 1, max_points, dtype=int)
    return time_axis[indices], signal[indices]
