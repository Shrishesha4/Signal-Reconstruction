"""
advanced_reconstruction.py – Enhanced Signal Reconstruction with Classical DSP
==============================================================================

This module implements advanced signal reconstruction techniques using only
classical signal processing algorithms (no machine learning).

Architecture (5-Stage Pipeline)
-------------------------------
  1. **Noise Reduction (Pre-Interpolation)**
     - Spectral subtraction in frequency domain
     - Savitzky-Golay filtering for structure preservation
     - Median filtering for impulse noise removal
     - Wiener filtering for optimal noise suppression

  2. **Damage Analysis & Segmentation**
     - Classify damage types: dropouts, clipping, glitches, noise
     - Segment signal into damaged/undamaged blocks
     - Detect clipping zones (flat peaks at ±threshold)
     - Estimate local signal characteristics

  3. **Model-Based Reconstruction**
     - Sinusoidal modeling for harmonic signals
     - Estimate dominant frequencies via FFT
     - Parametric model fitting for gap filling
     - Least-squares estimation over good segments

  4. **Adaptive Interpolation**
     - Gap size adaptive: Linear (1-3), PCHIP (4-50), Regularized Spline (>50)
     - Tikhonov regularization for noise stability
     - Clipping repair via envelope extrapolation
     - Edge blending with Hann windows

  5. **Perceptual Post-Processing**
     - Low-pass filtering (0.45 × Nyquist)
     - Dynamic range compression
     - Soft clipping (tanh limiter)
     - Envelope matching to original segments

Performance Targets
-------------------
  - SNR improvement: +4-8 dB over baseline methods
  - MSE reduction: 40-60% lower than basic interpolation
  - Audible quality: Natural sound with minimal artifacts

Mathematical Background
-----------------------
Each technique is documented with its mathematical formulation.
All operations are vectorized NumPy/SciPy for O(N) or O(N log N) complexity.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from numpy.fft import fft, ifft, fftfreq
from scipy import interpolate as sp_interp
from scipy import signal as sp_signal
from scipy.ndimage import uniform_filter1d, median_filter as nd_median
from scipy.linalg import solve_banded
from scipy.optimize import least_squares
from typing import Tuple, List, Dict, Optional
import warnings


# ═══════════════════════════════════════════════════════════════════
# Configuration Constants
# ═══════════════════════════════════════════════════════════════════

# Gap size thresholds for adaptive interpolation
GAP_TINY = 3        # ≤3 samples: linear interpolation
GAP_SMALL = 10      # ≤10 samples: PCHIP
GAP_MEDIUM = 50     # ≤50 samples: smooth PCHIP with regularization
GAP_LARGE = 200     # ≤200 samples: model-guided interpolation
# > 200 samples: sinusoidal modeling + regularized spline

# Noise reduction parameters
SPECTRAL_SUBTRACTION_ALPHA = 2.0  # Over-subtraction factor
SPECTRAL_SUBTRACTION_BETA = 0.01  # Spectral floor
WIENER_NOISE_EST_PERCENTILE = 10  # Percentile for noise estimation

# Sinusoidal modeling
MAX_HARMONICS = 8           # Maximum harmonics to model
MIN_HARMONIC_MAGNITUDE = 0.05  # Minimum relative magnitude
FREQ_RESOLUTION_HZ = 2.0    # FFT frequency resolution target

# Tikhonov regularization
TIKHONOV_LAMBDA = 1e-4      # Regularization strength
SMOOTHING_WEIGHT = 0.1      # Weight for smoothness term

# Clipping detection
CLIPPING_THRESHOLD = 0.95   # Amplitude threshold for clipping detection
CLIPPING_MIN_SAMPLES = 3    # Minimum samples to consider as clipping

# Post-processing
POST_LP_CUTOFF = 0.45       # Low-pass cutoff (fraction of Nyquist)
POST_LP_ORDER = 4           # Butterworth filter order
SOFT_CLIP_THRESH = 0.95     # Soft clipping threshold
ENVELOPE_WINDOW = 128       # Envelope estimation window

# Dynamic range compression
COMPRESSION_THRESHOLD = 0.7  # Compression threshold
COMPRESSION_RATIO = 3.0      # Compression ratio above threshold


# ═══════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════

def advanced_reconstruct(
    time_axis: NDArray[np.float64],
    spoiled: NDArray[np.float64],
    mask: NDArray[np.bool_],
    sample_rate: int = 8000,
    method: str = "pchip",
    use_sinusoidal_model: bool = True,
    use_spectral_subtraction: bool = True,
    use_tikhonov: bool = True,
) -> NDArray[np.float64]:
    """
    Advanced signal reconstruction with classical DSP techniques.
    
    Parameters
    ----------
    time_axis : 1-D float64 - Monotonic time vector
    spoiled   : 1-D float64 - Degraded signal
    mask      : 1-D bool    - True = valid sample, False = damaged
    sample_rate : int       - Sampling rate in Hz
    method    : str         - Base method: 'pchip', 'spline', 'linear', 'moving_average'
    use_sinusoidal_model : bool - Enable harmonic modeling for large gaps
    use_spectral_subtraction : bool - Enable frequency-domain noise reduction
    use_tikhonov : bool     - Enable Tikhonov regularization for splines
    
    Returns
    -------
    1-D float64 array clamped to [-1, 1]
    """
    # ─── Input validation ────────────────────────────────────
    n = len(spoiled)
    if n < 4:
        return np.clip(spoiled.copy(), -1.0, 1.0)
    
    valid_mask = mask.copy()
    missing_mask = ~valid_mask
    
    if not np.any(missing_mask):
        # No damage - just apply light noise reduction
        return _light_denoise(spoiled, sample_rate)
    
    # ─── Stage 1: Noise Reduction (Pre-Interpolation) ────────
    # Clean the valid samples before using them as anchors
    valid_samples = spoiled[valid_mask].copy()
    
    if use_spectral_subtraction and len(valid_samples) >= 64:
        # Estimate noise spectrum from valid samples
        noise_estimate = _estimate_noise_spectrum(valid_samples, sample_rate)
        valid_samples = _spectral_subtraction(valid_samples, noise_estimate)
    
    # Median filter for impulse noise
    if len(valid_samples) >= 5:
        valid_samples = nd_median(valid_samples, size=3, mode="reflect")
    
    # Light Savitzky-Golay for structure preservation
    if len(valid_samples) >= 7:
        smoothed = sp_signal.savgol_filter(valid_samples, 7, 3, mode="mirror")
        # Blend: 80% original, 20% smoothed (preserve detail)
        valid_samples = 0.8 * valid_samples + 0.2 * smoothed
    
    # Update spoiled array with cleaned valid samples
    cleaned = spoiled.copy()
    cleaned[valid_mask] = valid_samples
    
    # ─── Stage 2: Damage Analysis & Segmentation ─────────────
    gaps = _analyze_gaps(missing_mask, spoiled, sample_rate)
    clipping_regions = _detect_clipping_regions(spoiled, valid_mask)
    
    # ─── Stage 3: Model-Based Reconstruction ─────────────────
    # For large gaps, use sinusoidal modeling if enabled
    harmonic_model = None
    if use_sinusoidal_model and len(valid_samples) >= 128:
        harmonic_model = _estimate_harmonic_model(
            time_axis[valid_mask], valid_samples, sample_rate
        )
    
    # ─── Stage 4: Adaptive Interpolation ─────────────────────
    reconstructed = _adaptive_interpolate(
        time_axis, cleaned, valid_mask, 
        gaps, harmonic_model,
        method, use_tikhonov, sample_rate
    )
    
    # ─── Repair clipping regions ─────────────────────────────
    if len(clipping_regions) > 0:
        reconstructed = _repair_clipping(
            reconstructed, clipping_regions, time_axis, sample_rate
        )
    
    # ─── Stage 5: Perceptual Post-Processing ─────────────────
    reconstructed = _perceptual_postprocess(
        reconstructed, missing_mask, sample_rate
    )
    
    return reconstructed


# ═══════════════════════════════════════════════════════════════════
# Stage 1: Noise Reduction
# ═══════════════════════════════════════════════════════════════════

def _estimate_noise_spectrum(
    signal: NDArray[np.float64],
    sample_rate: int,
    frame_size: int = 256,
    hop_size: int = 64,
) -> NDArray[np.float64]:
    """
    Estimate noise power spectrum using minimum statistics.
    
    Mathematical Background:
    The noise spectrum is estimated by tracking the minimum power
    in each frequency bin over time. This works because:
    - Speech/music has pauses where only noise is present
    - Minimum tracking captures noise floor during these pauses
    
    Formula: N(f) = min_t { |X(f,t)|² } averaged over low-energy frames
    """
    n = len(signal)
    if n < frame_size:
        # Fallback: estimate noise as 10th percentile of signal energy
        return np.ones(frame_size // 2 + 1) * np.percentile(signal ** 2, 10)
    
    # Window function (Hann for spectral analysis)
    window = np.hanning(frame_size)
    
    # Compute STFT frames
    n_frames = max(1, (n - frame_size) // hop_size + 1)
    power_spectra = []
    
    for i in range(n_frames):
        start = i * hop_size
        end = min(start + frame_size, n)
        frame = np.zeros(frame_size)
        frame[:end - start] = signal[start:end]
        
        # Apply window and compute power spectrum
        windowed = frame * window
        spectrum = np.abs(fft(windowed)[:frame_size // 2 + 1]) ** 2
        power_spectra.append(spectrum)
    
    power_spectra = np.array(power_spectra)
    
    # Identify low-energy frames (likely noise-only)
    frame_energies = np.sum(power_spectra, axis=1)
    threshold = np.percentile(frame_energies, WIENER_NOISE_EST_PERCENTILE)
    noise_frames = power_spectra[frame_energies <= threshold]
    
    if len(noise_frames) == 0:
        # Fallback: use minimum across all frames
        return np.min(power_spectra, axis=0)
    
    # Average noise spectrum from low-energy frames
    noise_spectrum = np.mean(noise_frames, axis=0)
    
    return noise_spectrum


def _spectral_subtraction(
    signal: NDArray[np.float64],
    noise_spectrum: NDArray[np.float64],
    alpha: float = SPECTRAL_SUBTRACTION_ALPHA,
    beta: float = SPECTRAL_SUBTRACTION_BETA,
) -> NDArray[np.float64]:
    """
    Spectral subtraction for noise reduction.
    
    Mathematical Background:
    Given noisy signal Y(f) = S(f) + N(f), estimate clean signal as:
    
        |Ŝ(f)|² = max( |Y(f)|² - α|N̂(f)|², β|Y(f)|² )
    
    where:
    - α is over-subtraction factor (reduces musical noise)
    - β is spectral floor (prevents negative power)
    
    Phase is preserved from original signal.
    """
    n = len(signal)
    
    # Zero-pad to next power of 2 for efficient FFT
    nfft = 1 << (n - 1).bit_length()
    
    # Compute spectrum
    X = fft(signal, nfft)
    X_mag = np.abs(X)
    X_phase = np.angle(X)
    X_power = X_mag ** 2
    
    # Resize noise spectrum to match FFT size
    noise_power = np.zeros(nfft)
    noise_len = min(len(noise_spectrum), nfft // 2 + 1)
    noise_power[:noise_len] = noise_spectrum[:noise_len]
    noise_power[nfft - noise_len + 1:] = noise_spectrum[1:noise_len][::-1]
    
    # Spectral subtraction with flooring
    clean_power = np.maximum(
        X_power - alpha * noise_power,
        beta * X_power
    )
    
    # Reconstruct with original phase
    clean_mag = np.sqrt(clean_power)
    Y = clean_mag * np.exp(1j * X_phase)
    
    # Inverse FFT and truncate
    cleaned = np.real(ifft(Y))[:n]
    
    return cleaned


def _wiener_filter(
    signal: NDArray[np.float64],
    noise_power: float,
) -> NDArray[np.float64]:
    """
    Wiener filter for optimal noise suppression.
    
    Mathematical Background:
    The Wiener filter minimizes MSE between estimated and true signal:
    
        H(f) = |S(f)|² / (|S(f)|² + |N(f)|²)
    
    Approximated as:
        H(f) = max(1 - σ²_n / |X(f)|², 0)
    
    This is optimal in the least-squares sense for stationary signals.
    """
    n = len(signal)
    nfft = 1 << (n - 1).bit_length()
    
    X = fft(signal, nfft)
    X_power = np.abs(X) ** 2
    
    # Wiener gain (SNR-based)
    # H = max(1 - noise_power / signal_power, 0)
    H = np.maximum(1 - noise_power / (X_power + 1e-10), 0)
    
    # Apply filter
    Y = X * H
    
    return np.real(ifft(Y))[:n]


def _light_denoise(
    signal: NDArray[np.float64],
    sample_rate: int,
) -> NDArray[np.float64]:
    """
    Light denoising for signals with no missing samples.
    Uses gentle Savitzky-Golay filtering.
    """
    n = len(signal)
    if n < 7:
        return np.clip(signal, -1.0, 1.0)
    
    # Light smoothing
    smoothed = sp_signal.savgol_filter(signal, 7, 3, mode="mirror")
    
    # Blend: mostly original
    out = 0.9 * signal + 0.1 * smoothed
    
    return np.clip(out, -1.0, 1.0)


# ═══════════════════════════════════════════════════════════════════
# Stage 2: Damage Analysis & Segmentation
# ═══════════════════════════════════════════════════════════════════

def _analyze_gaps(
    missing_mask: NDArray[np.bool_],
    signal: NDArray[np.float64],
    sample_rate: int,
) -> List[Dict]:
    """
    Analyze gaps and classify damage types.
    
    Returns list of gap descriptors with:
    - start: start index
    - length: gap length in samples
    - type: 'dropout', 'glitch', 'clipping'
    - context: surrounding signal characteristics
    """
    gaps = []
    
    # Find contiguous gaps
    starts, lengths = _find_gaps(missing_mask)
    
    for start, length in zip(starts, lengths):
        end = start + length
        
        # Analyze context (samples before/after gap)
        ctx_start = max(0, start - 20)
        ctx_end = min(len(signal), end + 20)
        
        before_ctx = signal[ctx_start:start] if start > ctx_start else np.array([])
        after_ctx = signal[end:ctx_end] if end < ctx_end else np.array([])
        
        # Classify gap type
        gap_type = 'dropout'  # default
        
        # Check for clipping context
        if len(before_ctx) > 0 and len(after_ctx) > 0:
            max_before = np.max(np.abs(before_ctx)) if len(before_ctx) > 0 else 0
            max_after = np.max(np.abs(after_ctx)) if len(after_ctx) > 0 else 0
            
            if max_before > CLIPPING_THRESHOLD or max_after > CLIPPING_THRESHOLD:
                gap_type = 'clipping_related'
        
        # Estimate local frequency content
        local_freq = _estimate_local_frequency(
            signal, sample_rate, max(0, start - 100), min(len(signal), end + 100)
        )
        
        gaps.append({
            'start': start,
            'length': length,
            'end': end,
            'type': gap_type,
            'local_freq': local_freq,
            'context_rms': np.sqrt(np.mean(
                np.concatenate([before_ctx, after_ctx]) ** 2
            )) if len(before_ctx) + len(after_ctx) > 0 else 0.1
        })
    
    return gaps


def _estimate_local_frequency(
    signal: NDArray[np.float64],
    sample_rate: int,
    start: int,
    end: int,
) -> float:
    """
    Estimate dominant frequency in a local region using zero-crossing rate.
    
    Formula: f ≈ (ZCR × sample_rate) / 2
    """
    segment = signal[start:end]
    n = len(segment)
    
    if n < 10:
        return 440.0  # default to A4
    
    # Zero-crossing rate
    zero_crossings = np.sum(np.abs(np.diff(np.signbit(segment))))
    zcr = zero_crossings / (n - 1)
    
    # Estimated frequency
    freq = zcr * sample_rate / 2
    
    # Clamp to reasonable audio range
    return np.clip(freq, 20, sample_rate / 2)


def _detect_clipping_regions(
    signal: NDArray[np.float64],
    valid_mask: NDArray[np.bool_],
) -> List[Dict]:
    """
    Detect regions where signal is clipped (saturated at threshold).
    
    Clipping manifests as flat peaks at ±threshold.
    These regions need special reconstruction to recover the peak shape.
    """
    n = len(signal)
    regions = []
    
    # Find samples at clipping threshold
    clipped_high = signal >= CLIPPING_THRESHOLD
    clipped_low = signal <= -CLIPPING_THRESHOLD
    clipped = clipped_high | clipped_low
    
    # Find contiguous clipped regions
    if not np.any(clipped):
        return regions
    
    starts, lengths = _find_gaps(clipped)
    
    for start, length in zip(starts, lengths):
        if length >= CLIPPING_MIN_SAMPLES:
            end = start + length
            polarity = 1 if np.mean(signal[start:end]) > 0 else -1
            
            regions.append({
                'start': start,
                'end': end,
                'length': length,
                'polarity': polarity,
                'original_peak': signal[start] if valid_mask[start] else CLIPPING_THRESHOLD * polarity
            })
    
    return regions


# ═══════════════════════════════════════════════════════════════════
# Stage 3: Model-Based Reconstruction (Sinusoidal Modeling)
# ═══════════════════════════════════════════════════════════════════

def _estimate_harmonic_model(
    time: NDArray[np.float64],
    signal: NDArray[np.float64],
    sample_rate: int,
    max_harmonics: int = MAX_HARMONICS,
) -> Optional[Dict]:
    """
    Estimate harmonic model parameters using FFT peak detection.
    
    Mathematical Background:
    Model signal as sum of sinusoids:
    
        x(t) = Σ_k A_k sin(2π f_k t + φ_k)
    
    where (A_k, f_k, φ_k) are amplitude, frequency, phase of k-th harmonic.
    
    Estimation:
    1. Compute FFT and find magnitude peaks
    2. Refine frequency estimates via parabolic interpolation
    3. Estimate phase from complex FFT coefficients
    4. Fit amplitudes via least squares
    """
    n = len(signal)
    if n < 64:
        return None
    
    # Zero-pad for better frequency resolution
    nfft = max(512, 1 << (n - 1).bit_length())
    
    # Window and compute FFT
    window = np.hanning(n)
    windowed = signal * window
    X = fft(windowed, nfft)
    
    # Magnitude spectrum (positive frequencies only)
    mag = np.abs(X[:nfft // 2])
    freqs = fftfreq(nfft, 1 / sample_rate)[:nfft // 2]
    
    # Find peaks in magnitude spectrum
    # Peak: local maximum above threshold
    threshold = np.max(mag) * MIN_HARMONIC_MAGNITUDE
    peaks = []
    
    for i in range(1, len(mag) - 1):
        if mag[i] > mag[i-1] and mag[i] > mag[i+1] and mag[i] > threshold:
            # Parabolic interpolation for sub-bin frequency accuracy
            alpha = mag[i-1]
            beta = mag[i]
            gamma = mag[i+1]
            
            p = 0.5 * (alpha - gamma) / (alpha - 2*beta + gamma + 1e-10)
            
            refined_freq = freqs[i] + p * (freqs[1] - freqs[0])
            refined_mag = beta - 0.25 * (alpha - gamma) * p
            phase = np.angle(X[i])
            
            peaks.append({
                'freq': refined_freq,
                'amplitude': refined_mag * 2 / n,  # Scale factor
                'phase': phase
            })
    
    # Sort by magnitude and keep top harmonics
    peaks.sort(key=lambda x: x['amplitude'], reverse=True)
    harmonics = peaks[:max_harmonics]
    
    if len(harmonics) == 0:
        return None
    
    # Estimate fundamental (lowest strong frequency)
    freqs_sorted = sorted([h['freq'] for h in harmonics])
    fundamental = freqs_sorted[0] if freqs_sorted[0] > 20 else 100
    
    return {
        'fundamental': fundamental,
        'harmonics': harmonics,
        'sample_rate': sample_rate
    }


def _synthesize_from_model(
    time: NDArray[np.float64],
    model: Dict,
    amplitude_envelope: Optional[NDArray[np.float64]] = None,
) -> NDArray[np.float64]:
    """
    Synthesize signal from harmonic model.
    
    Formula:
        x(t) = Σ_k A_k sin(2π f_k t + φ_k) × envelope(t)
    """
    if model is None or len(model['harmonics']) == 0:
        return np.zeros_like(time)
    
    signal = np.zeros_like(time)
    
    for harmonic in model['harmonics']:
        freq = harmonic['freq']
        amp = harmonic['amplitude']
        phase = harmonic['phase']
        
        signal += amp * np.sin(2 * np.pi * freq * time + phase)
    
    # Apply amplitude envelope if provided
    if amplitude_envelope is not None:
        signal *= amplitude_envelope
    
    return signal


def _model_guided_interpolation(
    time: NDArray[np.float64],
    signal: NDArray[np.float64],
    mask: NDArray[np.bool_],
    gap_start: int,
    gap_end: int,
    model: Dict,
) -> NDArray[np.float64]:
    """
    Fill gap using harmonic model with residual interpolation.
    
    Strategy:
    1. Synthesize model prediction for gap region
    2. Compute residual at gap boundaries
    3. Interpolate residual through gap
    4. Combine model + interpolated residual
    """
    gap_time = time[gap_start:gap_end]
    gap_len = gap_end - gap_start
    
    # Synthesize model for gap
    model_signal = _synthesize_from_model(gap_time, model)
    
    # Get boundary values from original signal
    before_idx = max(0, gap_start - 1)
    after_idx = min(len(signal) - 1, gap_end)
    
    # Compute residuals at boundaries
    before_val = signal[before_idx]
    after_val = signal[after_idx]
    
    before_model = _synthesize_from_model(time[before_idx:before_idx+1], model)[0]
    after_model = _synthesize_from_model(time[after_idx:after_idx+1], model)[0]
    
    residual_before = before_val - before_model
    residual_after = after_val - after_model
    
    # Linear interpolation of residual
    t_norm = np.linspace(0, 1, gap_len)
    residual_interp = residual_before * (1 - t_norm) + residual_after * t_norm
    
    # Combine model and residual
    filled = model_signal + residual_interp
    
    # Apply Hann fade at edges for smooth transition
    if gap_len >= 8:
        fade_len = min(gap_len // 4, 16)
        fade_in = np.hanning(2 * fade_len)[:fade_len]
        fade_out = np.hanning(2 * fade_len)[fade_len:]
        
        # Blend with boundary values at edges
        filled[:fade_len] = fade_in * filled[:fade_len] + (1 - fade_in) * before_val
        filled[-fade_len:] = fade_out * filled[-fade_len:] + (1 - fade_out) * after_val
    
    return filled


# ═══════════════════════════════════════════════════════════════════
# Stage 4: Adaptive Interpolation with Regularization
# ═══════════════════════════════════════════════════════════════════

def _adaptive_interpolate(
    time_axis: NDArray[np.float64],
    signal: NDArray[np.float64],
    valid_mask: NDArray[np.bool_],
    gaps: List[Dict],
    harmonic_model: Optional[Dict],
    method: str,
    use_tikhonov: bool,
    sample_rate: int,
) -> NDArray[np.float64]:
    """
    Adaptive interpolation based on gap size and signal characteristics.
    
    Strategy by gap size:
    - Tiny (≤3): Linear interpolation
    - Small (4-10): PCHIP
    - Medium (11-50): PCHIP with moving average blend
    - Large (51-200): Regularized spline or PCHIP
    - Very large (>200): Model-guided interpolation
    """
    reconstructed = signal.copy()
    n = len(signal)
    
    valid_t = time_axis[valid_mask]
    valid_y = signal[valid_mask]
    
    if len(valid_t) < 2:
        return reconstructed
    
    # Create interpolators
    linear_fn = sp_interp.interp1d(
        valid_t, valid_y, kind='linear',
        bounds_error=False, fill_value='extrapolate'
    )
    
    pchip_fn = sp_interp.PchipInterpolator(valid_t, valid_y, extrapolate=True)
    
    # Create regularized spline if enabled
    if use_tikhonov and len(valid_t) >= 10:
        spline_fn = _create_regularized_spline(valid_t, valid_y, TIKHONOV_LAMBDA)
    else:
        spline_fn = sp_interp.CubicSpline(
            valid_t, valid_y, bc_type='natural', extrapolate=True
        ) if len(valid_t) >= 4 else pchip_fn
    
    # Process each gap
    for gap in gaps:
        start = gap['start']
        end = gap['end']
        length = gap['length']
        gap_time = time_axis[start:end]
        
        if length <= GAP_TINY:
            # Tiny gap: linear interpolation
            filled = linear_fn(gap_time)
            
        elif length <= GAP_SMALL:
            # Small gap: PCHIP
            filled = pchip_fn(gap_time)
            
        elif length <= GAP_MEDIUM:
            # Medium gap: PCHIP with smoothing
            filled = pchip_fn(gap_time)
            # Apply light smoothing
            if len(filled) >= 5:
                win = min(5, len(filled))
                win = win if win % 2 == 1 else win - 1
                smoothed = uniform_filter1d(filled, size=win)
                filled = 0.7 * filled + 0.3 * smoothed
                
        elif length <= GAP_LARGE:
            # Large gap: regularized spline
            filled = spline_fn(gap_time)
            # Post-filter for stability
            if len(filled) >= 9:
                filled = sp_signal.savgol_filter(filled, 9, 3, mode="mirror")
                
        else:
            # Very large gap: model-guided interpolation
            if harmonic_model is not None and len(harmonic_model['harmonics']) > 0:
                filled = _model_guided_interpolation(
                    time_axis, signal, valid_mask, start, end, harmonic_model
                )
            else:
                # Fallback to regularized spline with heavy smoothing
                filled = spline_fn(gap_time)
                if len(filled) >= 15:
                    filled = sp_signal.savgol_filter(filled, 15, 3, mode="mirror")
        
        # Ensure filled values are valid
        filled = np.clip(np.nan_to_num(filled, nan=0.0), -1.0, 1.0)
        reconstructed[start:end] = filled
    
    # Apply edge blending for smooth transitions
    reconstructed = _blend_gap_edges(reconstructed, valid_mask, margin=12)
    
    return reconstructed


def _create_regularized_spline(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    lambda_reg: float,
) -> sp_interp.CubicSpline:
    """
    Create cubic spline with Tikhonov regularization.
    
    Mathematical Background:
    Standard spline minimizes: Σ(y_i - S(x_i))² + λ ∫ S''(x)² dx
    
    The regularization term λ penalizes curvature, preventing oscillations
    in noisy data. This is solved via tridiagonal system.
    
    For simplicity, we use smoothing spline approximation:
    Blend between data points and smoothed version.
    """
    n = len(x)
    
    if n < 4:
        return sp_interp.interp1d(x, y, kind='linear', fill_value='extrapolate')
    
    # Compute smoothed version using Savitzky-Golay
    win = min(11, n if n % 2 == 1 else n - 1)
    win = max(5, win)
    y_smooth = sp_signal.savgol_filter(y, win, 3, mode="mirror")
    
    # Regularized fit: blend original and smoothed based on lambda
    # Higher lambda = more smoothing
    blend = min(0.5, lambda_reg * 1000)  # Scale lambda to blend factor
    y_reg = (1 - blend) * y + blend * y_smooth
    
    # Create spline on regularized data
    try:
        spline = sp_interp.CubicSpline(x, y_reg, bc_type='natural', extrapolate=True)
    except ValueError:
        spline = sp_interp.PchipInterpolator(x, y_reg, extrapolate=True)
    
    return spline


def _repair_clipping(
    signal: NDArray[np.float64],
    clipping_regions: List[Dict],
    time_axis: NDArray[np.float64],
    sample_rate: int,
) -> NDArray[np.float64]:
    """
    Repair clipped regions by extrapolating the peak shape.
    
    Strategy:
    1. Estimate the slope at clip boundaries
    2. Fit parabolic peak shape
    3. Replace clipped samples with estimated peak
    """
    repaired = signal.copy()
    
    for region in clipping_regions:
        start = region['start']
        end = region['end']
        length = region['length']
        polarity = region['polarity']
        
        if length < 3 or length > 100:
            continue  # Skip very short or very long clipping
        
        # Get context samples
        ctx_before = max(0, start - 5)
        ctx_after = min(len(signal), end + 5)
        
        # Estimate derivatives at boundaries
        if start > 0:
            slope_before = signal[start] - signal[start - 1]
        else:
            slope_before = 0
            
        if end < len(signal):
            slope_after = signal[end] - signal[end - 1] if end > 0 else 0
        else:
            slope_after = 0
        
        # Fit parabolic peak: y = a*x² + b*x + c
        # Peak at center of clipped region
        center = length // 2
        
        # Estimate peak amplitude (extrapolate from slopes)
        # Assume symmetric parabola: peak = clip_value + margin
        margin = abs(slope_before) * length / 4
        peak_amp = min(1.0, CLIPPING_THRESHOLD + margin) * polarity
        
        # Generate parabolic shape
        t = np.arange(length) - center
        # Parabola: peak at center, clip_value at edges
        scale = (CLIPPING_THRESHOLD * polarity - peak_amp) / (center ** 2 + 1e-10)
        parabola = peak_amp + scale * (t ** 2)
        
        # Blend with original clipped values at edges
        if length >= 6:
            fade_len = min(length // 3, 4)
            fade_in = np.linspace(0, 1, fade_len)
            fade_out = np.linspace(1, 0, fade_len)
            
            parabola[:fade_len] = fade_in * parabola[:fade_len] + (1 - fade_in) * signal[start:start + fade_len]
            parabola[-fade_len:] = fade_out * parabola[-fade_len:] + (1 - fade_out) * signal[end - fade_len:end]
        
        repaired[start:end] = np.clip(parabola, -1.0, 1.0)
    
    return repaired


def _blend_gap_edges(
    signal: NDArray[np.float64],
    valid_mask: NDArray[np.bool_],
    margin: int = 12,
) -> NDArray[np.float64]:
    """
    Apply Hann-windowed cross-fade at gap boundaries.
    
    Eliminates discontinuities that cause audible clicks.
    """
    out = signal.copy()
    n = len(out)
    
    if n < margin * 2:
        return out
    
    # Find gap boundaries
    missing = ~valid_mask
    starts, lengths = _find_gaps(missing)
    
    if len(starts) == 0:
        return out
    
    # Create blend weight mask
    blend_weight = np.zeros(n)
    
    for start, length in zip(starts, lengths):
        end = start + length
        
        if length < 4:
            continue
        
        # Mark boundary regions
        blend_start = max(0, start - margin // 2)
        blend_end_start = min(n, end + margin // 2)
        
        # Gradual weight at start boundary
        if start > margin // 2:
            idx = np.arange(blend_start, min(start + margin // 2, n))
            if len(idx) > 0:
                weight = 0.5 * (1 - np.cos(np.pi * np.linspace(0, 1, len(idx))))
                blend_weight[idx] = np.maximum(blend_weight[idx], weight)
        
        # Gradual weight at end boundary  
        if end < n - margin // 2:
            idx = np.arange(max(0, end - margin // 2), blend_end_start)
            if len(idx) > 0:
                weight = 0.5 * (1 - np.cos(np.pi * np.linspace(1, 0, len(idx))))
                blend_weight[idx] = np.maximum(blend_weight[idx], weight)
    
    # Smooth blend weight
    if np.any(blend_weight > 0):
        blend_weight = uniform_filter1d(blend_weight, size=margin // 2)
        blend_weight = np.clip(blend_weight, 0, 1)
        
        # Blend with smoothed version at boundaries
        smoothed = uniform_filter1d(out, size=margin)
        out = (1 - blend_weight) * out + blend_weight * smoothed
    
    return out


# ═══════════════════════════════════════════════════════════════════
# Stage 5: Perceptual Post-Processing
# ═══════════════════════════════════════════════════════════════════

def _perceptual_postprocess(
    signal: NDArray[np.float64],
    missing_mask: NDArray[np.bool_],
    sample_rate: int,
) -> NDArray[np.float64]:
    """
    Perceptual post-processing for natural sound quality.
    
    Stages:
    1. Low-pass filter (removes HF artifacts)
    2. Envelope normalization (consistent loudness)
    3. Dynamic range compression (prevents peaks)
    4. Soft clipping (tanh limiter)
    """
    out = signal.copy()
    n = len(out)
    
    if n < 16:
        return np.clip(out, -1.0, 1.0)
    
    # ─── Stage 1: Low-pass filter ────────────────────────────
    out = _apply_lowpass(out, missing_mask, sample_rate)
    
    # ─── Stage 2: Envelope normalization ─────────────────────
    out = _envelope_normalize(out, missing_mask)
    
    # ─── Stage 3: Dynamic range compression ──────────────────
    out = _dynamic_range_compress(out)
    
    # ─── Stage 4: Soft clipping ──────────────────────────────
    out = _soft_clip(out, SOFT_CLIP_THRESH)
    
    return out


def _apply_lowpass(
    signal: NDArray[np.float64],
    missing_mask: NDArray[np.bool_],
    sample_rate: int,
) -> NDArray[np.float64]:
    """
    Apply Butterworth low-pass filter to filled regions.
    
    Cutoff at 0.45 × Nyquist removes interpolation artifacts
    without audible dulling.
    """
    out = signal.copy()
    n = len(out)
    
    if n < 16 or not np.any(missing_mask):
        return out
    
    try:
        # Design filter
        sos = sp_signal.butter(
            POST_LP_ORDER,
            POST_LP_CUTOFF,
            btype='low',
            output='sos'
        )
        
        # Zero-phase filtering
        filtered = sp_signal.sosfiltfilt(sos, out)
        
        # Apply only to filled regions
        out[missing_mask] = filtered[missing_mask]
        
    except Exception:
        # Fallback: Savitzky-Golay
        if n >= 11:
            smoothed = sp_signal.savgol_filter(out, 11, 3, mode="mirror")
            out[missing_mask] = smoothed[missing_mask]
    
    return out


def _envelope_normalize(
    signal: NDArray[np.float64],
    missing_mask: NDArray[np.bool_],
) -> NDArray[np.float64]:
    """
    Envelope-based amplitude normalization.
    
    Prevents energy discontinuities at gap boundaries by
    matching local amplitude to surrounding context.
    """
    out = signal.copy()
    n = len(out)
    
    if n < ENVELOPE_WINDOW:
        return out
    
    # Compute envelope via Hilbert transform magnitude
    envelope = np.abs(signal)
    envelope = uniform_filter1d(envelope, size=ENVELOPE_WINDOW)
    envelope = np.maximum(envelope, 1e-10)
    
    # Target envelope (global smoothed)
    target = uniform_filter1d(envelope, size=ENVELOPE_WINDOW * 2)
    target = np.maximum(target, 1e-10)
    
    # Gain to match target
    gain = target / envelope
    gain = np.clip(gain, 0.7, 1.4)  # ±3 dB max
    
    # Smooth gain curve
    gain = uniform_filter1d(gain, size=ENVELOPE_WINDOW // 2)
    gain = np.clip(gain, 0.7, 1.4)
    
    # Apply gain to filled regions
    blend = np.zeros(n)
    blend[missing_mask] = 1.0
    blend = uniform_filter1d(blend, size=32)
    
    normalized = out * gain
    out = (1 - blend) * out + blend * normalized
    
    return out


def _dynamic_range_compress(
    signal: NDArray[np.float64],
    threshold: float = COMPRESSION_THRESHOLD,
    ratio: float = COMPRESSION_RATIO,
) -> NDArray[np.float64]:
    """
    Soft-knee dynamic range compression.
    
    Mathematical Background:
    For |x| > threshold:
        y = sign(x) * (threshold + (|x| - threshold) / ratio)
    
    This reduces peaks while preserving dynamics below threshold.
    """
    out = signal.copy()
    
    # Find samples above threshold
    above = np.abs(out) > threshold
    
    if not np.any(above):
        return out
    
    # Apply compression
    excess = np.abs(out[above]) - threshold
    compressed = threshold + excess / ratio
    
    out[above] = np.sign(out[above]) * compressed
    
    return out


def _soft_clip(
    signal: NDArray[np.float64],
    threshold: float = 0.95,
) -> NDArray[np.float64]:
    """
    Soft clipping using tanh saturation.
    
    Mathematical Background:
    For |x| > threshold:
        y = threshold + (1 - threshold) * tanh((|x| - threshold) / (1 - threshold))
    
    This provides gentle saturation instead of hard clipping.
    """
    out = signal.copy()
    
    above_thresh = np.abs(out) > threshold
    
    if not np.any(above_thresh):
        return np.clip(out, -1.0, 1.0)
    
    scale = 1.0 - threshold
    
    positive = out > threshold
    negative = out < -threshold
    
    out[positive] = threshold + scale * np.tanh(
        (out[positive] - threshold) / scale
    )
    out[negative] = -threshold - scale * np.tanh(
        (-out[negative] - threshold) / scale
    )
    
    return out


# ═══════════════════════════════════════════════════════════════════
# Utility Functions
# ═══════════════════════════════════════════════════════════════════

def _find_gaps(
    missing_mask: NDArray[np.bool_],
) -> Tuple[NDArray[np.intp], NDArray[np.intp]]:
    """
    Find contiguous runs of True (missing/damaged) samples.
    
    Returns
    -------
    starts  : 1-D int array - index where each gap begins
    lengths : 1-D int array - length of each gap
    """
    if not np.any(missing_mask):
        return np.array([], dtype=np.intp), np.array([], dtype=np.intp)
    
    padded = np.concatenate(([False], missing_mask, [False]))
    d = np.diff(padded.astype(np.int8))
    starts = np.where(d == 1)[0]
    ends = np.where(d == -1)[0]
    
    return starts, ends - starts


# ═══════════════════════════════════════════════════════════════════
# Comparison and Metrics
# ═══════════════════════════════════════════════════════════════════

def compute_reconstruction_metrics(
    original: NDArray[np.float64],
    reconstructed: NDArray[np.float64],
) -> Dict[str, float]:
    """
    Compute comprehensive reconstruction quality metrics.
    
    Metrics:
    - MSE: Mean Squared Error
    - RMSE: Root Mean Squared Error
    - SNR: Signal-to-Noise Ratio (dB)
    - MAE: Mean Absolute Error
    - Correlation: Pearson correlation coefficient
    """
    # Ensure same length
    n = min(len(original), len(reconstructed))
    orig = original[:n]
    recon = reconstructed[:n]
    
    # Remove any NaN/Inf
    valid = np.isfinite(orig) & np.isfinite(recon)
    orig = orig[valid]
    recon = recon[valid]
    
    if len(orig) == 0:
        return {'mse': np.inf, 'rmse': np.inf, 'snr_db': 0, 'mae': np.inf, 'correlation': 0}
    
    # Error
    error = orig - recon
    
    # MSE and RMSE
    mse = float(np.mean(error ** 2))
    rmse = float(np.sqrt(mse))
    
    # MAE
    mae = float(np.mean(np.abs(error)))
    
    # SNR (dB)
    signal_power = float(np.mean(orig ** 2))
    if mse > 0 and signal_power > 0:
        snr_db = float(10 * np.log10(signal_power / mse))
    else:
        snr_db = float('inf') if mse == 0 else 0.0
    
    # Correlation
    if np.std(orig) > 1e-10 and np.std(recon) > 1e-10:
        correlation = float(np.corrcoef(orig, recon)[0, 1])
    else:
        correlation = 1.0 if np.allclose(orig, recon) else 0.0
    
    return {
        'mse': round(mse, 8),
        'rmse': round(rmse, 8),
        'snr_db': round(snr_db, 2),
        'mae': round(mae, 8),
        'correlation': round(correlation, 4)
    }


def compare_methods(
    time_axis: NDArray[np.float64],
    original: NDArray[np.float64],
    spoiled: NDArray[np.float64],
    mask: NDArray[np.bool_],
    sample_rate: int = 8000,
) -> Dict[str, Dict]:
    """
    Compare reconstruction quality across all methods.
    
    Returns dictionary with metrics for each method.
    """
    from interpolation import reconstruct as baseline_reconstruct
    
    results = {}
    
    methods = ['linear', 'pchip', 'spline', 'moving_average']
    
    for method in methods:
        # Baseline reconstruction
        baseline = baseline_reconstruct(time_axis, spoiled, mask, method=method)
        baseline_metrics = compute_reconstruction_metrics(original, baseline)
        
        # Advanced reconstruction
        advanced = advanced_reconstruct(
            time_axis, spoiled, mask, sample_rate, method=method
        )
        advanced_metrics = compute_reconstruction_metrics(original, advanced)
        
        results[method] = {
            'baseline': baseline_metrics,
            'advanced': advanced_metrics,
            'improvement': {
                'snr_db': advanced_metrics['snr_db'] - baseline_metrics['snr_db'],
                'mse_reduction': (baseline_metrics['mse'] - advanced_metrics['mse']) / (baseline_metrics['mse'] + 1e-10) * 100
            }
        }
    
    return results
