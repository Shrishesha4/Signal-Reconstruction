"""
damaged_audio.py – Damaged Audio Detection and Repair
====================================================
Detects corrupted/missing regions in audio files and reconstructs them
using interpolation algorithms.

Detection methods:
  - Zero crossing density anomalies
  - Amplitude discontinuities
  - Silence/dropout detection
  - Clipping detection
  - DC offset / baseline drift

All reconstruction uses the scipy-backed interpolation from interpolation.py
"""

import numpy as np
from numpy.typing import NDArray
from typing import Tuple, Dict, List, Optional
from scipy import signal as sp_signal
from scipy.ndimage import uniform_filter1d, median_filter

from interpolation import reconstruct as _interpolation_reconstruct


# ═══════════════════════════════════════════════════════════════════
# Detection Parameters
# ═══════════════════════════════════════════════════════════════════

# Silence/dropout detection
_SILENCE_THRESHOLD = 0.001       # Amplitude below this is "silent"
_MIN_DROPOUT_LENGTH = 10         # Minimum consecutive samples for dropout
_MAX_NATURAL_SILENCE_MS = 100    # Longer silence considered dropout

# Amplitude discontinuity
_DISCONTINUITY_THRESHOLD = 0.5   # Jump exceeding this is suspicious
_DISCONTINUITY_WINDOW = 5        # Samples to check around jump

# Clipping detection
_CLIP_THRESHOLD = 0.99           # Values above this may be clipped
_MIN_CLIP_LENGTH = 3             # Minimum samples at max for clipping

# Zero crossing anomaly
_ZC_BLOCK_SIZE = 256             # Block size for zero-crossing rate
_ZC_ANOMALY_FACTOR = 3.0         # Factor above/below median is anomaly


# ═══════════════════════════════════════════════════════════════════
# Damage Detection
# ═══════════════════════════════════════════════════════════════════

def detect_dropouts(
    signal: np.ndarray,
    sample_rate: int,
    silence_threshold: float = _SILENCE_THRESHOLD,
    min_length: int = _MIN_DROPOUT_LENGTH,
) -> List[Tuple[int, int]]:
    """
    Detect dropout regions (consecutive near-zero samples).
    
    Returns:
        List of (start_idx, end_idx) tuples for dropout regions
    """
    n = len(signal)
    dropouts = []
    
    # Find samples below threshold
    is_silent = np.abs(signal) < silence_threshold
    
    # Find runs of silence
    in_dropout = False
    dropout_start = 0
    
    for i in range(n):
        if is_silent[i]:
            if not in_dropout:
                in_dropout = True
                dropout_start = i
        else:
            if in_dropout:
                dropout_length = i - dropout_start
                # Check if long enough to be a dropout (not natural silence)
                max_natural = int(sample_rate * _MAX_NATURAL_SILENCE_MS / 1000)
                if dropout_length >= min_length and dropout_length < max_natural:
                    dropouts.append((dropout_start, i))
                in_dropout = False
    
    # Handle dropout at end of signal
    if in_dropout:
        dropout_length = n - dropout_start
        max_natural = int(sample_rate * _MAX_NATURAL_SILENCE_MS / 1000)
        if dropout_length >= min_length and dropout_length < max_natural:
            dropouts.append((dropout_start, n))
    
    return dropouts


def detect_clipping(
    signal: np.ndarray,
    clip_threshold: float = _CLIP_THRESHOLD,
    min_length: int = _MIN_CLIP_LENGTH,
) -> List[Tuple[int, int]]:
    """
    Detect clipped regions (flat lines at maximum amplitude).
    
    Returns:
        List of (start_idx, end_idx) tuples for clipped regions
    """
    n = len(signal)
    clipped = []
    
    # Find samples at or near maximum
    is_clipped = np.abs(signal) >= clip_threshold
    
    # Find runs of clipping
    in_clip = False
    clip_start = 0
    
    for i in range(n):
        if is_clipped[i]:
            if not in_clip:
                in_clip = True
                clip_start = i
        else:
            if in_clip:
                if i - clip_start >= min_length:
                    clipped.append((clip_start, i))
                in_clip = False
    
    if in_clip and n - clip_start >= min_length:
        clipped.append((clip_start, n))
    
    return clipped


def detect_discontinuities(
    signal: np.ndarray,
    threshold: float = _DISCONTINUITY_THRESHOLD,
) -> List[int]:
    """
    Detect amplitude discontinuities (sudden jumps).
    
    Returns:
        List of sample indices where discontinuities occur
    """
    # Compute first derivative
    diff = np.abs(np.diff(signal))
    
    # Find large jumps
    discontinuities = np.where(diff > threshold)[0]
    
    return discontinuities.tolist()


def detect_zero_crossing_anomalies(
    signal: np.ndarray,
    block_size: int = _ZC_BLOCK_SIZE,
) -> List[Tuple[int, int]]:
    """
    Detect regions with abnormal zero-crossing rate.
    
    Very high ZC rate suggests noise/corruption.
    Very low ZC rate in audio suggests missing modulation.
    
    Returns:
        List of (start_idx, end_idx) tuples for anomalous regions
    """
    n = len(signal)
    n_blocks = n // block_size
    
    if n_blocks < 3:
        return []
    
    # Calculate zero-crossing rate per block
    zc_rates = []
    for i in range(n_blocks):
        start = i * block_size
        end = start + block_size
        block = signal[start:end]
        
        # Count zero crossings
        zc = np.sum(np.abs(np.diff(np.sign(block))) > 0)
        zc_rates.append(zc)
    
    zc_rates = np.array(zc_rates)
    median_zc = np.median(zc_rates)
    
    if median_zc < 1:
        return []
    
    # Find anomalous blocks
    anomalies = []
    for i in range(n_blocks):
        if zc_rates[i] > median_zc * _ZC_ANOMALY_FACTOR or \
           zc_rates[i] < median_zc / _ZC_ANOMALY_FACTOR:
            start = i * block_size
            end = start + block_size
            anomalies.append((start, end))
    
    # Merge adjacent anomalies
    if not anomalies:
        return []
    
    merged = [anomalies[0]]
    for start, end in anomalies[1:]:
        if start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    
    return merged


def detect_dc_offset_drift(
    signal: np.ndarray,
    window_size: int = 1024,
) -> Tuple[bool, float]:
    """
    Detect DC offset or baseline drift.
    
    Returns:
        (has_drift, estimated_offset)
    """
    # Calculate moving mean
    if len(signal) < window_size:
        mean_val = np.mean(signal)
        return abs(mean_val) > 0.05, mean_val
    
    moving_mean = uniform_filter1d(signal, size=window_size)
    
    # Check for significant drift
    drift_range = np.max(moving_mean) - np.min(moving_mean)
    mean_offset = np.mean(moving_mean)
    
    has_drift = drift_range > 0.1 or abs(mean_offset) > 0.05
    
    return has_drift, mean_offset


# ═══════════════════════════════════════════════════════════════════
# Combined Analysis
# ═══════════════════════════════════════════════════════════════════

def analyze_damage(
    signal: np.ndarray,
    sample_rate: int,
) -> Dict:
    """
    Comprehensive damage analysis of an audio signal.
    
    Returns:
        Dictionary with:
          - damage_mask: boolean array (True = valid, False = damaged)
          - damage_regions: list of (start, end, type) tuples
          - summary: text description of damage
          - damage_percent: percentage of signal that is damaged
    """
    n = len(signal)
    damage_mask = np.ones(n, dtype=bool)
    damage_regions = []
    
    # Detect dropouts
    dropouts = detect_dropouts(signal, sample_rate)
    for start, end in dropouts:
        damage_mask[start:end] = False
        damage_regions.append((start, end, "dropout"))
    
    # Detect clipping
    clipped = detect_clipping(signal)
    for start, end in clipped:
        damage_mask[start:end] = False
        damage_regions.append((start, end, "clipping"))
    
    # Detect discontinuities (mark small region around each)
    discontinuities = detect_discontinuities(signal)
    for idx in discontinuities:
        start = max(0, idx - 2)
        end = min(n, idx + 3)
        damage_mask[start:end] = False
        damage_regions.append((start, end, "discontinuity"))
    
    # Detect zero-crossing anomalies
    zc_anomalies = detect_zero_crossing_anomalies(signal)
    for start, end in zc_anomalies:
        damage_mask[start:end] = False
        damage_regions.append((start, end, "noise"))
    
    # Calculate statistics
    n_damaged = np.sum(~damage_mask)
    damage_percent = 100.0 * n_damaged / n
    
    # Build summary
    summary_parts = []
    dropout_count = len(dropouts)
    clip_count = len(clipped)
    disc_count = len(discontinuities)
    noise_count = len(zc_anomalies)
    
    if dropout_count > 0:
        summary_parts.append(f"{dropout_count} dropout region(s)")
    if clip_count > 0:
        summary_parts.append(f"{clip_count} clipped region(s)")
    if disc_count > 0:
        summary_parts.append(f"{disc_count} discontinuity point(s)")
    if noise_count > 0:
        summary_parts.append(f"{noise_count} noisy region(s)")
    
    if not summary_parts:
        summary = "No significant damage detected"
    else:
        summary = "Detected: " + ", ".join(summary_parts)
    
    return {
        "damage_mask": damage_mask,
        "damage_regions": damage_regions,
        "summary": summary,
        "damage_percent": round(damage_percent, 2),
        "stats": {
            "dropouts": dropout_count,
            "clipping": clip_count,
            "discontinuities": disc_count,
            "noise_regions": noise_count,
        }
    }


# ═══════════════════════════════════════════════════════════════════
# Repair / Reconstruction
# ═══════════════════════════════════════════════════════════════════

def repair_audio(
    signal: np.ndarray,
    sample_rate: int,
    damage_mask: Optional[np.ndarray] = None,
    method: str = "pchip",
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Repair damaged audio by detecting damage and interpolating.
    
    Args:
        signal: Input audio signal (normalized to [-1, 1])
        sample_rate: Sample rate in Hz
        damage_mask: Optional pre-computed damage mask (True = valid)
        method: Interpolation method ('pchip', 'spline', 'linear')
    
    Returns:
        (repaired_signal, damage_mask, analysis_info)
    """
    # Analyze damage if mask not provided
    if damage_mask is None:
        analysis = analyze_damage(signal, sample_rate)
        damage_mask = analysis["damage_mask"]
        info = analysis
    else:
        n_damaged = np.sum(~damage_mask)
        info = {
            "summary": f"Using provided mask ({n_damaged} damaged samples)",
            "damage_percent": round(100.0 * n_damaged / len(signal), 2),
            "damage_regions": [],
            "stats": {"custom_mask": True}
        }
    
    # Create time axis
    time_axis = np.arange(len(signal)) / sample_rate
    
    # Prepare signal for reconstruction
    # (interpolation expects damaged samples to be ~0)
    damaged_signal = signal.copy()
    damaged_signal[~damage_mask] = 0.0
    
    # Reconstruct
    repaired = _interpolation_reconstruct(time_axis, damaged_signal, damage_mask, method=method)
    
    return repaired, damage_mask, info


def apply_declip(
    signal: np.ndarray,
    clip_regions: List[Tuple[int, int]],
    method: str = "pchip",
) -> np.ndarray:
    """
    Specifically repair clipped regions using interpolation.
    
    For clipping, we need to extend the waveform beyond the clipped maximum.
    """
    if not clip_regions:
        return signal.copy()
    
    result = signal.copy()
    n = len(signal)
    
    for start, end in clip_regions:
        # Get context around clipped region
        context_size = min(50, start, n - end)
        
        if context_size < 10:
            # Not enough context, skip
            continue
        
        # Create mask for this region
        local_start = max(0, start - context_size)
        local_end = min(n, end + context_size)
        local_signal = signal[local_start:local_end]
        
        local_mask = np.ones(len(local_signal), dtype=bool)
        local_mask[start - local_start:end - local_start] = False
        
        # Time axis for local region
        local_time = np.arange(len(local_signal), dtype=np.float64)
        
        # Reconstruct
        local_repaired = _interpolation_reconstruct(
            local_time, local_signal, local_mask, method=method
        )
        
        # Insert repaired section
        result[start:end] = local_repaired[start - local_start:end - local_start]
    
    return result


# ═══════════════════════════════════════════════════════════════════
# High-Level API
# ═══════════════════════════════════════════════════════════════════

def process_damaged_audio(
    signal: np.ndarray,
    sample_rate: int,
    method: str = "pchip",
    auto_detect: bool = True,
    custom_mask: Optional[np.ndarray] = None,
) -> Dict:
    """
    Full pipeline for damaged audio processing.
    
    Args:
        signal: Audio signal (will be normalized if needed)
        sample_rate: Sample rate
        method: Interpolation method
        auto_detect: Whether to auto-detect damage
        custom_mask: Optional manual damage mask
    
    Returns:
        Dictionary with damaged, repaired signals and metadata
    """
    # Normalize if needed
    peak = np.max(np.abs(signal))
    if peak > 1.0:
        signal = signal / peak
    elif peak < 0.001:
        # Very quiet signal, amplify
        signal = signal / max(peak, 1e-6)
    
    # Determine damage mask
    if custom_mask is not None:
        damage_mask = custom_mask.astype(bool)
        analysis = {
            "summary": "Custom mask provided",
            "damage_percent": round(100.0 * np.sum(~damage_mask) / len(signal), 2),
            "damage_regions": [],
            "stats": {}
        }
    elif auto_detect:
        analysis = analyze_damage(signal, sample_rate)
        damage_mask = analysis["damage_mask"]
    else:
        # Assume no damage, return as-is
        return {
            "original": signal,
            "damaged": signal,
            "repaired": signal,
            "mask": np.ones(len(signal), dtype=bool),
            "analysis": {
                "summary": "No damage detection performed",
                "damage_percent": 0.0,
            }
        }
    
    # Store original (which is already damaged)
    damaged = signal.copy()
    
    # Repair
    time_axis = np.arange(len(signal)) / sample_rate
    
    # Zero out damaged regions for interpolation
    signal_for_interp = signal.copy()
    signal_for_interp[~damage_mask] = 0.0
    
    repaired = _interpolation_reconstruct(time_axis, signal_for_interp, damage_mask, method=method)
    
    # Clip output
    repaired = np.clip(repaired, -1.0, 1.0)
    
    return {
        "original": None,  # We don't have the original clean signal
        "damaged": damaged,
        "repaired": repaired,
        "mask": damage_mask,
        "analysis": analysis,
    }
