"""
scidata.py – Scientific Time-Series Data Processing
===================================================
Handles loading, preprocessing, and reconstruction of general scientific
time-series data (IoT sensors, ECG, radio signals, environmental data, etc.)

Supports:
  - CSV format: time,value or header-based
  - JSON format: {"time": [...], "values": [...]} or array of objects
  
All numerical reconstruction uses the same scipy-backed algorithms as audio.
"""

import json
import csv
import io
import numpy as np
from numpy.typing import NDArray
from typing import Tuple, Dict, Any, Optional, List

from interpolation import reconstruct as _interpolation_reconstruct


# ═══════════════════════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════════════════════

def load_csv_data(raw_bytes: bytes) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load time-series data from CSV bytes.
    
    Supported formats:
      - Two columns: time, value (with or without header)
      - Named columns: time/t/x, value/y/signal/data
    
    Returns:
        (time_array, value_array) as float64
    """
    text = raw_bytes.decode('utf-8', errors='ignore')
    lines = text.strip().split('\n')
    
    if not lines:
        raise ValueError("Empty CSV file")
    
    # Try to detect if first line is header
    reader = csv.reader(io.StringIO(text))
    rows = list(reader)
    
    if not rows:
        raise ValueError("No data rows found in CSV")
    
    # Check if first row looks like a header
    first_row = rows[0]
    has_header = False
    time_col = 0
    value_col = 1
    
    # Detect header and column indices
    if len(first_row) >= 2:
        header_candidates = ['time', 't', 'x', 'timestamp', 'index']
        value_candidates = ['value', 'y', 'signal', 'data', 'amplitude', 'reading']
        
        # Check if first row contains non-numeric strings
        try:
            float(first_row[0])
            float(first_row[1])
        except ValueError:
            # First row is likely a header
            has_header = True
            first_lower = [c.lower().strip() for c in first_row]
            
            # Find time column
            for i, col in enumerate(first_lower):
                if col in header_candidates:
                    time_col = i
                    break
            
            # Find value column
            for i, col in enumerate(first_lower):
                if col in value_candidates:
                    value_col = i
                    break
                elif i != time_col:
                    value_col = i  # Use first non-time column
    
    # Parse data rows
    data_rows = rows[1:] if has_header else rows
    
    if len(data_rows) < 2:
        raise ValueError("Need at least 2 data points")
    
    time_values = []
    signal_values = []
    
    for i, row in enumerate(data_rows):
        if len(row) < 2:
            continue
        try:
            t = float(row[time_col].strip())
            v = float(row[value_col].strip())
            time_values.append(t)
            signal_values.append(v)
        except (ValueError, IndexError):
            continue  # Skip malformed rows
    
    if len(time_values) < 2:
        raise ValueError("Could not parse enough valid data points")
    
    time_arr = np.array(time_values, dtype=np.float64)
    value_arr = np.array(signal_values, dtype=np.float64)
    
    # Sort by time
    sort_idx = np.argsort(time_arr)
    time_arr = time_arr[sort_idx]
    value_arr = value_arr[sort_idx]
    
    return time_arr, value_arr


def load_json_data(raw_bytes: bytes) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load time-series data from JSON bytes.
    
    Supported formats:
      - {"time": [...], "values": [...]}
      - {"time": [...], "data": [...]}
      - {"t": [...], "y": [...]}
      - [{"time": 0, "value": 1}, {"time": 1, "value": 2}, ...]
      - [[0, 1], [1, 2], ...]  (array of [time, value] pairs)
    
    Returns:
        (time_array, value_array) as float64
    """
    text = raw_bytes.decode('utf-8', errors='ignore')
    data = json.loads(text)
    
    time_values: List[float] = []
    signal_values: List[float] = []
    
    if isinstance(data, dict):
        # Find time array
        time_keys = ['time', 't', 'x', 'timestamp', 'times']
        value_keys = ['values', 'value', 'y', 'data', 'signal', 'readings', 'amplitude']
        
        time_arr = None
        value_arr = None
        
        for key in time_keys:
            if key in data and isinstance(data[key], list):
                time_arr = data[key]
                break
        
        for key in value_keys:
            if key in data and isinstance(data[key], list):
                value_arr = data[key]
                break
        
        if time_arr is None or value_arr is None:
            # Try to use first two list-like values
            lists = [(k, v) for k, v in data.items() if isinstance(v, list)]
            if len(lists) >= 2:
                time_arr = lists[0][1]
                value_arr = lists[1][1]
            else:
                raise ValueError("Could not find time and value arrays in JSON")
        
        time_values = [float(t) for t in time_arr]
        signal_values = [float(v) for v in value_arr]
        
    elif isinstance(data, list):
        if len(data) == 0:
            raise ValueError("Empty JSON array")
        
        # Check format: array of objects or array of arrays
        if isinstance(data[0], dict):
            # Array of objects
            time_keys = ['time', 't', 'x', 'timestamp']
            value_keys = ['value', 'y', 'signal', 'data', 'amplitude']
            
            time_key = None
            value_key = None
            
            for key in time_keys:
                if key in data[0]:
                    time_key = key
                    break
            
            for key in value_keys:
                if key in data[0]:
                    value_key = key
                    break
            
            if time_key is None or value_key is None:
                raise ValueError("Could not identify time and value keys in objects")
            
            for obj in data:
                time_values.append(float(obj[time_key]))
                signal_values.append(float(obj[value_key]))
                
        elif isinstance(data[0], (list, tuple)):
            # Array of [time, value] pairs
            for pair in data:
                if len(pair) >= 2:
                    time_values.append(float(pair[0]))
                    signal_values.append(float(pair[1]))
        else:
            raise ValueError("Unrecognized JSON array format")
    else:
        raise ValueError("JSON must be an object or array")
    
    if len(time_values) < 2:
        raise ValueError("Need at least 2 data points")
    
    time_arr_np = np.array(time_values, dtype=np.float64)
    value_arr_np = np.array(signal_values, dtype=np.float64)
    
    # Sort by time
    sort_idx = np.argsort(time_arr_np)
    time_arr_np = time_arr_np[sort_idx]
    value_arr_np = value_arr_np[sort_idx]
    
    return time_arr_np, value_arr_np


def load_scidata(raw_bytes: bytes, filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Auto-detect file format and load scientific data.
    
    Args:
        raw_bytes: File content
        filename: Original filename (for format detection)
    
    Returns:
        (time_array, value_array)
    """
    lower_name = filename.lower()
    
    if lower_name.endswith('.json'):
        return load_json_data(raw_bytes)
    elif lower_name.endswith('.csv'):
        return load_csv_data(raw_bytes)
    else:
        # Try JSON first, then CSV
        try:
            return load_json_data(raw_bytes)
        except (json.JSONDecodeError, ValueError):
            return load_csv_data(raw_bytes)


# ═══════════════════════════════════════════════════════════════════
# Preprocessing
# ═══════════════════════════════════════════════════════════════════

def normalize_signal(signal: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """
    Normalize signal to [-1, 1] range.
    
    Returns:
        (normalized_signal, original_min, original_max)
    """
    sig_min = np.min(signal)
    sig_max = np.max(signal)
    
    if sig_max - sig_min < 1e-10:
        # Constant signal
        return np.zeros_like(signal), sig_min, sig_max
    
    # Scale to [-1, 1]
    normalized = 2 * (signal - sig_min) / (sig_max - sig_min) - 1
    
    return normalized, sig_min, sig_max


def denormalize_signal(normalized: np.ndarray, sig_min: float, sig_max: float) -> np.ndarray:
    """Inverse of normalize_signal."""
    return (normalized + 1) / 2 * (sig_max - sig_min) + sig_min


def resample_to_uniform(time: np.ndarray, values: np.ndarray, 
                         target_points: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resample non-uniform time series to uniform spacing.
    
    Args:
        time: Original time array (may be non-uniform)
        values: Signal values
        target_points: Number of output points (default: same as input)
    
    Returns:
        (uniform_time, resampled_values)
    """
    if target_points is None:
        target_points = len(time)
    
    target_points = max(2, min(target_points, 50000))  # Limit size
    
    uniform_time = np.linspace(time[0], time[-1], target_points)
    
    # Use linear interpolation for resampling (preserves data character)
    resampled = np.interp(uniform_time, time, values)
    
    return uniform_time, resampled


# ═══════════════════════════════════════════════════════════════════
# Degradation (for demo/testing)
# ═══════════════════════════════════════════════════════════════════

def degrade_scidata(
    original: np.ndarray,
    dropout_pct: float = 20.0,
    noise_level: float = 0.02,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Degrade scientific signal with dropouts and noise.
    
    Same algorithm as audio degradation for consistency.
    
    Args:
        original: Clean normalized signal [-1, 1]
        dropout_pct: Percentage of samples to zero out (0-80)
        noise_level: Std deviation of Gaussian noise (0-0.5)
        seed: Random seed for reproducibility
    
    Returns:
        (spoiled_signal, mask)
        mask[i] = True means sample is valid
    """
    n = len(original)
    spoiled = original.copy()
    
    # Clamp parameters
    dropout_pct = np.clip(dropout_pct, 0.0, 80.0)
    noise_level = np.clip(noise_level, 0.0, 0.5)
    
    # Create dropout mask
    rng = np.random.default_rng(seed)
    n_drop = int(n * dropout_pct / 100.0)
    drop_indices = rng.choice(n, size=n_drop, replace=False)
    
    mask = np.ones(n, dtype=bool)
    mask[drop_indices] = False
    spoiled[drop_indices] = 0.0
    
    # Add noise
    if noise_level > 0:
        noise = rng.normal(0, noise_level, size=n)
        spoiled[mask] += noise[mask]
    
    spoiled = np.clip(spoiled, -1.0, 1.0)
    
    return spoiled, mask


# ═══════════════════════════════════════════════════════════════════
# Reconstruction
# ═══════════════════════════════════════════════════════════════════

def reconstruct_scidata(
    time: np.ndarray,
    spoiled: np.ndarray,
    mask: np.ndarray,
    method: str = "pchip",
) -> np.ndarray:
    """
    Reconstruct missing/corrupted samples in scientific data.
    
    Delegates to the same scipy-backed algorithms used for audio.
    
    Args:
        time: Time axis
        spoiled: Degraded signal
        mask: Boolean mask (True = valid sample)
        method: Interpolation method ('linear', 'spline', 'pchip', 'moving_average')
    
    Returns:
        Reconstructed signal
    """
    return _interpolation_reconstruct(time, spoiled, mask, method=method)


# ═══════════════════════════════════════════════════════════════════
# Demo Signal Generators
# ═══════════════════════════════════════════════════════════════════

def generate_ecg_demo(duration: float = 3.0, sample_rate: int = 500) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a synthetic ECG-like signal.
    
    Creates a realistic PQRST complex waveform with noise.
    """
    n_samples = int(duration * sample_rate)
    t = np.linspace(0, duration, n_samples)
    
    # Heart rate ~75 bpm = 0.8 s per beat
    beat_period = 0.8
    n_beats = int(duration / beat_period) + 1
    
    signal = np.zeros(n_samples)
    
    for beat in range(n_beats):
        beat_start = beat * beat_period
        
        # P wave (small bump before QRS)
        p_center = beat_start + 0.1
        signal += 0.15 * np.exp(-((t - p_center) ** 2) / (2 * 0.01 ** 2))
        
        # QRS complex (sharp peak)
        qrs_center = beat_start + 0.2
        # Q wave (small negative)
        signal -= 0.1 * np.exp(-((t - (qrs_center - 0.02)) ** 2) / (2 * 0.005 ** 2))
        # R wave (tall positive)
        signal += 1.0 * np.exp(-((t - qrs_center) ** 2) / (2 * 0.008 ** 2))
        # S wave (negative dip)
        signal -= 0.2 * np.exp(-((t - (qrs_center + 0.02)) ** 2) / (2 * 0.005 ** 2))
        
        # T wave (rounded bump after QRS)
        t_center = beat_start + 0.4
        signal += 0.25 * np.exp(-((t - t_center) ** 2) / (2 * 0.04 ** 2))
    
    # Normalize
    signal = signal / np.max(np.abs(signal))
    
    return t, signal


def generate_radio_demo(duration: float = 1.0, sample_rate: int = 10000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a synthetic amplitude-modulated radio signal.
    
    Carrier + modulating signal with fading effects.
    """
    n_samples = int(duration * sample_rate)
    t = np.linspace(0, duration, n_samples)
    
    # Carrier frequency
    f_carrier = 1000  # Hz
    
    # Modulating signal (speech-like envelope)
    f_mod = 5  # Hz
    modulator = 0.5 + 0.5 * np.sin(2 * np.pi * f_mod * t)
    
    # Add some "words" - amplitude variations
    words = np.zeros_like(t)
    word_times = [0.1, 0.35, 0.6, 0.85]
    for wt in word_times:
        words += 0.3 * np.exp(-((t - wt) ** 2) / (2 * 0.05 ** 2))
    modulator += words
    
    # AM signal
    signal = modulator * np.sin(2 * np.pi * f_carrier * t)
    
    # Add fading
    fading = 0.8 + 0.2 * np.sin(2 * np.pi * 0.5 * t)
    signal *= fading
    
    # Normalize
    signal = signal / np.max(np.abs(signal))
    
    return t, signal


def generate_sensor_demo(duration: float = 60.0, sample_rate: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic IoT temperature sensor data.
    
    Daily cycle with noise and occasional spikes.
    """
    n_samples = int(duration * sample_rate)
    t = np.linspace(0, duration, n_samples)
    
    # Base temperature: daily cycle (scaled to fit in duration)
    cycle_period = duration / 2  # Two "days" in the signal
    base = 20 + 5 * np.sin(2 * np.pi * t / cycle_period)
    
    # Add trend
    trend = 0.02 * t
    
    # Add noise
    rng = np.random.default_rng(42)
    noise = rng.normal(0, 0.5, n_samples)
    
    # Occasional spikes (e.g., heater turning on)
    spikes = np.zeros(n_samples)
    spike_times = [15, 35, 50]
    for st in spike_times:
        idx = min(int(st * sample_rate), n_samples - 1)
        spikes[max(0, idx-2):min(n_samples, idx+5)] = 3
    
    signal = base + trend + noise + spikes
    
    # Normalize to [-1, 1]
    signal = 2 * (signal - signal.min()) / (signal.max() - signal.min()) - 1
    
    return t, signal


def generate_wifi_demo(duration: float = 30.0, sample_rate: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic WiFi signal strength (RSSI) data.
    
    Simulates walking around with a device.
    """
    n_samples = int(duration * sample_rate)
    t = np.linspace(0, duration, n_samples)
    
    # Base signal strength with slow variations (movement)
    base = -50 + 15 * np.sin(2 * np.pi * t / 20)  # -65 to -35 dBm range
    
    # Fast fading (multipath)
    rng = np.random.default_rng(123)
    fast_fading = rng.normal(0, 3, n_samples)
    
    # Occasional deep fades (obstacles)
    deep_fades = np.zeros(n_samples)
    fade_positions = [8, 18, 25]
    for fp in fade_positions:
        idx = min(int(fp * sample_rate), n_samples - 1)
        deep_fades[max(0, idx-3):min(n_samples, idx+3)] = -15
    
    signal = base + fast_fading + deep_fades
    
    # Clip to realistic range
    signal = np.clip(signal, -90, -30)
    
    # Normalize to [-1, 1]
    signal = 2 * (signal - signal.min()) / (signal.max() - signal.min()) - 1
    
    return t, signal


def generate_accelerometer_demo(duration: float = 5.0, sample_rate: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic accelerometer data (single axis, walking motion).
    """
    n_samples = int(duration * sample_rate)
    t = np.linspace(0, duration, n_samples)
    
    # Walking frequency ~2 Hz (steps per second)
    f_walk = 2.0
    
    # Primary walking motion
    signal = 0.5 * np.sin(2 * np.pi * f_walk * t)
    
    # Add harmonics
    signal += 0.2 * np.sin(2 * np.pi * 2 * f_walk * t)  # 2nd harmonic
    signal += 0.1 * np.sin(2 * np.pi * 3 * f_walk * t)  # 3rd harmonic
    
    # Impact spikes (heel strikes)
    for i in range(int(duration * f_walk)):
        strike_time = i / f_walk + 0.1
        strike_idx = int(strike_time * sample_rate)
        if 0 <= strike_idx < n_samples:
            # Sharp spike
            width = 3
            for j in range(max(0, strike_idx - width), min(n_samples, strike_idx + width)):
                signal[j] += 0.8 * np.exp(-((j - strike_idx) ** 2) / 2)
    
    # Add sensor noise
    rng = np.random.default_rng(456)
    noise = rng.normal(0, 0.05, n_samples)
    signal += noise
    
    # Normalize
    signal = signal / np.max(np.abs(signal))
    
    return t, signal


# Available demo presets
DEMO_PRESETS = {
    "ecg": {
        "name": "ECG / Biomedical",
        "description": "Simulated electrocardiogram with PQRST complexes",
        "generator": generate_ecg_demo,
    },
    "radio": {
        "name": "AM Radio Signal",
        "description": "Amplitude-modulated carrier with fading",
        "generator": generate_radio_demo,
    },
    "temperature": {
        "name": "Temperature Sensor",
        "description": "IoT temperature readings with daily cycle",
        "generator": generate_sensor_demo,
    },
    "wifi": {
        "name": "WiFi RSSI",
        "description": "Signal strength with multipath fading",
        "generator": generate_wifi_demo,
    },
    "accelerometer": {
        "name": "Accelerometer",
        "description": "Walking motion with impact detection",
        "generator": generate_accelerometer_demo,
    },
}


def get_demo_signal(preset: str) -> Tuple[np.ndarray, np.ndarray]:
    """Get a demo signal by preset name."""
    if preset not in DEMO_PRESETS:
        raise ValueError(f"Unknown preset: {preset}. Available: {list(DEMO_PRESETS.keys())}")
    
    return DEMO_PRESETS[preset]["generator"]()


# ═══════════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════════

def compute_metrics(original: np.ndarray, reconstructed: np.ndarray) -> Dict[str, float]:
    """
    Compute reconstruction quality metrics.
    
    Same as audio metrics for consistency.
    """
    diff = original - reconstructed
    finite_diff = diff[np.isfinite(diff)]
    
    if len(finite_diff) == 0:
        return {"mse": 0.0, "rmse": 0.0, "mae": 0.0, "snr_db": 0.0}
    
    mse_val = float(np.mean(finite_diff ** 2))
    rmse_val = float(np.sqrt(mse_val))
    mae_val = float(np.mean(np.abs(finite_diff)))
    
    # Signal-to-Noise Ratio
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
