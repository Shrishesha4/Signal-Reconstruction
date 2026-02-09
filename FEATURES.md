# Signal Reconstruction Application - Feature Documentation

This document describes the features of the expanded Signal Reconstruction application, suitable for final-year project demonstration.

---

## Table of Contents

1. [Overview](#overview)
2. [Audio Reconstruction Tab](#audio-reconstruction-tab)
3. [Scientific Data Tab](#scientific-data-tab)
4. [Demo Signals Tab](#demo-signals-tab)
5. [API Reference](#api-reference)
6. [Running the Application](#running-the-application)

---

## Overview

The Signal Reconstruction application demonstrates numerical interpolation techniques for reconstructing damaged or incomplete signals. It supports:

- **Audio Signals**: WAV files with simulated or real damage
- **Scientific Time-Series Data**: CSV/JSON data from sensors, medical devices, communications
- **Demo Presets**: Pre-configured signals for demonstration

### Reconstruction Methods

| Method | Description | Best For |
|--------|-------------|----------|
| **PCHIP** | Piecewise Cubic Hermite Interpolating Polynomial | Preserves monotonicity, prevents overshoot |
| **Spline** | Cubic spline interpolation | Smooth curves, continuous derivatives |
| **Linear** | Linear interpolation | Simple, fast, predictable |
| **Moving Average** | Windowed averaging | Noise reduction with smoothing |

---

## Audio Reconstruction Tab

### Normal Mode

Upload a clean WAV file and simulate damage:

1. **Upload/Record**: Select a WAV file or record from microphone
2. **Configure Degradation**: Adjust loss percentage (5-50%)
3. **Select Method**: Choose reconstruction algorithm
4. **Process**: View original, degraded, and reconstructed waveforms

### Repair Mode (New!)

Repair actually damaged audio files with automatic damage detection:

1. **Upload Damaged Audio**: Select a corrupted WAV file
2. **Auto-Detection**: System identifies:
   - Dropouts (silent regions)
   - Clipping (amplitude saturation)
   - Discontinuities (sudden jumps)
   - Zero-crossing anomalies (corruption artifacts)
3. **Repair**: Reconstructs damaged regions using interpolation
4. **Compare**: View before/after waveforms and metrics

**Supported Damage Types:**
- Silent dropouts and gaps
- Audio clipping distortion
- Signal discontinuities
- Corrupted data segments

---

## Scientific Data Tab

Process general time-series data from various sources.

### Supported Formats

**CSV Files:**
```csv
time,value
0.0,1.234
0.001,1.256
...
```

**JSON Files:**
```json
{
  "time": [0.0, 0.001, ...],
  "value": [1.234, 1.256, ...]
}
```

### Demo Presets

| Preset | Description | Characteristics |
|--------|-------------|-----------------|
| **ECG** | Electrocardiogram | PQRST complex, heart rate ~72 BPM |
| **Radio** | AM Radio Signal | Carrier modulation with fading |
| **Temperature** | IoT Sensor | Daily thermal cycle with noise |
| **WiFi RSSI** | Signal Strength | Multipath fading, varying distance |
| **Accelerometer** | Motion Data | Walking gait with impact peaks |

### Workflow

1. **Select Preset** or **Upload File** (CSV/JSON)
2. **Configure Degradation**: Set loss percentage
3. **Select Method**: Choose reconstruction algorithm
4. **Process**: View original, degraded, and reconstructed signals
5. **Analyze**: Review quality metrics (MSE, RMSE, SNR, correlation)

---

## Demo Signals Tab

Pre-configured demonstrations for quick showcase:

- **Audio Demo**: Pure tone at 440 Hz with configurable degradation
- **Scientific Demos**: All preset signals with one-click generation
- Adjustable parameters: loss percentage, reconstruction method
- Instant visualization with quality metrics

---

## API Reference

### Audio Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/demo` | GET | Generate demo audio signal |
| `/api/process` | POST | Process uploaded WAV file |
| `/api/reconstruct` | POST | Re-run reconstruction |
| `/api/audio/repair` | POST | **New!** Repair damaged audio |

### Scientific Data Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/scidata/presets` | GET | List available presets |
| `/api/scidata/demo` | GET | Generate demo with preset |
| `/api/scidata/process` | POST | Process uploaded data file |
| `/api/scidata/reconstruct` | POST | Re-run reconstruction |

### Response Format

All endpoints return standardized JSON:

```json
{
  "name": "Signal Name",
  "totalSamples": 44100,
  "plot": {
    "time": [...],
    "original": [...],
    "spoiled": [...],
    "reconstructed": [...]
  },
  "metrics": {
    "mse": 0.0012,
    "rmse": 0.0346,
    "snr_db": 34.5,
    "correlation": 0.9987
  },
  "audio_base64": "UklGRi..." // Audio only
}
```

---

## Running the Application

### Prerequisites

- Python 3.9+
- Node.js 18+
- npm

### Backend Setup

```bash
cd backend
pip install -r requirements.txt
python -m uvicorn main:app --reload --port 8000
```

### Frontend Setup

```bash
npm install
npm run dev
```

### Access

- **Frontend**: http://localhost:5173
- **API Docs**: http://localhost:8000/docs

---

## Technical Implementation

### Backend (Python/FastAPI)

- **interpolation.py**: Core reconstruction algorithms (scipy.interpolate)
- **processing.py**: Audio I/O, degradation simulation
- **scidata.py**: Scientific data loading, demo generators
- **damaged_audio.py**: Damage detection and repair algorithms

### Frontend (SvelteKit/TypeScript)

- **Svelte 5 Runes**: $state, $derived, $props for reactivity
- **Chart.js**: Interactive waveform visualization
- **TailwindCSS**: Responsive UI styling
- **HTML5 Audio**: Playback with base64 encoding

### Key Algorithms

**Damage Detection (Audio):**
1. Rolling window analysis for dropout detection
2. Amplitude threshold for clipping detection
3. First-order difference for discontinuity detection
4. Zero-crossing rate analysis for corruption detection

**Reconstruction:**
1. Identify valid (undamaged) sample indices
2. Extract known values at valid positions
3. Interpolate missing values using selected method
4. Apply smoothing at region boundaries

---

## Quality Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **MSE** | Mean Squared Error | Lower is better |
| **RMSE** | Root MSE | Same units as signal |
| **SNR** | Signal-to-Noise Ratio (dB) | Higher is better (>20 dB good) |
| **Correlation** | Pearson coefficient | Closer to 1.0 is better |

---

## Demonstration Tips

1. **Start with Demo tab** for quick capabilities overview
2. **Show Audio tab** with clean file first, then repair mode
3. **Use Scientific tab** to show versatility across domains
4. **Compare methods** - PCHIP excels at preserving peaks
5. **Vary loss percentage** to show algorithm robustness

---

*Generated for Final Year Project Demonstration*
