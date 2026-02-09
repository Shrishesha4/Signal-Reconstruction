# Signal Reconstruction Pipeline

**Restore damaged signals with precision — Audio and scientific data reconstruction using advanced interpolation techniques.**

<p align="center">
  <img src="https://img.shields.io/badge/python-3.9+-blue.svg" alt="Python 3.9+">
  <img src="https://img.shields.io/badge/node-18+-green.svg" alt="Node 18+">
  <img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="MIT License">
  <img src="https://img.shields.io/badge/FastAPI-0.115-009688.svg" alt="FastAPI">
  <img src="https://img.shields.io/badge/SvelteKit-2.0-FF3E00.svg" alt="SvelteKit">
</p>

---

## Table of Contents

- [About](#about)
- [Features](#features)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Requirements](#requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [API Documentation](#api-documentation)
- [Usage](#usage)
- [Screenshots](#screenshots)
- [Data Formats](#data-formats)
- [Degradation Options](#degradation-options)
- [Interpolation Methods](#interpolation-methods)
- [Performance](#performance)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## About

The **Signal Reconstruction Pipeline** is a full-stack web application that demonstrates advanced numerical interpolation techniques for reconstructing damaged or incomplete signals. The system handles both audio and general scientific time-series data through a sophisticated 5-stage processing pipeline.

### Real-World Use Cases

| Domain | Application |
|--------|-------------|
| **Audio Restoration** | Repair damaged recordings, scratched vinyl digitization, lossy transmission recovery |
| **Medical Data** | ECG/EKG signal reconstruction, gap filling in patient monitoring data |
| **IoT & Sensors** | Missing sensor readings interpolation, network dropout compensation |
| **Communications** | Radio signal recovery, WiFi RSSI smoothing, telemetry data repair |
| **Scientific Research** | Time-series gap filling, experimental data reconstruction |

---

## Features

### Core Capabilities

- **Audio Signal Reconstruction**
  - Upload WAV files or use built-in demo signals
  - Simulate realistic damage patterns (dropouts, clipping, glitches, noise)
  - Repair actually damaged audio with automatic damage detection
  - Real-time waveform comparison and audio playback

- **Scientific Data Processing**
  - CSV and JSON file support
  - Built-in demo presets (ECG, Radio, Temperature, WiFi RSSI, Accelerometer)
  - Configurable degradation and reconstruction methods

- **Interactive Visualization**
  - Zoomable and pannable waveform charts (Chart.js + zoom plugin)
  - Side-by-side comparison of original, degraded, and reconstructed signals
  - Real-time quality metrics display

- **Advanced Reconstruction Pipeline**
  - 5-stage DSP processing with spectral subtraction
  - Sinusoidal modeling for large gaps
  - Tikhonov regularization for stability
  - Multiple interpolation algorithms (PCHIP, Spline, Linear, Moving Average)

### UI Features

- Three-tab interface: Audio | Scientific Data | Demo
- Normal mode and Repair mode for audio
- Adjustable degradation parameters with sliders
- Audio playback with play/pause controls
- Responsive design with TailwindCSS

---

## Architecture

### High-Level Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Client Browser                                │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │                    SvelteKit Frontend                         │  │
│  │  ┌──────────┐  ┌───────────┐  ┌──────────┐  ┌─────────────┐  │  │
│  │  │   Tab    │  │ Waveform  │  │  Audio   │  │   Metrics   │  │  │
│  │  │Navigation│  │   Chart   │  │  Player  │  │   Panel     │  │  │
│  │  └──────────┘  └───────────┘  └──────────┘  └─────────────┘  │  │
│  │  ┌────────────────────────────────────────────────────────┐  │  │
│  │  │              Degradation Controls                       │  │  │
│  │  └────────────────────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────────────────┘  │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ HTTP/JSON (REST API)
                               │ Audio: base64-encoded WAV
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      FastAPI Backend (Python)                        │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │                      API Endpoints                             │  │
│  │   /api/demo  /api/process  /api/reconstruct  /api/audio/repair │  │
│  │   /api/scidata/demo  /api/scidata/process  /api/compare       │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                               │                                      │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌─────────────┐   │
│  │ processing │  │interpolation│  │ advanced_  │  │ damaged_    │   │
│  │    .py     │  │    .py     │  │reconstruction│ │  audio.py   │   │
│  └────────────┘  └────────────┘  └────────────┘  └─────────────┘   │
│                               │                                      │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │              NumPy / SciPy (Vectorized DSP)                    │ │
│  └────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Input**: User uploads WAV/CSV/JSON or triggers demo signal
2. **Degradation**: Backend applies configurable damage (dropouts, noise, clipping)
3. **Reconstruction**: 5-stage advanced pipeline processes damaged signal
4. **Response**: JSON with downsampled plot data + base64 audio + metrics
5. **Visualization**: Frontend renders interactive charts and enables playback

### Reconstruction Pipeline (5 Stages)

```
Input Signal                                           Output Signal
     │                                                       ▲
     ▼                                                       │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Stage 1:        │    │ Stage 2:        │    │ Stage 3:        │
│ Noise Reduction │───▶│ Damage Analysis │───▶│ Model-Based     │
│ - Spectral sub  │    │ - Classify type │    │ Reconstruction  │
│ - Wiener filter │    │ - Segment gaps  │    │ - Sinusoidal    │
│ - Median filter │    │ - Detect clips  │    │ - Parametric    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                      │
                                                      ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │ Stage 5:        │    │ Stage 4:        │
                       │ Post-Processing │◀───│ Adaptive        │
                       │ - Low-pass LP   │    │ Interpolation   │
                       │ - Soft clipping │    │ - PCHIP/Spline  │
                       │ - Envelope match│    │ - Tikhonov reg  │
                       └─────────────────┘    └─────────────────┘
```

---

## Tech Stack

### Frontend

| Technology | Version | Purpose |
|------------|---------|---------|
| **SvelteKit** | 2.50+ | Full-stack framework with SSR |
| **Svelte 5** | 5.x | UI components with runes ($state, $derived) |
| **TailwindCSS** | 4.1+ | Utility-first styling |
| **Chart.js** | 4.4+ | Waveform visualization |
| **chartjs-plugin-zoom** | 2.2+ | Pan & zoom interactions |
| **TypeScript** | 5.x | Type safety |
| **Vite** | 7.x | Build tool & dev server |

### Backend

| Technology | Version | Purpose |
|------------|---------|---------|
| **FastAPI** | 0.115+ | Async REST API framework |
| **Uvicorn** | 0.34+ | ASGI server |
| **NumPy** | 2.2+ | Numerical arrays |
| **SciPy** | 1.15+ | Interpolation algorithms |
| **python-multipart** | 0.0.20 | File upload handling |

---

## Requirements

### System Requirements

- **OS**: macOS, Linux, or Windows 10+
- **RAM**: 4GB minimum (8GB recommended)
- **Disk**: 500MB free space

### Software Requirements

| Software | Minimum Version | Recommended |
|----------|-----------------|-------------|
| Python | 3.9 | 3.11+ |
| Node.js | 18.x | 20.x LTS |
| npm | 9.x | 10.x |

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/Signal-Reconstruction.git
cd Signal-Reconstruction
```

### 2. Backend Setup

```bash
# Navigate to backend directory
cd backend

# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import numpy, scipy, fastapi; print('Backend dependencies OK')"
```

### 3. Frontend Setup

```bash
# Return to project root
cd ..

# Install Node.js dependencies
npm install

# Verify installation
npm run check
```

### 4. Start Development Servers

**Terminal 1 — Backend:**

```bash
cd backend
uvicorn main:app --reload --port 8000
```

**Terminal 2 — Frontend:**

```bash
npm run dev -- --port 5173
```

### 5. Access the Application

| Service | URL |
|---------|-----|
| **Frontend** | http://localhost:5173 |
| **API Documentation** | http://localhost:8000/docs |
| **API Health Check** | http://localhost:8000/api/health |

---

## Configuration

### Environment Variables

Create a `.env` file in the project root (optional):

```bash
# Backend Configuration
BACKEND_PORT=8000
BACKEND_HOST=0.0.0.0

# Frontend Configuration
VITE_API_URL=http://localhost:8000

# Development
DEBUG=true
```

### Port Configuration

| Service | Default Port | Environment Variable |
|---------|--------------|---------------------|
| Backend API | 8000 | `BACKEND_PORT` |
| Frontend Dev | 5173 | `--port` flag |

### API URL Configuration

The frontend connects to the backend via the API base URL. In development, requests are proxied through Vite. For production, configure `VITE_API_URL`.

```typescript
// src/lib/api.ts
const API_BASE = import.meta.env.VITE_API_URL || '';
```

---

## API Documentation

### Base URL

```
http://localhost:8000
```

### Endpoints Overview

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Health check |
| GET | `/api/demo` | Demo signal with degradation |
| POST | `/api/process` | Upload & process WAV file |
| POST | `/api/reconstruct` | Re-run reconstruction |
| POST | `/api/audio/repair` | Repair damaged audio |
| GET | `/api/compare` | Compare baseline vs advanced |
| GET | `/api/scidata/presets` | List demo presets |
| GET | `/api/scidata/demo` | Generate demo scientific signal |
| POST | `/api/scidata/process` | Process uploaded data file |
| POST | `/api/scidata/reconstruct` | Re-run scientific data reconstruction |

---

### GET `/api/health`

Health check endpoint.

**Response:**

```json
{
  "status": "ok"
}
```

**Example:**

```bash
curl http://localhost:8000/api/health
```

---

### GET `/api/demo`

Generate a demo audio signal with degradation and reconstruction.

**Query Parameters:**

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `dropout_pct` | float | 10.0 | 0-50 | Percentage of audio to drop as silence |
| `dropout_length_ms` | float | 100.0 | 10-500 | Average dropout segment length (ms) |
| `glitch_pct` | float | 1.0 | 0-20 | Percentage of audio with glitch artifacts |
| `clip_pct` | float | 10.0 | 0-30 | Percentage of audio with amplitude clipping |
| `noise_level` | float | 0.02 | 0-0.1 | Gaussian noise amplitude |
| `method` | string | "pchip" | pchip/spline/linear/moving_average | Interpolation method |
| `seed` | int | null | — | Random seed for reproducibility |

**Response Schema:**

```json
{
  "sampleRate": 8000,
  "totalSamples": 8000,
  "plot": {
    "time": [0.0, 0.000125, ...],
    "original": [0.0, 0.034, ...],
    "spoiled": [0.0, 0.0, ...],
    "reconstructed": [0.0, 0.033, ...]
  },
  "audio": {
    "original": "UklGRi...(base64 WAV)",
    "spoiled": "UklGRi...",
    "reconstructed": "UklGRi..."
  },
  "metrics": {
    "mse": 0.00042,
    "rmse": 0.0205,
    "mae": 0.0156,
    "snr_db": 28.5
  },
  "mask": [1.0, 1.0, 0.0, 0.0, ...]
}
```

**Example — cURL:**

```bash
curl "http://localhost:8000/api/demo?dropout_pct=15&method=spline&seed=42"
```

**Example — JavaScript:**

```javascript
const response = await fetch('/api/demo?dropout_pct=10&method=pchip');
const data = await response.json();
console.log(`SNR: ${data.metrics.snr_db} dB`);
```

---

### POST `/api/process`

Upload a WAV file, apply degradation, and reconstruct.

**Content-Type:** `multipart/form-data`

**Form Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `file` | File | required | WAV file (max 10 MB) |
| `dropout_pct` | float | 10.0 | Dropout percentage |
| `dropout_length_ms` | float | 100.0 | Dropout length |
| `glitch_pct` | float | 1.0 | Glitch percentage |
| `clip_pct` | float | 10.0 | Clipping percentage |
| `noise_level` | float | 0.02 | Noise level |
| `method` | string | "pchip" | Interpolation method |

**Response:** Same schema as `/api/demo`

**Example — cURL:**

```bash
curl -X POST "http://localhost:8000/api/process" \
  -F "file=@audio.wav" \
  -F "dropout_pct=15" \
  -F "method=pchip"
```

**Example — JavaScript:**

```javascript
const formData = new FormData();
formData.append('file', audioFile);
formData.append('dropout_pct', '15');
formData.append('method', 'pchip');

const response = await fetch('/api/process', {
  method: 'POST',
  body: formData
});
const result = await response.json();
```

---

### POST `/api/reconstruct`

Re-run reconstruction with a different method on already-degraded data.

**Content-Type:** `application/json`

**Request Body:**

```json
{
  "time": [0.0, 0.000125, ...],
  "original": [0.0, 0.034, ...],
  "spoiled": [0.0, 0.0, ...],
  "mask": [true, true, false, ...],
  "method": "spline",
  "sampleRate": 8000
}
```

**Response:** Same schema as `/api/demo`

**Example:**

```bash
curl -X POST "http://localhost:8000/api/reconstruct" \
  -H "Content-Type: application/json" \
  -d '{"time":[0,0.001],"original":[0,0.5],"spoiled":[0,0],"mask":[true,false],"method":"pchip","sampleRate":8000}'
```

---

### POST `/api/audio/repair`

Repair a damaged audio file with automatic damage detection.

**Content-Type:** `multipart/form-data`

**Form Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `file` | File | required | Damaged WAV file |
| `method` | string | "pchip" | Interpolation method |
| `auto_detect` | bool | true | Auto-detect damaged regions |

**Response Schema:**

```json
{
  "sampleRate": 44100,
  "totalSamples": 220500,
  "plot": {
    "time": [...],
    "damaged": [...],
    "reconstructed": [...]
  },
  "audio": {
    "damaged": "UklGRi...(base64)",
    "reconstructed": "UklGRi..."
  },
  "metrics": {
    "damage_percent": 12.5,
    "samples_repaired": 27562,
    "summary": "Detected 3 dropout regions, 2 clipping zones"
  },
  "mask": [...],
  "analysis": {
    "summary": "Detected 3 dropout regions, 2 clipping zones",
    "damage_percent": 12.5,
    "stats": {
      "dropouts": 3,
      "clipping_zones": 2,
      "discontinuities": 1
    }
  }
}
```

**Example:**

```bash
curl -X POST "http://localhost:8000/api/audio/repair" \
  -F "file=@damaged_recording.wav" \
  -F "method=spline"
```

---

### GET `/api/compare`

Compare baseline vs advanced reconstruction across all methods.

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dropout_pct` | float | 15.0 | Dropout percentage |
| `dropout_length_ms` | float | 100.0 | Dropout length |
| `glitch_pct` | float | 1.0 | Glitch percentage |
| `clip_pct` | float | 10.0 | Clipping percentage |
| `noise_level` | float | 0.02 | Noise level |
| `seed` | int | null | Random seed |

**Response Schema:**

```json
{
  "methods": {
    "pchip": {
      "baseline": { "mse": 0.0045, "snr_db": 18.5 },
      "advanced": { "mse": 0.0012, "snr_db": 24.2 },
      "improvement": { "snr_db": 5.7, "mse_reduction": 73.3 }
    },
    "spline": { ... },
    "linear": { ... },
    "moving_average": { ... }
  },
  "summary": {
    "damage_config": { ... },
    "best_baseline_method": "pchip",
    "best_baseline_snr": 18.5,
    "best_advanced_method": "pchip",
    "best_advanced_snr": 24.2,
    "average_snr_improvement": 4.8,
    "average_mse_reduction_pct": 58.2
  }
}
```

**Example:**

```bash
curl "http://localhost:8000/api/compare?seed=42&dropout_pct=10"
```

---

### GET `/api/scidata/presets`

List available demo signal presets.

**Response:**

```json
{
  "presets": [
    { "id": "ecg", "name": "ECG Signal", "description": "Electrocardiogram with PQRST complex" },
    { "id": "radio", "name": "AM Radio", "description": "AM carrier with fading" },
    { "id": "temperature", "name": "Temperature Sensor", "description": "Daily thermal cycle" },
    { "id": "wifi", "name": "WiFi RSSI", "description": "Signal strength with multipath" },
    { "id": "accelerometer", "name": "Accelerometer", "description": "Walking gait data" }
  ]
}
```

---

### GET `/api/scidata/demo`

Generate a demo scientific signal with degradation.

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `preset` | string | "ecg" | Preset name (from `/api/scidata/presets`) |
| `dropout_pct` | float | 15.0 | Dropout percentage (0-80) |
| `noise_level` | float | 0.02 | Noise level (0-0.5) |
| `method` | string | "pchip" | Interpolation method |

**Response Schema:**

```json
{
  "name": "ECG Signal",
  "totalSamples": 5000,
  "plot": {
    "time": [...],
    "original": [...],
    "spoiled": [...],
    "reconstructed": [...]
  },
  "metrics": {
    "mse": 0.0023,
    "rmse": 0.048,
    "mae": 0.035,
    "snr_db": 26.3
  },
  "mask": [...],
  "unitInfo": { "min": -1.2, "max": 1.5 }
}
```

**Example:**

```bash
curl "http://localhost:8000/api/scidata/demo?preset=ecg&dropout_pct=20&method=spline"
```

---

### POST `/api/scidata/process`

Upload and process a scientific data file.

**Content-Type:** `multipart/form-data`

**Form Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `file` | File | required | CSV or JSON file (max 5 MB) |
| `dropout_pct` | float | 15.0 | Dropout percentage |
| `noise_level` | float | 0.02 | Noise level |
| `method` | string | "pchip" | Interpolation method |

**Response:** Same schema as `/api/scidata/demo`

---

## Usage

### Audio Reconstruction (Normal Mode)

1. **Navigate to the Audio tab**
2. **Select input method:**
   - Click "Load Demo" for a built-in test signal
   - Click "Upload WAV" to select your own file
3. **Configure degradation parameters:**
   - Dropout %: Percentage of audio replaced with silence
   - Dropout Length: Duration of each dropout segment
   - Glitch %: Percentage with random glitch artifacts
   - Clip %: Percentage with amplitude clipping
   - Noise Level: Gaussian noise amplitude
4. **Select reconstruction method** (PCHIP recommended)
5. **Click "Process"**
6. **Analyze results:**
   - View waveform comparison (original / degraded / reconstructed)
   - Check quality metrics (SNR, MSE)
   - Play audio samples using the audio player
7. **Try different methods** using the method dropdown (instant re-reconstruction)

### Audio Repair Mode

1. **Switch to Repair Mode** using the toggle
2. **Upload a damaged WAV file**
3. **Select reconstruction method**
4. **Click "Repair"**
5. **Review damage analysis** (detected dropouts, clipping zones, etc.)
6. **Compare damaged vs repaired** waveforms and audio

### Scientific Data Processing

1. **Navigate to the Scientific Data tab**
2. **Choose input:**
   - Select a preset (ECG, Radio, Temperature, WiFi, Accelerometer)
   - Or upload your own CSV/JSON file
3. **Configure degradation** (dropout %, noise level)
4. **Select reconstruction method**
5. **Click "Process"**
6. **Analyze** the reconstructed signal and metrics

### Using the Chart

- **Zoom**: Scroll wheel or pinch gesture
- **Pan**: Click and drag
- **Reset**: Double-click to reset zoom
- **Legend**: Click legend items to toggle series visibility

### Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `Space` | Play/pause audio |
| `1-4` | Switch reconstruction method |

---

## Screenshots

> **Note:** Replace these placeholders with actual screenshots.

### Audio Reconstruction Tab

```
┌─────────────────────────────────────────────────────────────────┐
│  [Screenshot: Audio tab with waveform comparison]               │
│                                                                 │
│  Show: Upload controls, degradation sliders, waveform chart    │
│        with original (blue), degraded (red), reconstructed     │
│        (green) signals, audio players, and metrics panel       │
└─────────────────────────────────────────────────────────────────┘
```

### Damage Repair Mode

```
┌─────────────────────────────────────────────────────────────────┐
│  [Screenshot: Repair mode with damage detection results]        │
│                                                                 │
│  Show: Damaged audio upload, detection summary, before/after   │
│        waveform comparison, repair metrics                     │
└─────────────────────────────────────────────────────────────────┘
```

### Scientific Data Tab

```
┌─────────────────────────────────────────────────────────────────┐
│  [Screenshot: Scientific data tab with ECG signal]              │
│                                                                 │
│  Show: Preset selector, ECG waveform with PQRST complex,       │
│        degradation controls, reconstruction comparison          │
└─────────────────────────────────────────────────────────────────┘
```

### Interactive Chart

```
┌─────────────────────────────────────────────────────────────────┐
│  [Screenshot: Zoomed-in waveform detail]                        │
│                                                                 │
│  Show: Chart with zoom applied, visible interpolation detail   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Data Formats

### Audio Files

**Supported:** WAV format only

| Property | Supported Values |
|----------|-----------------|
| Sample Width | 8-bit, 16-bit, 24-bit, 32-bit |
| Channels | Mono, Stereo (auto-converted to mono) |
| Sample Rate | Any (preserved) |
| Duration | Up to 5 seconds (auto-truncated) |
| File Size | Max 10 MB |

### CSV Format

```csv
time,value
0.0,1.234
0.001,1.256
0.002,1.301
```

**Recognized column headers:**
- Time: `time`, `t`, `x`, `timestamp`, `index`
- Value: `value`, `y`, `signal`, `data`, `amplitude`, `reading`

### JSON Format (Array)

```json
{
  "time": [0.0, 0.001, 0.002, 0.003],
  "value": [1.234, 1.256, 1.301, 1.245]
}
```

### JSON Format (Objects)

```json
[
  { "time": 0.0, "value": 1.234 },
  { "time": 0.001, "value": 1.256 },
  { "time": 0.002, "value": 1.301 }
]
```

---

## Degradation Options

The system simulates realistic signal damage using multiple degradation types:

### Dropout (Silence)

Replaces segments with silence, simulating:
- Transmission dropouts
- Disk read errors
- Network packet loss

| Parameter | Range | Effect |
|-----------|-------|--------|
| `dropout_pct` | 0-50% | Total signal replaced |
| `dropout_length_ms` | 10-500ms | Average gap duration |

### Clipping

Flattens peaks above threshold, simulating:
- Amplifier saturation
- ADC overflow
- Volume overloading

| Parameter | Range | Effect |
|-----------|-------|--------|
| `clip_pct` | 0-30% | Percentage of peaks clipped |

### Glitches

Adds random spike artifacts, simulating:
- Electrical interference
- Digital corruption
- Bit flips

| Parameter | Range | Effect |
|-----------|-------|--------|
| `glitch_pct` | 0-20% | Percentage with glitches |

### Gaussian Noise

Adds random noise floor, simulating:
- Thermal noise
- Quantization noise
- Environmental interference

| Parameter | Range | Effect |
|-----------|-------|--------|
| `noise_level` | 0-0.1 | Noise amplitude (0-10% of signal) |

---

## Interpolation Methods

### Method Comparison

| Method | Best For | Pros | Cons |
|--------|----------|------|------|
| **PCHIP** | General use | Shape-preserving, no overshoot | Slightly less smooth |
| **Cubic Spline** | Smooth signals | C² continuity, very smooth | Can overshoot at peaks |
| **Linear** | Small gaps | Fast, predictable | Visible discontinuities |
| **Moving Average** | Noise reduction | Simple denoising | Smooths details |

### PCHIP (Recommended)

**Piecewise Cubic Hermite Interpolating Polynomial**

- Preserves monotonicity between data points
- No Runge phenomenon (overshoot at edges)
- Best for signals with sharp features (ECG, transients)

```
Gap Size → Method Selection:
  ≤3 samples:   Linear interpolation
  4-50 samples: PCHIP with regularization
  >50 samples:  Model-guided + regularized spline
```

### Cubic Spline

- Natural boundary conditions
- C² continuous (smooth second derivative)
- Best for smooth, continuous signals

### Linear

- Simple point-to-point interpolation
- Blended with moving average for smoothness
- Fast computation

### Moving Average

- Adaptive window size based on signal characteristics
- Primary role: denoising (combined with other methods)
- Not recommended as standalone reconstruction

---

## Performance

### Computational Complexity

| Operation | Complexity | Notes |
|-----------|------------|-------|
| FFT-based noise reduction | O(N log N) | SciPy FFTPACK |
| Interpolation | O(N) | Vectorized NumPy |
| Gap detection | O(N) | Single pass |
| Metrics computation | O(N) | Parallel operations |

### Benchmarks

| Signal Length | Processing Time | Memory Usage |
|---------------|-----------------|--------------|
| 8,000 samples (1s @ 8kHz) | ~50ms | ~2 MB |
| 44,100 samples (1s @ 44.1kHz) | ~150ms | ~8 MB |
| 220,500 samples (5s @ 44.1kHz) | ~500ms | ~35 MB |

### Optimization Tips

1. **Limit signal length**: Auto-truncated to 5 seconds
2. **Use lower sample rates**: 8kHz sufficient for demos
3. **Downsampling**: Plot data automatically downsampled to 8,000 points
4. **Batch processing**: Use `/api/compare` for method benchmarking

---

## Troubleshooting

### Common Issues

#### Backend won't start

**Error:** `Address already in use`

```bash
# Kill existing process on port 8000
lsof -ti :8000 | xargs kill -9
uvicorn main:app --reload --port 8000
```

**Error:** `ModuleNotFoundError`

```bash
# Ensure virtual environment is activated
source backend/venv/bin/activate
pip install -r backend/requirements.txt
```

#### Frontend can't connect to backend

**Symptom:** "Failed to fetch" errors

1. Verify backend is running: `curl http://localhost:8000/api/health`
2. Check CORS settings in `main.py`
3. Ensure both servers are on expected ports

#### Audio playback issues

**Symptom:** No sound or distorted audio

- Check browser audio permissions
- Verify WAV file format (16-bit PCM recommended)
- Try a different browser (Chrome/Firefox recommended)

#### Chart rendering issues

**Symptom:** Empty or frozen chart

- Clear browser cache
- Check console for JavaScript errors
- Reduce data points by using shorter audio

### FAQ

**Q: What audio formats are supported?**
A: WAV only (8/16/24/32-bit, mono or stereo).

**Q: Can I process MP3 files?**
A: No. Convert to WAV first using FFmpeg: `ffmpeg -i input.mp3 output.wav`

**Q: Why is my audio truncated?**
A: Files are limited to 5 seconds for performance. Split longer files.

**Q: What's the maximum file size?**
A: 10 MB for audio, 5 MB for CSV/JSON data.

**Q: Which interpolation method is best?**
A: PCHIP is recommended for most cases. Use Spline for very smooth signals.

---

## Contributing

Contributions are welcome! Please follow these guidelines:

### Getting Started

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Make your changes
4. Run tests and linting:
   ```bash
   # Frontend
   npm run check
   npm run lint
   
   # Backend
   cd backend
   python -m pytest tests/
   ```
5. Commit with clear messages: `git commit -m "Add: new interpolation method"`
6. Push and open a Pull Request

### Code Style

**Frontend:**
- Use TypeScript for all new code
- Follow ESLint + Prettier configuration
- Use Svelte 5 runes ($state, $derived, $props)

**Backend:**
- Follow PEP 8 style guide
- Use type hints for all functions
- Document functions with docstrings

### Pull Request Guidelines

- Include description of changes
- Add tests for new features
- Update documentation as needed
- Ensure CI passes

---

## License

This project is licensed under the **MIT License**.

```
MIT License

Copyright (c) 2026 Shrishesha

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
```

---

## Acknowledgments

### Technologies

- [FastAPI](https://fastapi.tiangolo.com/) — Modern Python web framework
- [SvelteKit](https://kit.svelte.dev/) — Full-stack Svelte framework
- [SciPy](https://scipy.org/) — Scientific computing library
- [Chart.js](https://www.chartjs.org/) — JavaScript charting library
- [TailwindCSS](https://tailwindcss.com/) — Utility-first CSS framework

### Algorithms & References

- **PCHIP Interpolation**: Fritsch, F. N. & Carlson, R. E. (1980). *Monotone Piecewise Cubic Interpolation*. SIAM Journal on Numerical Analysis.
- **Spectral Subtraction**: Boll, S. (1979). *Suppression of Acoustic Noise in Speech Using Spectral Subtraction*. IEEE Transactions on ASSP.
- **Tikhonov Regularization**: Tikhonov, A. N. (1963). *Solution of Incorrectly Formulated Problems and the Regularization Method*. Soviet Mathematics.
- **Wiener Filtering**: Wiener, N. (1949). *Extrapolation, Interpolation, and Smoothing of Stationary Time Series*. MIT Press.

### Inspiration

- Signal processing courses and numerical methods research
- Real-world audio restoration challenges
- Scientific data quality improvement needs

---

<p align="center">
  <strong>Built for signal restoration | Capstone Project | UBA1032</strong>
  <br>
  <sub>If you find this project useful, consider giving it a ⭐</sub>
</p>
