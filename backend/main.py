"""
Signal Reconstruction Backend – FastAPI
=======================================
Handles audio loading, degradation, interpolation-based reconstruction,
and error metrics for the SvelteKit frontend.

Supports:
  - Audio signal reconstruction (with degradation simulation)
  - Damaged audio detection and repair
  - General scientific time-series data reconstruction (CSV/JSON)

API Endpoints:
  Audio:
    GET  /api/demo              – Demo signal with degradation
    POST /api/process           – Upload WAV + degrade + reconstruct
    POST /api/reconstruct       – Re-run reconstruction only
    POST /api/audio/repair      – Repair damaged audio file
  
  Scientific Data:
    GET  /api/scidata/presets   – List available demo presets
    GET  /api/scidata/demo      – Demo with preset signal
    POST /api/scidata/process   – Upload CSV/JSON + process

  Standardized JSON output:
    {
      "time": [...],
      "raw": [...],
      "damaged": [...],
      "reconstructed": [...],
      "audio_base64": "..." (for audio only)
    }
"""

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Optional, List
import numpy as np

from processing import (
    load_wav_bytes,
    degrade_signal,
    reconstruct_signal,
    reconstruct_signal_advanced,
    compare_reconstruction_methods,
    compute_metrics,
    generate_demo_signal,
    signal_to_wav_base64,
    downsample_for_plot,
)

from scidata import (
    load_scidata,
    normalize_signal,
    denormalize_signal,
    degrade_scidata,
    reconstruct_scidata,
    compute_metrics as compute_scidata_metrics,
    DEMO_PRESETS,
    get_demo_signal,
)

from damaged_audio import (
    analyze_damage,
    process_damaged_audio,
)

app = FastAPI(
    title="Signal Reconstruction API",
    description="Audio & Scientific Signal Processing with Numerical Interpolation",
    version="2.0.0",
)

# CORS – allow SvelteKit dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Maximum samples we send to the frontend for plotting ──
MAX_PLOT_POINTS = 8000


@app.get("/api/health")
async def health():
    return {"status": "ok"}


@app.get("/api/demo")
async def demo_signal(
    dropout_pct: float = 10.0,
    dropout_length_ms: float = 100.0,
    glitch_pct: float = 5.0,
    clip_pct: float = 10.0,
    noise_level: float = 0.02,
    method: str = "pchip",
    seed: Optional[int] = None,
):
    """
    Return a demo sine-composite signal, degraded and reconstructed.
    
    Uses ADVANCED reconstruction pipeline with:
    - Spectral subtraction for noise reduction
    - Sinusoidal modeling for large gaps
    - Tikhonov regularization for stability
    
    Degradation Parameters:
        dropout_pct:       % of audio to drop as silence (0-50)
        dropout_length_ms: average length of dropout segments in ms (10-500)
        glitch_pct:        % of audio with glitch artifacts (0-20)
        clip_pct:          % of audio with amplitude clipping (0-30)
        noise_level:       Gaussian noise amplitude (0-0.1)
        seed:              Random seed for reproducible degradation (optional)
    """
    sample_rate, samples = generate_demo_signal(duration=1.0, sr=8000)

    time_axis = np.arange(len(samples)) / sample_rate
    original = samples.astype(np.float64)

    spoiled, mask = degrade_signal(
        original,
        sample_rate=sample_rate,
        dropout_pct=dropout_pct,
        dropout_length_ms=dropout_length_ms,
        glitch_pct=glitch_pct,
        clip_pct=clip_pct,
        noise_level=noise_level,
        seed=seed,
    )
    # Use advanced reconstruction for better quality
    reconstructed = reconstruct_signal_advanced(
        time_axis, spoiled, mask,
        sample_rate=sample_rate,
        method=method,
        use_sinusoidal_model=True,
        use_spectral_subtraction=True,
        use_tikhonov=True,
    )
    metrics = compute_metrics(original, reconstructed)

    return _build_response(time_axis, original, spoiled, reconstructed, mask, metrics, sample_rate)


@app.post("/api/process")
async def process_audio(
    file: UploadFile = File(...),
    dropout_pct: float = Form(10.0),
    dropout_length_ms: float = Form(100.0),
    glitch_pct: float = Form(5.0),
    clip_pct: float = Form(10.0),
    noise_level: float = Form(0.02),
    method: str = Form("pchip"),
):
    """
    Full pipeline: load WAV → degrade → reconstruct → return.
    
    Degradation Parameters:
        dropout_pct:       % of audio to drop as silence (0-50)
        dropout_length_ms: average length of dropout segments in ms (10-500)
        glitch_pct:        % of audio with glitch artifacts (0-20)
        clip_pct:          % of audio with amplitude clipping (0-30)
        noise_level:       Gaussian noise amplitude (0-0.1)
    """
    if not file.filename or not file.filename.lower().endswith(".wav"):
        raise HTTPException(status_code=400, detail="Only .wav files are supported.")

    raw = await file.read()
    if len(raw) > 10 * 1024 * 1024:  # 10 MB limit
        raise HTTPException(status_code=400, detail="File too large (max 10 MB).")

    try:
        sample_rate, samples = load_wav_bytes(raw)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read WAV: {e}")

    time_axis = np.arange(len(samples)) / sample_rate
    original = samples.astype(np.float64)

    # Normalize to [-1, 1]
    peak = np.max(np.abs(original))
    if peak > 0:
        original = original / peak

    spoiled, mask = degrade_signal(
        original,
        sample_rate=sample_rate,
        dropout_pct=dropout_pct,
        dropout_length_ms=dropout_length_ms,
        glitch_pct=glitch_pct,
        clip_pct=clip_pct,
        noise_level=noise_level,
    )
    # Use advanced reconstruction for better quality
    reconstructed = reconstruct_signal_advanced(
        time_axis, spoiled, mask,
        sample_rate=sample_rate,
        method=method,
        use_sinusoidal_model=True,
        use_spectral_subtraction=True,
        use_tikhonov=True,
    )
    metrics = compute_metrics(original, reconstructed)

    return _build_response(time_axis, original, spoiled, reconstructed, mask, metrics, sample_rate)


@app.post("/api/reconstruct")
async def reconstruct_only(body: dict):
    """Re-run reconstruction with a different method on already-spoiled data."""
    try:
        time_axis = np.array(body["time"], dtype=np.float64)
        spoiled = np.array(body["spoiled"], dtype=np.float64)
        mask = np.array(body["mask"], dtype=bool)
        original = np.array(body["original"], dtype=np.float64)
        method = body.get("method", "pchip")
        sample_rate = body.get("sampleRate", 8000)
    except (KeyError, ValueError) as e:
        raise HTTPException(status_code=400, detail=f"Invalid payload: {e}")

    # Use advanced reconstruction for better quality
    reconstructed = reconstruct_signal_advanced(
        time_axis, spoiled, mask,
        sample_rate=sample_rate,
        method=method,
        use_sinusoidal_model=True,
        use_spectral_subtraction=True,
        use_tikhonov=True,
    )
    metrics = compute_metrics(original, reconstructed)

    return _build_response(time_axis, original, spoiled, reconstructed, mask, metrics, sample_rate)


# ═══════════════════════════════════════════════════════════════════
# Advanced Reconstruction Endpoints
# ═══════════════════════════════════════════════════════════════════

@app.get("/api/demo/advanced")
async def demo_signal_advanced(
    dropout_pct: float = 10.0,
    dropout_length_ms: float = 100.0,
    glitch_pct: float = 5.0,
    clip_pct: float = 10.0,
    noise_level: float = 0.02,
    method: str = "pchip",
    use_sinusoidal_model: bool = True,
    use_spectral_subtraction: bool = True,
    use_tikhonov: bool = True,
):
    """
    Demo with ADVANCED reconstruction using classical DSP techniques.
    
    This endpoint uses the enhanced 5-stage pipeline:
    1. Noise reduction (spectral subtraction, Wiener filtering)
    2. Damage analysis & segmentation
    3. Model-based reconstruction (sinusoidal modeling for large gaps)
    4. Adaptive interpolation with Tikhonov regularization
    5. Perceptual post-processing (compression, soft clipping)
    
    Compare with /api/demo for baseline reconstruction quality.
    """
    sample_rate, samples = generate_demo_signal(duration=1.0, sr=8000)

    time_axis = np.arange(len(samples)) / sample_rate
    original = samples.astype(np.float64)

    spoiled, mask = degrade_signal(
        original,
        sample_rate=sample_rate,
        dropout_pct=dropout_pct,
        dropout_length_ms=dropout_length_ms,
        glitch_pct=glitch_pct,
        clip_pct=clip_pct,
        noise_level=noise_level,
    )
    
    # Use advanced reconstruction
    reconstructed = reconstruct_signal_advanced(
        time_axis, spoiled, mask,
        sample_rate=sample_rate,
        method=method,
        use_sinusoidal_model=use_sinusoidal_model,
        use_spectral_subtraction=use_spectral_subtraction,
        use_tikhonov=use_tikhonov,
    )
    metrics = compute_metrics(original, reconstructed)

    return _build_response(time_axis, original, spoiled, reconstructed, mask, metrics, sample_rate)


@app.post("/api/process/advanced")
async def process_audio_advanced(
    file: UploadFile = File(...),
    dropout_pct: float = Form(10.0),
    dropout_length_ms: float = Form(100.0),
    glitch_pct: float = Form(5.0),
    clip_pct: float = Form(10.0),
    noise_level: float = Form(0.02),
    method: str = Form("pchip"),
    use_sinusoidal_model: bool = Form(True),
    use_spectral_subtraction: bool = Form(True),
    use_tikhonov: bool = Form(True),
):
    """
    Process audio with ADVANCED reconstruction pipeline.
    
    Full pipeline: load WAV → degrade → advanced reconstruct → return.
    """
    if not file.filename or not file.filename.lower().endswith(".wav"):
        raise HTTPException(status_code=400, detail="Only .wav files are supported.")

    raw = await file.read()
    if len(raw) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large (max 10 MB).")

    try:
        sample_rate, samples = load_wav_bytes(raw)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read WAV: {e}")

    time_axis = np.arange(len(samples)) / sample_rate
    original = samples.astype(np.float64)

    peak = np.max(np.abs(original))
    if peak > 0:
        original = original / peak

    spoiled, mask = degrade_signal(
        original,
        sample_rate=sample_rate,
        dropout_pct=dropout_pct,
        dropout_length_ms=dropout_length_ms,
        glitch_pct=glitch_pct,
        clip_pct=clip_pct,
        noise_level=noise_level,
    )
    
    reconstructed = reconstruct_signal_advanced(
        time_axis, spoiled, mask,
        sample_rate=sample_rate,
        method=method,
        use_sinusoidal_model=use_sinusoidal_model,
        use_spectral_subtraction=use_spectral_subtraction,
        use_tikhonov=use_tikhonov,
    )
    metrics = compute_metrics(original, reconstructed)

    return _build_response(time_axis, original, spoiled, reconstructed, mask, metrics, sample_rate)


@app.get("/api/compare")
async def compare_methods(
    dropout_pct: float = 15.0,
    dropout_length_ms: float = 100.0,
    glitch_pct: float = 5.0,
    clip_pct: float = 10.0,
    noise_level: float = 0.02,
    seed: Optional[int] = None,
):
    """
    Compare baseline vs advanced reconstruction across all methods.
    
    Returns detailed metrics showing improvement for each method.
    Useful for demonstrating the effectiveness of advanced DSP techniques.
    """
    sample_rate, samples = generate_demo_signal(duration=1.0, sr=8000)
    
    time_axis = np.arange(len(samples)) / sample_rate
    original = samples.astype(np.float64)
    
    spoiled, mask = degrade_signal(
        original,
        sample_rate=sample_rate,
        dropout_pct=dropout_pct,
        dropout_length_ms=dropout_length_ms,
        glitch_pct=glitch_pct,
        clip_pct=clip_pct,
        noise_level=noise_level,
        seed=seed,
    )
    
    # Compare all methods
    comparison = compare_reconstruction_methods(
        time_axis, original, spoiled, mask, sample_rate
    )
    
    # Add summary statistics
    summary = {
        "damage_config": {
            "dropout_pct": dropout_pct,
            "dropout_length_ms": dropout_length_ms,
            "glitch_pct": glitch_pct,
            "clip_pct": clip_pct,
            "noise_level": noise_level,
        },
        "best_baseline_method": None,
        "best_baseline_snr": -float('inf'),
        "best_advanced_method": None,
        "best_advanced_snr": -float('inf'),
        "average_snr_improvement": 0.0,
        "average_mse_reduction_pct": 0.0,
    }
    
    snr_improvements = []
    mse_reductions = []
    
    for method, data in comparison.items():
        baseline_snr = data['baseline']['snr_db']
        advanced_snr = data['advanced']['snr_db']
        
        if baseline_snr > summary['best_baseline_snr']:
            summary['best_baseline_snr'] = baseline_snr
            summary['best_baseline_method'] = method
        
        if advanced_snr > summary['best_advanced_snr']:
            summary['best_advanced_snr'] = advanced_snr
            summary['best_advanced_method'] = method
        
        snr_improvements.append(data['improvement']['snr_db'])
        mse_reductions.append(data['improvement']['mse_reduction'])
    
    summary['average_snr_improvement'] = round(sum(snr_improvements) / len(snr_improvements), 2)
    summary['average_mse_reduction_pct'] = round(sum(mse_reductions) / len(mse_reductions), 2)
    
    return {
        "methods": comparison,
        "summary": summary,
    }


def _build_response(
    time_axis: np.ndarray,
    original: np.ndarray,
    spoiled: np.ndarray,
    reconstructed: np.ndarray,
    mask: np.ndarray,
    metrics: dict,
    sample_rate: int,
) -> dict:
    """Package signals for the frontend – includes downsampled plot data + audio."""

    # Downsample for plotting (keeps shape, reduces JSON size)
    t_plot, orig_plot = downsample_for_plot(time_axis, original, MAX_PLOT_POINTS)
    _, spoil_plot = downsample_for_plot(time_axis, spoiled, MAX_PLOT_POINTS)
    _, recon_plot = downsample_for_plot(time_axis, reconstructed, MAX_PLOT_POINTS)

    # Generate playable audio (base64-encoded WAV)
    orig_wav = signal_to_wav_base64(original, sample_rate)
    spoil_wav = signal_to_wav_base64(spoiled, sample_rate)
    recon_wav = signal_to_wav_base64(reconstructed, sample_rate)

    return {
        "sampleRate": int(sample_rate),
        "totalSamples": int(len(original)),
        "plot": {
            "time": t_plot.tolist(),
            "original": orig_plot.tolist(),
            "spoiled": spoil_plot.tolist(),
            "reconstructed": recon_plot.tolist(),
        },
        "audio": {
            "original": orig_wav,
            "spoiled": spoil_wav,
            "reconstructed": recon_wav,
        },
        "metrics": metrics,
        "mask": downsample_for_plot(time_axis, mask.astype(np.float64), MAX_PLOT_POINTS)[1].tolist(),
    }


# ═══════════════════════════════════════════════════════════════════
# Damaged Audio Repair Endpoints
# ═══════════════════════════════════════════════════════════════════

@app.post("/api/audio/repair")
async def repair_damaged_audio(
    file: UploadFile = File(...),
    method: str = Form("pchip"),
    auto_detect: bool = Form(True),
):
    """
    Repair a damaged audio file.
    
    - Detects corrupted/missing regions automatically
    - Reconstructs using interpolation
    - Returns repaired audio + visualization data
    """
    if not file.filename or not file.filename.lower().endswith(".wav"):
        raise HTTPException(status_code=400, detail="Only .wav files are supported.")

    raw = await file.read()
    if len(raw) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large (max 10 MB).")

    try:
        sample_rate, samples = load_wav_bytes(raw)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read WAV: {e}")

    # Normalize
    original = samples.astype(np.float64)
    peak = np.max(np.abs(original))
    if peak > 0:
        original = original / peak

    # Process damaged audio
    result = process_damaged_audio(
        original, 
        sample_rate, 
        method=method, 
        auto_detect=auto_detect
    )
    
    damaged = result["damaged"]
    repaired = result["repaired"]
    mask = result["mask"]
    analysis = result["analysis"]
    
    # Compute metrics (comparing damaged to repaired, since we don't have original)
    # For damaged audio repair, metrics show the "repair magnitude"
    repair_metrics = {
        "damage_percent": analysis["damage_percent"],
        "samples_repaired": int(np.sum(~mask)),
        "summary": analysis["summary"],
    }
    
    time_axis = np.arange(len(original)) / sample_rate
    
    return _build_repair_response(
        time_axis, damaged, repaired, mask, repair_metrics, sample_rate, analysis
    )


def _build_repair_response(
    time_axis: np.ndarray,
    damaged: np.ndarray,
    repaired: np.ndarray,
    mask: np.ndarray,
    metrics: dict,
    sample_rate: int,
    analysis: dict,
) -> dict:
    """Build response for damaged audio repair."""
    
    # Downsample for plotting
    t_plot, damaged_plot = downsample_for_plot(time_axis, damaged, MAX_PLOT_POINTS)
    _, repaired_plot = downsample_for_plot(time_axis, repaired, MAX_PLOT_POINTS)
    mask_plot = downsample_for_plot(time_axis, mask.astype(np.float64), MAX_PLOT_POINTS)[1]

    # Generate playable audio
    damaged_wav = signal_to_wav_base64(damaged, sample_rate)
    repaired_wav = signal_to_wav_base64(repaired, sample_rate)

    return {
        "sampleRate": int(sample_rate),
        "totalSamples": int(len(damaged)),
        "plot": {
            "time": t_plot.tolist(),
            "damaged": damaged_plot.tolist(),
            "reconstructed": repaired_plot.tolist(),
        },
        "audio": {
            "damaged": damaged_wav,
            "reconstructed": repaired_wav,
        },
        "metrics": metrics,
        "mask": mask_plot.tolist(),
        "analysis": {
            "summary": analysis.get("summary", ""),
            "damage_percent": analysis.get("damage_percent", 0),
            "stats": analysis.get("stats", {}),
        },
    }


# ═══════════════════════════════════════════════════════════════════
# Scientific Data Endpoints
# ═══════════════════════════════════════════════════════════════════

@app.get("/api/scidata/presets")
async def list_scidata_presets():
    """List available demo signal presets."""
    presets = []
    for key, info in DEMO_PRESETS.items():
        presets.append({
            "id": key,
            "name": info["name"],
            "description": info["description"],
        })
    return {"presets": presets}


@app.get("/api/scidata/demo")
async def scidata_demo(
    preset: str = Query("ecg", description="Demo preset name"),
    dropout_pct: float = Query(15.0, ge=0, le=80),
    noise_level: float = Query(0.02, ge=0, le=0.5),
    method: str = Query("pchip"),
):
    """
    Generate a demo scientific signal with degradation and reconstruction.
    """
    try:
        time_axis, signal = get_demo_signal(preset)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    # Normalize
    normalized, sig_min, sig_max = normalize_signal(signal)
    
    # Degrade
    spoiled, mask = degrade_scidata(normalized, dropout_pct, noise_level)
    
    # Reconstruct
    reconstructed = reconstruct_scidata(time_axis, spoiled, mask, method=method)
    
    # Compute metrics
    metrics = compute_scidata_metrics(normalized, reconstructed)
    
    return _build_scidata_response(
        time_axis, normalized, spoiled, reconstructed, mask, metrics,
        preset_name=DEMO_PRESETS[preset]["name"],
        unit_info={"min": float(sig_min), "max": float(sig_max)},
    )


@app.post("/api/scidata/process")
async def process_scidata(
    file: UploadFile = File(...),
    dropout_pct: float = Form(15.0),
    noise_level: float = Form(0.02),
    method: str = Form("pchip"),
):
    """
    Process uploaded scientific time-series data.
    
    Accepts CSV or JSON format.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided.")
    
    filename = file.filename.lower()
    if not (filename.endswith(".csv") or filename.endswith(".json")):
        raise HTTPException(status_code=400, detail="Only .csv and .json files supported.")

    raw = await file.read()
    if len(raw) > 5 * 1024 * 1024:  # 5 MB limit for data files
        raise HTTPException(status_code=400, detail="File too large (max 5 MB).")

    try:
        time_axis, signal = load_scidata(raw, file.filename)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse file: {e}")
    
    # Limit to 50000 samples
    if len(signal) > 50000:
        step = len(signal) // 50000
        time_axis = time_axis[::step]
        signal = signal[::step]
    
    # Normalize
    normalized, sig_min, sig_max = normalize_signal(signal)
    
    # Degrade
    spoiled, mask = degrade_scidata(normalized, dropout_pct, noise_level)
    
    # Reconstruct
    reconstructed = reconstruct_scidata(time_axis, spoiled, mask, method=method)
    
    # Compute metrics
    metrics = compute_scidata_metrics(normalized, reconstructed)
    
    return _build_scidata_response(
        time_axis, normalized, spoiled, reconstructed, mask, metrics,
        preset_name=file.filename,
        unit_info={"min": float(sig_min), "max": float(sig_max)},
    )


@app.post("/api/scidata/reconstruct")
async def scidata_reconstruct_only(body: dict):
    """Re-run reconstruction with different method on scientific data."""
    try:
        time_axis = np.array(body["time"], dtype=np.float64)
        spoiled = np.array(body["spoiled"], dtype=np.float64)
        mask = np.array(body["mask"], dtype=bool)
        original = np.array(body["original"], dtype=np.float64)
        method = body.get("method", "pchip")
    except (KeyError, ValueError) as e:
        raise HTTPException(status_code=400, detail=f"Invalid payload: {e}")

    reconstructed = reconstruct_scidata(time_axis, spoiled, mask, method=method)
    metrics = compute_scidata_metrics(original, reconstructed)

    return _build_scidata_response(
        time_axis, original, spoiled, reconstructed, mask, metrics,
        preset_name=body.get("name", "Custom"),
        unit_info=body.get("unitInfo", {"min": -1, "max": 1}),
    )


def _build_scidata_response(
    time_axis: np.ndarray,
    original: np.ndarray,
    spoiled: np.ndarray,
    reconstructed: np.ndarray,
    mask: np.ndarray,
    metrics: dict,
    preset_name: str = "Signal",
    unit_info: dict = None,
) -> dict:
    """Build standardized response for scientific data."""
    
    # Downsample for plotting
    t_plot, orig_plot = downsample_for_plot(time_axis, original, MAX_PLOT_POINTS)
    _, spoil_plot = downsample_for_plot(time_axis, spoiled, MAX_PLOT_POINTS)
    _, recon_plot = downsample_for_plot(time_axis, reconstructed, MAX_PLOT_POINTS)
    mask_plot = downsample_for_plot(time_axis, mask.astype(np.float64), MAX_PLOT_POINTS)[1]

    return {
        "name": preset_name,
        "totalSamples": int(len(original)),
        "plot": {
            "time": t_plot.tolist(),
            "original": orig_plot.tolist(),
            "spoiled": spoil_plot.tolist(),
            "reconstructed": recon_plot.tolist(),
        },
        "metrics": metrics,
        "mask": mask_plot.tolist(),
        "unitInfo": unit_info or {"min": -1, "max": 1},
    }

