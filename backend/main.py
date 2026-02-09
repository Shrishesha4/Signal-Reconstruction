"""
Signal Reconstruction Backend – FastAPI
=======================================
Handles audio loading, degradation, interpolation-based reconstruction,
and error metrics for the SvelteKit frontend.

API Contract:
  POST /api/process
    - Accepts: multipart form with audio file (.wav) + degradation params
    - Returns: JSON with original, spoiled, reconstructed signals + metrics

  GET /api/demo
    - Returns: pre-loaded demo signal processed with default params

  POST /api/reconstruct
    - Accepts: JSON with spoiled signal + method choice
    - Returns: reconstructed signal + metrics
"""

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np

from .processing import (
    load_wav_bytes,
    degrade_signal,
    reconstruct_signal,
    compute_metrics,
    generate_demo_signal,
    signal_to_wav_base64,
    downsample_for_plot,
)

app = FastAPI(
    title="Signal Reconstruction API",
    description="Audio signal degradation and numerical interpolation reconstruction",
    version="1.0.0",
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
    dropout_pct: float = 20.0,
    noise_level: float = 0.02,
    method: str = "pchip",
):
    """Return a demo sine-composite signal, degraded and reconstructed."""
    sample_rate, samples = generate_demo_signal(duration=1.0, sr=8000)

    time_axis = np.arange(len(samples)) / sample_rate
    original = samples.astype(np.float64)

    spoiled, mask = degrade_signal(original, dropout_pct=dropout_pct, noise_level=noise_level)
    reconstructed = reconstruct_signal(time_axis, spoiled, mask, method=method)
    metrics = compute_metrics(original, reconstructed)

    return _build_response(time_axis, original, spoiled, reconstructed, mask, metrics, sample_rate)


@app.post("/api/process")
async def process_audio(
    file: UploadFile = File(...),
    dropout_pct: float = Form(20.0),
    noise_level: float = Form(0.02),
    method: str = Form("pchip"),
):
    """Full pipeline: load WAV → degrade → reconstruct → return."""
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

    spoiled, mask = degrade_signal(original, dropout_pct=dropout_pct, noise_level=noise_level)
    reconstructed = reconstruct_signal(time_axis, spoiled, mask, method=method)
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

    reconstructed = reconstruct_signal(time_axis, spoiled, mask, method=method)
    metrics = compute_metrics(original, reconstructed)

    return _build_response(time_axis, original, spoiled, reconstructed, mask, metrics, sample_rate)


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
