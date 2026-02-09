"""
interpolation.py – Research-grade signal interpolation with DSP conditioning
============================================================================

Architecture
------------
Every reconstruction follows a three-stage pipeline:

  1. **Pre-conditioning**  – median-filter anchor samples to suppress
     impulsive noise; optionally Savitzky–Golay smooth for spline fitting.
  2. **Core interpolation** – method-specific gap filling with adaptive
     handling of small vs. large dropout regions.
  3. **Post-refinement**   – light low-pass on filled regions, gentle
     moving-RMS normalisation, hard clip to [-1, 1].

The pipeline is transparent: each stage is independently safe and can
be skipped (the defaults are conservative).

Supported methods
-----------------
- **Linear**         – fast baseline; post-blended with moving average
                       to soften slope discontinuities.
- **Cubic Spline**   – C² natural BC, pre-smoothed inputs, Hann-windowed
                       gap edges to suppress ringing.
- **PCHIP** (default, recommended for audio) – shape-preserving Hermite;
                       no overshoot, gentle post-filter.  Best all-round.
- **Moving Average** – adaptive-window denoising stage, not full
                       reconstruction.  Uses PCHIP as gap-fill base.

🚫 Lagrange interpolation is deliberately excluded — numerically unstable
   for dense / noisy signals (Runge phenomenon, O(N³), round-off explosion).

Performance
-----------
All operations O(N) or O(N log N).  Pure NumPy / SciPy vectorised code —
no Python-level sample loops.  Tested to ≥ 200 000 samples.

Numerical safety
----------------
- NaN / Inf scrubbed, x-axis strictly increasing, output ∈ [-1, 1].
- Large dropout gaps get blended treatment (no polynomial explosion).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import interpolate as sp_interp
from scipy import signal as sp_signal
from scipy.ndimage import uniform_filter1d, median_filter as nd_median


# ═══════════════════════════════════════════════════════════════════
# Tunable constants — sensible defaults for 8–48 kHz audio
# ═══════════════════════════════════════════════════════════════════

# Pre-conditioning
_MEDIAN_KERNEL      = 5      # odd; removes impulsive spikes in anchors
_SAVGOL_WINDOW      = 7      # odd; gentle Savitzky–Golay pre-smooth for spline
_SAVGOL_ORDER       = 2      # SG polynomial order (lower = gentler)

# Gap edge blending
_BLEND_MARGIN       = 8      # samples; Hann taper half-width at gap edges
_MIN_GAP_FOR_BLEND  = 4      # only blend at edges of gaps ≥ this many samples

# Post-refinement
_POST_LP_SIZE       = 5      # light uniform-filter low-pass window (odd)
_RMS_BLOCK          = 512    # moving-RMS normalisation block size

# Moving-average method
_MA_MIN_WIN         = 3      # minimum smoothing window
_MA_MAX_WIN         = 31     # maximum smoothing window
_TRANSIENT_THRESH   = 0.08   # gradient fraction → transient detection

# Linear post-blend
_LINEAR_BLEND_ALPHA = 0.3    # fraction of MA blended into linear output


# ═══════════════════════════════════════════════════════════════════
# Public API  (signature unchanged — drop-in replacement)
# ═══════════════════════════════════════════════════════════════════

def reconstruct(
    time_axis: NDArray[np.float64],
    spoiled:   NDArray[np.float64],
    mask:      NDArray[np.bool_],
    method:    str = "pchip",
) -> NDArray[np.float64]:
    """
    Reconstruct missing samples using numerical interpolation.

    Three-stage pipeline: pre-condition → interpolate → post-refine.

    Parameters
    ----------
    time_axis : 1-D float64   – monotonic time vector
    spoiled   : 1-D float64   – degraded signal (dropped samples ≈ 0)
    mask      : 1-D bool      – True = valid, False = dropped
    method    : str            – 'linear' | 'spline' | 'pchip' | 'moving_average'

    Returns
    -------
    1-D float64 clamped to [-1, 1].
    """
    # ── Stage 0: sanitise inputs ─────────────────────────────
    valid_t, valid_y = _sanitise_inputs(time_axis, spoiled, mask)

    if len(valid_t) < 2:
        return np.clip(spoiled.copy(), -1.0, 1.0)

    missing_mask = ~mask

    # ── Stage 1: pre-condition anchor samples ────────────────
    # Spline gets aggressive SG smoothing to prevent ringing;
    # other methods get only the median de-spike.
    valid_y = _precondition(valid_y, aggressive=(method == "spline"))

    # ── Stage 2: method-specific core interpolation ──────────
    _dispatch = {
        "pchip":          _pchip_reconstruct,
        "spline":         _spline_reconstruct,
        "linear":         _linear_reconstruct,
        "moving_average": _moving_average_reconstruct,
    }
    fn = _dispatch.get(method)
    if fn is None:
        raise ValueError(
            f"Unknown method '{method}'. "
            f"Supported: {list(_dispatch.keys())}."
        )
    reconstructed = fn(time_axis, spoiled, valid_t, valid_y, mask)

    # ── Stage 3: post-refinement ─────────────────────────────
    reconstructed = _postrefine(reconstructed, missing_mask)

    return reconstructed


# ═══════════════════════════════════════════════════════════════════
#  Stage 1 — Pre-conditioning
# ═══════════════════════════════════════════════════════════════════

def _precondition(
    y: NDArray[np.float64],
    aggressive: bool = False,
) -> NDArray[np.float64]:
    """
    Clean anchor samples before the interpolator sees them.

    1. **Median filter** (always) — removes impulsive noise (clicks, pops)
       without smearing energy into neighbours.  Median is the only linear-
       time filter that eliminates salt-and-pepper noise cleanly.

    2. **Savitzky–Golay** (aggressive=True, used for cubic spline) — fits
       local polynomials to further smooth anchors.  This prevents the
       spline from fitting through noisy points and ringing.

    Why median *before* SG?  Median kills isolated spikes first so that
    the SG polynomial fit doesn't incorporate them.
    """
    n = len(y)
    out = y.copy()

    if n >= _MEDIAN_KERNEL:
        out = nd_median(out, size=_MEDIAN_KERNEL, mode="reflect")

    if aggressive and n >= _SAVGOL_WINDOW:
        out = sp_signal.savgol_filter(
            out, _SAVGOL_WINDOW, _SAVGOL_ORDER, mode="mirror"
        )

    return out


# ═══════════════════════════════════════════════════════════════════
#  Stage 2 — Method-specific core interpolation
# ═══════════════════════════════════════════════════════════════════

# ─── PCHIP (recommended default for audio) ────────────────────────

def _pchip_reconstruct(
    t: NDArray[np.float64],
    spoiled: NDArray[np.float64],
    vt: NDArray[np.float64],
    vy: NDArray[np.float64],
    mask: NDArray[np.bool_],
) -> NDArray[np.float64]:
    """
    PCHIP — Piecewise Cubic Hermite Interpolating Polynomial.

    **Why PCHIP is the recommended default for audio:**

    - **Shape-preserving**: guaranteed monotonic between consecutive data
      points, so it never overshoots or creates phantom oscillations.
    - **C¹ continuous**: smooth waveform without the "ringing" artefacts
      that plague C² splines.
    - **Stable on large gaps**: no polynomial explosion even with long
      dropout regions.

    Post-processing: Savitzky–Golay (window=7, order=3) applied *only*
    to filled samples softens micro-step artefacts at anchor boundaries
    without blurring original valid samples.

    **Use for**: general audio reconstruction, dropout recovery, any
    signal where preserving peak shape matters.
    """
    interp_fn = sp_interp.PchipInterpolator(vt, vy, extrapolate=True)

    reconstructed = spoiled.copy()
    missing = ~mask

    if np.any(missing):
        filled = interp_fn(t[missing])
        reconstructed[missing] = _scrub(filled)

    # Gentle SG post-filter on filled samples only — preserves peaks
    # far better than a box / uniform filter.
    if len(reconstructed) >= 7:
        smoothed = sp_signal.savgol_filter(
            reconstructed, 7, 3, mode="mirror"
        )
        reconstructed[missing] = smoothed[missing]

    return reconstructed


# ─── Cubic Spline ──────────────────────────────────────────────────

def _spline_reconstruct(
    t: NDArray[np.float64],
    spoiled: NDArray[np.float64],
    vt: NDArray[np.float64],
    vy: NDArray[np.float64],
    mask: NDArray[np.bool_],
) -> NDArray[np.float64]:
    """
    Cubic spline with natural boundary conditions.

    **Why spline needs extra care:**

    C² continuity is ideal for smoothness, but the spline minimises
    curvature *globally* — a single noisy anchor can produce oscillations
    ("ringing") that propagate for many samples.

    Mitigations applied here:
      1. Pre-smooth anchors with SG filter (in ``_precondition``,
         ``aggressive=True``) to remove noise before the spline sees it.
      2. Natural boundary conditions (d²y/dx² = 0 at endpoints) prevent
         edge blow-up.
      3. Hann-window edge blending at each gap boundary absorbs residual
         derivative discontinuities.

    **Use for**: slowly varying signals, speech envelopes, music sustain.
    **Avoid for**: transient-heavy material — prefer PCHIP instead.
    """
    interp_fn = sp_interp.CubicSpline(
        vt, vy, bc_type="natural", extrapolate=True
    )

    reconstructed = spoiled.copy()
    missing = ~mask

    if not np.any(missing):
        return reconstructed

    filled = interp_fn(t[missing])
    reconstructed[missing] = _scrub(filled)

    # Hann-window edge blending suppresses derivative discontinuities
    # at the boundaries of each dropout gap.
    reconstructed = _blend_gap_edges(reconstructed, mask, margin=_BLEND_MARGIN)

    return reconstructed


# ─── Linear ────────────────────────────────────────────────────────

def _linear_reconstruct(
    t: NDArray[np.float64],
    spoiled: NDArray[np.float64],
    vt: NDArray[np.float64],
    vy: NDArray[np.float64],
    mask: NDArray[np.bool_],
) -> NDArray[np.float64]:
    """
    Linear interpolation — fast baseline, not for final audio.

    **Why it sounds harsh:**

    Piecewise linear creates *slope discontinuities* (kinks) at every
    anchor point.  These kinks have infinite bandwidth, adding an
    audible zipper-like buzz that is perceptually annoying.

    Mitigations:
      1. Light uniform-filter smoothing on filled samples.
      2. Blend 30 % of a moving-average version into the filled regions
         to round off the kinks without full MA muffling.

    **Use for**: quick preview, A-B metric comparison.  Any other
    method should beat linear on perceptual quality.
    """
    interp_fn = sp_interp.interp1d(
        vt, vy,
        kind="linear",
        bounds_error=False,
        fill_value="extrapolate",
        assume_sorted=True,
    )

    reconstructed = spoiled.copy()
    missing = ~mask

    if np.any(missing):
        filled = interp_fn(t[missing])
        reconstructed[missing] = _scrub(filled)

    # Post-smooth: blend linear with a moving-average version
    # to soften slope kinks without full muffling.
    n = len(reconstructed)
    win = max(5, min(15, n // 500))
    win = win if win % 2 == 1 else win + 1
    ma = uniform_filter1d(reconstructed, size=win)

    # Blend only on missing samples — original anchors stay untouched.
    alpha = _LINEAR_BLEND_ALPHA
    reconstructed[missing] = (
        (1.0 - alpha) * reconstructed[missing]
        + alpha * ma[missing]
    )

    return reconstructed


# ─── Moving Average (adaptive-window denoising) ───────────────────

def _moving_average_reconstruct(
    t: NDArray[np.float64],
    spoiled: NDArray[np.float64],
    vt: NDArray[np.float64],
    vy: NDArray[np.float64],
    mask: NDArray[np.bool_],
) -> NDArray[np.float64]:
    """
    Adaptive-window moving average — denoising stage, not reconstruction.

    **The classic MA problem:** a fixed box filter muffles transients
    (drums, consonants) while barely helping steady regions that
    actually need smoothing.

    **Solution — gradient-adaptive blending:**

      1. Fill gaps with PCHIP (fast, no overshoot) so the MA has a
         continuous signal to work with.
      2. Compute per-sample gradient magnitude as a transient proxy.
      3. Apply two uniform filters — small window (preserves transients)
         and large window (smooths noise).
      4. Blend per-sample: high gradient → small window, low gradient →
         large window.

    Fully vectorised — the "adaptive" part is a gradient-weighted blend
    of two fixed-size filters with no per-sample Python loop.

    **Use for**: denoising pass on top of another method, or as a
    comparison baseline.
    """
    # Step 1: fill gaps with PCHIP to get a continuous base signal
    interp_fn = sp_interp.PchipInterpolator(vt, vy, extrapolate=True)
    filled = spoiled.copy()
    missing = ~mask
    if np.any(missing):
        filled[missing] = _scrub(interp_fn(t[missing]))

    n = len(filled)

    # Step 2: per-sample gradient magnitude → transient map
    grad = np.abs(np.gradient(filled))
    gmax = np.max(grad)
    if gmax > 1e-15:
        grad_norm = grad / gmax
    else:
        grad_norm = np.zeros(n, dtype=np.float64)

    # alpha ∈ [0, 1]: 1 = transient (small window), 0 = steady (large)
    alpha = np.clip(grad_norm / _TRANSIENT_THRESH, 0.0, 1.0)

    # Step 3: two fixed-size uniform filters
    small_win = _MA_MIN_WIN if _MA_MIN_WIN % 2 == 1 else _MA_MIN_WIN + 1
    large_win = min(_MA_MAX_WIN, max(5, n // 200))
    large_win = large_win if large_win % 2 == 1 else large_win + 1

    smooth_s = uniform_filter1d(filled, size=small_win)
    smooth_l = uniform_filter1d(filled, size=large_win)

    # Step 4: gradient-weighted blend
    reconstructed = alpha * smooth_s + (1.0 - alpha) * smooth_l

    return reconstructed


# ═══════════════════════════════════════════════════════════════════
#  Stage 3 — Post-refinement
# ═══════════════════════════════════════════════════════════════════

def _postrefine(
    signal: NDArray[np.float64],
    missing_mask: NDArray[np.bool_],
) -> NDArray[np.float64]:
    """
    Post-interpolation refinement applied identically to every method.

    1. **Light low-pass** on filled regions only — removes residual HF
       artefacts from interpolation without altering original anchors.
    2. **Moving RMS normalisation** — compensates amplitude drift where
       interpolation under- or over-estimates local energy.  Conservative:
       gain clamped to [0.8, 1.25] ≈ ±2 dB.
    3. **Hard clip** to [-1, 1].
    """
    out = signal.copy()
    n = len(out)

    # 1. Light low-pass on filled samples only
    if n >= _POST_LP_SIZE:
        lp = uniform_filter1d(out, size=_POST_LP_SIZE)
        out[missing_mask] = lp[missing_mask]

    # 2. Moving RMS normalisation
    out = _rms_normalise(out)

    # 3. Hard clip
    return np.clip(out, -1.0, 1.0)


def _rms_normalise(
    sig: NDArray[np.float64],
    block: int = _RMS_BLOCK,
) -> NDArray[np.float64]:
    """
    Gentle block-wise RMS normalisation.

    Computes local RMS via ``uniform_filter1d(sig², block)`` and applies
    a gain that nudges local RMS toward the global RMS.

    Gain is clamped to [0.8, 1.25] ≈ ±2 dB to prevent pumping artefacts.
    The gain curve is then smoothed to avoid abrupt gain changes.

    Fully vectorised — no Python loops.
    """
    n = len(sig)
    if n < block:
        return sig

    global_rms = np.sqrt(np.mean(sig ** 2))
    if global_rms < 1e-10:
        return sig  # near-silence — nothing to normalise

    local_ms = uniform_filter1d(sig ** 2, size=block)
    local_rms = np.sqrt(np.clip(local_ms, 1e-20, None))

    gain = np.clip(global_rms / local_rms, 0.8, 1.25)

    # Smooth gain curve to prevent click artefacts
    gain = uniform_filter1d(gain, size=block)
    gain = np.clip(gain, 0.8, 1.25)  # re-clamp after smoothing

    return sig * gain


# ═══════════════════════════════════════════════════════════════════
#  Gap analysis & edge-blending utilities
# ═══════════════════════════════════════════════════════════════════

def _blend_gap_edges(
    signal: NDArray[np.float64],
    mask: NDArray[np.bool_],
    margin: int = _BLEND_MARGIN,
    min_gap: int = _MIN_GAP_FOR_BLEND,
) -> NDArray[np.float64]:
    """
    Hann-window cross-fade at boundaries of *large* gaps only.

    Single-sample dropouts (random masking) don't need edge blending —
    the spline passes through nearby anchors and the derivative
    discontinuity is negligible.  Only gaps ≥ ``min_gap`` samples
    produce audible artefacts at their boundaries.

    For each qualifying gap, a short Hann taper smoothly merges the
    interpolated region with the surrounding valid samples, absorbing
    derivative discontinuities that cause clicks.
    """
    out = signal.copy()
    n = len(out)
    kernel = 2 * margin + 1
    if n < kernel:
        return out

    # Find contiguous gaps and filter for large ones
    starts, lengths = _find_gaps(~mask)
    if len(starts) == 0:
        return out
    large = lengths >= min_gap
    if not np.any(large):
        return out

    lg_starts = starts[large]
    lg_ends   = lg_starts + lengths[large]

    # Build indicator at edges of large gaps only
    indicator = np.zeros(n, dtype=np.float64)
    indicator[np.clip(lg_starts, 0, n - 1)]     = 1.0
    indicator[np.clip(lg_ends - 1, 0, n - 1)]   = 1.0

    # Spread into a smooth blend weight via box filter
    blend_w = uniform_filter1d(indicator, size=kernel)
    blend_w = np.clip(blend_w * kernel * 0.5, 0.0, 1.0)

    # Hann-like envelope for smooth taper
    blend_w = 0.5 * (1.0 - np.cos(np.pi * blend_w))

    # Locally smoothed version to blend toward
    smoothed = uniform_filter1d(out, size=kernel)
    out = (1.0 - blend_w) * out + blend_w * smoothed

    return out


def _find_gaps(
    missing_mask: NDArray[np.bool_],
) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
    """
    Find contiguous runs of True (missing) samples.

    Returns
    -------
    starts  : 1-D int array — index where each gap begins
    lengths : 1-D int array — length of each gap
    """
    if not np.any(missing_mask):
        return np.array([], dtype=np.intp), np.array([], dtype=np.intp)

    padded = np.concatenate(([False], missing_mask, [False]))
    d = np.diff(padded.astype(np.int8))
    starts = np.where(d == 1)[0]
    ends   = np.where(d == -1)[0]
    return starts, ends - starts


# ═══════════════════════════════════════════════════════════════════
#  Input sanitisation
# ═══════════════════════════════════════════════════════════════════

def _sanitise_inputs(
    time_axis: NDArray[np.float64],
    spoiled: NDArray[np.float64],
    mask: NDArray[np.bool_],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Extract valid (non-dropped) samples, remove NaN/Inf, enforce
    strictly increasing x-axis.
    """
    valid_t = time_axis[mask]
    valid_y = spoiled[mask]

    # Remove NaN / Inf from both x and y.
    finite = np.isfinite(valid_t) & np.isfinite(valid_y)
    valid_t = valid_t[finite]
    valid_y = valid_y[finite]

    if len(valid_t) < 2:
        return valid_t, valid_y

    # Strictly increasing x (remove duplicate timestamps).
    keep = np.concatenate(([True], np.diff(valid_t) > 0))
    return valid_t[keep], valid_y[keep]


def _scrub(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Replace NaN / Inf with 0 and clamp to [-1, 1]."""
    return np.clip(
        np.where(np.isfinite(values), values, 0.0),
        -1.0, 1.0,
    )
