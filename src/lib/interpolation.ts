/**
 * interpolation.ts — DEPRECATED
 * ==============================
 *
 * All numerical interpolation has been moved to the **Python backend**
 * (``backend/interpolation.py``) using proven SciPy implementations:
 *
 *   - Linear          → scipy.interpolate.interp1d(kind='linear')
 *   - Cubic Spline    → scipy.interpolate.CubicSpline (natural BC)
 *   - PCHIP           → scipy.interpolate.PchipInterpolator
 *   - Moving Average  → linear fill + scipy.ndimage.uniform_filter1d
 *
 * 🚫  Lagrange interpolation is numerically unstable for dense and
 *     noisy signals such as audio and has been removed entirely.
 *
 * The frontend now acts as a **visualization-only** layer.
 * Reconstructed signals are fetched from the backend via the API
 * defined in ``$lib/api.ts``.
 *
 * These stubs are retained **only** to avoid import errors from any
 * remaining references.  They are never called in the production flow.
 */

export type Point = { x: number; y: number };

/** @deprecated — interpolation runs on the Python backend. */
export function linearInterpolation(_xs: number[], _ys: number[]) {
	console.warn('linearInterpolation: use the Python backend instead.');
	return (_x: number) => 0;
}

/** @deprecated — interpolation runs on the Python backend. */
export function cubicSplineInterpolation(_xs: number[], _ys: number[]) {
	console.warn('cubicSplineInterpolation: use the Python backend instead.');
	return (_x: number) => 0;
}

/** @deprecated — interpolation runs on the Python backend. */
export function pchipInterpolation(_xs: number[], _ys: number[]) {
	console.warn('pchipInterpolation: use the Python backend instead.');
	return (_x: number) => 0;
}

/** @deprecated — interpolation runs on the Python backend. */
export function movingAverageInterpolation(_xs: number[], _ys: number[], _windowSize = 5) {
	console.warn('movingAverageInterpolation: use the Python backend instead.');
	return (_x: number) => 0;
}
