// Interpolation utilities: Linear, Barycentric Lagrange, and Natural Cubic Spline
// All functions accept arrays of x and y of equal length and return a function f(x) to evaluate

export type Point = { x: number; y: number };

/**
 * Maximum safe polynomial degree for Lagrange interpolation.
 * Beyond this, Runge phenomenon causes extreme oscillation.
 * Set conservatively to prevent visualization issues.
 */
export const LAGRANGE_MAX_SAFE_DEGREE = 12;

/**
 * Absolute value clamp — any interpolated y beyond this is treated
 * as numerical blowup and clamped to keep charts / metrics sane.
 */
const VALUE_CLAMP = 1e12;

function clamp(v: number): number {
  if (!Number.isFinite(v)) return 0;
  return Math.max(-VALUE_CLAMP, Math.min(VALUE_CLAMP, v));
}

/**
 * Range-aware clamp for Lagrange interpolation.
 * Prevents extreme oscillations from breaking visualization.
 */
function clampToRange(v: number, ys: number[]): number {
  if (!Number.isFinite(v)) return 0;
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  const range = maxY - minY;
  const margin = Math.max(range * 3, 0.1); // Allow 3x range, minimum 0.1
  const lowerBound = minY - margin;
  const upperBound = maxY + margin;
  return Math.max(lowerBound, Math.min(upperBound, v));
}

// ── Linear Interpolation (piecewise) ────────────────────────────────

export function linearInterpolation(xs: number[], ys: number[]) {
  return (x: number) => {
    const n = xs.length;
    if (n === 0) return 0;
    if (n === 1) return ys[0];
    if (x <= xs[0]) return ys[0];
    if (x >= xs[n - 1]) return ys[n - 1];
    let i = 0;
    while (i < n - 1 && xs[i + 1] < x) i++;
    const x0 = xs[i];
    const x1 = xs[i + 1];
    const y0 = ys[i];
    const y1 = ys[i + 1];
    const denom = x1 - x0;
    if (denom === 0) return y0;
    const t = (x - x0) / denom;
    return y0 * (1 - t) + y1 * t;
  };
}

// ── Barycentric Lagrange Interpolation (numerically stable) ─────────
// O(n) evaluation after O(n²) precomputation of weights.
// Avoids the catastrophic cancellation of the classic formulation.
// Uses range-aware clamping to prevent Runge oscillations from breaking charts.

export function lagrangeInterpolation(xs: number[], ys: number[]) {
  const n = xs.length;
  if (n === 0) return (_x: number) => 0;
  if (n === 1) return (_x: number) => ys[0];

  // Precompute barycentric weights
  const w = new Array(n);
  for (let j = 0; j < n; j++) {
    let wj = 1;
    for (let i = 0; i < n; i++) {
      if (i === j) continue;
      const diff = xs[j] - xs[i];
      // guard against duplicate x-values
      if (diff === 0) { wj = 0; break; }
      wj /= diff;
    }
    w[j] = wj;
  }

  return (x: number) => {
    let numerator = 0;
    let denominator = 0;

    for (let j = 0; j < n; j++) {
      const diff = x - xs[j];
      // If x is exactly at a node, return that node's y
      if (diff === 0) return ys[j];
      const term = w[j] / diff;
      numerator += term * ys[j];
      denominator += term;
    }

    if (denominator === 0) return 0;
    const result = numerator / denominator;
    // Use range-aware clamping to prevent extreme Runge oscillations
    return clampToRange(result, ys);
  };
}

/**
 * Check whether Lagrange interpolation is safe for the given dataset.
 * Returns { safe, reason } indicating whether it should be used.
 * Compares the polynomial degree (n-1) against the safe limit.
 */
export function isLagrangeSafe(n: number): { safe: boolean; reason: string } {
  const degree = n - 1;
  if (degree <= LAGRANGE_MAX_SAFE_DEGREE) {
    return { safe: true, reason: '' };
  }
  return {
    safe: false,
    reason: `Polynomial degree ${degree} exceeds safe limit (${LAGRANGE_MAX_SAFE_DEGREE}). Runge phenomenon may cause extreme oscillation.`
  };
}

// ── Natural Cubic Spline ────────────────────────────────────────────

export function cubicSplineInterpolation(xs: number[], ys: number[]) {
  const n = xs.length;
  if (n === 0) return (_x: number) => 0;
  if (n === 1) return (_x: number) => ys[0];
  if (n === 2) return linearInterpolation(xs, ys);

  const h = new Array(n - 1);
  for (let i = 0; i < n - 1; i++) {
    h[i] = xs[i + 1] - xs[i];
    if (h[i] === 0) h[i] = 1e-10; // guard zero-width intervals
  }

  const alpha = new Array(n).fill(0);
  for (let i = 1; i < n - 1; i++) {
    alpha[i] = (3 / h[i]) * (ys[i + 1] - ys[i]) - (3 / h[i - 1]) * (ys[i] - ys[i - 1]);
  }

  const l = new Array(n).fill(0);
  const mu = new Array(n).fill(0);
  const z = new Array(n).fill(0);

  l[0] = 1;

  for (let i = 1; i < n - 1; i++) {
    l[i] = 2 * (xs[i + 1] - xs[i - 1]) - h[i - 1] * mu[i - 1];
    if (l[i] === 0) l[i] = 1e-10; // guard
    mu[i] = h[i] / l[i];
    z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i];
  }

  l[n - 1] = 1;
  z[n - 1] = 0;

  const b = new Array(n - 1).fill(0);
  const c = new Array(n).fill(0);
  const d = new Array(n - 1).fill(0);

  for (let j = n - 2; j >= 0; j--) {
    c[j] = z[j] - mu[j] * c[j + 1];
    b[j] = (ys[j + 1] - ys[j]) / h[j] - (h[j] * (c[j + 1] + 2 * c[j])) / 3;
    d[j] = (c[j + 1] - c[j]) / (3 * h[j]);
  }

  return (x: number) => {
    const i = Math.max(0, Math.min(n - 2, findInterval(xs, x)));
    const dx = x - xs[i];
    return clamp(ys[i] + b[i] * dx + c[i] * dx * dx + d[i] * dx * dx * dx);
  };
}

function findInterval(xs: number[], x: number) {
  const n = xs.length;
  if (x <= xs[0]) return 0;
  if (x >= xs[n - 1]) return n - 2;
  // binary search for efficiency with large datasets
  let lo = 0, hi = n - 2;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (xs[mid + 1] < x) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}
