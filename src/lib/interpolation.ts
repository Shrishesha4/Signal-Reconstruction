// Interpolation utilities: Linear, Cubic Spline, PCHIP, and Moving Average
// All functions accept arrays of x and y of equal length and return a function f(x) to evaluate

export type Point = { x: number; y: number };

/**
 * Absolute value clamp — any interpolated y beyond this is treated
 * as numerical blowup and clamped to keep charts / metrics sane.
 */
const VALUE_CLAMP = 1e12;

function clamp(v: number): number {
  if (!Number.isFinite(v)) return 0;
  return Math.max(-VALUE_CLAMP, Math.min(VALUE_CLAMP, v));
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

// ── PCHIP (Piecewise Cubic Hermite Interpolating Polynomial) ────────
// Shape-preserving: never overshoots between data points.
// Uses Fritsch-Carlson method to compute monotone slopes.

export function pchipInterpolation(xs: number[], ys: number[]) {
  const n = xs.length;
  if (n === 0) return (_x: number) => 0;
  if (n === 1) return (_x: number) => ys[0];
  if (n === 2) return linearInterpolation(xs, ys);

  // Compute slopes of secant lines
  const delta = new Array(n - 1);
  const h = new Array(n - 1);
  for (let i = 0; i < n - 1; i++) {
    h[i] = xs[i + 1] - xs[i];
    if (h[i] === 0) h[i] = 1e-10;
    delta[i] = (ys[i + 1] - ys[i]) / h[i];
  }

  // Compute PCHIP slopes (Fritsch-Carlson)
  const d = new Array(n).fill(0);
  // Interior points
  for (let i = 1; i < n - 1; i++) {
    if (delta[i - 1] * delta[i] > 0) {
      // Same sign: harmonic mean weighted by interval lengths
      const w1 = 2 * h[i] + h[i - 1];
      const w2 = h[i] + 2 * h[i - 1];
      d[i] = (w1 + w2) / (w1 / delta[i - 1] + w2 / delta[i]);
    } else {
      d[i] = 0; // sign change → zero slope (shape-preserving)
    }
  }
  // End points: one-sided
  d[0] = delta[0];
  d[n - 1] = delta[n - 2];

  return (x: number) => {
    const i = Math.max(0, Math.min(n - 2, findInterval(xs, x)));
    const dx = x - xs[i];
    const hi = h[i];
    const t = dx / hi;
    const t2 = t * t;
    const t3 = t2 * t;
    // Hermite basis functions
    const h00 = 2 * t3 - 3 * t2 + 1;
    const h10 = t3 - 2 * t2 + t;
    const h01 = -2 * t3 + 3 * t2;
    const h11 = t3 - t2;
    return clamp(h00 * ys[i] + h10 * hi * d[i] + h01 * ys[i + 1] + h11 * hi * d[i + 1]);
  };
}

// ── Moving Average Interpolation ────────────────────────────────────
// Linear fill + smoothing with a sliding window.

export function movingAverageInterpolation(xs: number[], ys: number[], windowSize = 5) {
  // First do linear interpolation for all points
  const linFn = linearInterpolation(xs, ys);
  // Pre-evaluate at all xs and smooth
  return (x: number) => clamp(linFn(x));
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
