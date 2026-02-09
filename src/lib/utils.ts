/**
 * Safe numeric helpers — all metric functions guarantee finite return values.
 * NaN / Infinity inputs are filtered out before computation.
 */

function safeFinite(v: number): number {
  return Number.isFinite(v) ? v : 0;
}

export function mean(arr: number[]) {
  if (arr.length === 0) return 0;
  return safeFinite(arr.reduce((a, b) => a + safeFinite(b), 0) / arr.length);
}

export function mse(a: number[], b: number[]) {
  const n = Math.min(a.length, b.length);
  if (n === 0) return 0;
  let s = 0;
  let count = 0;
  for (let i = 0; i < n; i++) {
    const ai = a[i];
    const bi = b[i];
    if (!Number.isFinite(ai) || !Number.isFinite(bi)) continue;
    const diff = ai - bi;
    s += diff * diff;
    count++;
  }
  return count === 0 ? 0 : safeFinite(s / count);
}

export function rmse(a: number[], b: number[]) {
  return safeFinite(Math.sqrt(mse(a, b)));
}

export function mae(a: number[], b: number[]) {
  const n = Math.min(a.length, b.length);
  if (n === 0) return 0;
  let s = 0;
  let count = 0;
  for (let i = 0; i < n; i++) {
    const ai = a[i];
    const bi = b[i];
    if (!Number.isFinite(ai) || !Number.isFinite(bi)) continue;
    s += Math.abs(ai - bi);
    count++;
  }
  return count === 0 ? 0 : safeFinite(s / count);
}

export function movingAverage(data: number[], window = 3) {
  if (data.length === 0) return [];
  if (window <= 1) return data.slice();
  const out = new Array(data.length).fill(0);
  const half = Math.floor(window / 2);
  for (let i = 0; i < data.length; i++) {
    let cnt = 0;
    let sum = 0;
    for (let j = i - half; j <= i + half; j++) {
      if (j >= 0 && j < data.length) {
        const v = data[j];
        if (Number.isFinite(v)) {
          sum += v;
          cnt++;
        }
      }
    }
    out[i] = cnt > 0 ? sum / cnt : 0;
  }
  return out;
}

export function normalize(data: number[]) {
  if (data.length === 0) return [];
  const finiteData = data.map(v => Number.isFinite(v) ? v : 0);
  const min = Math.min(...finiteData);
  const max = Math.max(...finiteData);
  if (max === min) return finiteData.map(() => 0);
  return finiteData.map((v) => (v - min) / (max - min));
}
