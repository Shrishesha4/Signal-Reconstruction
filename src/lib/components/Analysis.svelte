<script lang="ts">
  /**
   * Analysis.svelte – Section 4: Signal analysis & validation.
   * Computes MSE / RMSE / MAE with safe numeric handling.
   * Shows warning when metrics are unreliable.
   */
  import Chart from './Chart.svelte';
  import { mse, rmse, mae } from '$lib/utils';

  let {
    original = [],
    reconstructed = []
  }: {
    original?: Array<{ x: number; y: number }>;
    reconstructed?: Array<{ x: number; y: number }>;
  } = $props();

  /**
   * For each original sample, find the nearest reconstructed y-value.
   * Uses binary search for large reconstructed arrays.
   */
  let metrics = $derived.by(() => {
    if (!original.length || !reconstructed.length)
      return { mse: 0, rmse: 0, mae: 0, valid: false, invalidCount: 0 };

    const xsR = reconstructed.map(p => p.x);
    const ysR = reconstructed.map(p => p.y);
    let invalidCount = 0;

    const recAt = original.map(o => {
      if (!Number.isFinite(o.x) || !Number.isFinite(o.y)) {
        invalidCount++;
        return 0;
      }
      // Binary search for closest x
      let lo = 0, hi = xsR.length - 1;
      while (lo < hi) {
        const mid = (lo + hi) >> 1;
        if (xsR[mid] < o.x) lo = mid + 1;
        else hi = mid;
      }
      // Check neighbors for closest
      let best = lo;
      if (lo > 0 && Math.abs(xsR[lo - 1] - o.x) < Math.abs(xsR[lo] - o.x)) {
        best = lo - 1;
      }
      const v = ysR[best];
      if (!Number.isFinite(v)) {
        invalidCount++;
        return 0;
      }
      return v;
    });

    const origYs = original.map(p => Number.isFinite(p.y) ? p.y : 0);
    const m = mse(origYs, recAt);
    const r = rmse(origYs, recAt);
    const a = mae(origYs, recAt);

    return {
      mse:  Number.isFinite(m) ? m : 0,
      rmse: Number.isFinite(r) ? r : 0,
      mae:  Number.isFinite(a) ? a : 0,
      valid: invalidCount === 0 && Number.isFinite(m),
      invalidCount
    };
  });

  let hasData = $derived(original.length > 0 && reconstructed.length > 0);

  let chartData = $derived({
    datasets: [
      {
        label: 'Reconstructed Signal',
        data: reconstructed.map(p => ({ x: p.x, y: p.y })),
        borderColor: '#f97316',
        backgroundColor: 'rgba(249,115,22,0.06)',
        borderWidth: 2.5,
        pointRadius: 0,
        showLine: true,
        tension: 0.2,
        fill: true,
        order: 0
      },
      {
        label: 'Original Samples',
        data: original.map(p => ({ x: p.x, y: p.y })),
        borderColor: '#3b82f6',
        backgroundColor: '#3b82f6',
        showLine: false,
        pointRadius: 6,
        pointHoverRadius: 8,
        pointStyle: 'circle',
        order: 1
      }
    ]
  });

  function formatMetric(v: number): string {
    if (!Number.isFinite(v)) return '—';
    if (v === 0) return '0.000000';
    if (v < 0.000001) return v.toExponential(2);
    if (v > 1e6) return v.toExponential(2);
    return v.toFixed(6);
  }
</script>

<div class="card">
  <div class="section-header">
    <span class="section-number">4</span>
    <div>
      <span class="section-title">Analysis & Validation</span>
      <span class="section-subtitle">— error metrics & comparison</span>
    </div>
  </div>

  {#if !hasData}
    <div class="px-4 py-8 text-center rounded-lg bg-slate-50 border border-slate-200">
      <svg class="w-10 h-10 mx-auto text-slate-300 mb-2" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="1.5">
        <path d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" stroke-linecap="round" stroke-linejoin="round"/>
      </svg>
      <p class="text-sm text-slate-500">Add data points and run interpolation to see analysis.</p>
    </div>
  {:else}
    <!-- Metric cards row -->
    <div class="grid grid-cols-3 gap-3 mb-5">
      <div class="metric-card border-l-4 border-l-blue-500">
        <div class="metric-label">MSE</div>
        <div class="metric-value text-xl">{formatMetric(metrics.mse)}</div>
        <p class="text-xs text-slate-400 mt-1">Mean Squared Error</p>
      </div>
      <div class="metric-card border-l-4 border-l-emerald-500">
        <div class="metric-label">RMSE</div>
        <div class="metric-value text-xl">{formatMetric(metrics.rmse)}</div>
        <p class="text-xs text-slate-400 mt-1">Root Mean Squared Error</p>
      </div>
      <div class="metric-card border-l-4 border-l-orange-500">
        <div class="metric-label">MAE</div>
        <div class="metric-value text-xl">{formatMetric(metrics.mae)}</div>
        <p class="text-xs text-slate-400 mt-1">Mean Absolute Error</p>
      </div>
    </div>

    {#if metrics.invalidCount > 0}
      <div class="mb-4 px-4 py-2.5 rounded-lg bg-amber-50/70 border border-amber-200 text-amber-700 text-xs flex items-start gap-2">
        <svg class="w-4 h-4 shrink-0 mt-0.5" fill="currentColor" viewBox="0 0 20 20">
          <path fill-rule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clip-rule="evenodd"/>
        </svg>
        <span>
          <strong>{metrics.invalidCount} invalid point(s)</strong> were excluded from metric computation
          (NaN or Infinity values). Consider switching to Cubic Spline interpolation.
        </span>
      </div>
    {/if}

    <!-- Comparison chart - prominent -->
    <div>
      <h4 class="text-sm font-medium text-slate-700 mb-2">Reconstruction Accuracy</h4>
      <div class="w-full h-72 sm:h-80 rounded-lg bg-slate-50 border border-slate-200 p-2">
        <Chart data={chartData} height={310} />
      </div>
    </div>
  {/if}
</div>
