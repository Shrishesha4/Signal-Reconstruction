<script lang="ts">
  /**
   * MetricsPanel.svelte – Display reconstruction quality metrics.
   * Shows MSE, RMSE, MAE, and SNR with explanations.
   */
  import type { ProcessingMetrics } from '$lib/api';

  let {
    metrics = null,
    totalSamples = 0,
    sampleRate = 0,
    method = 'spline',
  }: {
    metrics?: ProcessingMetrics | null;
    totalSamples?: number;
    sampleRate?: number;
    method?: string;
  } = $props();

  function fmt(v: number | undefined): string {
    if (v == null || !Number.isFinite(v)) return '—';
    if (v === 0) return '0.000000';
    if (v < 0.000001) return v.toExponential(2);
    if (v > 1e6) return v.toExponential(2);
    return v.toFixed(6);
  }

  function fmtSNR(v: number | undefined): string {
    if (v == null || !Number.isFinite(v)) return '—';
    return `${v.toFixed(1)} dB`;
  }

  let methodLabel = $derived(
    method === 'spline' ? 'Cubic Spline'
      : method === 'linear' ? 'Linear'
      : method === 'pchip' ? 'PCHIP'
      : method === 'moving_average' ? 'Moving Avg'
      : method
  );

  let qualityColor = $derived(() => {
    if (!metrics) return 'text-slate-400';
    if (metrics.snr_db > 30) return 'text-emerald-600';
    if (metrics.snr_db > 15) return 'text-blue-600';
    if (metrics.snr_db > 5) return 'text-amber-600';
    return 'text-red-600';
  });

  let qualityLabel = $derived(() => {
    if (!metrics) return 'N/A';
    if (metrics.snr_db > 30) return 'Excellent';
    if (metrics.snr_db > 15) return 'Good';
    if (metrics.snr_db > 5) return 'Fair';
    return 'Poor';
  });
</script>

<div class="card">
  <div class="section-header">
    <span class="section-number">4</span>
    <div>
      <span class="section-title">Analysis & Metrics</span>
      <span class="section-subtitle">— reconstruction quality</span>
    </div>
  </div>

  {#if !metrics}
    <div class="px-4 py-8 text-center rounded-lg bg-slate-50 border border-slate-200">
      <svg class="w-10 h-10 mx-auto text-slate-300 mb-2" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="1.5">
        <path d="M9 19v-6a2 2 0 0 0-2-2H5a2 2 0 0 0-2 2v6a2 2 0 0 0 2 2h2a2 2 0 0 0 2-2zm0 0V9a2 2 0 0 1 2-2h2a2 2 0 0 1 2 2v10m-6 0a2 2 0 0 0 2 2h2a2 2 0 0 0 2-2m0 0V5a2 2 0 0 1 2-2h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2h-2a2 2 0 0 1-2-2z" stroke-linecap="round" stroke-linejoin="round"/>
      </svg>
      <p class="text-sm text-slate-500">Process audio to see error metrics.</p>
    </div>
  {:else}
    <!-- Quality badge -->
    <div class="mb-4 flex items-center justify-between">
      <div class="flex items-center gap-2">
        <span class="text-sm text-slate-600">Method:</span>
        <span class="badge badge-blue">{methodLabel}</span>
      </div>
      <div class="flex items-center gap-2">
        <span class="text-sm text-slate-600">Quality:</span>
        <span class="text-sm font-semibold {qualityColor()}">{qualityLabel()}</span>
      </div>
    </div>

    <!-- Metric cards -->
    <div class="grid grid-cols-2 lg:grid-cols-4 gap-3 mb-5">
      <div class="metric-card border-l-4 border-l-blue-500">
        <div class="metric-label">MSE</div>
        <div class="metric-value text-lg">{fmt(metrics.mse)}</div>
        <p class="text-xs text-slate-400 mt-1">Mean Squared Error</p>
      </div>
      <div class="metric-card border-l-4 border-l-emerald-500">
        <div class="metric-label">RMSE</div>
        <div class="metric-value text-lg">{fmt(metrics.rmse)}</div>
        <p class="text-xs text-slate-400 mt-1">Root Mean Squared Error</p>
      </div>
      <div class="metric-card border-l-4 border-l-orange-500">
        <div class="metric-label">MAE</div>
        <div class="metric-value text-lg">{fmt(metrics.mae)}</div>
        <p class="text-xs text-slate-400 mt-1">Mean Absolute Error</p>
      </div>
      <div class="metric-card border-l-4 border-l-purple-500">
        <div class="metric-label">SNR</div>
        <div class="metric-value text-lg">{fmtSNR(metrics.snr_db)}</div>
        <p class="text-xs text-slate-400 mt-1">Signal-to-Noise Ratio</p>
      </div>
    </div>

    <!-- Signal info -->
    <div class="grid grid-cols-2 gap-3">
      <div class="px-3 py-2 rounded-lg bg-slate-50 border border-slate-200">
        <span class="text-xs text-slate-500">Total Samples</span>
        <p class="text-sm font-semibold text-slate-700 tabular-nums">{totalSamples.toLocaleString()}</p>
      </div>
      <div class="px-3 py-2 rounded-lg bg-slate-50 border border-slate-200">
        <span class="text-xs text-slate-500">Sample Rate</span>
        <p class="text-sm font-semibold text-slate-700 tabular-nums">{sampleRate.toLocaleString()} Hz</p>
      </div>
    </div>

    <!-- Why spline works best -->
    <details class="mt-4">
      <summary class="text-sm font-medium text-slate-600 cursor-pointer hover:text-slate-800">
        Why Cubic Spline works best for audio ▸
      </summary>
      <div class="mt-2 text-xs text-slate-600 space-y-2 pl-3 border-l-2 border-blue-200">
        <p>
          <strong>Audio signals are smooth & continuous.</strong> Cubic splines fit piecewise 
          cubic polynomials with C² continuity (continuous second derivative), matching the 
          inherent smoothness of audio waveforms.
        </p>
        <p>
          <strong>Linear interpolation creates corners</strong> at every join point, introducing 
          high-frequency artifacts (clicks and harshness) that degrade perceived audio quality.
        </p>
        <p>
          <strong>PCHIP (Piecewise Cubic Hermite)</strong> preserves the shape of data — it 
          never overshoots between adjacent samples. Excellent when the signal shape must be 
          strictly preserved, though it sacrifices C² smoothness.
        </p>
        <p>
          <strong>Moving Average</strong> fills gaps linearly then smooths with a sliding window.
          Useful as a baseline to show the benefit of higher-order methods, but blurs transients.
        </p>
        <p>
          <strong>Cubic Spline</strong> avoids all these problems: smooth joins, local support (errors 
          don't propagate globally), and O(n) computation. It's the same method used in professional 
          DAWs and codec error concealment (e.g., Opus PLC).
        </p>
      </div>
    </details>
  {/if}
</div>
