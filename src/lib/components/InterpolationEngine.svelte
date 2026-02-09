<script lang="ts">
  /**
   * InterpolationEngine.svelte – Section 3: Interpolation method selector.
   *
   * All numerical interpolation now runs on the **Python backend**
   * (backend/interpolation.py) using scipy.  This component is a
   * UI-only method picker that notifies the parent when the user
   * switches methods.
   *
   * 🚫  No client-side interpolation is performed here.
   * 🚫  Lagrange interpolation is excluded — it is numerically
   *     unstable for dense and noisy signals such as audio.
   */

  let {
    method = $bindable('spline'),
    totalSamples = 0,
    processing = false,
    onmethodchange,
  }: {
    method?: string;
    totalSamples?: number;
    processing?: boolean;
    onmethodchange?: (method: string) => void;
  } = $props();

  const methods = [
    { id: 'linear'         as const, label: 'Linear',       desc: 'scipy.interpolate.interp1d — fast baseline' },
    { id: 'spline'         as const, label: 'Cubic Spline', desc: 'scipy.interpolate.CubicSpline — C² smooth, best for audio' },
    { id: 'pchip'          as const, label: 'PCHIP',        desc: 'scipy.interpolate.PchipInterpolator — shape-preserving, no overshoot' },
    { id: 'moving_average' as const, label: 'Moving Avg',   desc: 'Linear fill + uniform_filter1d — baseline comparison' },
  ];

  function selectMethod(id: string) {
    method = id;
    onmethodchange?.(id);
  }
</script>

<div class="card">
  <div class="section-header">
    <span class="section-number">3</span>
    <div class="flex items-center gap-2">
      <span class="section-title">Interpolation Engine</span>
      <span class="section-subtitle">— Python / SciPy backend</span>
      {#if processing}
        <span class="inline-flex items-center gap-1.5 ml-2">
          <svg class="w-4 h-4 animate-spin text-blue-500" fill="none" viewBox="0 0 24 24">
            <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
            <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"></path>
          </svg>
          <span class="text-xs text-blue-500 font-medium">Processing…</span>
        </span>
      {/if}
    </div>
  </div>

  <!-- Method selector -->
  <div>
    <span class="text-sm font-medium text-slate-700 mb-2 block">Method</span>
    <div class="flex gap-2 flex-wrap">
      {#each methods as m}
        <button
          class="method-pill {method === m.id ? 'method-pill-active' : 'method-pill-inactive'}"
          onclick={() => selectMethod(m.id)}
          title={m.desc}
          disabled={processing}>
          {m.label}
        </button>
      {/each}
    </div>
    <p class="text-xs text-slate-500 mt-2">
      {methods.find(m => m.id === method)?.desc}
    </p>
  </div>

  <!-- Status bar -->
  <div class="mt-4 px-4 py-2.5 rounded-lg bg-slate-50 border border-slate-200 flex items-center justify-between text-sm">
    {#if totalSamples > 0}
      <span class="text-slate-600">
        Reconstructing <strong>{totalSamples.toLocaleString()}</strong> samples via <strong>{method}</strong> (server-side)
      </span>
      <span class="badge badge-green">SciPy</span>
    {:else}
      <span class="text-slate-400">No signal loaded yet</span>
    {/if}
  </div>
</div>
