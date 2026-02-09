<script lang="ts">
  /**
   * DegradationControls.svelte – Section 2: Control signal spoiling parameters.
   * Manages dropout percentage, noise level, and interpolation method.
   */

  let {
    dropoutPct = $bindable(20),
    noiseLevel = $bindable(0.02),
    method = $bindable('spline'),
    processing = false,
    onprocess,
  }: {
    dropoutPct: number;
    noiseLevel: number;
    method: string;
    processing?: boolean;
    onprocess?: () => void;
  } = $props();

  const methods = [
    { id: 'linear', label: 'Linear', desc: 'Piecewise linear — fast, may produce clicks' },
    { id: 'spline', label: 'Cubic Spline', desc: 'C² smooth piecewise cubics — best for audio' },
    { id: 'pchip', label: 'PCHIP', desc: 'Shape-preserving Hermite — no overshoot' },
    { id: 'moving_average', label: 'Moving Avg', desc: 'Smoothed linear fill — baseline comparison' },
  ];
</script>

<div class="card h-full">
  <div class="section-header">
    <span class="section-number">2</span>
    <div>
      <span class="section-title">Degradation & Method</span>
      <span class="section-subtitle">— configure spoiling + reconstruction</span>
    </div>
  </div>

  <div class="space-y-5">
    <!-- Dropout control -->
    <div>
      <div class="flex items-center justify-between mb-2">
        <span class="text-sm font-medium text-slate-700">Sample Dropout</span>
        <span class="badge badge-orange">{dropoutPct}%</span>
      </div>
      <input type="range" min="0" max="80" step="5" bind:value={dropoutPct}
             disabled={processing}
             class="w-full" />
      <div class="flex justify-between text-xs text-slate-400 mt-1">
        <span>0% (none)</span>
        <span>80% (severe)</span>
      </div>
      <p class="text-xs text-slate-500 mt-1">Simulates packet loss / sample corruption</p>
    </div>

    <!-- Noise control -->
    <div>
      <div class="flex items-center justify-between mb-2">
        <span class="text-sm font-medium text-slate-700">Noise Level</span>
        <span class="badge badge-blue">{noiseLevel.toFixed(3)}</span>
      </div>
      <input type="range" min="0" max="0.2" step="0.005" bind:value={noiseLevel}
             disabled={processing}
             class="w-full" />
      <div class="flex justify-between text-xs text-slate-400 mt-1">
        <span>0 (clean)</span>
        <span>0.2 (heavy)</span>
      </div>
      <p class="text-xs text-slate-500 mt-1">Additive Gaussian noise (σ)</p>
    </div>

    <!-- Method selector -->
    <div>
      <span class="text-sm font-medium text-slate-700 mb-2 block">Interpolation Method</span>
      <div class="flex gap-2 flex-wrap">
        {#each methods as m}
          <button
            class="method-pill {method === m.id ? 'method-pill-active' : 'method-pill-inactive'}"
            onclick={() => { method = m.id; }}
            title={m.desc}
            disabled={processing}>
            {m.label}
          </button>
        {/each}
      </div>
      <p class="text-xs text-slate-500 mt-2">
        {methods.find(m => m.id === method)?.desc ?? ''}
      </p>
    </div>

    <!-- Process button -->
    <button
      onclick={() => onprocess?.()}
      disabled={processing}
      class="w-full px-4 py-3 rounded-lg text-sm font-semibold transition-all
             bg-blue-600 text-white hover:bg-blue-700 active:bg-blue-800
             shadow-sm hover:shadow disabled:opacity-50 disabled:cursor-not-allowed"
    >
      {#if processing}
        <svg class="w-4 h-4 inline animate-spin -mt-0.5 mr-2" fill="none" viewBox="0 0 24 24">
          <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
          <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"></path>
        </svg>
        Processing…
      {:else}
        <svg class="w-4 h-4 inline -mt-0.5 mr-1.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
          <path d="M13 10V3L4 14h7v7l9-11h-7z" stroke-linecap="round" stroke-linejoin="round"/>
        </svg>
        Degrade & Reconstruct
      {/if}
    </button>

    {#if method !== 'spline'}
      <div class="px-3 py-2 rounded-lg bg-blue-50/70 border border-blue-200 text-blue-700 text-xs flex items-start gap-2">
        <svg class="w-4 h-4 shrink-0 mt-0.5" fill="currentColor" viewBox="0 0 20 20">
          <path fill-rule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clip-rule="evenodd"/>
        </svg>
        <span>
          <strong>Tip:</strong> Cubic Spline is recommended for audio signals —
          it provides C² smoothness and matches how professional DAWs reconstruct lost samples.
        </span>
      </div>
    {/if}
  </div>
</div>
