<script lang="ts">
  /**
   * DegradationControls.svelte – Control realistic audio degradation parameters.
   * Manages dropouts, glitches, clipping, noise, and interpolation method.
   */
  import type { DegradationParams } from '$lib/api';

  let {
    degradation = $bindable<DegradationParams>({
      dropoutPct: 10,
      dropoutLengthMs: 100,
      glitchPct: 5,
      clipPct: 10,
      noiseLevel: 0.02
    }),
    method = $bindable('spline'),
    processing = false,
    onprocess,
  }: {
    degradation: DegradationParams;
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

  // Track expanded sections
  let showAdvanced = $state(false);
</script>

<div class="card h-full">
  <div class="section-header">
    <span class="section-number">2</span>
    <div>
      <span class="section-title">Degradation & Method</span>
      <span class="section-subtitle">— realistic audio damage simulation</span>
    </div>
  </div>

  <div class="space-y-4">
    <!-- ═══════════════ SEGMENT DROPOUTS ═══════════════ -->
    <div class="border border-slate-200 rounded-lg p-3 bg-slate-50/50">
      <div class="flex items-center gap-2 mb-2">
        <svg class="w-4 h-4 text-red-500" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
          <path stroke-linecap="round" stroke-linejoin="round" d="M18.364 18.364A9 9 0 005.636 5.636m12.728 12.728A9 9 0 015.636 5.636m12.728 12.728L5.636 5.636"/>
        </svg>
        <span class="text-sm font-semibold text-slate-700">Segment Dropouts</span>
        <span class="px-2 py-0.5 rounded-full text-xs font-medium bg-red-100 text-red-700 ml-auto">{degradation.dropoutPct}%</span>
      </div>
      
      <div class="space-y-2">
        <div>
          <div class="flex items-center justify-between text-xs text-slate-500 mb-1">
            <span>Dropout Amount</span>
            <span>{degradation.dropoutPct}% of audio</span>
          </div>
          <input type="range" min="0" max="50" step="2" bind:value={degradation.dropoutPct}
                 disabled={processing} class="w-full" />
          <div class="flex justify-between text-xs text-slate-400">
            <span>0%</span>
            <span>50%</span>
          </div>
        </div>
        
        <div>
          <div class="flex items-center justify-between text-xs text-slate-500 mb-1">
            <span>Segment Length</span>
            <span>~{degradation.dropoutLengthMs}ms each</span>
          </div>
          <input type="range" min="10" max="500" step="10" bind:value={degradation.dropoutLengthMs}
                 disabled={processing} class="w-full" />
          <div class="flex justify-between text-xs text-slate-400">
            <span>10ms</span>
            <span>500ms</span>
          </div>
        </div>
      </div>
      <p class="text-xs text-slate-400 mt-1">Simulates packet loss, tape dropouts, or transmission gaps</p>
    </div>

    <!-- ═══════════════ GLITCHES ═══════════════ -->
    <div class="border border-slate-200 rounded-lg p-3 bg-slate-50/50">
      <div class="flex items-center gap-2 mb-2">
        <svg class="w-4 h-4 text-yellow-500" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
          <path stroke-linecap="round" stroke-linejoin="round" d="M13 10V3L4 14h7v7l9-11h-7z"/>
        </svg>
        <span class="text-sm font-semibold text-slate-700">Glitch Bursts</span>
        <span class="px-2 py-0.5 rounded-full text-xs font-medium bg-yellow-100 text-yellow-700 ml-auto">{degradation.glitchPct}%</span>
      </div>
      
      <input type="range" min="0" max="20" step="1" bind:value={degradation.glitchPct}
             disabled={processing} class="w-full" />
      <div class="flex justify-between text-xs text-slate-400 mt-1">
        <span>0% (none)</span>
        <span>20% (heavy)</span>
      </div>
      <p class="text-xs text-slate-400 mt-1">Short bursts of stuck samples, repeats, or noise artifacts</p>
    </div>

    <!-- ═══════════════ CLIPPING ═══════════════ -->
    <div class="border border-slate-200 rounded-lg p-3 bg-slate-50/50">
      <div class="flex items-center gap-2 mb-2">
        <svg class="w-4 h-4 text-orange-500" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
          <path stroke-linecap="round" stroke-linejoin="round" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"/>
        </svg>
        <span class="text-sm font-semibold text-slate-700">Amplitude Clipping</span>
        <span class="px-2 py-0.5 rounded-full text-xs font-medium bg-orange-100 text-orange-700 ml-auto">{degradation.clipPct}%</span>
      </div>
      
      <input type="range" min="0" max="30" step="2" bind:value={degradation.clipPct}
             disabled={processing} class="w-full" />
      <div class="flex justify-between text-xs text-slate-400 mt-1">
        <span>0% (none)</span>
        <span>30% (severe)</span>
      </div>
      <p class="text-xs text-slate-400 mt-1">Hard clipping at ±60% amplitude (distortion)</p>
    </div>

    <!-- ═══════════════ NOISE ═══════════════ -->
    <div class="border border-slate-200 rounded-lg p-3 bg-slate-50/50">
      <div class="flex items-center gap-2 mb-2">
        <svg class="w-4 h-4 text-blue-500" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
          <path stroke-linecap="round" stroke-linejoin="round" d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z"/>
        </svg>
        <span class="text-sm font-semibold text-slate-700">Background Noise</span>
        <span class="px-2 py-0.5 rounded-full text-xs font-medium bg-blue-100 text-blue-700 ml-auto">{degradation.noiseLevel.toFixed(3)}</span>
      </div>
      
      <input type="range" min="0" max="0.1" step="0.005" bind:value={degradation.noiseLevel}
             disabled={processing} class="w-full" />
      <div class="flex justify-between text-xs text-slate-400 mt-1">
        <span>0 (clean)</span>
        <span>0.1 (heavy)</span>
      </div>
      <p class="text-xs text-slate-400 mt-1">Gaussian noise (σ) added to surviving samples</p>
    </div>

    <!-- ═══════════════ METHOD SELECTOR ═══════════════ -->
    <div>
      <span class="text-sm font-medium text-slate-700 mb-2 block">Reconstruction Method</span>
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

    <!-- Damage preview summary -->
    <div class="px-3 py-2 rounded-lg bg-slate-100 border border-slate-200 text-slate-600 text-xs">
      <div class="font-medium mb-1">Damage Summary:</div>
      <div class="grid grid-cols-2 gap-x-4 gap-y-0.5">
        <span>Dropouts: <span class="font-mono">{degradation.dropoutPct}%</span> @ {degradation.dropoutLengthMs}ms</span>
        <span>Glitches: <span class="font-mono">{degradation.glitchPct}%</span></span>
        <span>Clipping: <span class="font-mono">{degradation.clipPct}%</span></span>
        <span>Noise: <span class="font-mono">σ={degradation.noiseLevel.toFixed(3)}</span></span>
      </div>
    </div>
  </div>
</div>
