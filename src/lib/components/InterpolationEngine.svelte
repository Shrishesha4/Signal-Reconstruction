<script lang="ts">
  /**
   * InterpolationEngine.svelte – Section 3: Numerical Interpolation.
   *
   * Four selectable methods with safety guards, debounced sliders,
   * and a computing spinner.
   */
  import {
    linearInterpolation,
    cubicSplineInterpolation,
    pchipInterpolation,
    movingAverageInterpolation,
  } from '$lib/interpolation';

  let {
    points = [],
    onreconstructed
  }: {
    points?: Array<{ x: number; y: number }>;
    onreconstructed?: (pts: Array<{ x: number; y: number }>) => void;
  } = $props();

  let method = $state<'linear' | 'spline' | 'pchip' | 'moving_average'>('spline');
  let samplesSlider = $state(300);  // raw slider value
  let samples = $state(300);        // debounced value used in computation
  let computing = $state(false);

  // Debounce slider
  let debounceTimer: ReturnType<typeof setTimeout> | undefined;
  function onSamplesInput(val: number) {
    samplesSlider = val;
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(() => { samples = val; }, 120);
  }

  const methods = [
    { id: 'linear'         as const, label: 'Linear',       desc: 'Piecewise linear — fast, no oscillation' },
    { id: 'spline'         as const, label: 'Cubic Spline', desc: 'C² smooth piecewise cubics — best general choice' },
    { id: 'pchip'          as const, label: 'PCHIP',        desc: 'Shape-preserving Hermite — no overshoot' },
    { id: 'moving_average' as const, label: 'Moving Avg',   desc: 'Smoothed linear fill — baseline comparison' },
  ];

  // ── derived computations ──────────────────────────────────────────
  let xs = $derived(points.map(p => p.x));
  let ys = $derived(points.map(p => p.y));

  let interpFn = $derived.by(() => {
    if (points.length < 2) return (_x: number) => 0;
    if (method === 'linear')         return linearInterpolation(xs, ys);
    if (method === 'pchip')          return pchipInterpolation(xs, ys);
    if (method === 'moving_average') return movingAverageInterpolation(xs, ys);
    return cubicSplineInterpolation(xs, ys);
  });

  let reconstructed = $derived.by(() => {
    if (points.length < 2) return [];
    const min = xs[0];
    const max = xs[xs.length - 1];
    if (min === max) return [{ x: min, y: ys[0] }];
    const out: Array<{ x: number; y: number }> = [];
    for (let i = 0; i < samples; i++) {
      const t = i / (samples - 1);
      const x = min + t * (max - min);
      const y = interpFn(x);
      if (Number.isFinite(y)) out.push({ x, y });
    }
    return out;
  });

  // Notify parent + spinner management
  $effect(() => {
    computing = true;
    const pts = reconstructed;
    onreconstructed?.(pts);
    // Use microtask to keep spinner visible briefly for user feedback
    queueMicrotask(() => { computing = false; });
  });
</script>

<div class="card">
  <div class="section-header">
    <span class="section-number">3</span>
    <div class="flex items-center gap-2">
      <span class="section-title">Interpolation Engine</span>
      <span class="section-subtitle">— reconstruct continuous signal</span>
      {#if computing}
        <span class="inline-flex items-center gap-1.5 ml-2">
          <svg class="w-4 h-4 animate-spin text-blue-500" fill="none" viewBox="0 0 24 24">
            <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
            <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"></path>
          </svg>
          <span class="text-xs text-blue-500 font-medium">Computing…</span>
        </span>
      {/if}
    </div>
  </div>

  <div class="grid grid-cols-1 md:grid-cols-2 gap-5">
    <!-- Method selector -->
    <div>
      <span class="text-sm font-medium text-slate-700 mb-2 block">Method</span>
      <div class="flex gap-2 flex-wrap">
        {#each methods as m}
          <button
            class="method-pill {method === m.id ? 'method-pill-active' : 'method-pill-inactive'}"
            onclick={() => { method = m.id; }}
            title={m.desc}
            disabled={false}>
            {m.label}
          </button>
        {/each}
      </div>
      <p class="text-xs text-slate-500 mt-2">
        {methods.find(m => m.id === method)?.desc}
      </p>
    </div>

    <!-- Resolution slider (debounced) -->
    <div>
      <div class="flex items-center justify-between mb-2">
        <span class="text-sm font-medium text-slate-700">Sample Resolution</span>
        <span class="badge badge-orange">{samplesSlider} pts</span>
      </div>
      <label class="sr-only" for="samples-slider">Sample resolution</label>
      <input id="samples-slider" type="range" min="50" max="1000" step="50"
             value={samplesSlider}
             oninput={(e: Event) => onSamplesInput(parseInt((e.target as HTMLInputElement).value))} />
      <div class="flex justify-between text-xs text-slate-400 mt-1">
        <span>50 (coarse)</span>
        <span>1000 (fine)</span>
      </div>
    </div>
  </div>

  <!-- Status bar -->
  {#if points.length < 2}
    <div class="mt-4 px-4 py-3 rounded-lg bg-amber-50 border border-amber-200 text-amber-800 text-sm flex items-center gap-2">
      <svg class="w-4 h-4 shrink-0" fill="currentColor" viewBox="0 0 20 20">
        <path fill-rule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clip-rule="evenodd"/>
      </svg>
      Need at least 2 data points for interpolation.
    </div>
  {:else}
    <div class="mt-4 px-4 py-2.5 rounded-lg bg-slate-50 border border-slate-200 flex items-center justify-between text-sm">
      <span class="text-slate-600">
        Interpolating <strong>{points.length}</strong> sample points → <strong>{reconstructed.length}</strong> reconstructed points
      </span>
      <span class="badge badge-green">Active</span>
    </div>
  {/if}
</div>
