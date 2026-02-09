<script lang="ts">
  /**
   * Preprocessing.svelte – Section 2: Noise handling & normalization.
   * Moving-average smoother + optional min-max normalization.
   */
  import { movingAverage, normalize } from '$lib/utils';

  let {
    points = [],
    onprocessed
  }: {
    points?: Array<{ x: number; y: number }>;
    onprocessed?: (pts: Array<{ x: number; y: number }>) => void;
  } = $props();

  let windowSize = $state(3);
  let doNormalize = $state(false);

  // Derived chain: raw ys → smoothed → (optionally) normalized → final points
  let ys        = $derived(points.map(p => p.y));
  let smoothed  = $derived(movingAverage(ys, windowSize));
  let normed    = $derived(doNormalize ? normalize(smoothed) : smoothed);
  let processed = $derived(points.map((p, i) => ({ x: p.x, y: normed[i] ?? p.y })));

  // Notify parent whenever preprocessed output changes
  $effect(() => {
    const pts = processed;
    onprocessed?.(pts);
  });

  // Compute change summary
  let maxDelta = $derived.by(() => {
    if (!points.length) return 0;
    let maxD = 0;
    for (let i = 0; i < points.length; i++) {
      const d = Math.abs((normed[i] ?? 0) - points[i].y);
      if (d > maxD) maxD = d;
    }
    return maxD;
  });
</script>

<div class="card h-full">
  <div class="section-header">
    <span class="section-number">2</span>
    <div>
      <span class="section-title">Preprocessing</span>
      <span class="section-subtitle">— smoothing & normalization</span>
    </div>
  </div>

  <div class="space-y-4">
    <!-- Smoothing slider -->
    <div>
      <div class="flex items-center justify-between mb-2">
        <span class="text-sm font-medium text-slate-700">Moving Average Window</span>
        <span class="badge badge-blue">{windowSize}</span>
      </div>
      <label class="sr-only" for="window-slider">Smoothing window size</label>
      <input id="window-slider" type="range" min="1" max="11" step="2" bind:value={windowSize} />
      <div class="flex justify-between text-xs text-slate-400 mt-1">
        <span>1 (none)</span>
        <span>11 (heavy)</span>
      </div>
    </div>

    <!-- Normalize toggle -->
    <div class="flex items-center justify-between p-3 rounded-lg bg-slate-50 border border-slate-200">
      <div>
        <span class="text-sm font-medium text-slate-700">Normalization</span>
        <p class="text-xs text-slate-500 mt-0.5">Scale y-values to [0, 1] range</p>
      </div>
      <label class="relative inline-flex items-center cursor-pointer">
        <input type="checkbox" bind:checked={doNormalize} class="sr-only peer" />
        <div class="w-9 h-5 bg-slate-300 peer-focus:ring-2 peer-focus:ring-blue-500/20
                    rounded-full peer peer-checked:bg-blue-600 transition-colors
                    after:content-[''] after:absolute after:top-[2px] after:left-[2px]
                    after:bg-white after:rounded-full after:h-4 after:w-4
                    after:transition-all peer-checked:after:translate-x-full"></div>
      </label>
    </div>

    <!-- Status summary -->
    <div class="pt-3 border-t border-slate-100">
      <div class="flex items-center justify-between text-xs">
        <span class="text-slate-500">Processing {points.length} points</span>
        <span class="text-slate-500">Max &Delta;y: <span class="font-mono text-slate-700">{maxDelta.toFixed(4)}</span></span>
      </div>
      {#if doNormalize || windowSize > 1}
        <div class="flex gap-2 mt-2">
          {#if windowSize > 1}
            <span class="badge badge-green">Smoothed (w={windowSize})</span>
          {/if}
          {#if doNormalize}
            <span class="badge badge-green">Normalized</span>
          {/if}
        </div>
      {:else}
        <span class="badge mt-2" style="background: #f1f5f9; color: #64748b;">No processing applied</span>
      {/if}
    </div>
  </div>
</div>
