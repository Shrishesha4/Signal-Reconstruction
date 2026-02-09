<script lang="ts">
  /**
   * +page.svelte – root page that wires the four visual sections:
   *   1. Signal Input
   *   2. Preprocessing
   *   3. Interpolation Engine
   *   4. Analysis & Validation
   */
  import SignalInput          from '$lib/components/SignalInput.svelte';
  import Preprocessing        from '$lib/components/Preprocessing.svelte';
  import InterpolationEngine  from '$lib/components/InterpolationEngine.svelte';
  import Analysis             from '$lib/components/Analysis.svelte';
  import Chart                from '$lib/components/Chart.svelte';

  // ── Demo signal: a sampled damped sine that clearly benefits from interpolation ──
  function generateDemoData(): Array<{ x: number; y: number }> {
    return [
      { x: 0.0, y:  0.00 },
      { x: 0.6, y:  0.82 },
      { x: 1.3, y:  0.96 },
      { x: 2.0, y:  0.35 },
      { x: 2.7, y: -0.52 },
      { x: 3.4, y: -0.90 },
      { x: 4.1, y: -0.38 },
      { x: 5.0, y:  0.48 }
    ];
  }

  // ── reactive state ────────────────────────────────────────────────
  let rawPoints: Array<{ x: number; y: number }> = $state(generateDemoData());
  let processedPoints: Array<{ x: number; y: number }> = $state(generateDemoData());
  let reconstructedPoints: Array<{ x: number; y: number }> = $state([]);

  let resetKey = $state(0);

  // ── callbacks from child components ───────────────────────────────
  function onRawChange(pts: Array<{ x: number; y: number }>) {
    rawPoints = pts;
  }
  function onProcessed(pts: Array<{ x: number; y: number }>) {
    processedPoints = pts;
  }
  function onReconstructed(pts: Array<{ x: number; y: number }>) {
    reconstructedPoints = pts;
  }
  function resetToDemo() {
    const demo = generateDemoData();
    rawPoints = demo;
    processedPoints = demo;
    reconstructedPoints = [];
    resetKey++;
  }

  // ── chart data for the overview panel ─────────────────────────────
  let comparisonChart = $derived({
    datasets: [
      {
        label: 'Raw Samples',
        data: rawPoints.map(p => ({ x: p.x, y: p.y })),
        showLine: false,
        pointRadius: 6,
        pointHoverRadius: 8,
        borderColor: '#3b82f6',
        backgroundColor: '#3b82f6',
        pointStyle: 'circle',
        order: 2
      },
      {
        label: 'Preprocessed',
        data: processedPoints.map(p => ({ x: p.x, y: p.y })),
        showLine: false,
        pointRadius: 5,
        pointHoverRadius: 7,
        borderColor: '#10b981',
        backgroundColor: '#10b981',
        pointStyle: 'triangle',
        order: 1
      },
      {
        label: 'Reconstructed Signal',
        data: reconstructedPoints.map(p => ({ x: p.x, y: p.y })),
        borderColor: '#f97316',
        backgroundColor: 'rgba(249,115,22,0.08)',
        pointRadius: 0,
        showLine: true,
        borderWidth: 2.5,
        tension: 0.2,
        fill: true,
        order: 0
      }
    ]
  });
</script>

<!-- Hero overview chart -->
<section class="mb-6">
  <div class="card">
    <div class="flex items-center justify-between mb-3">
      <div>
        <h2 class="text-lg font-semibold text-slate-800">Signal Overview</h2>
        <p class="text-xs text-slate-500 mt-0.5">
          Comparison of raw samples, preprocessed points, and reconstructed curve
        </p>
      </div>
      <button onclick={resetToDemo}
        class="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm font-medium
               text-blue-600 bg-blue-50 hover:bg-blue-100 border border-blue-200 transition-colors">
        <svg class="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
          <path d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" stroke-linecap="round" stroke-linejoin="round"/>
        </svg>
        Reset Demo
      </button>
    </div>
    <div class="w-full h-72 sm:h-80">
      <Chart data={comparisonChart} height={320} />
    </div>
    <div class="flex gap-4 mt-3 justify-center flex-wrap">
      <span class="flex items-center gap-1.5 text-xs text-slate-600">
        <span class="w-3 h-3 rounded-full bg-blue-500"></span> Raw Samples
      </span>
      <span class="flex items-center gap-1.5 text-xs text-slate-600">
        <span class="w-3 h-3 rounded-full bg-emerald-500"></span> Preprocessed
      </span>
      <span class="flex items-center gap-1.5 text-xs text-slate-600">
        <span class="w-3 h-3 rounded-full bg-orange-500"></span> Reconstructed
      </span>
    </div>
  </div>
</section>

<!-- Sections 1 & 2: Signal Input + Preprocessing side by side -->
<section class="grid grid-cols-1 lg:grid-cols-2 gap-5 mb-5">
  {#key resetKey}
    <SignalInput initial={rawPoints} onchange={onRawChange} />
    <Preprocessing points={rawPoints} onprocessed={onProcessed} />
  {/key}
</section>

<!-- Section 3: Interpolation Engine -->
<section class="mb-5">
  <InterpolationEngine points={processedPoints} onreconstructed={onReconstructed} />
</section>

<!-- Section 4: Analysis -->
<section class="mb-5">
  <Analysis original={processedPoints} reconstructed={reconstructedPoints} />
</section>