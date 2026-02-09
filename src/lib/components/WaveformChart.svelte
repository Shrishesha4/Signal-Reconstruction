<script lang="ts">
  /**
   * WaveformChart.svelte – Specialized chart for audio waveform display.
   * Shows three overlaid waveforms (original, spoiled, reconstructed)
   * with time-domain x-axis, zoom/pan support, and large dataset handling.
   */
  import { onMount, onDestroy } from 'svelte';
  import { Chart as ChartJS } from 'chart.js/auto';

  let {
    time = [],
    original = [],
    spoiled = [],
    reconstructed = [],
    height = 320,
    title = 'Waveform Comparison',
  }: {
    time?: number[];
    original?: number[];
    spoiled?: number[];
    reconstructed?: number[];
    height?: number;
    title?: string;
  } = $props();

  let canvas = $state<HTMLCanvasElement>();
  let chartInstance: ChartJS | undefined;
  let isZoomed = $state(false);
  let pluginReady = $state(false);

  function resetZoom() {
    chartInstance?.resetZoom();
    isZoomed = false;
  }

  onMount(() => {
    import('chartjs-plugin-zoom').then(({ default: zoomPlugin }) => {
      ChartJS.register(zoomPlugin);
      pluginReady = true;
      // Don't call initChart here — the $effect below handles it
      // once canvas is actually in the DOM.
    });
  });

  onDestroy(() => {
    chartInstance?.destroy();
    chartInstance = undefined;
  });

  function buildData() {
    return {
      datasets: [
        {
          label: 'Original',
          data: time.map((t, i) => ({ x: t, y: original[i] ?? 0 })),
          borderColor: '#3b82f6',
          backgroundColor: 'rgba(59,130,246,0.05)',
          borderWidth: 1.5,
          pointRadius: 0,
          showLine: true,
          fill: false,
          order: 2,
        },
        {
          label: 'Spoiled',
          data: time.map((t, i) => ({ x: t, y: spoiled[i] ?? 0 })),
          borderColor: '#ef4444',
          backgroundColor: 'rgba(239,68,68,0.05)',
          borderWidth: 1,
          pointRadius: 0,
          showLine: true,
          fill: false,
          order: 1,
        },
        {
          label: 'Reconstructed',
          data: time.map((t, i) => ({ x: t, y: reconstructed[i] ?? 0 })),
          borderColor: '#10b981',
          backgroundColor: 'rgba(16,185,129,0.08)',
          borderWidth: 1.5,
          pointRadius: 0,
          showLine: true,
          fill: false,
          order: 0,
        },
      ],
    };
  }

  function initChart() {
    if (!canvas || !pluginReady) return;
    chartInstance = new ChartJS(canvas.getContext('2d')!, {
      type: 'scatter',
      data: buildData(),
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: 'nearest', intersect: false },
        elements: {
          point: { radius: 0 },
          line: { borderWidth: 1.5 },
        },
        plugins: {
          legend: {
            position: 'top',
            align: 'end',
            labels: {
              usePointStyle: true,
              pointStyleWidth: 10,
              padding: 16,
              font: { size: 11, family: 'system-ui' },
              color: '#64748b',
            },
          },
          tooltip: {
            backgroundColor: 'rgba(15,23,42,0.9)',
            titleFont: { size: 12, family: 'system-ui' },
            bodyFont: { size: 11, family: 'system-ui' },
            padding: 10,
            cornerRadius: 8,
            callbacks: {
              label: (ctx: any) => {
                const x = ctx.parsed?.x;
                const y = ctx.parsed?.y;
                if (x == null || y == null) return '';
                return `${ctx.dataset.label}: ${y.toFixed(4)} @ ${(x * 1000).toFixed(1)}ms`;
              },
            },
          },
          zoom: {
            pan: { enabled: true, mode: 'xy' },
            zoom: {
              wheel: { enabled: true, speed: 0.08 },
              pinch: { enabled: true },
              drag: {
                enabled: true,
                modifierKey: 'shift',
                backgroundColor: 'rgba(59,130,246,0.1)',
                borderColor: 'rgba(59,130,246,0.4)',
                borderWidth: 1,
              },
              mode: 'xy',
              onZoom: () => { isZoomed = true; },
            },
            limits: { x: { minRange: 0.001 }, y: { minRange: 0.01 } },
          },
        },
        scales: {
          x: {
            type: 'linear',
            title: {
              display: true,
              text: 'Time (seconds)',
              font: { size: 11, family: 'system-ui', weight: 500 },
              color: '#94a3b8',
            },
            grid: { color: 'rgba(148,163,184,0.12)' },
            ticks: {
              font: { size: 10, family: 'system-ui' },
              color: '#94a3b8',
              maxTicksLimit: 12,
              callback: (val: any) => `${(val as number).toFixed(2)}s`,
            },
            border: { color: 'rgba(148,163,184,0.2)' },
          },
          y: {
            title: {
              display: true,
              text: 'Amplitude',
              font: { size: 11, family: 'system-ui', weight: 500 },
              color: '#94a3b8',
            },
            grid: { color: 'rgba(148,163,184,0.12)' },
            ticks: {
              font: { size: 10, family: 'system-ui' },
              color: '#94a3b8',
              maxTicksLimit: 8,
            },
            border: { color: 'rgba(148,163,184,0.2)' },
            min: -1.1,
            max: 1.1,
          },
        },
        animation: false,
      },
    });
  }

  // Unified effect: initializes chart when canvas + plugin are ready,
  // and updates data on every prop change.
  $effect(() => {
    // Subscribe to all reactive dependencies
    const _t = time;
    const _o = original;
    const _s = spoiled;
    const _r = reconstructed;
    const _canvas = canvas;
    const _ready = pluginReady;

    if (!_ready || !_canvas) return;

    if (chartInstance) {
      chartInstance.data = buildData();
      chartInstance.update('none');
    } else {
      initChart();
    }
  });

  let hasData = $derived(time.length > 0);
</script>

<div class="card">
  <div class="flex items-center justify-between mb-3">
    <div>
      <h3 class="text-lg font-semibold text-slate-800">{title}</h3>
      <p class="text-xs text-slate-500 mt-0.5">
        {#if hasData}
          {time.length} plot points · Scroll to zoom · Drag to pan
        {:else}
          Process audio to see waveforms
        {/if}
      </p>
    </div>
    {#if isZoomed}
      <button onclick={resetZoom}
        class="px-2 py-1 text-xs font-medium bg-white/90 border border-slate-300
               rounded-md shadow-sm text-slate-600 hover:bg-slate-100 transition-colors">
        Reset Zoom
      </button>
    {/if}
  </div>

  {#if !hasData}
    <div class="h-64 flex items-center justify-center rounded-lg bg-slate-50 border border-slate-200">
      <div class="text-center">
        <svg class="w-10 h-10 mx-auto text-slate-300 mb-2" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="1.5">
          <path d="M3 17l4-8 4 6 4-10 4 12" stroke-linecap="round" stroke-linejoin="round"/>
        </svg>
        <p class="text-sm text-slate-400">No waveform data yet</p>
      </div>
    </div>
  {:else}
    <div style="width:100%;height:{height}px;position:relative">
      <canvas bind:this={canvas}></canvas>
    </div>
    <div class="flex gap-4 mt-3 justify-center flex-wrap">
      <span class="flex items-center gap-1.5 text-xs text-slate-600">
        <span class="w-3 h-3 rounded-full bg-blue-500"></span> Original
      </span>
      <span class="flex items-center gap-1.5 text-xs text-slate-600">
        <span class="w-3 h-3 rounded-full bg-red-500"></span> Spoiled
      </span>
      <span class="flex items-center gap-1.5 text-xs text-slate-600">
        <span class="w-3 h-3 rounded-full bg-emerald-500"></span> Reconstructed
      </span>
    </div>
  {/if}
</div>
