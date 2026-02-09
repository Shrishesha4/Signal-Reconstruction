<script lang="ts">
  /**
   * Chart.svelte – Chart.js wrapper with zoom/pan, responsive sizing,
   * decimation for large datasets, and reset-zoom button.
   * Dynamically imports zoom plugin to avoid SSR `window` errors.
   */
  import { onMount, onDestroy } from 'svelte';
  import { Chart as ChartJS } from 'chart.js/auto';

  let {
    data,
    options = {},
    height = 300
  }: {
    data: any;
    options?: any;
    height?: number;
  } = $props();

  let canvas: HTMLCanvasElement;
  let chartInstance: ChartJS | undefined;
  let isZoomed = $state(false);
  let pluginReady = $state(false);

  function resetZoom() {
    chartInstance?.resetZoom();
    isZoomed = false;
  }

  onMount(() => {
    // Dynamic import — runs only in browser, avoids SSR window error
    import('chartjs-plugin-zoom').then(({ default: zoomPlugin }) => {
      ChartJS.register(zoomPlugin);
      pluginReady = true;
      initChart();
    });
  });

  onDestroy(() => {
    chartInstance?.destroy();
    chartInstance = undefined;
  });

  function initChart() {
    if (!canvas) return;
    chartInstance = new ChartJS(canvas.getContext('2d')!, {
      type: 'scatter',
      data,
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: {
          mode: 'nearest',
          intersect: false
        },
        elements: {
          point: {
            radius: (ctx: any) => {
              const count = ctx.dataset?.data?.length ?? 0;
              if (count > 2000) return 0;
              if (count > 500) return ctx.dataset?.pointRadius ?? 2;
              return ctx.dataset?.pointRadius ?? 4;
            }
          },
          line: {
            borderWidth: 2
          }
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
              color: '#64748b'
            }
          },
          tooltip: {
            backgroundColor: 'rgba(15,23,42,0.9)',
            titleFont: { size: 12, family: 'system-ui' },
            bodyFont: { size: 11, family: 'system-ui' },
            padding: 10,
            cornerRadius: 8,
            displayColors: true,
            callbacks: {
              label: (ctx: any) => {
                const xv = ctx.parsed?.x;
                const yv = ctx.parsed?.y;
                if (xv == null || yv == null) return '';
                return `${ctx.dataset.label}: (${xv.toFixed(3)}, ${yv.toFixed(4)})`;
              }
            }
          },
          zoom: {
            pan: {
              enabled: true,
              mode: 'xy',
              modifierKey: undefined
            },
            zoom: {
              wheel: { enabled: true, speed: 0.08 },
              pinch: { enabled: true },
              drag: {
                enabled: true,
                modifierKey: 'shift',
                backgroundColor: 'rgba(59,130,246,0.1)',
                borderColor: 'rgba(59,130,246,0.4)',
                borderWidth: 1
              },
              mode: 'xy',
              onZoom: () => { isZoomed = true; }
            },
            limits: {
              x: { minRange: 0.01 },
              y: { minRange: 0.01 }
            }
          }
        },
        scales: {
          x: {
            type: 'linear',
            title: {
              display: true,
              text: 'x (time / position)',
              font: { size: 11, family: 'system-ui', weight: 500 },
              color: '#94a3b8'
            },
            grid: { color: 'rgba(148,163,184,0.12)' },
            ticks: {
              font: { size: 10, family: 'system-ui' },
              color: '#94a3b8',
              maxTicksLimit: 12
            },
            border: { color: 'rgba(148,163,184,0.2)' }
          },
          y: {
            title: {
              display: true,
              text: 'y (amplitude)',
              font: { size: 11, family: 'system-ui', weight: 500 },
              color: '#94a3b8'
            },
            grid: { color: 'rgba(148,163,184,0.12)' },
            ticks: {
              font: { size: 10, family: 'system-ui' },
              color: '#94a3b8',
              maxTicksLimit: 8
            },
            border: { color: 'rgba(148,163,184,0.2)' }
          }
        },
        animation: false,
        ...options
      }
    });
  }

  // Efficient data update — avoid full re-init
  $effect(() => {
    const d = data;
    if (chartInstance) {
      chartInstance.data = d;
      chartInstance.update('none');
    }
  });
</script>

<div style="width:100%;height:{height}px;position:relative">
  <canvas bind:this={canvas}></canvas>
  {#if isZoomed}
    <button onclick={resetZoom}
      class="absolute top-2 right-2 z-10 px-2 py-1 text-xs font-medium
             bg-white/90 border border-slate-300 rounded-md shadow-sm
             text-slate-600 hover:bg-slate-100 transition-colors backdrop-blur-sm">
      Reset Zoom
    </button>
  {/if}
  <div class="absolute bottom-1 right-2 text-[10px] text-slate-400 select-none pointer-events-none">
    Scroll to zoom · Drag to pan · Shift+drag to select
  </div>
</div>
