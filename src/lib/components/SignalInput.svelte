<script lang="ts">
  /**
   * SignalInput.svelte – Section 1: Signal Acquisition.
   * Clean tabular point entry with inline editing, CSV upload, and delete icons.
   */
  import Papa from 'papaparse';
  import { untrack } from 'svelte';

  let {
    initial = [] as Array<{ x: number; y: number }>,
    onchange
  }: {
    initial?: Array<{ x: number; y: number }>;
    onchange?: (pts: Array<{ x: number; y: number }>) => void;
  } = $props();

  let points = $state(untrack(() => initial.map(p => ({ ...p }))));
  let xVal = $state('');
  let yVal = $state('');
  let error = $state('');

  function addPoint() {
    const x = parseFloat(xVal);
    const y = parseFloat(yVal);
    if (!Number.isFinite(x) || !Number.isFinite(y)) {
      error = 'Enter valid numbers for both x and y.';
      return;
    }
    if (points.some(p => p.x === x)) {
      error = `x = ${x} already exists. Use a unique x value.`;
      return;
    }
    error = '';
    points = [...points, { x, y }].sort((a, b) => a.x - b.x);
    onchange?.(points);
    xVal = '';
    yVal = '';
  }

  function remove(i: number) {
    if (points.length <= 2) {
      error = 'Need at least 2 points for interpolation.';
      return;
    }
    error = '';
    points = points.slice(0, i).concat(points.slice(i + 1));
    onchange?.(points);
  }

  function updatePoint(i: number, field: 'x' | 'y', value: string) {
    const v = parseFloat(value);
    if (!Number.isFinite(v)) return;
    const updated = [...points];
    updated[i] = { ...updated[i], [field]: v };
    if (field === 'x') updated.sort((a, b) => a.x - b.x);
    points = updated;
    onchange?.(points);
  }

  function onFile(e: Event) {
    const input = e.target as HTMLInputElement;
    if (!input.files?.length) return;
    const file = input.files[0];
    Papa.parse(file, {
      complete: (res: any) => {
        const data = res.data as any[];
        const parsed: Array<{ x: number; y: number }> = [];
        for (const row of data) {
          if (!row || row.length < 2) continue;
          const x = parseFloat(row[0]);
          const y = parseFloat(row[1]);
          if (Number.isFinite(x) && Number.isFinite(y)) parsed.push({ x, y });
        }
        if (parsed.length >= 2) {
          error = '';
          points = parsed.sort((a, b) => a.x - b.x);
          onchange?.(points);
        } else {
          error = 'CSV must contain at least 2 valid (x, y) rows.';
        }
      }
    });
  }

  function handleKeydown(e: KeyboardEvent) {
    if (e.key === 'Enter') addPoint();
  }
</script>

<div class="card h-full">
  <div class="section-header">
    <span class="section-number">1</span>
    <div>
      <span class="section-title">Signal Input</span>
      <span class="section-subtitle">— discrete sample points</span>
    </div>
  </div>

  <!-- Add point row -->
  <div class="flex items-end gap-2 mb-3">
    <div class="flex-1 min-w-0">
      <label class="block text-xs font-medium text-slate-500 mb-1">
        x value
        <input type="number" step="any" class="w-full px-3 py-1.5 mt-0.5" placeholder="0.0"
               bind:value={xVal} onkeydown={handleKeydown} />
      </label>
    </div>
    <div class="flex-1 min-w-0">
      <label class="block text-xs font-medium text-slate-500 mb-1">
        y value
        <input type="number" step="any" class="w-full px-3 py-1.5 mt-0.5" placeholder="0.0"
               bind:value={yVal} onkeydown={handleKeydown} />
      </label>
    </div>
    <button onclick={addPoint}
      class="shrink-0 px-4 py-1.5 rounded-lg bg-blue-600 text-white text-sm font-medium
             hover:bg-blue-700 active:bg-blue-800 transition-colors shadow-sm">
      <svg class="w-4 h-4 inline -mt-0.5 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
        <path d="M12 4v16m8-8H4" stroke-linecap="round"/>
      </svg>
      Add
    </button>
  </div>

  {#if error}
    <p class="text-xs text-red-600 mb-2 flex items-center gap-1">
      <svg class="w-3.5 h-3.5 shrink-0" fill="currentColor" viewBox="0 0 20 20">
        <path fill-rule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z" clip-rule="evenodd"/>
      </svg>
      {error}
    </p>
  {/if}

  <!-- Points table -->
  <div class="overflow-auto max-h-52 rounded-lg border border-slate-200">
    <table class="w-full text-sm">
      <thead class="bg-slate-50 sticky top-0">
        <tr>
          <th class="text-left px-3 py-2 text-xs font-semibold text-slate-500 uppercase tracking-wider">#</th>
          <th class="text-left px-3 py-2 text-xs font-semibold text-slate-500 uppercase tracking-wider">x</th>
          <th class="text-left px-3 py-2 text-xs font-semibold text-slate-500 uppercase tracking-wider">y</th>
          <th class="w-10 px-2 py-2"></th>
        </tr>
      </thead>
      <tbody class="divide-y divide-slate-100">
        {#each points as p, i}
          <tr class="hover:bg-slate-50 transition-colors">
            <td class="px-3 py-1.5 text-xs text-slate-400 tabular-nums">{i + 1}</td>
            <td class="px-3 py-1.5">
              <input type="number" step="any" value={p.x}
                     class="w-full border-0 bg-transparent p-0 text-sm tabular-nums focus:ring-0"
                     onchange={(e: Event) => updatePoint(i, 'x', (e.target as HTMLInputElement).value)} />
            </td>
            <td class="px-3 py-1.5">
              <input type="number" step="any" value={p.y}
                     class="w-full border-0 bg-transparent p-0 text-sm tabular-nums focus:ring-0"
                     onchange={(e: Event) => updatePoint(i, 'y', (e.target as HTMLInputElement).value)} />
            </td>
            <td class="px-2 py-1.5 text-center">
              <button class="btn-icon" onclick={() => remove(i)}
                      title="Remove point" aria-label="Remove point {i + 1}">
                <svg class="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
                  <path d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" stroke-linecap="round" stroke-linejoin="round"/>
                </svg>
              </button>
            </td>
          </tr>
        {/each}
      </tbody>
    </table>
  </div>

  <!-- Footer: CSV upload + count -->
  <div class="flex items-center justify-between mt-3 pt-3 border-t border-slate-100">
    <label class="flex items-center gap-2 text-xs text-blue-600 cursor-pointer hover:text-blue-700">
      <svg class="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
        <path d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" stroke-linecap="round" stroke-linejoin="round"/>
      </svg>
      Upload CSV
      <input type="file" accept=".csv" onchange={onFile} class="hidden" />
    </label>
    <span class="badge badge-blue">{points.length} points</span>
  </div>
</div>