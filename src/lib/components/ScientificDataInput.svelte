<script lang="ts">
  /**
   * ScientificDataInput.svelte – Upload interface for scientific time-series data.
   * Supports CSV and JSON formats for IoT, ECG, radio signals, etc.
   */

  let {
    onfileselected,
    onpreset,
    processing = false,
  }: {
    onfileselected?: (file: File) => void;
    onpreset?: (preset: string) => void;
    processing?: boolean;
  } = $props();

  let dragOver = $state(false);
  let selectedFile: File | null = $state(null);
  let error = $state('');

  // Available data types with icons and descriptions
  const dataTypes = [
    { id: 'ecg', name: 'ECG / Biomedical', icon: '❤️', desc: 'Heart signals, pulse data' },
    { id: 'radio', name: 'Radio / RF', icon: '📡', desc: 'AM/FM, wireless signals' },
    { id: 'temperature', name: 'Temperature', icon: '🌡️', desc: 'IoT sensor readings' },
    { id: 'wifi', name: 'WiFi RSSI', icon: '📶', desc: 'Signal strength data' },
    { id: 'accelerometer', name: 'Accelerometer', icon: '📱', desc: 'Motion sensor data' },
  ];

  function handleFile(file: File) {
    const name = file.name.toLowerCase();
    if (!name.endsWith('.csv') && !name.endsWith('.json')) {
      error = 'Please select a .csv or .json file.';
      return;
    }
    if (file.size > 5 * 1024 * 1024) {
      error = 'File too large (max 5 MB).';
      return;
    }
    error = '';
    selectedFile = file;
    onfileselected?.(file);
  }

  function onFileInput(e: Event) {
    const input = e.target as HTMLInputElement;
    if (input.files?.length) {
      handleFile(input.files[0]);
    }
  }

  function onDrop(e: DragEvent) {
    e.preventDefault();
    dragOver = false;
    if (e.dataTransfer?.files.length) {
      handleFile(e.dataTransfer.files[0]);
    }
  }

  function onDragOver(e: DragEvent) {
    e.preventDefault();
    dragOver = true;
  }

  function selectPreset(preset: string) {
    selectedFile = null;
    error = '';
    onpreset?.(preset);
  }
</script>

<div class="card h-full">
  <div class="section-header">
    <span class="section-number">1</span>
    <div>
      <span class="section-title">Data Input</span>
      <span class="section-subtitle">— upload CSV/JSON or use preset</span>
    </div>
  </div>

  <!-- Drop zone -->
  <!-- svelte-ignore a11y_no_static_element_interactions -->
  <div
    class="relative border-2 border-dashed rounded-xl p-6 text-center transition-colors mb-4
           {dragOver ? 'border-blue-500 bg-blue-50/50' : 'border-slate-300 bg-slate-50/50'}
           {processing ? 'opacity-60 pointer-events-none' : ''}"
    ondrop={onDrop}
    ondragover={onDragOver}
    ondragleave={() => dragOver = false}
  >
    <svg class="w-10 h-10 mx-auto text-slate-400 mb-3" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="1.5">
      <path stroke-linecap="round" stroke-linejoin="round" d="M9 17v-2m3 2v-4m3 4v-6m2 10H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
    </svg>

    <p class="text-sm text-slate-600 mb-1">
      <label class="font-medium text-blue-600 cursor-pointer hover:text-blue-700 hover:underline">
        Choose a data file
        <input type="file" accept=".csv,.json" class="hidden" onchange={onFileInput} />
      </label>
      or drag & drop
    </p>
    <p class="text-xs text-slate-400">CSV or JSON format · Max 5 MB · 50K samples max</p>
  </div>

  {#if error}
    <p class="mb-3 text-xs text-red-600 flex items-center gap-1">
      <svg class="w-3.5 h-3.5 shrink-0" fill="currentColor" viewBox="0 0 20 20">
        <path fill-rule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z" clip-rule="evenodd"/>
      </svg>
      {error}
    </p>
  {/if}

  {#if selectedFile}
    <div class="mb-4 px-3 py-2 rounded-lg bg-emerald-50 border border-emerald-200 flex items-center gap-2">
      <svg class="w-4 h-4 text-emerald-600 shrink-0" fill="currentColor" viewBox="0 0 20 20">
        <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clip-rule="evenodd"/>
      </svg>
      <span class="text-sm text-emerald-800 truncate flex-1">{selectedFile.name}</span>
      <span class="text-xs text-emerald-600">{(selectedFile.size / 1024).toFixed(1)} KB</span>
    </div>
  {/if}

  <!-- Demo Presets -->
  <div class="border-t border-slate-200 pt-4 mt-2">
    <p class="text-sm font-medium text-slate-700 mb-3">Or try a demo preset:</p>
    <div class="grid grid-cols-2 gap-2">
      {#each dataTypes as dtype}
        <button
          class="flex items-center gap-2 px-3 py-2 rounded-lg text-left text-sm transition-colors
                 bg-slate-50 hover:bg-slate-100 border border-slate-200 hover:border-slate-300
                 disabled:opacity-50 disabled:cursor-not-allowed"
          onclick={() => selectPreset(dtype.id)}
          disabled={processing}
          title={dtype.desc}
        >
          <span class="text-lg">{dtype.icon}</span>
          <span class="truncate text-slate-700">{dtype.name}</span>
        </button>
      {/each}
    </div>
  </div>
</div>

<!-- Expected Data Format Info -->
<details class="mt-4 text-xs text-slate-500">
  <summary class="cursor-pointer hover:text-slate-700">Expected data format</summary>
  <div class="mt-2 p-3 bg-slate-50 rounded-lg font-mono text-xs">
    <p class="mb-2"><strong>CSV:</strong></p>
    <pre class="bg-slate-100 p-2 rounded mb-3">time,value
0.0,1.23
0.1,1.45
...</pre>
    <p class="mb-2"><strong>JSON:</strong></p>
    <pre class="bg-slate-100 p-2 rounded">{"{"}"time": [0, 0.1, ...], "values": [1.23, 1.45, ...]{"}"}</pre>
  </div>
</details>
