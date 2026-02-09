<script lang="ts">
  /**
   * DamagedAudioInput.svelte – Upload interface for already-damaged audio files.
   * Allows direct upload of corrupted audio for repair.
   */

  let {
    onfileselected,
    processing = false,
  }: {
    onfileselected?: (file: File, mode: 'repair') => void;
    processing?: boolean;
  } = $props();

  let dragOver = $state(false);
  let selectedFile: File | null = $state(null);
  let error = $state('');

  function handleFile(file: File) {
    if (!file.name.toLowerCase().endsWith('.wav')) {
      error = 'Please select a .wav audio file.';
      return;
    }
    if (file.size > 10 * 1024 * 1024) {
      error = 'File too large (max 10 MB).';
      return;
    }
    error = '';
    selectedFile = file;
    onfileselected?.(file, 'repair');
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
</script>

<div class="card h-full">
  <div class="section-header">
    <span class="section-number">
      <svg class="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
        <path stroke-linecap="round" stroke-linejoin="round" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
      </svg>
    </span>
    <div>
      <span class="section-title">Damaged Audio Repair</span>
      <span class="section-subtitle">— upload corrupted file for repair</span>
    </div>
  </div>

  <p class="text-sm text-slate-600 mb-4">
    Upload an audio file with missing samples, dropouts, or corruption.
    The system will automatically detect damaged regions and repair them.
  </p>

  <!-- Drop zone -->
  <!-- svelte-ignore a11y_no_static_element_interactions -->
  <div
    class="relative border-2 border-dashed rounded-xl p-6 text-center transition-colors
           {dragOver ? 'border-orange-500 bg-orange-50/50' : 'border-slate-300 bg-slate-50/50'}
           {processing ? 'opacity-60 pointer-events-none' : ''}"
    ondrop={onDrop}
    ondragover={onDragOver}
    ondragleave={() => dragOver = false}
  >
    <svg class="w-10 h-10 mx-auto text-orange-400 mb-3" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="1.5">
      <path stroke-linecap="round" stroke-linejoin="round" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
    </svg>

    <p class="text-sm text-slate-600 mb-1">
      <label class="font-medium text-orange-600 cursor-pointer hover:text-orange-700 hover:underline">
        Choose a damaged WAV file
        <input type="file" accept=".wav,audio/wav" class="hidden" onchange={onFileInput} />
      </label>
      or drag & drop
    </p>
    <p class="text-xs text-slate-400">Mono WAV recommended · Max 10 MB · Truncated to 5s</p>
  </div>

  {#if error}
    <p class="mt-2 text-xs text-red-600 flex items-center gap-1">
      <svg class="w-3.5 h-3.5 shrink-0" fill="currentColor" viewBox="0 0 20 20">
        <path fill-rule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z" clip-rule="evenodd"/>
      </svg>
      {error}
    </p>
  {/if}

  {#if selectedFile}
    <div class="mt-3 px-3 py-2 rounded-lg bg-orange-50 border border-orange-200 flex items-center gap-2">
      <svg class="w-4 h-4 text-orange-600 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
        <path stroke-linecap="round" stroke-linejoin="round" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
      </svg>
      <span class="text-sm text-orange-800 truncate flex-1">{selectedFile.name}</span>
      <span class="text-xs text-orange-600">{(selectedFile.size / 1024).toFixed(1)} KB</span>
    </div>
  {/if}

  <!-- Damage Detection Info -->
  <div class="mt-4 p-3 rounded-lg bg-slate-50 border border-slate-200">
    <p class="text-xs font-medium text-slate-700 mb-2">Auto-detection capabilities:</p>
    <ul class="text-xs text-slate-500 space-y-1">
      <li class="flex items-center gap-2">
        <span class="w-1.5 h-1.5 rounded-full bg-slate-400"></span>
        Sample dropouts (silent gaps)
      </li>
      <li class="flex items-center gap-2">
        <span class="w-1.5 h-1.5 rounded-full bg-slate-400"></span>
        Clipped audio regions
      </li>
      <li class="flex items-center gap-2">
        <span class="w-1.5 h-1.5 rounded-full bg-slate-400"></span>
        Amplitude discontinuities
      </li>
      <li class="flex items-center gap-2">
        <span class="w-1.5 h-1.5 rounded-full bg-slate-400"></span>
        Noise/corruption anomalies
      </li>
    </ul>
  </div>
</div>
