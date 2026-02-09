<script lang="ts">
  /**
   * AudioInput.svelte – Audio file upload or demo signal selection.
   * Section 1 of the pipeline.
   */

  let {
    onfileselected,
    ondemo,
    processing = false,
  }: {
    onfileselected?: (file: File) => void;
    ondemo?: () => void;
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

  function useDemoSignal() {
    selectedFile = null;
    error = '';
    ondemo?.();
  }
</script>

<div class="card h-full">
  <div class="section-header">
    <span class="section-number">1</span>
    <div>
      <span class="section-title">Audio Input</span>
      <span class="section-subtitle">— upload WAV or use demo</span>
    </div>
  </div>

  <!-- Drop zone -->
  <!-- svelte-ignore a11y_no_static_element_interactions -->
  <div
    class="relative border-2 border-dashed rounded-xl p-8 text-center transition-colors
           {dragOver ? 'border-blue-500 bg-blue-50/50' : 'border-slate-300 bg-slate-50/50'}
           {processing ? 'opacity-60 pointer-events-none' : ''}"
    ondrop={onDrop}
    ondragover={onDragOver}
    ondragleave={() => dragOver = false}
  >
    <svg class="w-10 h-10 mx-auto text-slate-400 mb-3" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="1.5">
      <path d="M9 19V6l12-3v13M9 19c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zm12-3c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zM9 10l12-3" stroke-linecap="round" stroke-linejoin="round"/>
    </svg>

    <p class="text-sm text-slate-600 mb-1">
      <label class="font-medium text-blue-600 cursor-pointer hover:text-blue-700 hover:underline">
        Choose a WAV file
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
    <div class="mt-3 px-3 py-2 rounded-lg bg-blue-50 border border-blue-200 text-sm text-blue-700 flex items-center gap-2">
      <svg class="w-4 h-4 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
        <path d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" stroke-linecap="round" stroke-linejoin="round"/>
      </svg>
      <span class="truncate font-medium">{selectedFile.name}</span>
      <span class="shrink-0 text-xs text-blue-500">({(selectedFile.size / 1024).toFixed(0)} KB)</span>
    </div>
  {/if}

  <!-- Divider -->
  <div class="flex items-center gap-3 my-4">
    <div class="flex-1 border-t border-slate-200"></div>
    <span class="text-xs text-slate-400 font-medium">OR</span>
    <div class="flex-1 border-t border-slate-200"></div>
  </div>

  <!-- Demo button -->
  <button
    onclick={useDemoSignal}
    disabled={processing}
    class="w-full px-4 py-3 rounded-lg text-sm font-medium transition-colors
           bg-gradient-to-r from-indigo-50 to-blue-50 text-indigo-700
           border border-indigo-200 hover:from-indigo-100 hover:to-blue-100
           disabled:opacity-50 disabled:cursor-not-allowed"
  >
    <svg class="w-4 h-4 inline -mt-0.5 mr-1.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
      <path d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" stroke-linecap="round" stroke-linejoin="round"/>
      <path d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" stroke-linecap="round" stroke-linejoin="round"/>
    </svg>
    Use Demo Signal (440 Hz + harmonics)
  </button>
</div>
