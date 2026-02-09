<script lang="ts">
  /**
   * +page.svelte – Audio Signal Reconstruction Pipeline
   *
   * Flow: Audio Input → Degradation → Reconstruction → Analysis + Playback
   *
   * All heavy processing runs on the Python backend (FastAPI).
   * The frontend handles UI, visualization, and audio playback.
   */
  import AudioInput           from '$lib/components/AudioInput.svelte';
  import DegradationControls  from '$lib/components/DegradationControls.svelte';
  import WaveformChart        from '$lib/components/WaveformChart.svelte';
  import AudioPlayer          from '$lib/components/AudioPlayer.svelte';
  import MetricsPanel         from '$lib/components/MetricsPanel.svelte';
  import {
    processAudio,
    loadDemo,
    reconstructOnly,
    checkHealth,
    type ProcessingResult,
    type ProcessingMetrics,
  } from '$lib/api';
  import { onMount } from 'svelte';

  // ── State ────────────────────────────────────────────────────────
  let backendOnline = $state(false);
  let processing = $state(false);
  let error = $state('');

  // Degradation / method parameters
  let dropoutPct = $state(20);
  let noiseLevel = $state(0.02);
  let method = $state('spline');

  // Current audio file (null = use demo)
  let audioFile: File | null = $state(null);

  // Processing results
  let result = $state<ProcessingResult | null>(null);

  // Derived from result
  let plotTime         = $derived(result?.plot.time ?? []);
  let plotOriginal     = $derived(result?.plot.original ?? []);
  let plotSpoiled      = $derived(result?.plot.spoiled ?? []);
  let plotReconstructed = $derived(result?.plot.reconstructed ?? []);
  let metrics          = $derived(result?.metrics ?? null);
  let totalSamples     = $derived(result?.totalSamples ?? 0);
  let sampleRate       = $derived(result?.sampleRate ?? 0);

  // Audio playback data
  let audioOriginal     = $derived(result?.audio.original ?? '');
  let audioSpoiled      = $derived(result?.audio.spoiled ?? '');
  let audioReconstructed = $derived(result?.audio.reconstructed ?? '');

  // ── Backend health check ──────────────────────────────────────────
  onMount(() => {
    checkHealth().then(ok => backendOnline = ok);
    const interval = setInterval(async () => {
      const ok = await checkHealth();
      backendOnline = ok;
      if (ok) clearInterval(interval);
    }, 5000);
    return () => clearInterval(interval);
  });

  // ── Actions ───────────────────────────────────────────────────────
  function onFileSelected(file: File) {
    audioFile = file;
    error = '';
  }

  async function onDemoSelected() {
    audioFile = null;
    error = '';
    await runProcessing();
  }

  async function runProcessing() {
    if (!backendOnline) {
      error = 'Backend is offline. Start the Python server first.';
      return;
    }
    processing = true;
    error = '';
    try {
      if (audioFile) {
        result = await processAudio(audioFile, dropoutPct, noiseLevel, method);
      } else {
        result = await loadDemo(dropoutPct, noiseLevel, method);
      }
    } catch (e: any) {
      error = e.message || 'Processing failed.';
      result = null;
    } finally {
      processing = false;
    }
  }

  // Re-run with different method (without re-uploading)
  async function switchMethod(newMethod: string) {
    if (!result || !backendOnline) return;
    method = newMethod;
    processing = true;
    error = '';
    try {
      result = await reconstructOnly(
        result.plot.time,
        result.plot.original,
        result.plot.spoiled,
        result.mask,
        newMethod,
        result.sampleRate
      );
    } catch (e: any) {
      error = e.message || 'Reconstruction failed.';
    } finally {
      processing = false;
    }
  }
</script>

<!-- Backend status banner -->
{#if !backendOnline}
  <div class="mb-5 px-4 py-3 rounded-xl bg-amber-50 border border-amber-300 text-amber-800 text-sm flex items-start gap-3">
    <svg class="w-5 h-5 shrink-0 mt-0.5 text-amber-500" fill="currentColor" viewBox="0 0 20 20">
      <path fill-rule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clip-rule="evenodd"/>
    </svg>
    <div>
      <strong>Backend Offline</strong>
      <p class="mt-0.5 text-xs">
        Start the Python server: <code class="px-1.5 py-0.5 rounded bg-amber-100 font-mono text-xs">cd backend && pip install -r requirements.txt && uvicorn backend.main:app --reload</code>
      </p>
    </div>
  </div>
{/if}

<!-- Error banner -->
{#if error}
  <div class="mb-5 px-4 py-3 rounded-xl bg-red-50 border border-red-300 text-red-700 text-sm flex items-center gap-2">
    <svg class="w-4 h-4 shrink-0" fill="currentColor" viewBox="0 0 20 20">
      <path fill-rule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z" clip-rule="evenodd"/>
    </svg>
    {error}
  </div>
{/if}

<!-- Sections 1 & 2: Audio Input + Degradation Controls side by side -->
<section class="grid grid-cols-1 lg:grid-cols-2 gap-5 mb-5">
  <AudioInput onfileselected={onFileSelected} ondemo={onDemoSelected} {processing} />
  <DegradationControls
    bind:dropoutPct
    bind:noiseLevel
    bind:method
    {processing}
    onprocess={runProcessing}
  />
</section>

<!-- Section 3: Waveform Visualization -->
<section class="mb-5">
  <WaveformChart
    time={plotTime}
    original={plotOriginal}
    spoiled={plotSpoiled}
    reconstructed={plotReconstructed}
    title="Signal Waveform Comparison"
    height={360}
  />
</section>

<!-- Section: Audio Playback -->
<section class="mb-5">
  <AudioPlayer
    originalB64={audioOriginal}
    spoiledB64={audioSpoiled}
    reconstructedB64={audioReconstructed}
  />
</section>

<!-- Section 4: Metrics & Analysis -->
<section class="mb-5">
  <MetricsPanel
    {metrics}
    {totalSamples}
    {sampleRate}
    {method}
  />
</section>