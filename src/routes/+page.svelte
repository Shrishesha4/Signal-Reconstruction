<script lang="ts">
  /**
   * +page.svelte – Multi-Mode Signal Reconstruction Pipeline
   *
   * Tabs:
   *   1. Audio Reconstruction – Upload/demo + degradation + reconstruction
   *   2. Scientific Data – CSV/JSON time-series reconstruction
   *   3. Demo Signals – Quick access to preset demo signals
   *
   * All heavy processing runs on the Python backend (FastAPI).
   */
  import { onMount } from 'svelte';
  
  // Components
  import TabNavigation from '$lib/components/TabNavigation.svelte';
  import AudioInput from '$lib/components/AudioInput.svelte';
  import DamagedAudioInput from '$lib/components/DamagedAudioInput.svelte';
  import ScientificDataInput from '$lib/components/ScientificDataInput.svelte';
  import DegradationControls from '$lib/components/DegradationControls.svelte';
  import WaveformChart from '$lib/components/WaveformChart.svelte';
  import AudioPlayer from '$lib/components/AudioPlayer.svelte';
  import MetricsPanel from '$lib/components/MetricsPanel.svelte';
  
  // API
  import {
    processAudio,
    loadDemo,
    reconstructOnly,
    repairDamagedAudio,
    loadSciDataDemo,
    processSciData,
    reconstructSciDataOnly,
    checkHealth,
    type ProcessingResult,
    type RepairResult,
    type SciDataResult,
  } from '$lib/api';
  import type { Tab } from '$lib/components/TabNavigation.svelte';

  // ── Global State ─────────────────────────────────────────────────
  let backendOnline = $state(false);
  let processing = $state(false);
  let error = $state('');
  let activeTab = $state<Tab>('audio');

  // ── Audio Tab State ──────────────────────────────────────────────
  let audioMode = $state<'normal' | 'repair'>('normal');
  let dropoutPct = $state(20);
  let noiseLevel = $state(0.02);
  let method = $state('pchip');
  let audioFile: File | null = $state(null);
  let audioResult = $state<ProcessingResult | null>(null);
  let repairResult = $state<RepairResult | null>(null);

  // Audio derived values
  let audioPlotTime = $derived(
    audioMode === 'repair' 
      ? (repairResult?.plot.time ?? [])
      : (audioResult?.plot.time ?? [])
  );
  let audioPlotOriginal = $derived(
    audioMode === 'repair' 
      ? (repairResult?.plot.damaged ?? [])
      : (audioResult?.plot.original ?? [])
  );
  let audioPlotSpoiled = $derived(
    audioMode === 'repair' 
      ? []
      : (audioResult?.plot.spoiled ?? [])
  );
  let audioPlotReconstructed = $derived(
    audioMode === 'repair'
      ? (repairResult?.plot.reconstructed ?? [])
      : (audioResult?.plot.reconstructed ?? [])
  );
  let audioMetrics = $derived(audioResult?.metrics ?? null);
  let audioTotalSamples = $derived(
    audioMode === 'repair'
      ? (repairResult?.totalSamples ?? 0)
      : (audioResult?.totalSamples ?? 0)
  );
  let audioSampleRate = $derived(
    audioMode === 'repair'
      ? (repairResult?.sampleRate ?? 0)
      : (audioResult?.sampleRate ?? 0)
  );

  // Audio playback data
  let audioOriginalB64 = $derived(
    audioMode === 'repair' 
      ? ''
      : (audioResult?.audio.original ?? '')
  );
  let audioSpoiledB64 = $derived(
    audioMode === 'repair'
      ? (repairResult?.audio.damaged ?? '')
      : (audioResult?.audio.spoiled ?? '')
  );
  let audioReconstructedB64 = $derived(
    audioMode === 'repair'
      ? (repairResult?.audio.reconstructed ?? '')
      : (audioResult?.audio.reconstructed ?? '')
  );

  // ── Scientific Data Tab State ────────────────────────────────────
  let sciDropoutPct = $state(15);
  let sciNoiseLevel = $state(0.02);
  let sciMethod = $state('pchip');
  let sciDataFile: File | null = $state(null);
  let sciResult = $state<SciDataResult | null>(null);

  // Scientific data derived values
  let sciPlotTime = $derived(sciResult?.plot.time ?? []);
  let sciPlotOriginal = $derived(sciResult?.plot.original ?? []);
  let sciPlotSpoiled = $derived(sciResult?.plot.spoiled ?? []);
  let sciPlotReconstructed = $derived(sciResult?.plot.reconstructed ?? []);
  let sciMetrics = $derived(sciResult?.metrics ?? null);
  let sciTotalSamples = $derived(sciResult?.totalSamples ?? 0);
  let sciName = $derived(sciResult?.name ?? 'Signal');

  // ── Demo Tab State ───────────────────────────────────────────────
  let demoPreset = $state('ecg');
  let demoDropoutPct = $state(15);
  let demoNoiseLevel = $state(0.02);
  let demoMethod = $state('pchip');
  let demoResult = $state<SciDataResult | null>(null);

  let demoPlotTime = $derived(demoResult?.plot.time ?? []);
  let demoPlotOriginal = $derived(demoResult?.plot.original ?? []);
  let demoPlotSpoiled = $derived(demoResult?.plot.spoiled ?? []);
  let demoPlotReconstructed = $derived(demoResult?.plot.reconstructed ?? []);
  let demoMetrics = $derived(demoResult?.metrics ?? null);
  let demoTotalSamples = $derived(demoResult?.totalSamples ?? 0);
  let demoName = $derived(demoResult?.name ?? 'Demo');

  const demoPresets = [
    { id: 'ecg', name: 'ECG / Biomedical', icon: '❤️' },
    { id: 'radio', name: 'AM Radio Signal', icon: '📡' },
    { id: 'temperature', name: 'Temperature Sensor', icon: '🌡️' },
    { id: 'wifi', name: 'WiFi RSSI', icon: '📶' },
    { id: 'accelerometer', name: 'Accelerometer', icon: '📱' },
  ];

  // ── Backend Health Check ─────────────────────────────────────────
  onMount(() => {
    checkHealth().then(ok => backendOnline = ok);
    const interval = setInterval(async () => {
      const ok = await checkHealth();
      backendOnline = ok;
      if (ok) clearInterval(interval);
    }, 5000);
    return () => clearInterval(interval);
  });

  // ── Tab Change Handler ───────────────────────────────────────────
  function onTabChange(tab: Tab) {
    error = '';
  }

  // ── Audio Tab Actions ────────────────────────────────────────────
  function onAudioFileSelected(file: File) {
    audioFile = file;
    audioMode = 'normal';
    error = '';
  }

  function onDamagedFileSelected(file: File, mode: 'repair') {
    audioFile = file;
    audioMode = 'repair';
    error = '';
    runAudioRepair(file);
  }

  async function onAudioDemoSelected() {
    audioFile = null;
    audioMode = 'normal';
    error = '';
    await runAudioProcessing();
  }

  async function runAudioProcessing() {
    if (!backendOnline) {
      error = 'Backend is offline. Start the Python server first.';
      return;
    }
    processing = true;
    error = '';
    try {
      if (audioFile && audioMode === 'normal') {
        audioResult = await processAudio(audioFile, dropoutPct, noiseLevel, method);
      } else {
        audioResult = await loadDemo(dropoutPct, noiseLevel, method);
      }
      repairResult = null;
    } catch (e: any) {
      error = e.message || 'Processing failed.';
      audioResult = null;
    } finally {
      processing = false;
    }
  }

  async function runAudioRepair(file: File) {
    if (!backendOnline) {
      error = 'Backend is offline. Start the Python server first.';
      return;
    }
    processing = true;
    error = '';
    try {
      repairResult = await repairDamagedAudio(file, method);
      audioResult = null;
    } catch (e: any) {
      error = e.message || 'Repair failed.';
      repairResult = null;
    } finally {
      processing = false;
    }
  }

  async function switchAudioMethod(newMethod: string) {
    if (!audioResult || !backendOnline || audioMode === 'repair') return;
    method = newMethod;
    processing = true;
    error = '';
    try {
      audioResult = await reconstructOnly(
        audioResult.plot.time,
        audioResult.plot.original,
        audioResult.plot.spoiled,
        audioResult.mask,
        newMethod,
        audioResult.sampleRate
      );
    } catch (e: any) {
      error = e.message || 'Reconstruction failed.';
    } finally {
      processing = false;
    }
  }

  // ── Scientific Data Tab Actions ──────────────────────────────────
  function onSciFileSelected(file: File) {
    sciDataFile = file;
    error = '';
    runSciDataProcessing();
  }

  async function onSciPresetSelected(preset: string) {
    sciDataFile = null;
    error = '';
    await runSciDataDemo(preset);
  }

  async function runSciDataProcessing() {
    if (!backendOnline || !sciDataFile) {
      error = 'Backend is offline or no file selected.';
      return;
    }
    processing = true;
    error = '';
    try {
      sciResult = await processSciData(sciDataFile, sciDropoutPct, sciNoiseLevel, sciMethod);
    } catch (e: any) {
      error = e.message || 'Processing failed.';
      sciResult = null;
    } finally {
      processing = false;
    }
  }

  async function runSciDataDemo(preset: string) {
    if (!backendOnline) {
      error = 'Backend is offline.';
      return;
    }
    processing = true;
    error = '';
    try {
      sciResult = await loadSciDataDemo(preset, sciDropoutPct, sciNoiseLevel, sciMethod);
    } catch (e: any) {
      error = e.message || 'Failed to load demo.';
      sciResult = null;
    } finally {
      processing = false;
    }
  }

  async function switchSciMethod(newMethod: string) {
    if (!sciResult || !backendOnline) return;
    sciMethod = newMethod;
    processing = true;
    error = '';
    try {
      sciResult = await reconstructSciDataOnly(
        sciResult.plot.time,
        sciResult.plot.original,
        sciResult.plot.spoiled,
        sciResult.mask,
        newMethod,
        sciResult.name,
        sciResult.unitInfo
      );
    } catch (e: any) {
      error = e.message || 'Reconstruction failed.';
    } finally {
      processing = false;
    }
  }

  async function reprocessSciData() {
    if (sciDataFile) {
      await runSciDataProcessing();
    }
  }

  // ── Demo Tab Actions ─────────────────────────────────────────────
  async function runDemoSignal() {
    if (!backendOnline) {
      error = 'Backend is offline.';
      return;
    }
    processing = true;
    error = '';
    try {
      demoResult = await loadSciDataDemo(demoPreset, demoDropoutPct, demoNoiseLevel, demoMethod);
    } catch (e: any) {
      error = e.message || 'Failed to load demo.';
      demoResult = null;
    } finally {
      processing = false;
    }
  }

  async function switchDemoMethod(newMethod: string) {
    if (!demoResult || !backendOnline) return;
    demoMethod = newMethod;
    processing = true;
    error = '';
    try {
      demoResult = await reconstructSciDataOnly(
        demoResult.plot.time,
        demoResult.plot.original,
        demoResult.plot.spoiled,
        demoResult.mask,
        newMethod,
        demoResult.name,
        demoResult.unitInfo
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

<!-- Tab Navigation -->
<TabNavigation bind:activeTab onTabChange={onTabChange} />

<!-- ═══════════════════════════════════════════════════════════════ -->
<!-- AUDIO TAB -->
<!-- ═══════════════════════════════════════════════════════════════ -->
{#if activeTab === 'audio'}
  <!-- Mode toggle -->
  <div class="mb-5 flex items-center gap-4">
    <span class="text-sm text-slate-600">Mode:</span>
    <div class="flex rounded-lg bg-slate-100 p-1">
      <button
        class="px-4 py-1.5 text-sm font-medium rounded-md transition-all
               {audioMode === 'normal' ? 'bg-white text-blue-600 shadow-sm' : 'text-slate-500 hover:text-slate-700'}"
        onclick={() => { audioMode = 'normal'; }}
      >
        Normal Processing
      </button>
      <button
        class="px-4 py-1.5 text-sm font-medium rounded-md transition-all
               {audioMode === 'repair' ? 'bg-white text-orange-600 shadow-sm' : 'text-slate-500 hover:text-slate-700'}"
        onclick={() => { audioMode = 'repair'; }}
      >
        Repair Damaged Audio
      </button>
    </div>
  </div>

  {#if audioMode === 'normal'}
    <!-- Normal audio processing -->
    <section class="grid grid-cols-1 lg:grid-cols-2 gap-5 mb-5">
      <AudioInput onfileselected={onAudioFileSelected} ondemo={onAudioDemoSelected} {processing} />
      <DegradationControls
        bind:dropoutPct
        bind:noiseLevel
        bind:method
        {processing}
        onprocess={runAudioProcessing}
      />
    </section>
  {:else}
    <!-- Damaged audio repair -->
    <section class="grid grid-cols-1 lg:grid-cols-2 gap-5 mb-5">
      <DamagedAudioInput onfileselected={onDamagedFileSelected} {processing} />
      
      <!-- Repair settings card -->
      <div class="card h-full">
        <div class="section-header">
          <span class="section-number">2</span>
          <div>
            <span class="section-title">Repair Method</span>
            <span class="section-subtitle">— interpolation algorithm</span>
          </div>
        </div>

        <div class="space-y-4">
          <div>
            <span class="text-sm font-medium text-slate-700 mb-2 block">Interpolation Method</span>
            <div class="flex gap-2 flex-wrap">
              {#each [{id: 'pchip', label: 'PCHIP'}, {id: 'spline', label: 'Cubic Spline'}, {id: 'linear', label: 'Linear'}] as m}
                <button
                  class="method-pill {method === m.id ? 'method-pill-active' : 'method-pill-inactive'}"
                  onclick={() => { method = m.id; }}
                  disabled={processing}
                >
                  {m.label}
                </button>
              {/each}
            </div>
          </div>

          {#if repairResult}
            <div class="p-3 rounded-lg bg-orange-50 border border-orange-200">
              <p class="text-sm font-medium text-orange-800 mb-1">Damage Analysis</p>
              <p class="text-xs text-orange-700">{repairResult.analysis.summary}</p>
              <p class="text-xs text-orange-600 mt-1">
                {repairResult.metrics.damage_percent.toFixed(1)}% of samples repaired
              </p>
            </div>
          {/if}
        </div>
      </div>
    </section>
  {/if}

  <!-- Waveform Visualization -->
  <section class="mb-5">
    <WaveformChart
      time={audioPlotTime}
      original={audioPlotOriginal}
      spoiled={audioMode === 'repair' ? [] : audioPlotSpoiled}
      reconstructed={audioPlotReconstructed}
      title={audioMode === 'repair' ? 'Damaged vs Repaired Audio' : 'Signal Waveform Comparison'}
      height={360}
    />
  </section>

  <!-- Audio Playback -->
  <section class="mb-5">
    <AudioPlayer
      originalB64={audioOriginalB64}
      spoiledB64={audioSpoiledB64}
      reconstructedB64={audioReconstructedB64}
    />
  </section>

  <!-- Metrics Panel (only for normal mode) -->
  {#if audioMode === 'normal'}
    <section class="mb-5">
      <MetricsPanel
        metrics={audioMetrics}
        totalSamples={audioTotalSamples}
        sampleRate={audioSampleRate}
        {method}
      />
    </section>
  {/if}
{/if}

<!-- ═══════════════════════════════════════════════════════════════ -->
<!-- SCIENTIFIC DATA TAB -->
<!-- ═══════════════════════════════════════════════════════════════ -->
{#if activeTab === 'scidata'}
  <section class="grid grid-cols-1 lg:grid-cols-2 gap-5 mb-5">
    <ScientificDataInput 
      onfileselected={onSciFileSelected} 
      onpreset={onSciPresetSelected}
      {processing} 
    />
    
    <!-- Controls for scientific data -->
    <div class="card h-full">
      <div class="section-header">
        <span class="section-number">2</span>
        <div>
          <span class="section-title">Processing Settings</span>
          <span class="section-subtitle">— degradation + reconstruction</span>
        </div>
      </div>

      <div class="space-y-5">
        <!-- Dropout control -->
        <div>
          <div class="flex items-center justify-between mb-2">
            <span class="text-sm font-medium text-slate-700">Sample Dropout</span>
            <span class="badge badge-orange">{sciDropoutPct}%</span>
          </div>
          <input type="range" min="0" max="80" step="5" bind:value={sciDropoutPct}
                 disabled={processing} class="w-full" />
        </div>

        <!-- Noise control -->
        <div>
          <div class="flex items-center justify-between mb-2">
            <span class="text-sm font-medium text-slate-700">Noise Level</span>
            <span class="badge badge-blue">{sciNoiseLevel.toFixed(3)}</span>
          </div>
          <input type="range" min="0" max="0.2" step="0.005" bind:value={sciNoiseLevel}
                 disabled={processing} class="w-full" />
        </div>

        <!-- Method selector -->
        <div>
          <span class="text-sm font-medium text-slate-700 mb-2 block">Interpolation Method</span>
          <div class="flex gap-2 flex-wrap">
            {#each [{id: 'pchip', label: 'PCHIP'}, {id: 'spline', label: 'Spline'}, {id: 'linear', label: 'Linear'}, {id: 'moving_average', label: 'Moving Avg'}] as m}
              <button
                class="method-pill {sciMethod === m.id ? 'method-pill-active' : 'method-pill-inactive'}"
                onclick={() => switchSciMethod(m.id)}
                disabled={processing}
              >
                {m.label}
              </button>
            {/each}
          </div>
        </div>

        <!-- Re-process button -->
        <button
          onclick={reprocessSciData}
          disabled={processing || !sciDataFile}
          class="w-full px-4 py-3 rounded-lg text-sm font-semibold transition-all
                 bg-blue-600 text-white hover:bg-blue-700 active:bg-blue-800
                 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {#if processing}
            <svg class="w-4 h-4 inline animate-spin -mt-0.5 mr-2" fill="none" viewBox="0 0 24 24">
              <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
              <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"></path>
            </svg>
            Processing...
          {:else}
            Re-process Data
          {/if}
        </button>
      </div>
    </div>
  </section>

  <!-- Scientific Data Chart -->
  <section class="mb-5">
    <WaveformChart
      time={sciPlotTime}
      original={sciPlotOriginal}
      spoiled={sciPlotSpoiled}
      reconstructed={sciPlotReconstructed}
      title="{sciName} – Signal Comparison"
      height={360}
    />
  </section>

  <!-- Metrics Panel -->
  <section class="mb-5">
    <MetricsPanel
      metrics={sciMetrics}
      totalSamples={sciTotalSamples}
      sampleRate={0}
      method={sciMethod}
    />
  </section>
{/if}

<!-- ═══════════════════════════════════════════════════════════════ -->
<!-- DEMO SIGNALS TAB -->
<!-- ═══════════════════════════════════════════════════════════════ -->
{#if activeTab === 'demo'}
  <section class="grid grid-cols-1 lg:grid-cols-3 gap-5 mb-5">
    <!-- Preset Selection -->
    <div class="card">
      <div class="section-header">
        <span class="section-number">1</span>
        <div>
          <span class="section-title">Signal Type</span>
          <span class="section-subtitle">— choose a preset</span>
        </div>
      </div>

      <div class="space-y-2">
        {#each demoPresets as preset}
          <button
            class="w-full flex items-center gap-3 px-4 py-3 rounded-lg text-left transition-all
                   {demoPreset === preset.id 
                     ? 'bg-blue-50 border-2 border-blue-500 text-blue-800' 
                     : 'bg-slate-50 border-2 border-transparent hover:bg-slate-100 text-slate-700'}"
            onclick={() => { demoPreset = preset.id; }}
            disabled={processing}
          >
            <span class="text-xl">{preset.icon}</span>
            <span class="font-medium">{preset.name}</span>
          </button>
        {/each}
      </div>
    </div>

    <!-- Degradation Controls -->
    <div class="card">
      <div class="section-header">
        <span class="section-number">2</span>
        <div>
          <span class="section-title">Degradation</span>
          <span class="section-subtitle">— damage simulation</span>
        </div>
      </div>

      <div class="space-y-4">
        <div>
          <div class="flex items-center justify-between mb-2">
            <span class="text-sm font-medium text-slate-700">Dropout</span>
            <span class="badge badge-orange">{demoDropoutPct}%</span>
          </div>
          <input type="range" min="0" max="80" step="5" bind:value={demoDropoutPct}
                 disabled={processing} class="w-full" />
        </div>

        <div>
          <div class="flex items-center justify-between mb-2">
            <span class="text-sm font-medium text-slate-700">Noise</span>
            <span class="badge badge-blue">{demoNoiseLevel.toFixed(3)}</span>
          </div>
          <input type="range" min="0" max="0.2" step="0.005" bind:value={demoNoiseLevel}
                 disabled={processing} class="w-full" />
        </div>
      </div>
    </div>

    <!-- Method + Run -->
    <div class="card">
      <div class="section-header">
        <span class="section-number">3</span>
        <div>
          <span class="section-title">Reconstruct</span>
          <span class="section-subtitle">— method selection</span>
        </div>
      </div>

      <div class="space-y-4">
        <div>
          <span class="text-sm font-medium text-slate-700 mb-2 block">Method</span>
          <div class="flex gap-2 flex-wrap">
            {#each [{id: 'pchip', label: 'PCHIP'}, {id: 'spline', label: 'Spline'}, {id: 'linear', label: 'Linear'}] as m}
              <button
                class="method-pill {demoMethod === m.id ? 'method-pill-active' : 'method-pill-inactive'}"
                onclick={() => switchDemoMethod(m.id)}
                disabled={processing}
              >
                {m.label}
              </button>
            {/each}
          </div>
        </div>

        <button
          onclick={runDemoSignal}
          disabled={processing}
          class="w-full px-4 py-3 rounded-lg text-sm font-semibold transition-all
                 bg-emerald-600 text-white hover:bg-emerald-700 active:bg-emerald-800
                 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {#if processing}
            <svg class="w-4 h-4 inline animate-spin -mt-0.5 mr-2" fill="none" viewBox="0 0 24 24">
              <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
              <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"></path>
            </svg>
            Generating...
          {:else}
            Generate & Reconstruct
          {/if}
        </button>
      </div>
    </div>
  </section>

  <!-- Demo Chart -->
  <section class="mb-5">
    <WaveformChart
      time={demoPlotTime}
      original={demoPlotOriginal}
      spoiled={demoPlotSpoiled}
      reconstructed={demoPlotReconstructed}
      title="{demoName} – Demo Signal"
      height={360}
    />
  </section>

  <!-- Demo Metrics -->
  <section class="mb-5">
    <MetricsPanel
      metrics={demoMetrics}
      totalSamples={demoTotalSamples}
      sampleRate={0}
      method={demoMethod}
    />
  </section>
{/if}