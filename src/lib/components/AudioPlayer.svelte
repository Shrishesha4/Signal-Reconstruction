<script lang="ts">
  /**
   * AudioPlayer.svelte – Playback controls for original, spoiled, and reconstructed audio.
   * Each player shows a label, play/pause button, and waveform progress.
   */
  import { base64ToAudioUrl } from '$lib/api';
  import { onDestroy } from 'svelte';
  import { untrack } from 'svelte';

  let {
    originalB64 = '',
    spoiledB64 = '',
    reconstructedB64 = '',
  }: {
    originalB64?: string;
    spoiledB64?: string;
    reconstructedB64?: string;
  } = $props();

  type Track = {
    label: string;
    color: string;
    b64: string;
    dotColor: string;
  };

  let tracks = $derived<Track[]>([
    { label: 'Original', color: 'blue', b64: originalB64, dotColor: 'bg-blue-500' },
    { label: 'Spoiled', color: 'red', b64: spoiledB64, dotColor: 'bg-red-500' },
    { label: 'Reconstructed', color: 'emerald', b64: reconstructedB64, dotColor: 'bg-emerald-500' },
  ]);

  // Audio elements and state
  let audioElements = $state<(HTMLAudioElement | undefined)[]>([]);
  let playingIndex = $state(-1);
  let progress = $state([0, 0, 0]);
  let durations = $state([0, 0, 0]);
  let audioUrls = $state<string[]>([]);

  // Create/revoke object URLs when tracks change
  $effect(() => {
    const newUrls = tracks.map((t) => (t.b64 ? base64ToAudioUrl(t.b64) : ''));

    // Clean up old URLs without triggering reactivity
    untrack(() => {
      audioUrls.forEach((url) => { if (url) URL.revokeObjectURL(url); });
    });

    audioUrls = newUrls;
  });

  onDestroy(() => {
    audioUrls.forEach((url) => { if (url) URL.revokeObjectURL(url); });
    audioElements.forEach((el) => { el?.pause(); });
  });

  function togglePlay(idx: number) {
    const audio = audioElements[idx];
    if (!audio) return;

    if (playingIndex === idx) {
      audio.pause();
      playingIndex = -1;
    } else {
      // Stop any currently playing
      if (playingIndex >= 0) {
        audioElements[playingIndex]?.pause();
      }
      audio.currentTime = 0;
      audio.play();
      playingIndex = idx;
    }
  }

  function onTimeUpdate(idx: number) {
    const audio = audioElements[idx];
    if (!audio) return;
    progress[idx] = audio.currentTime;
    durations[idx] = audio.duration || 0;
  }

  function onEnded(idx: number) {
    playingIndex = -1;
    progress[idx] = 0;
  }

  function formatTime(s: number): string {
    if (!Number.isFinite(s)) return '0:00';
    const min = Math.floor(s / 60);
    const sec = Math.floor(s % 60);
    return `${min}:${sec.toString().padStart(2, '0')}`;
  }

  let hasAudio = $derived(tracks.some(t => t.b64));
</script>

<div class="card">
  <div class="section-header">
    <span class="section-number">
      <svg class="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
        <path d="M15.536 8.464a5 5 0 010 7.072M18.364 5.636a9 9 0 010 12.728M5.586 15H4a1 1 0 01-1-1v-4a1 1 0 011-1h1.586l4.707-4.707A1 1 0 0112 5.586v12.828a1 1 0 01-1.707.707L5.586 15z" stroke-linecap="round" stroke-linejoin="round"/>
      </svg>
    </span>
    <div>
      <span class="section-title">Audio Playback</span>
      <span class="section-subtitle">— compare original, spoiled & reconstructed</span>
    </div>
  </div>

  {#if !hasAudio}
    <div class="px-4 py-6 text-center rounded-lg bg-slate-50 border border-slate-200">
      <svg class="w-8 h-8 mx-auto text-slate-300 mb-2" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="1.5">
        <path d="M15.536 8.464a5 5 0 010 7.072M18.364 5.636a9 9 0 010 12.728M5.586 15H4a1 1 0 01-1-1v-4a1 1 0 011-1h1.586l4.707-4.707A1 1 0 0112 5.586v12.828a1 1 0 01-1.707.707L5.586 15z" stroke-linecap="round" stroke-linejoin="round"/>
      </svg>
      <p class="text-sm text-slate-500">Process audio to enable playback comparison.</p>
    </div>
  {:else}
    <div class="space-y-3">
      {#each tracks as track, idx}
        {#if track.b64}
          <div class="flex items-center gap-3 p-3 rounded-lg bg-slate-50 border border-slate-200">
            <!-- Hidden audio element -->
            <audio
              bind:this={audioElements[idx]}
              src={audioUrls[idx] || undefined}
              ontimeupdate={() => onTimeUpdate(idx)}
              onended={() => onEnded(idx)}
              preload="auto"
            ></audio>

            <!-- Color dot -->
            <span class="w-3 h-3 rounded-full shrink-0 {track.dotColor}"></span>

            <!-- Label -->
            <span class="text-sm font-medium text-slate-700 w-28 shrink-0">{track.label}</span>

            <!-- Play/Pause button -->
            <button
              onclick={() => togglePlay(idx)}
              class="shrink-0 w-8 h-8 rounded-full bg-white border border-slate-300
                     flex items-center justify-center shadow-sm
                     hover:bg-slate-50 hover:border-slate-400 transition-colors"
            >
              {#if playingIndex === idx}
                <svg class="w-4 h-4 text-slate-700" fill="currentColor" viewBox="0 0 20 20">
                  <path fill-rule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zM7 8a1 1 0 012 0v4a1 1 0 11-2 0V8zm5-1a1 1 0 00-1 1v4a1 1 0 102 0V8a1 1 0 00-1-1z" clip-rule="evenodd"/>
                </svg>
              {:else}
                <svg class="w-4 h-4 text-slate-700 ml-0.5" fill="currentColor" viewBox="0 0 20 20">
                  <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM9.555 7.168A1 1 0 008 8v4a1 1 0 001.555.832l3-2a1 1 0 000-1.664l-3-2z" clip-rule="evenodd"/>
                </svg>
              {/if}
            </button>

            <!-- Progress bar -->
            <div class="flex-1 min-w-0">
              <div class="w-full h-1.5 bg-slate-200 rounded-full overflow-hidden">
                <div
                  class="h-full rounded-full transition-all duration-100
                         {track.color === 'blue' ? 'bg-blue-500' : track.color === 'red' ? 'bg-red-500' : 'bg-emerald-500'}"
                  style="width: {durations[idx] > 0 ? (progress[idx] / durations[idx]) * 100 : 0}%"
                ></div>
              </div>
            </div>

            <!-- Time -->
            <span class="text-xs text-slate-400 tabular-nums shrink-0 w-12 text-right">
              {formatTime(progress[idx])} / {formatTime(durations[idx])}
            </span>
          </div>
        {/if}
      {/each}
    </div>
  {/if}
</div>
