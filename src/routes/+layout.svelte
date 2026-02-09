<script lang="ts">
	import '../app.css';
	import favicon from '$lib/assets/favicon.svg';

	let { children } = $props();
</script>

<svelte:head>
  <link rel="icon" href={favicon} />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
</svelte:head>

<header class="bg-white border-b border-slate-200">
  <div class="container py-4 flex items-center justify-between">
    <div class="flex items-center gap-3">
      <div class="w-9 h-9 rounded-lg bg-gradient-to-br from-blue-600 to-indigo-600 flex items-center justify-center">
        <svg class="w-5 h-5 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
          <path d="M3 17l4-8 4 6 4-10 4 12" stroke-linecap="round" stroke-linejoin="round"/>
        </svg>
      </div>
      <div>
        <h1 class="text-lg font-bold text-slate-800 tracking-tight">Signal Reconstruction</h1>
        <p class="text-xs text-slate-500">Numerical Interpolation &middot; Analysis &middot; Visualization</p>
      </div>
    </div>
    <span class="badge badge-blue">Final Year Project</span>
  </div>
</header>

<main class="container py-6">{@render children()}</main>

<footer class="border-t border-slate-200 bg-white mt-8">
  <div class="container py-4 text-center text-xs text-slate-400">
    Numerical Interpolation for Signal Reconstruction &amp; Analysis &middot; SvelteKit + Chart.js
  </div>
</footer>
