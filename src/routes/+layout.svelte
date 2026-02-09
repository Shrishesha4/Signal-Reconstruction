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
      <img src="/logo.png" alt="Signal Reconstruction Logo" class="h-16 w-auto" />
      <div>
        <h1 class="text-lg font-bold text-slate-800 tracking-tight">Signal Reconstruction</h1>
        <p class="text-xs text-slate-500">Advanced Audio Signal Processing &middot; Numerical Interpolation</p>
      </div>
    </div>
  </div>
</header>

<main class="container py-6">{@render children()}</main>

<footer class="border-t border-slate-200 bg-white mt-8">
  <div class="container py-4 text-center text-xs text-slate-400">
    Numerical Interpolation for Audio Signal Reconstruction &amp; Analysis &middot; SvelteKit + FastAPI
  </div>
</footer>
