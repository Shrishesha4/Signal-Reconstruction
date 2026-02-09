<script lang="ts" module>
  export type Tab = 'audio' | 'scidata' | 'demo';
</script>

<script lang="ts">
  /**
   * TabNavigation.svelte – Tab bar for switching between app sections.
   * Supports: Audio Reconstruction, Scientific Data, Demo Signals
   */

  let {
    activeTab = $bindable<Tab>('audio'),
    onTabChange,
  }: {
    activeTab: Tab;
    onTabChange?: (tab: Tab) => void;
  } = $props();

  const tabs: { id: Tab; label: string; icon: string; description: string }[] = [
    {
      id: 'audio',
      label: 'Audio',
      icon: 'M9 19V6l12-3v13M9 19c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zm12-3c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zM9 10l12-3',
      description: 'Upload & reconstruct audio signals',
    },
    {
      id: 'scidata',
      label: 'Scientific Data',
      icon: 'M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z',
      description: 'Reconstruct IoT, ECG, radio signals',
    },
    {
      id: 'demo',
      label: 'Demo Signals',
      icon: 'M13 10V3L4 14h7v7l9-11h-7z',
      description: 'Try with preset demo signals',
    },
  ];

  function selectTab(tab: Tab) {
    if (tab !== activeTab) {
      activeTab = tab;
      onTabChange?.(tab);
    }
  }
</script>

<nav class="mb-6">
  <!-- Desktop Tab Bar -->
  <div class="hidden sm:flex border-b border-slate-200">
    {#each tabs as tab}
      <button
        class="group relative px-6 py-4 text-sm font-medium transition-colors flex items-center gap-2
               {activeTab === tab.id 
                 ? 'text-blue-600 border-b-2 border-blue-600 -mb-px' 
                 : 'text-slate-500 hover:text-slate-800 hover:bg-slate-50'}"
        onclick={() => selectTab(tab.id)}
        title={tab.description}
      >
        <svg class="w-5 h-5 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="1.5">
          <path stroke-linecap="round" stroke-linejoin="round" d={tab.icon} />
        </svg>
        {tab.label}
        
        <!-- Active indicator dot -->
        {#if activeTab === tab.id}
          <span class="absolute top-2 right-2 w-1.5 h-1.5 rounded-full bg-blue-600"></span>
        {/if}
      </button>
    {/each}
  </div>

  <!-- Mobile Tab Bar (Segmented Control) -->
  <div class="sm:hidden flex rounded-xl bg-slate-100 p-1 gap-1">
    {#each tabs as tab}
      <button
        class="flex-1 px-3 py-2.5 text-xs font-medium rounded-lg transition-all flex flex-col items-center gap-1
               {activeTab === tab.id 
                 ? 'bg-white text-blue-600 shadow-sm' 
                 : 'text-slate-500 hover:text-slate-700'}"
        onclick={() => selectTab(tab.id)}
      >
        <svg class="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
          <path stroke-linecap="round" stroke-linejoin="round" d={tab.icon} />
        </svg>
        <span class="truncate max-w-full">{tab.label}</span>
      </button>
    {/each}
  </div>

  <!-- Tab Description -->
  <p class="mt-3 text-xs text-slate-500 text-center sm:text-left">
    {tabs.find(t => t.id === activeTab)?.description}
  </p>
</nav>
