# Numerical Interpolation for Signal Reconstruction and Analysis

This SvelteKit project implements a web-based toolkit for signal acquisition, preprocessing, numerical interpolation (linear, Lagrange polynomial, natural cubic spline), and validation (MSE, RMSE, MAE). The UI uses TailwindCSS and charts are rendered with Chart.js.

## Features

- Module 1 — Signal Acquisition & Preprocessing
  - Manual or CSV upload of discrete samples
  - Simple smoothing (moving average) and normalization
  - Raw vs preprocessed preview charts

- Module 2 — Numerical Interpolation Engine
  - Linear, Polynomial (Lagrange), Natural Cubic Spline
  - Adjustable resolution for continuous reconstruction

- Module 3 — Signal Analysis & Validation
  - MSE, RMSE and MAE calculation
  - Comparison plots between discrete and reconstructed signals

## Run locally

1. Install dependencies

```bash
npm install
```

2. Run dev server

```bash
npm run dev -- --open
```

This opens a local dev server (Vite + SvelteKit).

## Project structure (key files)

- `src/routes/+page.svelte` — main app layout and wiring
- `src/lib/components/SignalInput.svelte` — input & CSV upload
- `src/lib/components/Preprocessing.svelte` — smoothing & normalization
- `src/lib/components/InterpolationEngine.svelte` — interpolation controls and generation
- `src/lib/components/Analysis.svelte` — error metrics and comparison plots
- `src/lib/interpolation.ts` — numerical interpolation algorithms (well-commented)
- `src/lib/utils.ts` — helpers (MSE, RMSE, MAE, moving average, normalize)

## Notes

- All numerical computations run client-side in TypeScript.
- Interpolation implementations are educational and suitable for small to medium sized datasets.
- For very large datasets, consider streaming or WebWorker-based computation.

