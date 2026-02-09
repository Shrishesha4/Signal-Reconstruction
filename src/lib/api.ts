/**
 * api.ts – Frontend service layer for the Python backend.
 * Handles all HTTP communication with the signal processing API.
 * 
 * Endpoints:
 *   Audio:
 *     GET  /api/demo              – Demo signal with degradation
 *     POST /api/process           – Upload WAV + degrade + reconstruct
 *     POST /api/reconstruct       – Re-run reconstruction only
 *     POST /api/audio/repair      – Repair damaged audio file
 *   
 *   Scientific Data:
 *     GET  /api/scidata/presets   – List available demo presets
 *     GET  /api/scidata/demo      – Demo with preset signal
 *     POST /api/scidata/process   – Upload CSV/JSON + process
 */

const API_BASE = '';

// ═══════════════════════════════════════════════════════════════════
// Types – Audio
// ═══════════════════════════════════════════════════════════════════

export interface PlotData {
	time: number[];
	original: number[];
	spoiled: number[];
	reconstructed: number[];
}

export interface AudioData {
	original: string; // base64 WAV
	spoiled: string;
	reconstructed: string;
}

export interface ProcessingMetrics {
	mse: number;
	rmse: number;
	mae: number;
	snr_db: number;
}

export interface DegradationParams {
	dropoutPct: number;      // % of audio to drop (0-50)
	dropoutLengthMs: number; // average dropout segment length in ms (10-500)
	glitchPct: number;       // % of audio with glitches (0-20)
	clipPct: number;         // % of audio with clipping (0-30)
	noiseLevel: number;      // Gaussian noise amplitude (0-0.1)
}

export const DEFAULT_DEGRADATION: DegradationParams = {
	dropoutPct: 10,
	dropoutLengthMs: 100,
	glitchPct: 5,
	clipPct: 10,
	noiseLevel: 0.02
};

export interface ProcessingResult {
	sampleRate: number;
	totalSamples: number;
	plot: PlotData;
	audio: AudioData;
	metrics: ProcessingMetrics;
	mask: number[];
}

// ═══════════════════════════════════════════════════════════════════
// Types – Damaged Audio Repair
// ═══════════════════════════════════════════════════════════════════

export interface RepairPlotData {
	time: number[];
	damaged: number[];
	reconstructed: number[];
}

export interface RepairAudioData {
	damaged: string;
	reconstructed: string;
}

export interface RepairMetrics {
	damage_percent: number;
	samples_repaired: number;
	summary: string;
}

export interface RepairAnalysis {
	summary: string;
	damage_percent: number;
	stats: Record<string, number>;
}

export interface RepairResult {
	sampleRate: number;
	totalSamples: number;
	plot: RepairPlotData;
	audio: RepairAudioData;
	metrics: RepairMetrics;
	mask: number[];
	analysis: RepairAnalysis;
}

// ═══════════════════════════════════════════════════════════════════
// Types – Scientific Data
// ═══════════════════════════════════════════════════════════════════

export interface SciDataPlot {
	time: number[];
	original: number[];
	spoiled: number[];
	reconstructed: number[];
}

export interface SciDataResult {
	name: string;
	totalSamples: number;
	plot: SciDataPlot;
	metrics: ProcessingMetrics;
	mask: number[];
	unitInfo: { min: number; max: number };
}

export interface SciDataPreset {
	id: string;
	name: string;
	description: string;
}

// ═══════════════════════════════════════════════════════════════════
// Audio API Functions
// ═══════════════════════════════════════════════════════════════════

/**
 * Process an uploaded WAV file through the full pipeline.
 */
export async function processAudio(
	file: File,
	degradation: DegradationParams,
	method: string
): Promise<ProcessingResult> {
	const form = new FormData();
	form.append('file', file);
	form.append('dropout_pct', degradation.dropoutPct.toString());
	form.append('dropout_length_ms', degradation.dropoutLengthMs.toString());
	form.append('glitch_pct', degradation.glitchPct.toString());
	form.append('clip_pct', degradation.clipPct.toString());
	form.append('noise_level', degradation.noiseLevel.toString());
	form.append('method', method);

	const res = await fetch(`${API_BASE}/api/process`, {
		method: 'POST',
		body: form
	});

	if (!res.ok) {
		const err = await res.json().catch(() => ({ detail: 'Server error' }));
		throw new Error(err.detail || `HTTP ${res.status}`);
	}

	return res.json();
}

/**
 * Load the demo signal with given parameters.
 */
export async function loadDemo(
	degradation: DegradationParams,
	method: string
): Promise<ProcessingResult> {
	const params = new URLSearchParams({
		dropout_pct: degradation.dropoutPct.toString(),
		dropout_length_ms: degradation.dropoutLengthMs.toString(),
		glitch_pct: degradation.glitchPct.toString(),
		clip_pct: degradation.clipPct.toString(),
		noise_level: degradation.noiseLevel.toString(),
		method
	});

	const res = await fetch(`${API_BASE}/api/demo?${params}`);

	if (!res.ok) {
		const err = await res.json().catch(() => ({ detail: 'Server error' }));
		throw new Error(err.detail || `HTTP ${res.status}`);
	}

	return res.json();
}

/**
 * Re-run reconstruction with a different method (without re-uploading audio).
 */
export async function reconstructOnly(
	time: number[],
	original: number[],
	spoiled: number[],
	mask: number[],
	method: string,
	sampleRate: number
): Promise<ProcessingResult> {
	const res = await fetch(`${API_BASE}/api/reconstruct`, {
		method: 'POST',
		headers: { 'Content-Type': 'application/json' },
		body: JSON.stringify({
			time,
			original,
			spoiled,
			mask: mask.map((m) => m > 0.5),
			method,
			sampleRate
		})
	});

	if (!res.ok) {
		const err = await res.json().catch(() => ({ detail: 'Server error' }));
		throw new Error(err.detail || `HTTP ${res.status}`);
	}

	return res.json();
}

// ═══════════════════════════════════════════════════════════════════
// Damaged Audio Repair API Functions
// ═══════════════════════════════════════════════════════════════════

/**
 * Repair a damaged audio file.
 */
export async function repairDamagedAudio(
	file: File,
	method: string = 'pchip',
	autoDetect: boolean = true
): Promise<RepairResult> {
	const form = new FormData();
	form.append('file', file);
	form.append('method', method);
	form.append('auto_detect', autoDetect.toString());

	const res = await fetch(`${API_BASE}/api/audio/repair`, {
		method: 'POST',
		body: form
	});

	if (!res.ok) {
		const err = await res.json().catch(() => ({ detail: 'Server error' }));
		throw new Error(err.detail || `HTTP ${res.status}`);
	}

	return res.json();
}

// ═══════════════════════════════════════════════════════════════════
// Scientific Data API Functions
// ═══════════════════════════════════════════════════════════════════

/**
 * Get list of available demo presets.
 */
export async function getSciDataPresets(): Promise<SciDataPreset[]> {
	const res = await fetch(`${API_BASE}/api/scidata/presets`);

	if (!res.ok) {
		throw new Error('Failed to fetch presets');
	}

	const data = await res.json();
	return data.presets;
}

/**
 * Load a demo scientific signal with specified preset.
 */
export async function loadSciDataDemo(
	preset: string,
	dropoutPct: number,
	noiseLevel: number,
	method: string
): Promise<SciDataResult> {
	const params = new URLSearchParams({
		preset,
		dropout_pct: dropoutPct.toString(),
		noise_level: noiseLevel.toString(),
		method
	});

	const res = await fetch(`${API_BASE}/api/scidata/demo?${params}`);

	if (!res.ok) {
		const err = await res.json().catch(() => ({ detail: 'Server error' }));
		throw new Error(err.detail || `HTTP ${res.status}`);
	}

	return res.json();
}

/**
 * Process uploaded scientific data file (CSV or JSON).
 */
export async function processSciData(
	file: File,
	dropoutPct: number,
	noiseLevel: number,
	method: string
): Promise<SciDataResult> {
	const form = new FormData();
	form.append('file', file);
	form.append('dropout_pct', dropoutPct.toString());
	form.append('noise_level', noiseLevel.toString());
	form.append('method', method);

	const res = await fetch(`${API_BASE}/api/scidata/process`, {
		method: 'POST',
		body: form
	});

	if (!res.ok) {
		const err = await res.json().catch(() => ({ detail: 'Server error' }));
		throw new Error(err.detail || `HTTP ${res.status}`);
	}

	return res.json();
}

/**
 * Re-run reconstruction on scientific data with different method.
 */
export async function reconstructSciDataOnly(
	time: number[],
	original: number[],
	spoiled: number[],
	mask: number[],
	method: string,
	name: string = 'Custom',
	unitInfo: { min: number; max: number } = { min: -1, max: 1 }
): Promise<SciDataResult> {
	const res = await fetch(`${API_BASE}/api/scidata/reconstruct`, {
		method: 'POST',
		headers: { 'Content-Type': 'application/json' },
		body: JSON.stringify({
			time,
			original,
			spoiled,
			mask: mask.map((m) => m > 0.5),
			method,
			name,
			unitInfo
		})
	});

	if (!res.ok) {
		const err = await res.json().catch(() => ({ detail: 'Server error' }));
		throw new Error(err.detail || `HTTP ${res.status}`);
	}

	return res.json();
}

// ═══════════════════════════════════════════════════════════════════
// Utility Functions
// ═══════════════════════════════════════════════════════════════════

/**
 * Check if the backend is running.
 */
export async function checkHealth(): Promise<boolean> {
	try {
		const controller = new AbortController();
		const timeout = setTimeout(() => controller.abort(), 3000);
		const res = await fetch(`${API_BASE}/api/health`, { signal: controller.signal });
		clearTimeout(timeout);
		return res.ok;
	} catch {
		return false;
	}
}

/**
 * Convert a base64 WAV string to an audio URL for playback.
 */
export function base64ToAudioUrl(b64: string): string {
	const binary = atob(b64);
	const bytes = new Uint8Array(binary.length);
	for (let i = 0; i < binary.length; i++) {
		bytes[i] = binary.charCodeAt(i);
	}
	const blob = new Blob([bytes], { type: 'audio/wav' });
	return URL.createObjectURL(blob);
}

