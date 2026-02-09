/**
 * api.ts – Frontend service layer for the Python backend.
 * Handles all HTTP communication with the signal processing API.
 */

const API_BASE = '';

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

export interface ProcessingResult {
	sampleRate: number;
	totalSamples: number;
	plot: PlotData;
	audio: AudioData;
	metrics: ProcessingMetrics;
	mask: number[];
}

/**
 * Process an uploaded WAV file through the full pipeline.
 */
export async function processAudio(
	file: File,
	dropoutPct: number,
	noiseLevel: number,
	method: string
): Promise<ProcessingResult> {
	const form = new FormData();
	form.append('file', file);
	form.append('dropout_pct', dropoutPct.toString());
	form.append('noise_level', noiseLevel.toString());
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
	dropoutPct: number,
	noiseLevel: number,
	method: string
): Promise<ProcessingResult> {
	const params = new URLSearchParams({
		dropout_pct: dropoutPct.toString(),
		noise_level: noiseLevel.toString(),
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
