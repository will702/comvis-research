// Typed API client with abort-on-new-request pattern

export interface DemoImage {
  id: string;
  label: string;
  thumb: string; // base64 PNG
}

export interface PipelineParams {
  clahe_clip: number;
  clahe_tile: number;
  skin_close: number;
  skin_open: number;
  cr_k: number;
  a_k: number;
  cr_floor: number;
  a_floor: number;
  red_open: number;
  red_close: number;
  max_aspect: number;
  min_fill: number;
  max_area_pct: number;
  min_local_contrast: number;
}

export const DEFAULT_PARAMS: PipelineParams = {
  clahe_clip: 3.0,
  clahe_tile: 8,
  skin_close: 9,
  skin_open: 5,
  cr_k: 1.5,
  a_k: 1.2,
  cr_floor: 145,
  a_floor: 133,
  red_open: 3,
  red_close: 7,
  max_aspect: 3.5,
  min_fill: 0.25,
  max_area_pct: 0.04,
  min_local_contrast: 1.5,
};

export interface BlobInfo {
  bbox: [number, number, number, number];
  area: number;
  aspect: number;
  fill: number;
  local_contrast: number;
  circularity: number;
  kept: boolean;
  reject_reason: string | null;
}

export interface Feature {
  idx: number;
  name: string;
  value: number;
  description: string;
}

export interface GlcmProps {
  contrast: number;
  homogeneity: number;
  energy: number;
}

export interface Trace {
  // Stage images (base64 PNG data URIs — we prefix "data:image/png;base64,")
  img_resized: string;
  img_gray: string;
  img_clahe: string;
  img_face_box: string;
  img_roi_mask: string;
  img_skin_raw: string;
  img_skin_mask: string;
  img_masked_face: string;
  img_red_mask: string;
  img_blobs: string;
  img_lbp_r1: string;
  img_lbp_r2: string;
  img_lbp_r3: string;
  img_lbp_hist_r1: string;
  img_lbp_hist_r2: string;
  img_lbp_hist_r3: string;
  img_a_heat: string;
  img_cr_heat: string;
  // Metadata
  face_detected: boolean;
  cascade_attempt: number;
  face_box: [number, number, number, number] | null;
  roi_regions: { type: string; bbox: [number, number, number, number] }[];
  skin_pixel_count: number;
  thr_cr: number;
  thr_a: number;
  mu_cr: number; std_cr: number;
  mu_a: number;  std_a: number;
  blobs: BlobInfo[];
  lesion_count: number;
  kept_blobs: number;
  rejected_blobs: number;
  glcm: GlcmProps;
  feature_vector: number[];
  features: Feature[];
  feature_groups: Record<string, number[]>;
}

export interface Prediction {
  name: string;
  label: string;
  confidence: number | null;
  accuracy: number | null;
  f1: number | null;
  is_champion: boolean;
  error?: string;
}

export interface RunResult {
  trace: Trace;
  predictions: Prediction[];
  params: PipelineParams;
}

// Empty string = relative path (FastAPI serves frontend on same origin in prod).
// Set VITE_API_URL=http://localhost:8000 for local dev when running separately.
const BASE = (import.meta.env.VITE_API_URL as string | undefined) ?? "";

let _runController: AbortController | null = null;

export async function fetchDemos(): Promise<DemoImage[]> {
  const r = await fetch(`${BASE}/api/demos`);
  if (!r.ok) throw new Error("Failed to fetch demos");
  return r.json();
}

export async function runPipeline(opts: {
  demoId?: string;
  file?: File;
  params: PipelineParams;
}): Promise<RunResult> {
  // Cancel previous in-flight run
  if (_runController) _runController.abort();
  _runController = new AbortController();

  const fd = new FormData();
  if (opts.file) fd.append("file", opts.file);
  if (opts.demoId) fd.append("demo_id", opts.demoId);
  fd.append("params_json", JSON.stringify(opts.params));

  const r = await fetch(`${BASE}/api/run`, {
    method: "POST",
    body: fd,
    signal: _runController.signal,
  });
  if (!r.ok) {
    const err = await r.text();
    throw new Error(err || "Pipeline failed");
  }
  return r.json();
}

/** Prefix a raw b64 string with the data URI header */
export function b64img(raw: string): string {
  if (!raw) return "";
  if (raw.startsWith("data:")) return raw;
  return `data:image/png;base64,${raw}`;
}
