// Client for the FastAPI backend (floodseg.system.server), proxied at /api by Vite.

export type StageId = "select" | "compare" | "auth" | "catalog" | "download" | "read" | "mask" | "inference" | "write";
export type StageStatus = "pending" | "running" | "done" | "cached";

export interface Stage {
  status: StageStatus;
  fraction: number;
  detail: string;
  seconds: number | null;
}

export interface Job {
  id: string;
  event: string;
  tile: string;
  date: string;
  status: "queued" | "running" | "done" | "error";
  cached: boolean;
  error: string | null;
  received: string;
  stop?: string;
  seconds?: number;
  elapsed?: number;
  scene?: string;
  stages: Record<string, Stage>;
}

export interface Scene {
  scene: string;
  tile: string;
  date: string;
  product: string;
  pixels: { valid: number; cloud: number; nodata: number; water_valid: number };
  total_s: number;
  bounds: [number, number, number, number];
  comparison: { model_group: string; seconds: number; water_valid: number } | null;
}

export interface Acquisition {
  Id: string;
  Name: string;
  date: string;
  cloud_cover: number | null;
  cached: boolean;
}

export interface Pixel {
  inside: boolean;
  probability?: number | null;
  uncertainty?: number | null;
  mask?: number;
  state?: "valid" | "cloud" | "nodata";
}

export interface Metrics {
  available: boolean;
  pixels: number;
  iou: number | null;
  precision: number | null;
  recall: number | null;
  tp_km2: number;
  fp_km2: number;
  fn_km2: number;
  reference_water_km2: number;
  partition_share: Record<"train" | "dev" | "test" | "buffer" | "test_near_train", number>;
}

export interface Flood {
  available: boolean;
  is_reference?: boolean;
  reference_scene?: string;
  flooded_km2?: number;
  water_km2?: number;
  water_ref_km2?: number;
  valid_both_share?: number;
  record: { chosen: string; window: [string, string]; ranking: { date: string; thick_cloud: number; valid: number }[] } | null;
}

async function get<T>(url: string): Promise<T> {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
  return r.json();
}

export const api = {
  status: () => get<{ models_loaded: boolean; model: string; busy: boolean }>("/api/status"),
  scenes: () => get<Scene[]>("/api/scenes"),
  acquisitions: (tile: string, start: string, end: string) =>
    get<Acquisition[]>(`/api/acquisitions?tile=${tile}&start=${start}&end=${end}`),
  job: (id: string) => get<Job>(`/api/jobs/${id}`),
  pixel: (scene: string, lon: number, lat: number) => get<Pixel>(`/api/pixel/${scene}?lon=${lon}&lat=${lat}`),
  createJob: async (body: { event: string; tile: string; date: string; bounds?: number[]; compare?: boolean }): Promise<Job> => {
    const r = await fetch("/api/jobs", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
    return r.json();
  },
  metrics: (scene: string, bbox?: number[], part?: string) => {
    const q = new URLSearchParams();
    if (bbox) q.set("bbox", bbox.join(","));
    if (part) q.set("part", part);
    return get<Metrics>(`/api/metrics/${scene}?${q}`);
  },
  flood: (scene: string) => get<Flood>(`/api/flood/${scene}`),
  createPreevent: async (body: { event: string; tile: string }): Promise<Job> => {
    const r = await fetch("/api/preevent", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
    return r.json();
  },
  frozen: (event: string) => get<{ chips: number; iou_mean: number; iou_std: number; precision_mean: number; recall_mean: number }>(`/api/frozen-metrics?event=${event}`),
  tileUrl: (scene: string, layer: string) => `${location.origin}/api/tiles/${scene}/${layer}/{z}/{x}/{y}.png`,
};
