import type { AppSettings, ImageInfo } from "./types";

const BASE = "/api";

export function imageUrl(path: string): string {
  const norm = path.replace(/\\/g, "/");
  return `${BASE}/output/${norm}`;
}

// ── SSE helper ────────────────────────────────────────────────────

export interface SSEHandlers {
  onToken?: (content: string) => void;
  onProgress?: (current: number, total: number, message: string) => void;
  onPrompts?: (prompts: Array<{ prompt: string; concept: string; style: string }>) => void;
  onResult?: (data: Record<string, unknown>) => void;
  onError?: (message: string) => void;
}

export async function streamPost(
  url: string,
  body: unknown,
  handlers: SSEHandlers,
  options?: { signal?: AbortSignal },
): Promise<void> {
  let res: Response;
  try {
    res = await fetch(`${BASE}${url}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal: options?.signal,
    });
  } catch (e) {
    if ((e as DOMException)?.name === "AbortError") return;
    throw e;
  }

  if (!res.ok) {
    const err = await res.json().catch(() => ({ error: res.statusText }));
    handlers.onError?.(err.error || "Request failed");
    return;
  }

  const reader = res.body!.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop()!;

      for (const line of lines) {
        if (!line.startsWith("data: ")) continue;
        try {
          const data = JSON.parse(line.slice(6));
          switch (data.type) {
            case "token":
              handlers.onToken?.(data.content);
              break;
            case "progress":
              handlers.onProgress?.(data.current, data.total, data.message);
              break;
            case "prompts":
              handlers.onPrompts?.(data.prompts);
              break;
            case "done":
            case "result":
              handlers.onResult?.(data);
              break;
            case "error":
              handlers.onError?.(data.message);
              break;
          }
        } catch {
          // skip malformed lines
        }
      }
    }
  } catch (e) {
    if ((e as DOMException)?.name === "AbortError") return;
    throw e;
  }
}

// ── REST helpers ──────────────────────────────────────────────────

async function get<T>(url: string): Promise<T> {
  const res = await fetch(`${BASE}${url}`);
  if (!res.ok) throw new Error(`GET ${url} failed: ${res.statusText}`);
  return res.json();
}

async function post<T>(url: string, body?: unknown): Promise<T> {
  const res = await fetch(`${BASE}${url}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ error: res.statusText }));
    throw new Error(err.error || "Request failed");
  }
  return res.json();
}

// ── Upload ────────────────────────────────────────────────────────

export async function uploadImages(files: File[]): Promise<ImageInfo[]> {
  const form = new FormData();
  for (const f of files) form.append("images", f);
  const res = await fetch(`${BASE}/upload`, { method: "POST", body: form });
  if (!res.ok) throw new Error("Upload failed");
  return res.json();
}

// ── Chat ──────────────────────────────────────────────────────────

export const chatReset = () => post<{ ok: boolean }>("/chat/reset");
export const chatPrompts = () => get<{ has_prompts: boolean; prompts: unknown[] }>("/chat/prompts");

// ── Images ────────────────────────────────────────────────────────

export const fetchGenerated = () => get<ImageInfo[]>("/images/generated");
export const fetchUpscaled = () => get<ImageInfo[]>("/images/upscaled");
export const fetchCleaned = () => get<ImageInfo[]>("/images/cleaned");
export const fetchAllSources = () => get<ImageInfo[]>("/images/all-sources");

// ── Upscale ───────────────────────────────────────────────────────

export const fetchUpscaleOptions = () =>
  get<{ methods: Record<string, string>; scales: Record<string, number> }>("/upscale/options");

export const upscaleOne = (path: string, method: string, scale: number) =>
  post<ImageInfo>("/upscale", { path, method, scale });

// ── Background removal ───────────────────────────────────────────

export const fetchBgOptions = () =>
  get<{ models: Record<string, string>; default: string }>("/bg-remove/options");

export interface BgRemoveParams {
  path: string;
  method: "ai" | "color";
  erode_pixels: number;
  // AI method
  model?: string;
  alpha_matting?: boolean;
  // Color method
  tolerance?: number;
  auto_corners?: boolean;
  color?: string;  // hex like "#ffffff"
}

export const bgRemoveOne = (params: BgRemoveParams) =>
  post<ImageInfo>("/bg-remove", params);

// ── Settings ─────────────────────────────────────────────────────

export const fetchSettings = () => get<AppSettings>("/settings");
export const fetchDefaultSettings = () => get<AppSettings>("/settings/defaults");
export const updateSettings = (data: Partial<AppSettings>) => post<{ ok: boolean }>("/settings", data);
export const fetchClaudeCliStatus = () => get<{ available: boolean }>("/claude-cli/status");
