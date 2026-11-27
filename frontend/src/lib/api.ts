// Typed client for the logo-gen backend. All server state goes through
// TanStack Query (see queries.ts); this file only knows how to talk HTTP.

export class ApiError extends Error {
  status: number
  kind?: string
  constructor(message: string, status: number, kind?: string) {
    super(message)
    this.status = status
    this.kind = kind
  }
}

async function request<T>(method: string, url: string, body?: unknown): Promise<T> {
  const init: RequestInit = { method, headers: {} }
  if (body instanceof FormData) {
    init.body = body
  } else if (body !== undefined) {
    init.body = JSON.stringify(body)
    init.headers = { "Content-Type": "application/json" }
  }
  let res: Response
  try {
    res = await fetch(`/api${url}`, init)
  } catch {
    throw new ApiError("Cannot reach the logo-gen server. Is it still running?", 0, "network")
  }
  if (!res.ok) {
    const data = await res.json().catch(() => null)
    const msg = data?.error ?? data?.detail ?? `${res.status} ${res.statusText}`
    throw new ApiError(typeof msg === "string" ? msg : JSON.stringify(msg), res.status, data?.kind)
  }
  return res.json() as Promise<T>
}

export const api = {
  get: <T>(url: string) => request<T>("GET", url),
  post: <T>(url: string, body?: unknown) => request<T>("POST", url, body ?? {}),
  put: <T>(url: string, body?: unknown) => request<T>("PUT", url, body),
  patch: <T>(url: string, body?: unknown) => request<T>("PATCH", url, body),
  del: <T>(url: string) => request<T>("DELETE", url),
}

// ── Types ─────────────────────────────────────────────────────────

export type ReasoningEffort = "none" | "low" | "medium" | "high" | "xhigh" | "max"

export interface Settings {
  llm_backend: "openrouter" | "local" | "claude-cli"
  chat_model: string
  reasoning_effort: ReasoningEffort
  local_llm_base_url: string
  local_llm_model: string
  local_llm_api_key: string
  image_models: string[]
  images_per_prompt: number
  variations: number
  aspect_ratio: string
  transparent_background: boolean
  use_global_taste: boolean
  max_images_cloud: number
  max_images_local: number
  model_caps: Record<string, number>
  cloud_concurrency: number
  local_batch: Record<string, number>
}

export interface SecretStatus {
  set: boolean
  source: "env" | "file" | "none"
  masked: string
}

export interface SettingsResponse {
  settings: Settings
  secrets: Record<"OPENROUTER_KEY" | "HF_TOKEN", SecretStatus>
}

export interface KeyStatus {
  valid: boolean
  error?: string
  label?: string
  usage?: number
  limit?: number | null
  limit_remaining?: number | null
  is_free_tier?: boolean
}

export interface ChatModel {
  id: string
  name: string
  provider: string
  description: string
  context_length: number | null
  input_modalities: string[]
  output_modalities: string[]
  prompt_price: number | null
  completion_price: number | null
  supported_parameters: string[]
  reasoning: { supported_efforts?: string[]; mandatory?: boolean } | null
  created: number | null
  is_free: boolean
}

export interface ImageModel {
  id: string
  name: string
  provider: string
  description: string
  input_modalities: string[]
  output_modalities: string[]
  price_per_image: number | null
  price_unit: "image" | "megapixel" | "token" | ""
  supports_seed: boolean
  supports_transparent: boolean
  supports_references: boolean
  max_n: number
  vector: boolean
  created: number | null
  is_free: boolean
}

export interface Catalog {
  fetched_at: number
  stale: boolean
  chat: ChatModel[]
  image: ImageModel[]
}

export type Support = "ok" | "slow" | "no"

export interface LocalModel {
  id: string
  ref: string
  name: string
  repo: string
  page_url: string
  license: string
  license_url: string
  blurb: string
  size_bytes: number
  vram_gb: number
  resolution: number
  steps: number
  device: "cuda" | "mps" | "cpu"
  support: Record<"cuda" | "mps" | "cpu", Support>
  runs_here: Support | "tight"
  device_note: string
  gated: boolean
  access: { state: string; message: string } | null
  state: "installed" | "checking" | "runtime" | "downloading" | "partial" | "not_installed" | "error"
  downloaded_bytes: number
  speed_bps: number
  message: string
  installed: boolean
  batch: number
  batch_source: "calibrated" | "default"
  calibration: Calibration | null
  cap: number
}

export interface CalibrationRow {
  batch: number
  ok: boolean
  seconds?: number
  per_image?: number
  peak_gb?: number | null
  error?: string
  note?: string
}

export interface CalibrationOption extends CalibrationRow {
  key: "fastest" | "balanced" | "single"
  label: string
}

export interface Calibration {
  model_id: string
  device: string
  vram_gb: number | null
  load_seconds: number
  results: CalibrationRow[]
  options: CalibrationOption[]
  created_at?: number
}

export interface Spend {
  credits: number
  byok: number
  total: number
  chat_calls: number
  cloud_images: number
  local_images: number
  by_kind: Record<string, { credits: number; byok: number; calls: number }>
}

export interface SystemInfo {
  gpu: { name: string; vram_gb: number; driver: string } | null
  device: "cuda" | "mps" | "cpu"
  forced_device: string | null
  torch: string | null
  torch_variant: string
  local_stack: boolean
  disk_free_gb: number
  platform: string
  models_dir: string
  library?: { images: number; cost: number }
}

export interface RuntimeStatus {
  installed: boolean
  variant: string
  torch: string | null
  running: boolean
  error: string
  log: string[]
  elapsed: number
}

export interface LocalModelsResponse {
  models: LocalModel[]
  system: SystemInfo
  runtime: RuntimeStatus
}

export interface BriefColor {
  name: string
  hex: string
}

/** The model-agnostic design brief the designer writes for each direction. */
export interface Brief {
  concept: string
  mark_type: string
  format: string
  subject: string
  composition: string
  shape_language: string
  style: string
  palette: BriefColor[]
  background: string
  text: string
  avoid: string[]
}

export interface Prompt {
  prompt: string
  concept: string
  style: string
  brief?: Brief | null
}

/** What one model was (or would be) sent for one direction. */
export interface RenderedPrompt {
  model: string
  profile: string
  profile_name: string
  style: string
  prompt: string
  negative_prompt: string
  params: Record<string, unknown>
  notes: string[]
  direction: number
  concept?: string
}

export interface TasteProfile {
  scope: string
  liked: Record<string, string[]>
  disliked: Record<string, string[]>
  models: { model: string; score: number; likes: number; dislikes: number }[]
  more: string[]
  less: string[]
  counts: { like: number; dislike: number; pick: number; notes: number }
  empty: boolean
}

export interface PreferencesResponse {
  chat: TasteProfile | null
  global: TasteProfile
  global_enabled: boolean
  designer_context: string
}

export interface Job {
  id: string
  kind: "generate" | "quick" | "variations" | "calibrate"
  title: string
  status: "running" | "done" | "error" | "cancelled"
  done: number
  total: number
  message: string
  errors: string[]
  image_ids: string[]
  result: { prompts?: Prompt[]; rendered?: RenderedPrompt[]; images?: number; source_id?: string } & Partial<Calibration>
  created_at: number
  finished_at: number | null
  cost: number
  byok: number
  local_images: number
  chat_id: string | null
  notes: string[]
}

export interface ImageItem {
  id: string
  created_at: number
  path: string
  url: string
  kind: "generated" | "upscaled" | "cleaned"
  parent_id: string | null
  job_id: string | null
  chat_id: string | null
  prompt: string
  concept: string
  style: string
  model: string
  seed: number | null
  width: number | null
  height: number | null
  media_type: string
  cost: number
  gen_time: number
  favorite: boolean
  note: string
  vector: boolean
  primary_id: string | null
  primary?: ImageItem | null
  brief?: Brief | null
  negative_prompt?: string | null
  params?: Record<string, unknown> | null
  profile?: string | null
}

export interface ImageDetail extends ImageItem {
  children: ImageItem[]
  parent: ImageItem | null
}

export interface ChatSummary {
  id: string
  title: string
  created_at: number
  updated_at: number
  message_count: number
}

export interface ChatMessage {
  id: number
  role: "user" | "assistant"
  content: string
  images: string[]
  prompts: Prompt[]
  created_at: number
}

export interface Chat extends ChatSummary {
  messages: ChatMessage[]
  spend: Spend
}

export interface ToolOptions {
  upscale: { methods: Record<string, string>; scales: Record<string, number>; ai_available: boolean; device: string }
  background: { models: Record<string, string>; default: string }
}

export interface Upload {
  path: string
  url: string
  name: string
}

export const fileUrl = (path: string) => `/api/files/${path}`

export async function uploadImages(files: File[]): Promise<Upload[]> {
  const form = new FormData()
  files.forEach((f) => form.append("images", f))
  return request<Upload[]>("POST", "/uploads", form)
}

// ── Chat streaming (SSE over POST) ────────────────────────────────

export type ChatEvent =
  | { type: "start"; model: string }
  | { type: "token"; content: string }
  | { type: "reasoning"; content: string }
  | { type: "usage"; credits: number; byok: number }
  | { type: "done"; message_id: number; content: string; prompts: Prompt[] }
  | { type: "error"; message: string }

export async function streamChat(
  chatId: string,
  body: { content: string; images: string[] },
  onEvent: (e: ChatEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  const res = await fetch(`/api/chats/${chatId}/messages`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  })
  if (!res.ok || !res.body) {
    const data = await res.json().catch(() => null)
    throw new ApiError(data?.error ?? `Chat failed (${res.status})`, res.status)
  }
  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ""
  for (;;) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    const parts = buffer.split("\n\n")
    buffer = parts.pop() ?? ""
    for (const part of parts) {
      const line = part.split("\n").find((l) => l.startsWith("data: "))
      if (!line) continue
      try {
        onEvent(JSON.parse(line.slice(6)) as ChatEvent)
      } catch {
        // ignore a malformed event
      }
    }
  }
}
