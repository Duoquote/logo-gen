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
  image_sort: ImageSort
  variations: number
  aspect_ratio: string
  transparent_background: boolean
  use_global_taste: boolean
  ask_before_generating: boolean
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
  pinned: boolean
  streaming: boolean
  spend: number
  thumbnail: string | null
}

export type MessageStatus = "streaming" | "done" | "stopped" | "error"

export interface ChatOutput extends ImageItem {
  rating: "like" | "dislike" | null
}

export interface ChatMessage {
  id: number
  chat_id: string
  parent_id: number | null
  role: "user" | "assistant"
  content: string
  reasoning: string
  images: string[]
  prompts: Prompt[]
  status: MessageStatus
  error: string
  model: string
  created_at: number
  finished_at: number | null
  siblings: number[]
  cost: { credits: number; byok: number } | null
  outputs: ChatOutput[]
}

export interface Chat {
  id: string
  title: string
  created_at: number
  updated_at: number
  pinned: boolean
  ask_first: boolean | null // null follows Settings
  active_leaf: number | null
  messages: ChatMessage[]
  spend: Spend
  jobs: (Job & { message_id: number | null })[]
  running: number | null
}

export interface FeedbackEvent {
  id: number
  created_at: number
  kind: "like" | "dislike" | "unrate" | "pick"
  image_id: string | null
  chat_id: string | null
  message_id: number | null
  model: string
  prompt: string
  concept: string
  style: string
  data: Record<string, unknown>
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

// ── Following a designer reply (SSE with resume) ────────────────────

export type ReplyEvent =
  | { type: "snapshot"; id: number; content: string; reasoning: string; status: MessageStatus; error: string }
  | { type: "start"; id: number; model: string }
  | { type: "token"; id: number; content: string }
  | { type: "reasoning"; id: number; content: string }
  | { type: "usage"; id: number; credits: number; byok: number }
  | { type: "done"; id: number; status: MessageStatus; content: string; prompts?: Prompt[] }
  | { type: "error"; id?: number; status: "error"; message: string }
  | { type: "end"; status: MessageStatus }
  | { type: "ping" }

/**
 * Follow a reply that runs on the server. EventSource reconnects by itself and
 * sends Last-Event-ID, so a dropped connection resumes where it stopped. The
 * server ends the stream with an "end" event, and we close then (otherwise
 * EventSource would reconnect forever).
 */
export function followReply(chatId: string, messageId: number, onEvent: (e: ReplyEvent) => void): () => void {
  const es = new EventSource(`/api/chats/${chatId}/messages/${messageId}/stream`)
  es.onmessage = (m) => {
    let ev: ReplyEvent
    try {
      ev = JSON.parse(m.data) as ReplyEvent
    } catch {
      return
    }
    onEvent(ev)
    if (ev.type === "end") es.close()
  }
  return () => es.close()
}

// ── Model rankings ─────────────────────────────────────────────────

export type ImageSort = "quality" | "value" | "cheapest" | "popular" | "local" | "newest" | "name"

export interface ModelRank {
  ranked: boolean
  leaderboard_name: string | null
  category: "commercial_design" | "overall" | null
  rating: number | null
  ci: [number, number] | null
  votes: number | null
  arena_rank: number | null
  price: number | null
  value: number | null
  pareto: boolean
  rank: number | null
  popularity_rank: number | null
}

export interface Rankings {
  source: { name: string; url: string; dataset: string; license: string; published: string; bundled: boolean; fetched_at?: number }
  popularity: { name: string; url: string; license: string; as_of: string; citation: string } | null
  notes: string[]
  formula: { elo_per_doubling: number; price_ref: number; price_floor: number }
  models: Record<string, ModelRank>
  recent: string[]
}
