import { QueryClient, useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import { toast } from "sonner"

import {
  api,
  type Catalog,
  type Chat,
  type ChatSummary,
  type ImageDetail,
  type ImageItem,
  type Job,
  type KeyStatus,
  type LocalModel,
  type LocalModelsResponse,
  type PreferencesResponse,
  type Prompt,
  type RuntimeStatus,
  type Settings,
  type SettingsResponse,
  type Spend,
  type SystemInfo,
  type ToolOptions,
} from "@/lib/api"

export const queryClient = new QueryClient({
  defaultOptions: {
    queries: { staleTime: 10_000, retry: 1, refetchOnWindowFocus: false },
    mutations: {
      onError: (e) => toast.error(e instanceof Error ? e.message : String(e)),
    },
  },
})

export const keys = {
  settings: ["settings"] as const,
  key: ["openrouter-key"] as const,
  catalog: ["catalog"] as const,
  system: ["system"] as const,
  localModels: ["local-models"] as const,
  jobs: ["jobs"] as const,
  job: (id: string) => ["jobs", id] as const,
  images: (params: Record<string, unknown>) => ["images", params] as const,
  image: (id: string) => ["image", id] as const,
  chats: ["chats"] as const,
  chat: (id: string) => ["chats", id] as const,
  tools: ["tools"] as const,
  spend: (chatId?: string) => ["spend", chatId ?? "all"] as const,
  localLlm: ["local-llm"] as const,
  claudeCli: ["claude-cli"] as const,
  preferences: (chatId?: string | null) => ["preferences", chatId ?? "global"] as const,
}

// ── Settings ───────────────────────────────────────────────────────

export const useSettings = () =>
  useQuery({ queryKey: keys.settings, queryFn: () => api.get<SettingsResponse>("/settings"), staleTime: Infinity })

export function useUpdateSettings() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (patch: Partial<Settings>) => api.put<SettingsResponse>("/settings", patch),
    onSuccess: (data, patch) => {
      qc.setQueryData(keys.settings, data)
      // Caps and batch sizes are shown on the local model cards too.
      if ("local_batch" in patch || "model_caps" in patch || "max_images_local" in patch) {
        qc.invalidateQueries({ queryKey: keys.localModels })
      }
    },
  })
}

export function useResetSettings() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: () => api.post<SettingsResponse>("/settings/reset"),
    onSuccess: (data) => qc.setQueryData(keys.settings, data),
  })
}

export function useSetSecret() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (body: { name: "OPENROUTER_KEY" | "HF_TOKEN"; value: string }) =>
      api.put<{ secrets: SettingsResponse["secrets"] }>("/secrets", body),
    onSuccess: (data, vars) => {
      qc.setQueryData<SettingsResponse>(keys.settings, (old) => (old ? { ...old, secrets: data.secrets } : old))
      if (vars.name === "OPENROUTER_KEY") {
        qc.invalidateQueries({ queryKey: keys.key })
        qc.invalidateQueries({ queryKey: keys.catalog })
      } else {
        qc.invalidateQueries({ queryKey: keys.localModels })
      }
    },
  })
}

export const useKeyStatus = () =>
  useQuery({ queryKey: keys.key, queryFn: () => api.get<KeyStatus>("/openrouter/key"), staleTime: 60_000 })

// ── OpenRouter catalog (the server caches it for 10 minutes) ──────

export const useCatalog = (enabled = true) =>
  useQuery({
    queryKey: keys.catalog,
    queryFn: () => api.get<Catalog>("/openrouter/models"),
    staleTime: 5 * 60_000,
    enabled,
    retry: false,
  })

export function useRefreshCatalog() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: () => api.get<Catalog>("/openrouter/models?force=true"),
    onSuccess: (data) => qc.setQueryData(keys.catalog, data),
  })
}

export const useSystem = () =>
  useQuery({ queryKey: keys.system, queryFn: () => api.get<SystemInfo>("/system"), staleTime: 30_000 })

export const useTools = () =>
  useQuery({ queryKey: keys.tools, queryFn: () => api.get<ToolOptions>("/tools"), staleTime: 60_000 })

export const useLocalLlm = (enabled: boolean) =>
  useQuery({
    queryKey: keys.localLlm,
    queryFn: () => api.get<{ reachable: boolean; models: string[]; base_url: string }>("/llm/local"),
    enabled,
  })

export const useClaudeCli = (enabled: boolean) =>
  useQuery({ queryKey: keys.claudeCli, queryFn: () => api.get<{ available: boolean }>("/llm/claude-cli"), enabled })

// ── Local models: poll while anything is in flight ─────────────────

const busy = (m: LocalModel) => m.state === "checking" || m.state === "downloading" || m.state === "runtime"

export const useLocalModels = () =>
  useQuery({
    queryKey: keys.localModels,
    queryFn: () => api.get<LocalModelsResponse>("/local-models"),
    refetchInterval: (q) => (q.state.data?.models.some(busy) || q.state.data?.runtime.running ? 1000 : false),
  })

function useLocalModelAction(path: (id: string) => string, method: "post" | "del") {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (id: string) => (method === "post" ? api.post<LocalModel>(path(id)) : api.del<LocalModel>(path(id))),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: keys.localModels })
      qc.invalidateQueries({ queryKey: keys.system })
    },
  })
}

export const useCalibrate = () => useStartJob((id: string) => api.post<Job>(`/local-models/${id}/calibrate`))

export const useSpend = (chatId?: string) =>
  useQuery({
    queryKey: keys.spend(chatId),
    queryFn: () => api.get<Spend>(`/spend${chatId ? `?chat_id=${chatId}` : ""}`),
  })

export const useInstallModel = () => useLocalModelAction((id) => `/local-models/${id}/install`, "post")
export const useCancelInstall = () => useLocalModelAction((id) => `/local-models/${id}/cancel`, "post")
export const useUninstallModel = () => useLocalModelAction((id) => `/local-models/${id}`, "del")
export const useCheckAccess = () => useLocalModelAction((id) => `/local-models/${id}/check-access`, "post")

export function useInstallRuntime() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: () => api.post<RuntimeStatus>("/local-stack/install"),
    onSuccess: () => qc.invalidateQueries({ queryKey: keys.localModels }),
  })
}

// ── Jobs ───────────────────────────────────────────────────────────

export const useJobs = () =>
  useQuery({
    queryKey: keys.jobs,
    queryFn: () => api.get<Job[]>("/jobs"),
    refetchInterval: (q) => (q.state.data?.some((j) => j.status === "running") ? 1000 : 5000),
  })

export function useJob(id: string | null | undefined) {
  const qc = useQueryClient()
  return useQuery({
    queryKey: keys.job(id ?? ""),
    queryFn: async () => {
      const prev = qc.getQueryData<Job>(keys.job(id ?? ""))
      const job = await api.get<Job>(`/jobs/${id}`)
      // New images show up in the gallery while the job runs.
      if (!prev || prev.image_ids.length !== job.image_ids.length || prev.status !== job.status) {
        qc.invalidateQueries({ queryKey: ["images"] })
      }
      if (job.status !== "running" || job.cost !== prev?.cost) {
        qc.invalidateQueries({ queryKey: keys.system })
        qc.invalidateQueries({ queryKey: ["spend"] })
        if (job.chat_id) qc.invalidateQueries({ queryKey: keys.chat(job.chat_id) })
      }
      if (job.kind === "calibrate" && job.status !== "running") qc.invalidateQueries({ queryKey: keys.localModels })
      return job
    },
    enabled: !!id,
    refetchInterval: (q) => (q.state.data?.status === "running" ? 900 : false),
  })
}

function useStartJob<T>(fn: (body: T) => Promise<Job>) {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: fn,
    onSuccess: (job) => {
      qc.setQueryData(keys.job(job.id), job)
      qc.invalidateQueries({ queryKey: keys.jobs })
    },
  })
}

export interface GenerateBody {
  prompts: Prompt[]
  models?: string[]
  images_per_prompt?: number
  references?: string[]
  chat_id?: string
}

export const useGenerate = () => useStartJob((b: GenerateBody) => api.post<Job>("/generate", b))

export const useQuickGenerate = () =>
  useStartJob(
    (b: {
      brief: string
      count?: number
      enhance: boolean
      models?: string[]
      images_per_prompt?: number
      references?: string[]
    }) => api.post<Job>("/generate/quick", b),
  )

export const useVariations = () =>
  useStartJob(
    ({ id, ...b }: { id: string; count: number; mode: "remix" | "reroll"; instruction?: string; model?: string }) =>
      api.post<Job>(`/images/${id}/variations`, b),
  )

export const useRegenerate = () =>
  useStartJob(
    ({ id, ...b }: { id: string; prompt: string; negative_prompt?: string; count: number; model?: string }) =>
      api.post<Job>(`/images/${id}/regenerate`, b),
  )

export function useCancelJob() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (id: string) => api.post<Job>(`/jobs/${id}/cancel`),
    onSuccess: (job) => {
      qc.setQueryData(keys.job(job.id), job)
      qc.invalidateQueries({ queryKey: keys.jobs })
    },
  })
}

// ── Library ────────────────────────────────────────────────────────

export interface ImageQuery {
  kind?: string
  favorite?: boolean
  q?: string
  chat_id?: string
  job_id?: string
  ids?: string[]
  limit?: number
}

export function useImages(params: ImageQuery, enabled = true) {
  const search = new URLSearchParams()
  if (params.kind) search.set("kind", params.kind)
  if (params.favorite) search.set("favorite", "true")
  if (params.q) search.set("q", params.q)
  if (params.chat_id) search.set("chat_id", params.chat_id)
  if (params.job_id) search.set("job_id", params.job_id)
  if (params.ids) search.set("ids", params.ids.join(","))
  search.set("limit", String(params.limit ?? 300))
  return useQuery({
    queryKey: keys.images(params as Record<string, unknown>),
    queryFn: () => api.get<ImageItem[]>(`/images?${search}`),
    enabled,
    placeholderData: (prev) => prev,
  })
}

export const useImage = (id: string | null | undefined) =>
  useQuery({ queryKey: keys.image(id ?? ""), queryFn: () => api.get<ImageDetail>(`/images/${id}`), enabled: !!id })

function useImageMutation<T, R>(fn: (vars: T) => Promise<R>) {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: fn,
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["images"] })
      qc.invalidateQueries({ queryKey: ["image"] })
      qc.invalidateQueries({ queryKey: keys.system })
    },
  })
}

export function useToggleFavorite() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: ({ id, favorite }: { id: string; favorite: boolean }) => api.patch<ImageItem>(`/images/${id}`, { favorite }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["images"] })
      qc.invalidateQueries({ queryKey: ["image"] })
      qc.invalidateQueries({ queryKey: ["preferences"] })
    },
  })
}

export const useSetPrimary = () =>
  useImageMutation(({ id, childId }: { id: string; childId: string | null }) =>
    api.put<ImageItem>(`/images/${id}/primary`, { child_id: childId }),
  )

export const useDeleteImage = () => useImageMutation((id: string) => api.del(`/images/${id}`))

export const useUpscale = () =>
  useImageMutation(({ id, method, scale }: { id: string; method: string; scale: number }) =>
    api.post<ImageItem>(`/images/${id}/upscale`, { method, scale }),
  )

export const useRemoveBackground = () =>
  useImageMutation(
    ({ id, ...body }: { id: string; method: "ai" | "color"; model?: string; tolerance?: number; erode_pixels?: number; alpha_matting?: boolean }) =>
      api.post<ImageItem>(`/images/${id}/remove-background`, body),
  )

// ── Chats ──────────────────────────────────────────────────────────

export const useChats = () => useQuery({ queryKey: keys.chats, queryFn: () => api.get<ChatSummary[]>("/chats") })

export const useChat = (id: string) => useQuery({ queryKey: keys.chat(id), queryFn: () => api.get<Chat>(`/chats/${id}`) })

export function useCreateChat() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: () => api.post<Chat>("/chats", {}),
    onSuccess: () => qc.invalidateQueries({ queryKey: keys.chats }),
  })
}

export function useDeleteChat() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (id: string) => api.del(`/chats/${id}`),
    onSuccess: () => qc.invalidateQueries({ queryKey: keys.chats }),
  })
}

// ── Learned preferences ────────────────────────────────────────────

export const usePreferences = (chatId?: string | null) =>
  useQuery({
    queryKey: keys.preferences(chatId),
    queryFn: () => api.get<PreferencesResponse>(`/preferences${chatId ? `?chat_id=${chatId}` : ""}`),
  })

export function useClearPreferences() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: ({ chatId, global }: { chatId?: string | null; global?: boolean }) =>
      api.del<PreferencesResponse>(global ? "/preferences?scope=global" : `/preferences?chat_id=${chatId}`),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["preferences"] }),
  })
}

/** Like, dislike or pick an image, or clear the reaction. The chat UI can call this. */
export function useFeedback() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (b: { kind: "like" | "dislike" | "pick" | "clear"; image_id: string; chat_id?: string | null }) =>
      api.post("/preferences/events", b),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["preferences"] }),
  })
}
