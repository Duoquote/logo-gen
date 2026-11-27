import { useQuery } from "@tanstack/react-query"

import { api, type ImageSort, type ModelRank, type Rankings } from "@/lib/api"

export const useRankings = () =>
  useQuery({
    queryKey: ["rankings"],
    queryFn: () => api.get<Rankings>("/models/rankings"),
    staleTime: 60 * 60 * 1000,
  })

/** One image model as the pickers see it, cloud or local. */
export interface ModelEntry {
  id: string
  name: string
  price: number | null // USD for one 1024px image; 0 for free and local
  priceUnit: string // image | megapixel | token (token prices are estimates) | local
  local: boolean
  created?: number | null
}

export const SORTS: { value: ImageSort; label: string; hint: string }[] = [
  {
    value: "quality",
    label: "Best quality",
    hint: "By LMArena text-to-image score (the commercial design category where there is one, since it's closest to logo work). Unranked models go last.",
  },
  { value: "value", label: "Best value", hint: "" }, // filled in by valueHint()
  { value: "cheapest", label: "Cheapest", hint: "By estimated price for one 1024px image. Free and local first, unknown prices last." },
  {
    value: "popular",
    label: "Popular on OpenRouter",
    hint: "By image tokens generated on OpenRouter over the last 30 days. This is how much a model is used, not how good it is.",
  },
  { value: "local", label: "Local / free", hint: "Only models that cost nothing: installed local models and free OpenRouter models." },
]

export function valueHint(r?: Rankings): string {
  const k = r?.formula.elo_per_doubling ?? 50
  return (
    `Quality for the money. First the Pareto frontier (models nothing else beats on both score and price), then the rest. ` +
    `Within each, by score minus ${k} points for every doubling of the price per image above $0.01, ` +
    `so a model twice as expensive has to score ${k} points higher to rank the same. Free counts as $0.001. ` +
    `Unranked models and unknown prices go last.`
  )
}

const last = Number.POSITIVE_INFINITY

/** Sort (and for "local", filter) models. Unranked models and unknown prices never break the order: they go last, by name. */
export function sortModels(list: ModelEntry[], sort: ImageSort, r: Rankings | undefined): ModelEntry[] {
  const rank = (id: string): ModelRank | undefined => r?.models[id]
  const byName = (a: ModelEntry, b: ModelEntry) => a.name.localeCompare(b.name)
  const num = (v: number | null | undefined, desc = false) => (v == null || Number.isNaN(v) ? last : desc ? -v : v)
  let out = [...list]
  if (sort === "local") out = out.filter((m) => m.local || m.price === 0)
  const cmp: Record<ImageSort, (a: ModelEntry, b: ModelEntry) => number> = {
    quality: (a, b) => num(rank(a.id)?.rating, true) - num(rank(b.id)?.rating, true) || byName(a, b),
    value: (a, b) =>
      Number(!rank(a.id)?.pareto) - Number(!rank(b.id)?.pareto) ||
      num(rank(a.id)?.value, true) - num(rank(b.id)?.value, true) ||
      byName(a, b),
    cheapest: (a, b) => num(a.price) - num(b.price) || num(rank(a.id)?.rating, true) - num(rank(b.id)?.rating, true) || byName(a, b),
    popular: (a, b) => num(rank(a.id)?.popularity_rank) - num(rank(b.id)?.popularity_rank) || byName(a, b),
    local: (a, b) => Number(!a.local) - Number(!b.local) || num(rank(a.id)?.rating, true) - num(rank(b.id)?.rating, true) || byName(a, b),
    newest: (a, b) => (b.created ?? 0) - (a.created ?? 0) || byName(a, b),
    name: byName,
  }
  // Infinity - Infinity is NaN; treat ties between two unknowns as equal.
  return out.sort((a, b) => cmp[sort](a, b) || 0)
}

/** Best quality, best value and the user's recent models, in that order, without repeats. */
export function topPicks(list: ModelEntry[], r: Rankings | undefined): { entry: ModelEntry; why: string }[] {
  if (!list.length) return []
  const picks: { entry: ModelEntry; why: string }[] = []
  const add = (e: ModelEntry | undefined, why: string) => {
    if (e && !picks.some((p) => p.entry.id === e.id)) picks.push({ entry: e, why })
  }
  if (r) {
    const best = sortModels(list, "quality", r)[0]
    if (r.models[best?.id]?.ranked) add(best, "Best quality")
    const value = sortModels(list, "value", r)[0]
    if (r.models[value?.id]?.pareto) {
      if (picks[0]?.entry.id === value.id) picks[0].why = "Best quality and best value"
      else add(value, "Best value")
    }
    for (const id of r.recent) add(list.find((m) => m.id === id), "Recently used")
  }
  return picks.slice(0, 5)
}

export const ordinal = (n: number) => {
  const s = ["th", "st", "nd", "rd"]
  const v = n % 100
  return n + (s[(v - 20) % 10] || s[v] || s[0])
}
