import { InfoIcon } from "lucide-react"

import { Badge } from "@/components/ui/badge"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip"
import type { ImageSort, Rankings } from "@/lib/api"
import { formatUsd } from "@/lib/format"
import { ordinal, SORTS, useRankings, valueHint, type ModelEntry } from "@/lib/rankings"
import { useSettings, useUpdateSettings } from "@/lib/queries"
import { cn } from "@/lib/utils"

/** Rank, price per image, best value, local and unranked badges for one model. */
export function RankBadges({
  m,
  r,
  className,
  hidePrice = false,
}: {
  m: ModelEntry
  r: Rankings | undefined
  className?: string
  hidePrice?: boolean
}) {
  const k = r?.models[m.id]
  const category = k?.category === "commercial_design" ? "commercial design" : "overall"
  return (
    <span className={cn("flex shrink-0 flex-wrap items-center gap-1", className)}>
      {k?.ranked ? (
        <Badge
          variant="secondary"
          className="tabular-nums"
          title={`${ordinal(k.rank!)} of the ranked models here. LMArena ${category}: ${Math.round(k.rating!)} (${Math.round(k.ci![0])} to ${Math.round(k.ci![1])}), #${k.arena_rank} of the whole leaderboard, ${k.votes?.toLocaleString()} votes, as "${k.leaderboard_name}".`}
        >
          #{k.rank}
        </Badge>
      ) : (
        r && (
          <Badge variant="outline" className="text-muted-foreground" title="Not on the LMArena text-to-image leaderboard, or not matched to it.">
            unranked
          </Badge>
        )
      )}
      {k?.pareto && (
        <Badge variant="outline" className="border-primary text-primary" title="On the Pareto frontier: no other model here is both better and cheaper.">
          best value
        </Badge>
      )}
      {hidePrice ? null : m.local ? (
        <Badge variant="outline" title="Runs on this machine. Free.">
          local
        </Badge>
      ) : (
        <span
          className="text-[11px] text-muted-foreground tabular-nums"
          title={m.priceUnit === "token" ? "Priced per token; estimated for one 1024px image" : "Per 1024px image"}
        >
          {m.price == null ? "price n/a" : m.price === 0 ? "free" : `${m.priceUnit === "token" ? "≈" : ""}${formatUsd(m.price)}/img`}
        </span>
      )}
    </span>
  )
}

/** The sort for image model lists. Saved in settings, so every picker and page remembers it. */
export function SortSelect({ className, extra = false }: { className?: string; extra?: boolean }) {
  const settings = useSettings()
  const update = useUpdateSettings()
  const rankings = useRankings()
  const value = settings.data?.settings.image_sort ?? "quality"
  const options = SORTS.filter((s) => s.value !== "popular" || rankings.data?.popularity)
  const current = options.find((s) => s.value === value)
  const hint = value === "value" ? valueHint(rankings.data) : current?.hint
  return (
    <span className={cn("flex items-center gap-1", className)}>
      <Select value={value} onValueChange={(v) => update.mutate({ image_sort: v as ImageSort })}>
        <SelectTrigger size="sm" className="w-auto min-w-36" aria-label="Sort models">
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          {options.map((s) => (
            <SelectItem key={s.value} value={s.value}>
              {s.label}
            </SelectItem>
          ))}
          {extra && (
            <>
              <SelectItem value="newest">Newest first</SelectItem>
              <SelectItem value="name">Name</SelectItem>
            </>
          )}
        </SelectContent>
      </Select>
      {hint && (
        <Tooltip>
          <TooltipTrigger asChild>
            <button type="button" aria-label="How this sort works" className="text-muted-foreground hover:text-foreground">
              <InfoIcon className="size-3.5" />
            </button>
          </TooltipTrigger>
          <TooltipContent className="max-w-80">{hint}</TooltipContent>
        </Tooltip>
      )}
    </span>
  )
}

/** Where the scores come from, with links, as the licenses ask. */
export function RankingSource({ className }: { className?: string }) {
  const r = useRankings().data
  if (!r) return null
  return (
    <p className={cn("text-[11px] text-muted-foreground", className)}>
      Quality:{" "}
      <a href={r.source.url} target="_blank" rel="noreferrer" className="underline">
        {r.source.name}
      </a>{" "}
      (commercial design scores, published {r.source.published}
      {r.source.bundled && ", bundled copy"}, {r.source.license}).
      {r.popularity && (
        <>
          {" "}
          Popularity:{" "}
          <a href={r.popularity.url} target="_blank" rel="noreferrer" className="underline">
            OpenRouter usage
          </a>
          , as of {r.popularity.as_of.slice(0, 10)} ({r.popularity.license}).
        </>
      )}
      {r.notes.length > 0 && ` ${r.notes.join(" ")}`}
    </p>
  )
}
