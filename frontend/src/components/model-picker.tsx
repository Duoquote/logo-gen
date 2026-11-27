import { useMemo, useState } from "react"
import { CheckIcon, ChevronsUpDownIcon, CpuIcon } from "lucide-react"

import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import {
  Command,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
} from "@/components/ui/command"
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover"
import { formatUsd, shortModel } from "@/lib/format"
import { RankBadges, RankingSource, SortSelect } from "@/components/model-rank"
import { useCatalog, useLocalModels, useSettings } from "@/lib/queries"
import { SORTS, sortModels, topPicks, useRankings, type ModelEntry } from "@/lib/rankings"
import { cn } from "@/lib/utils"

export function imagePriceLabel(price: number | null, unit: string): string {
  if (price === null) return ""
  if (price === 0) return "free"
  return `${unit === "token" ? "≈" : ""}${formatUsd(price)}/img`
}

/** Every image model we can use right now: installed local models and the OpenRouter catalog. */
export function useImageModelEntries(): { entries: ModelEntry[]; loading: boolean; error: Error | null; hasKey: boolean } {
  const settings = useSettings()
  const hasKey = settings.data?.secrets.OPENROUTER_KEY.set ?? false
  const catalog = useCatalog(hasKey)
  const local = useLocalModels()
  const entries = useMemo(() => {
    const installed: ModelEntry[] = (local.data?.models ?? [])
      .filter((m) => m.installed && m.runs_here !== "no")
      .map((m) => ({ id: m.ref, name: m.name, price: 0, priceUnit: "local", local: true }))
    const cloud: ModelEntry[] = (catalog.data?.image ?? []).map((m) => ({
      id: m.id,
      name: m.name,
      price: m.price_per_image,
      priceUnit: m.price_unit,
      local: false,
      created: m.created,
    }))
    return [...installed, ...cloud]
  }, [local.data, catalog.data])
  return { entries, loading: catalog.isLoading, error: catalog.error as Error | null, hasKey }
}

/** Pick one or more image models, sorted by quality, value, price or popularity. */
export function ImageModelPicker({
  value,
  onChange,
  className,
  compact = false,
}: {
  value: string[]
  onChange: (v: string[]) => void
  className?: string
  compact?: boolean
}) {
  const [open, setOpen] = useState(false)
  const settings = useSettings()
  const rankings = useRankings()
  const { entries, loading, error, hasKey } = useImageModelEntries()
  const sort = settings.data?.settings.image_sort ?? "quality"
  const sorted = useMemo(() => sortModels(entries, sort, rankings.data), [entries, sort, rankings.data])
  const picks = useMemo(() => topPicks(entries, rankings.data), [entries, rankings.data])
  const toggle = (id: string) => onChange(value.includes(id) ? value.filter((v) => v !== id) : [...value, id])

  const item = (m: ModelEntry, key: string, why?: string) => (
    <CommandItem key={key} value={`${key} ${m.id} ${m.name}`} onSelect={() => toggle(m.id)} className="gap-2">
      <CheckIcon className={cn(value.includes(m.id) ? "opacity-100" : "opacity-0")} />
      <span className="flex min-w-0 flex-1 flex-col">
        <span className="truncate">
          {m.local && <CpuIcon className="mr-1 inline size-3" />}
          {m.name}
        </span>
        {why && <span className="text-[10px] text-muted-foreground">{why}</span>}
      </span>
      <RankBadges m={m} r={rankings.data} className="justify-end" />
    </CommandItem>
  )

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <Button
          variant="outline"
          size={compact ? "sm" : "default"}
          aria-label="Image models"
          className={cn("h-auto min-h-8 justify-between py-1", className)}
        >
          <span className="flex min-w-0 flex-wrap gap-1">
            {value.length === 0 && <span className="text-muted-foreground">Choose image models</span>}
            {value.map((v) => (
              <Badge key={v} variant="secondary">
                {v.startsWith("local/") && <CpuIcon />}
                {shortModel(v)}
              </Badge>
            ))}
          </span>
          <ChevronsUpDownIcon className="opacity-50" />
        </Button>
      </PopoverTrigger>
      <PopoverContent className="w-[min(480px,calc(100vw-1rem))] p-0" align="start">
        <Command>
          <div className="flex items-center gap-2 border-b px-2 py-1.5">
            <span className="text-xs text-muted-foreground">Sort</span>
            <SortSelect />
          </div>
          <CommandInput placeholder="Search image models" />
          <CommandList className="max-h-[min(24rem,60svh)]">
            <CommandEmpty>
              {sort === "local" ? "No free or local models. Install one on the Models page." : hasKey ? (loading ? "Loading models" : "No match") : "Add an OpenRouter key for cloud models"}
            </CommandEmpty>
            {error && <p className="p-3 text-xs text-destructive">{error.message}</p>}
            {picks.length > 0 && sort !== "local" && (
              <CommandGroup heading="Top picks">{picks.map((p) => item(p.entry, `top:${p.entry.id}`, p.why))}</CommandGroup>
            )}
            {sorted.length > 0 && (
              <CommandGroup heading={SORTS.find((s) => s.value === sort)?.label ?? "All models"}>
                {sorted.map((m) => item(m, m.id))}
              </CommandGroup>
            )}
          </CommandList>
          <RankingSource className="border-t px-2 py-1.5" />
        </Command>
      </PopoverContent>
    </Popover>
  )
}
