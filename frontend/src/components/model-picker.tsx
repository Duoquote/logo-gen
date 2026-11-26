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
import { useCatalog, useLocalModels, useSettings } from "@/lib/queries"
import { cn } from "@/lib/utils"

export function imagePriceLabel(price: number | null, unit: string): string {
  if (price === null) return ""
  if (price === 0) return "free"
  return `${unit === "token" ? "≈" : ""}${formatUsd(price)}/img`
}

/** Pick one or more image models: installed local models and OpenRouter image models. */
export function ImageModelPicker({
  value,
  onChange,
  className,
}: {
  value: string[]
  onChange: (v: string[]) => void
  className?: string
}) {
  const [open, setOpen] = useState(false)
  const settings = useSettings()
  const hasKey = settings.data?.secrets.OPENROUTER_KEY.set ?? false
  const catalog = useCatalog(hasKey)
  const local = useLocalModels()

  const installed = (local.data?.models ?? []).filter((m) => m.installed && m.runs_here !== "no")
  const cloud = useMemo(
    () =>
      [...(catalog.data?.image ?? [])].sort(
        (a, b) => (a.price_per_image ?? 99) - (b.price_per_image ?? 99) || a.name.localeCompare(b.name),
      ),
    [catalog.data],
  )

  const toggle = (id: string) => onChange(value.includes(id) ? value.filter((v) => v !== id) : [...value, id])

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <Button variant="outline" className={cn("h-auto min-h-8 justify-between py-1", className)}>
          <span className="flex flex-wrap gap-1">
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
      <PopoverContent className="w-[440px] p-0" align="start">
        <Command>
          <CommandInput placeholder="Search image models" />
          <CommandList className="max-h-80">
            <CommandEmpty>
              {hasKey ? (catalog.isLoading ? "Loading models" : "No match") : "Add an OpenRouter key for cloud models"}
            </CommandEmpty>
            {installed.length > 0 && (
              <CommandGroup heading="Local (free, on this machine)">
                {installed.map((m) => (
                  <CommandItem key={m.ref} value={`${m.ref} ${m.name}`} onSelect={() => toggle(m.ref)}>
                    <CheckIcon className={cn(value.includes(m.ref) ? "opacity-100" : "opacity-0")} />
                    <span className="flex-1">{m.name}</span>
                    <span className="text-xs text-muted-foreground">{m.resolution}px</span>
                  </CommandItem>
                ))}
              </CommandGroup>
            )}
            {catalog.error && (
              <p className="p-3 text-xs text-destructive">{(catalog.error as Error).message}</p>
            )}
            {cloud.length > 0 && (
              <CommandGroup heading="OpenRouter">
                {cloud.map((m) => (
                  <CommandItem key={m.id} value={`${m.id} ${m.name}`} onSelect={() => toggle(m.id)}>
                    <CheckIcon className={cn(value.includes(m.id) ? "opacity-100" : "opacity-0")} />
                    <span className="min-w-0 flex-1 truncate">
                      {m.name}
                      {m.vector && <span className="ml-1 text-xs text-muted-foreground">SVG</span>}
                    </span>
                    <span className="text-xs text-muted-foreground tabular-nums">
                      {imagePriceLabel(m.price_per_image, m.price_unit)}
                    </span>
                  </CommandItem>
                ))}
              </CommandGroup>
            )}
          </CommandList>
        </Command>
      </PopoverContent>
    </Popover>
  )
}
