import { useMemo, useState } from "react"
import { CheckIcon, Loader2Icon, PlusIcon, RefreshCwIcon, SearchIcon } from "lucide-react"

import { imagePriceLabel } from "@/components/model-picker"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { InputGroup, InputGroupAddon, InputGroupInput } from "@/components/ui/input-group"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { Toggle } from "@/components/ui/toggle"
import type { ChatModel, ImageModel } from "@/lib/api"
import { formatContext, perMillion, timeAgo } from "@/lib/format"
import { useCatalog, useRefreshCatalog, useSettings, useUpdateSettings } from "@/lib/queries"

type Sort = "price" | "name" | "newest" | "context"

function useFilters() {
  const [q, setQ] = useState("")
  const [provider, setProvider] = useState("all")
  const [freeOnly, setFreeOnly] = useState(false)
  const [sort, setSort] = useState<Sort>("price")
  return { q, setQ, provider, setProvider, freeOnly, setFreeOnly, sort, setSort }
}

function matches(m: { id: string; name: string; provider: string; is_free: boolean }, f: ReturnType<typeof useFilters>) {
  const q = f.q.toLowerCase()
  if (q && !m.id.toLowerCase().includes(q) && !m.name.toLowerCase().includes(q)) return false
  if (f.provider !== "all" && m.provider !== f.provider) return false
  if (f.freeOnly && !m.is_free) return false
  return true
}

function Toolbar({
  f,
  providers,
  count,
  total,
  extra,
}: {
  f: ReturnType<typeof useFilters>
  providers: string[]
  count: number
  total: number
  extra?: React.ReactNode
}) {
  const catalog = useCatalog()
  const refresh = useRefreshCatalog()
  return (
    <div className="flex flex-wrap items-center gap-2">
      <InputGroup className="w-64">
        <InputGroupAddon>
          <SearchIcon />
        </InputGroupAddon>
        <InputGroupInput placeholder="Search by name or id" value={f.q} onChange={(e) => f.setQ(e.target.value)} />
      </InputGroup>
      <Select value={f.provider} onValueChange={f.setProvider}>
        <SelectTrigger className="w-44">
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value="all">All providers</SelectItem>
          {providers.map((p) => (
            <SelectItem key={p} value={p}>
              {p}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
      <Select value={f.sort} onValueChange={(v) => f.setSort(v as Sort)}>
        <SelectTrigger className="w-36">
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value="price">Cheapest first</SelectItem>
          <SelectItem value="newest">Newest first</SelectItem>
          <SelectItem value="name">Name</SelectItem>
          <SelectItem value="context">Largest context</SelectItem>
        </SelectContent>
      </Select>
      <Toggle variant="outline" size="sm" pressed={f.freeOnly} onPressedChange={f.setFreeOnly}>
        Free only
      </Toggle>
      {extra}
      <span className="ml-auto text-xs text-muted-foreground">
        {count} of {total}
        {catalog.data && ` · fetched ${timeAgo(catalog.data.fetched_at)}`}
      </span>
      <Button variant="ghost" size="icon-sm" aria-label="Refresh model list" onClick={() => refresh.mutate()} disabled={refresh.isPending}>
        {refresh.isPending ? <Loader2Icon className="animate-spin" /> : <RefreshCwIcon />}
      </Button>
    </div>
  )
}

function sortBy<T extends { name: string; created: number | null }>(list: T[], sort: Sort, price: (m: T) => number, ctx: (m: T) => number) {
  const out = [...list]
  if (sort === "price") out.sort((a, b) => price(a) - price(b) || a.name.localeCompare(b.name))
  if (sort === "name") out.sort((a, b) => a.name.localeCompare(b.name))
  if (sort === "newest") out.sort((a, b) => (b.created ?? 0) - (a.created ?? 0))
  if (sort === "context") out.sort((a, b) => ctx(b) - ctx(a))
  return out
}

function CatalogState({ children }: { children: React.ReactNode }) {
  const settings = useSettings()
  const catalog = useCatalog(settings.data?.secrets.OPENROUTER_KEY.set ?? false)
  if (settings.data && !settings.data.secrets.OPENROUTER_KEY.set) {
    return (
      <Alert>
        <AlertTitle>No OpenRouter key</AlertTitle>
        <AlertDescription>Add your key in Settings to list and use OpenRouter models.</AlertDescription>
      </Alert>
    )
  }
  if (catalog.isLoading) return <Loader2Icon className="animate-spin text-muted-foreground" />
  if (catalog.error) {
    return (
      <Alert variant="destructive">
        <AlertTitle>Could not load models</AlertTitle>
        <AlertDescription>{(catalog.error as Error).message}</AlertDescription>
      </Alert>
    )
  }
  return <>{children}</>
}

export function ChatModelsTable() {
  return (
    <CatalogState>
      <ChatModelsInner />
    </CatalogState>
  )
}

function ChatModelsInner() {
  const catalog = useCatalog()
  const settings = useSettings()
  const update = useUpdateSettings()
  const f = useFilters()
  const [vision, setVision] = useState(false)
  const all = useMemo(() => catalog.data?.chat ?? [], [catalog.data])
  const providers = useMemo(() => [...new Set(all.map((m) => m.provider))].sort(), [all])
  const rows = sortBy(
    all.filter((m) => matches(m, f) && (!vision || m.input_modalities.includes("image"))),
    f.sort,
    (m) => (m.prompt_price ?? 0) + (m.completion_price ?? 0),
    (m) => m.context_length ?? 0,
  )
  const current = settings.data?.settings.chat_model

  return (
    <div className="flex flex-col gap-3">
      <Toolbar
        f={f}
        providers={providers}
        count={rows.length}
        total={all.length}
        extra={
          <Toggle variant="outline" size="sm" pressed={vision} onPressedChange={setVision}>
            Sees images
          </Toggle>
        }
      />
      <ModelTable
        head={["Model", "Provider", "Context", "Input / 1M", "Output / 1M", "Modalities", ""]}
        rows={rows.slice(0, 300).map((m: ChatModel) => [
          <ModelName key="n" name={m.name} id={m.id} />,
          m.provider,
          formatContext(m.context_length),
          perMillion(m.prompt_price),
          perMillion(m.completion_price),
          <Modalities key="m" input={m.input_modalities} output={m.output_modalities} />,
          current === m.id ? (
            <Badge key="u">
              <CheckIcon /> Chat model
            </Badge>
          ) : (
            <Button key="u" size="xs" variant="outline" onClick={() => update.mutate({ chat_model: m.id, llm_backend: "openrouter" })}>
              Use for chat
            </Button>
          ),
        ])}
      />
    </div>
  )
}

export function ImageModelsTable() {
  return (
    <CatalogState>
      <ImageModelsInner />
    </CatalogState>
  )
}

function ImageModelsInner() {
  const catalog = useCatalog()
  const settings = useSettings()
  const update = useUpdateSettings()
  const f = useFilters()
  const [feature, setFeature] = useState("any")
  const all = useMemo(() => catalog.data?.image ?? [], [catalog.data])
  const providers = useMemo(() => [...new Set(all.map((m) => m.provider))].sort(), [all])
  const rows = sortBy(
    all.filter(
      (m) =>
        matches(m, f) &&
        (feature === "any" ||
          (feature === "svg" && m.vector) ||
          (feature === "transparent" && m.supports_transparent) ||
          (feature === "references" && m.supports_references) ||
          (feature === "seed" && m.supports_seed)),
    ),
    f.sort,
    (m) => m.price_per_image ?? 99,
    () => 0,
  )
  const selected = settings.data?.settings.image_models ?? []
  const toggle = (id: string) =>
    update.mutate({ image_models: selected.includes(id) ? selected.filter((x) => x !== id) : [...selected, id] })

  return (
    <div className="flex flex-col gap-3">
      <Toolbar
        f={f}
        providers={providers}
        count={rows.length}
        total={all.length}
        extra={
          <Select value={feature} onValueChange={setFeature}>
            <SelectTrigger className="w-44">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="any">Any feature</SelectItem>
              <SelectItem value="svg">SVG output</SelectItem>
              <SelectItem value="transparent">Transparent background</SelectItem>
              <SelectItem value="references">Reference images</SelectItem>
              <SelectItem value="seed">Seed control</SelectItem>
            </SelectContent>
          </Select>
        }
      />
      <ModelTable
        head={["Model", "Provider", "Price", "Features", "Modalities", ""]}
        rows={rows.map((m: ImageModel) => [
          <ModelName key="n" name={m.name} id={m.id} />,
          m.provider,
          <span key="p" title={m.price_unit === "token" ? "Priced per output token; estimate for a 1024 px image" : undefined}>
            {imagePriceLabel(m.price_per_image, m.price_unit) || "n/a"}
          </span>,
          <span key="f" className="flex flex-wrap gap-1">
            {m.vector && <Badge variant="outline">SVG</Badge>}
            {m.supports_transparent && <Badge variant="outline">transparent</Badge>}
            {m.supports_references && <Badge variant="outline">references</Badge>}
            {m.supports_seed && <Badge variant="outline">seed</Badge>}
            {m.max_n > 1 && <Badge variant="outline">up to {m.max_n}/call</Badge>}
          </span>,
          <Modalities key="m" input={m.input_modalities} output={m.output_modalities} />,
          <Button key="u" size="xs" variant={selected.includes(m.id) ? "default" : "outline"} onClick={() => toggle(m.id)}>
            {selected.includes(m.id) ? <CheckIcon /> : <PlusIcon />}
            {selected.includes(m.id) ? "Selected" : "Use"}
          </Button>,
        ])}
      />
    </div>
  )
}

function ModelName({ name, id }: { name: string; id: string }) {
  return (
    <div className="flex min-w-0 flex-col">
      <span className="truncate font-medium">{name}</span>
      <span className="truncate font-mono text-[10px] text-muted-foreground">{id}</span>
    </div>
  )
}

function Modalities({ input, output }: { input: string[]; output: string[] }) {
  return (
    <span className="text-[11px] text-muted-foreground">
      {input.join(", ")} → {output.join(", ")}
    </span>
  )
}

function ModelTable({ head, rows }: { head: string[]; rows: React.ReactNode[][] }) {
  return (
    <div className="border">
      <Table>
        <TableHeader>
          <TableRow>
            {head.map((h, i) => (
              <TableHead key={i}>{h}</TableHead>
            ))}
          </TableRow>
        </TableHeader>
        <TableBody>
          {rows.length === 0 && (
            <TableRow>
              <TableCell colSpan={head.length} className="text-center text-muted-foreground">
                No models match.
              </TableCell>
            </TableRow>
          )}
          {rows.map((r, i) => (
            <TableRow key={i}>
              {r.map((c, j) => (
                <TableCell key={j} className={j === 0 ? "max-w-72" : undefined}>
                  {c}
                </TableCell>
              ))}
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  )
}
