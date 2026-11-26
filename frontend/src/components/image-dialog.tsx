import { useState } from "react"
import {
  CopyIcon,
  DownloadIcon,
  EraserIcon,
  HeartIcon,
  Loader2Icon,
  Maximize2Icon,
  SparklesIcon,
  StarIcon,
  Trash2Icon,
} from "lucide-react"
import { toast } from "sonner"

import { JobProgress } from "@/components/job-progress"
import { Button } from "@/components/ui/button"
import { Dialog, DialogContent, DialogDescription, DialogTitle } from "@/components/ui/dialog"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Separator } from "@/components/ui/separator"
import { Slider } from "@/components/ui/slider"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group"
import type { ImageDetail, ImageItem } from "@/lib/api"
import { formatUsd, shortModel } from "@/lib/format"
import {
  useDeleteImage,
  useImage,
  useJob,
  useRemoveBackground,
  useSetPrimary,
  useToggleFavorite,
  useTools,
  useUpscale,
  useVariations,
} from "@/lib/queries"
import { cn } from "@/lib/utils"

export function ImageDialog({
  imageId,
  onOpenChange,
  onSelect,
}: {
  imageId: string | null
  onOpenChange: (open: boolean) => void
  onSelect: (id: string) => void
}) {
  const { data: img, isLoading } = useImage(imageId)
  return (
    <Dialog open={!!imageId} onOpenChange={onOpenChange}>
      <DialogContent className="max-h-[92svh] overflow-y-auto sm:max-w-5xl">
        {isLoading || !img ? (
          <div className="flex h-96 items-center justify-center">
            <Loader2Icon className="animate-spin text-muted-foreground" />
          </div>
        ) : (
          <ImageDetailView img={img} onSelect={onSelect} onDeleted={() => onOpenChange(false)} />
        )}
      </DialogContent>
    </Dialog>
  )
}

function versionLabel(c: ImageItem): string {
  if (c.kind === "upscaled") return `Upscale: ${c.note || "upscaled"}${c.width ? ` (${c.width} px)` : ""}`
  if (c.kind === "cleaned") return `Background: ${c.note || "removed"}`
  return `Variation: ${c.concept || shortModel(c.model)}`
}

function ImageDetailView({
  img,
  onSelect,
  onDeleted,
}: {
  img: ImageDetail
  onSelect: (id: string) => void
  onDeleted: () => void
}) {
  const fav = useToggleFavorite()
  const del = useDeleteImage()
  const primary = useSetPrimary()
  const [compareId, setCompareId] = useState<string | null>(null)
  const [mode, setMode] = useState<"side" | "toggle">("side")
  const [showCandidate, setShowCandidate] = useState(true)
  const candidate = img.children.find((c) => c.id === compareId) ?? null

  const groups: [string, ImageItem[]][] = [
    ["Background removed", img.children.filter((c) => c.kind === "cleaned")],
    ["Upscaled", img.children.filter((c) => c.kind === "upscaled")],
    ["Variations", img.children.filter((c) => c.kind === "generated")],
  ]

  const view = (src: ImageItem, label: string) => (
    <figure className="flex min-w-0 flex-1 flex-col gap-1">
      <img
        src={src.url}
        alt={label}
        className={cn("aspect-square w-full border object-contain", src.kind === "cleaned" && "checker")}
      />
      <figcaption className="truncate text-xs text-muted-foreground">{label}</figcaption>
    </figure>
  )

  return (
    <div className="grid gap-6 md:grid-cols-[minmax(0,1fr)_320px]">
      <div className="flex min-w-0 flex-col gap-3">
        {candidate ? (
          <>
            <div className="flex flex-wrap items-center gap-2">
              <ToggleGroup
                type="single"
                variant="outline"
                size="sm"
                value={mode}
                onValueChange={(v) => v && setMode(v as "side" | "toggle")}
              >
                <ToggleGroupItem value="side">Side by side</ToggleGroupItem>
                <ToggleGroupItem value="toggle">Toggle</ToggleGroupItem>
              </ToggleGroup>
              {mode === "toggle" && (
                <Button size="sm" variant="outline" onClick={() => setShowCandidate((v) => !v)}>
                  Showing the {showCandidate ? "candidate" : "original"}. Switch
                </Button>
              )}
              <Button size="sm" variant="ghost" className="ml-auto" onClick={() => setCompareId(null)}>
                Close comparison
              </Button>
            </div>
            {mode === "side" ? (
              <div className="flex gap-3">
                {view(img, "Original")}
                {view(candidate, versionLabel(candidate))}
              </div>
            ) : showCandidate ? (
              view(candidate, versionLabel(candidate))
            ) : (
              view(img, "Original")
            )}
            <div className="flex flex-wrap items-center gap-2">
              {img.primary_id === candidate.id ? (
                <Button size="sm" variant="outline" onClick={() => primary.mutate({ id: img.id, childId: null })}>
                  <StarIcon className="fill-current text-primary" />
                  Primary version (click to unset)
                </Button>
              ) : (
                <Button size="sm" onClick={() => primary.mutate({ id: img.id, childId: candidate.id })}>
                  <StarIcon />
                  Use this version
                </Button>
              )}
              <Button asChild size="sm" variant="outline">
                <a href={`${candidate.url}?download=1`} download>
                  <DownloadIcon />
                  Download
                </a>
              </Button>
              <Button size="sm" variant="ghost" onClick={() => onSelect(candidate.id)}>
                Open
              </Button>
              <Button
                size="icon-sm"
                variant="ghost"
                aria-label="Delete this version"
                onClick={() => del.mutate(candidate.id, { onSuccess: () => setCompareId(null) })}
              >
                <Trash2Icon />
              </Button>
            </div>
          </>
        ) : (
          view(img, img.parent ? `Version of ${img.parent.concept || "an image"}${img.note ? ` (${img.note})` : ""}` : "Original")
        )}

        {img.parent && (
          <Button size="sm" variant="ghost" className="self-start" onClick={() => onSelect(img.parent!.id)}>
            Back to the original
          </Button>
        )}

        {groups.map(([title, items]) =>
          items.length ? (
            <div key={title} className="flex flex-col gap-1.5">
              <span className="text-xs text-muted-foreground">
                {title} ({items.length}){title !== "Variations" && ", click one to compare"}
              </span>
              <div className="flex flex-wrap gap-2">
                {items.map((c) => (
                  <button
                    key={c.id}
                    type="button"
                    onClick={() => (c.kind === "generated" ? onSelect(c.id) : setCompareId(c.id))}
                    className={cn("relative border hover:border-primary", compareId === c.id && "border-primary")}
                    title={versionLabel(c)}
                  >
                    <img src={c.url} alt="" className={cn("size-16 object-contain", c.kind === "cleaned" && "checker")} />
                    {img.primary_id === c.id && (
                      <StarIcon className="absolute top-0.5 right-0.5 size-3 fill-current text-primary" />
                    )}
                    <span className="block max-w-16 truncate px-0.5 text-[9px] text-muted-foreground">
                      {c.kind === "generated" ? shortModel(c.model) : c.note}
                    </span>
                  </button>
                ))}
              </div>
            </div>
          ) : null,
        )}
      </div>

      <div className="flex min-w-0 flex-col gap-4">
        <div>
          <DialogTitle>{img.concept || "Untitled"}</DialogTitle>
          <DialogDescription className="mt-1">
            {shortModel(img.model)}
            {img.width ? ` · ${img.width}×${img.height}` : ""}
            {img.model.startsWith("local/") ? " · local, $0" : img.cost ? ` · ${formatUsd(img.cost, 4)}` : ""}
            {img.note ? ` · ${img.note}` : ""}
          </DialogDescription>
        </div>

        <div className="flex flex-wrap gap-2">
          <Button asChild size="sm">
            <a href={`${(img.primary ?? img).url}?download=1`} download>
              <DownloadIcon />
              Download {img.vector ? "SVG" : "PNG"}
              {img.primary && " (primary version)"}
            </a>
          </Button>
          <Button size="sm" variant="outline" onClick={() => fav.mutate({ id: img.id, favorite: !img.favorite })}>
            <HeartIcon className={cn(img.favorite && "fill-current text-primary")} />
            {img.favorite ? "Favorite" : "Add to favorites"}
          </Button>
          {img.prompt && (
            <Button
              size="icon-sm"
              variant="ghost"
              aria-label="Copy prompt"
              onClick={() => navigator.clipboard.writeText(img.prompt).then(() => toast.success("Prompt copied"))}
            >
              <CopyIcon />
            </Button>
          )}
          <Button
            size="icon-sm"
            variant="ghost"
            aria-label="Delete image"
            onClick={() => {
              if (confirm("Delete this image and the versions made from it?")) {
                del.mutate(img.id, { onSuccess: onDeleted })
              }
            }}
          >
            <Trash2Icon />
          </Button>
        </div>

        {img.prompt && <p className="text-xs leading-relaxed text-muted-foreground">{img.prompt}</p>}

        <Separator />

        <Tabs defaultValue="variations">
          <TabsList className="w-full">
            <TabsTrigger value="variations">
              <SparklesIcon />
              Variations
            </TabsTrigger>
            <TabsTrigger value="upscale" disabled={img.vector}>
              <Maximize2Icon />
              Upscale
            </TabsTrigger>
            <TabsTrigger value="background" disabled={img.vector}>
              <EraserIcon />
              Background
            </TabsTrigger>
          </TabsList>
          <TabsContent value="variations">
            <VariationsPanel img={img} />
          </TabsContent>
          <TabsContent value="upscale">
            <UpscalePanel img={img} onDone={setCompareId} />
          </TabsContent>
          <TabsContent value="background">
            <BackgroundPanel img={img} onDone={setCompareId} />
          </TabsContent>
        </Tabs>
        <p className="text-xs text-muted-foreground">
          Upscales and background removals stay next to the original as candidates. Try a few, compare them, and pick
          one with Use this version. The original is never replaced.
        </p>
      </div>
    </div>
  )
}

function VariationsPanel({ img }: { img: ImageDetail }) {
  const [count, setCount] = useState("4")
  const [mode, setMode] = useState<"remix" | "reroll">("remix")
  const [instruction, setInstruction] = useState("")
  const [jobId, setJobId] = useState<string | null>(null)
  const start = useVariations()
  const job = useJob(jobId)

  return (
    <div className="flex flex-col gap-3 pt-2">
      <ToggleGroup
        type="single"
        variant="outline"
        size="sm"
        value={mode}
        onValueChange={(v) => v && setMode(v as "remix" | "reroll")}
      >
        <ToggleGroupItem value="remix">Close variations</ToggleGroupItem>
        <ToggleGroupItem value="reroll">Same prompt, new seeds</ToggleGroupItem>
      </ToggleGroup>
      {mode === "remix" && (
        <Input
          placeholder="Optional: what to change (e.g. thicker lines, warmer colors)"
          value={instruction}
          onChange={(e) => setInstruction(e.target.value)}
        />
      )}
      <div className="flex items-center gap-2">
        <Select value={count} onValueChange={setCount}>
          <SelectTrigger className="w-24">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {["2", "4", "6", "8"].map((n) => (
              <SelectItem key={n} value={n}>
                {n} images
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
        <Button
          size="sm"
          disabled={start.isPending || job.data?.status === "running" || !img.prompt}
          onClick={() =>
            start.mutate(
              { id: img.id, count: Number(count), mode, instruction },
              { onSuccess: (j) => setJobId(j.id) },
            )
          }
        >
          <SparklesIcon />
          Make variations
        </Button>
      </div>
      <p className="text-xs text-muted-foreground">
        Uses {shortModel(img.model)}. Cloud models that accept reference images also get this image as a reference.
        Results show up under derived versions and in the gallery.
      </p>
      {job.data && <JobProgress job={job.data} />}
    </div>
  )
}

function UpscalePanel({ img, onDone }: { img: ImageDetail; onDone: (id: string) => void }) {
  const tools = useTools()
  const upscale = useUpscale()
  const aiAvailable = tools.data?.upscale.ai_available ?? false
  const [method, setMethod] = useState(aiAvailable ? "realesrgan-anime" : "lanczos")
  const [scale, setScale] = useState("4")
  const methods: [string, string][] = Object.entries(tools.data?.upscale.methods ?? {})

  return (
    <div className="flex flex-col gap-3 pt-2">
      <div className="flex flex-col gap-1.5">
        <Label>Method</Label>
        <Select value={method} onValueChange={setMethod}>
          <SelectTrigger className="w-full">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {methods.map(([k, label]: [string, string]) => (
              <SelectItem key={k} value={k} disabled={k.startsWith("realesrgan") && !aiAvailable}>
                {label}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
        {!aiAvailable && (
          <p className="text-xs text-muted-foreground">
            Real-ESRGAN needs local model support. It installs with your first local model.
          </p>
        )}
      </div>
      <div className="flex items-center gap-2">
        <Select value={scale} onValueChange={setScale}>
          <SelectTrigger className="w-24">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {Object.entries<number>(tools.data?.upscale.scales ?? { "2x": 2, "4x": 4 }).map(([label, v]) => (
              <SelectItem key={v} value={String(v)}>
                {label}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
        <Button
          size="sm"
          disabled={upscale.isPending}
          onClick={() =>
            upscale.mutate({ id: img.id, method, scale: Number(scale) }, { onSuccess: (r) => onDone(r.id) })
          }
        >
          {upscale.isPending ? <Loader2Icon className="animate-spin" /> : <Maximize2Icon />}
          Upscale
        </Button>
      </div>
      {upscale.isPending && method.startsWith("realesrgan") && tools.data?.upscale.device === "cpu" && (
        <p className="text-xs text-muted-foreground">Real-ESRGAN on the CPU takes a few minutes for a 1024 px image.</p>
      )}
    </div>
  )
}

function BackgroundPanel({ img, onDone }: { img: ImageDetail; onDone: (id: string) => void }) {
  const tools = useTools()
  const remove = useRemoveBackground()
  const [method, setMethod] = useState<"color" | "ai">("color")
  const [model, setModel] = useState<string>("")
  const [tolerance, setTolerance] = useState(20)
  const [erode, setErode] = useState(0)
  const chosen = model || tools.data?.background.default || "isnet-general-use"

  return (
    <div className="flex flex-col gap-3 pt-2">
      <ToggleGroup
        type="single"
        variant="outline"
        size="sm"
        value={method}
        onValueChange={(v) => v && setMethod(v as "color" | "ai")}
      >
        <ToggleGroupItem value="color">Color key</ToggleGroupItem>
        <ToggleGroupItem value="ai">AI model</ToggleGroupItem>
      </ToggleGroup>
      {method === "color" ? (
        <div className="flex flex-col gap-2">
          <Label>Tolerance: {tolerance}</Label>
          <Slider min={1} max={80} value={[tolerance]} onValueChange={([v]) => setTolerance(v)} />
          <p className="text-xs text-muted-foreground">
            Clears the color found in the corners. Instant, and best for flat logos on a plain background.
          </p>
        </div>
      ) : (
        <Select value={chosen} onValueChange={setModel}>
          <SelectTrigger className="w-full">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {Object.entries<string>(tools.data?.background.models ?? {}).map(([k, label]) => (
              <SelectItem key={k} value={k}>
                {label}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      )}
      <div className="flex flex-col gap-2">
        <Label>Shrink edge: {erode}px</Label>
        <Slider min={0} max={6} value={[erode]} onValueChange={([v]) => setErode(v)} />
      </div>
      <Button
        size="sm"
        className="self-start"
        disabled={remove.isPending}
        onClick={() =>
          remove.mutate(
            { id: img.id, method, model: chosen, tolerance, erode_pixels: erode },
            { onSuccess: (r) => onDone(r.id) },
          )
        }
      >
        {remove.isPending ? <Loader2Icon className="animate-spin" /> : <EraserIcon />}
        Remove background
      </Button>
    </div>
  )
}
