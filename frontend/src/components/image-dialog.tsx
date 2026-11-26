import { useState } from "react"
import {
  CopyIcon,
  DownloadIcon,
  EraserIcon,
  HeartIcon,
  Loader2Icon,
  Maximize2Icon,
  SparklesIcon,
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
  const family: ImageItem[] = [...(img.parent ? [img.parent] : []), ...img.children]

  return (
    <div className="grid gap-6 md:grid-cols-[minmax(0,1fr)_320px]">
      <div className="flex flex-col gap-3">
        <img
          src={img.url}
          alt={img.concept || img.prompt}
          className={cn("aspect-square w-full border object-contain", img.kind === "cleaned" && "checker")}
        />
        {family.length > 0 && (
          <div className="flex flex-col gap-1.5">
            <span className="text-xs text-muted-foreground">
              {img.parent ? "Made from / derived versions" : "Derived versions"}
            </span>
            <div className="flex flex-wrap gap-2">
              {family.map((c) => (
                <button
                  key={c.id}
                  type="button"
                  onClick={() => onSelect(c.id)}
                  className="border hover:border-primary"
                  title={c.note || c.kind}
                >
                  <img src={c.url} alt="" className={cn("size-16 object-contain", c.kind === "cleaned" && "checker")} />
                </button>
              ))}
            </div>
          </div>
        )}
      </div>

      <div className="flex min-w-0 flex-col gap-4">
        <div>
          <DialogTitle>{img.concept || "Untitled"}</DialogTitle>
          <DialogDescription className="mt-1">
            {shortModel(img.model)}
            {img.width ? ` · ${img.width}×${img.height}` : ""}
            {img.cost ? ` · ${formatUsd(img.cost)}` : ""}
            {img.note ? ` · ${img.note}` : ""}
          </DialogDescription>
        </div>

        <div className="flex flex-wrap gap-2">
          <Button asChild size="sm">
            <a href={`${img.url}?download=1`} download>
              <DownloadIcon />
              Download {img.vector ? "SVG" : "PNG"}
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
            <UpscalePanel img={img} onDone={onSelect} />
          </TabsContent>
          <TabsContent value="background">
            <BackgroundPanel img={img} onDone={onSelect} />
          </TabsContent>
        </Tabs>
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
