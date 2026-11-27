import { useState } from "react"
import { createFileRoute, Link } from "@tanstack/react-router"
import { Loader2Icon, WandSparklesIcon } from "lucide-react"

import { ImageDialog } from "@/components/image-dialog"
import { ImageGrid } from "@/components/image-grid"
import { JobProgress } from "@/components/job-progress"
import { ImageModelPicker } from "@/components/model-picker"
import { RenderedPrompts } from "@/components/prompt-details"
import { ReferenceImages, useReferenceUploads } from "@/components/reference-images"
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Switch } from "@/components/ui/switch"
import { Textarea } from "@/components/ui/textarea"
import { maxPerDirection } from "@/lib/caps"
import { useCostEstimate } from "@/lib/cost"
import { formatUsd } from "@/lib/format"
import { useImages, useJob, useQuickGenerate, useSettings, useUpdateSettings } from "@/lib/queries"

export const Route = createFileRoute("/")({
  validateSearch: (s: Record<string, unknown>): { job?: string; image?: string; prompt?: string } => ({
    job: typeof s.job === "string" ? s.job : undefined,
    image: typeof s.image === "string" ? s.image : undefined,
    // A finished prompt from the designer's "Open in Generate". It goes out as written.
    prompt: typeof s.prompt === "string" ? s.prompt : undefined,
  }),
  component: GeneratePage,
})

function GeneratePage() {
  const { job: jobId, image, prompt: given } = Route.useSearch()
  const navigate = Route.useNavigate()
  const settings = useSettings()
  const update = useUpdateSettings()
  const start = useQuickGenerate()
  const job = useJob(jobId)
  const refs = useReferenceUploads()
  const [brief, setBrief] = useState(given ?? "")
  const [enhance, setEnhance] = useState(!given)

  const s = settings.data?.settings
  const directions = enhance ? (s?.variations ?? 4) : 1
  const maxPer = s ? maxPerDirection(s, s.image_models, directions) : 1
  const per = Math.min(s?.images_per_prompt ?? 1, maxPer)
  const estimate = useCostEstimate(s?.image_models ?? [], directions * per)
  const jobImages = useImages({ job_id: jobId }, !!jobId)
  const recent = useImages({ kind: "generated", limit: 24 })

  const running = job.data?.status === "running"
  const pending = running ? Math.max(0, (job.data?.total ?? 0) - (jobImages.data?.length ?? 0)) : 0

  const submit = () => {
    if (!brief.trim() || !s) return
    start.mutate(
      {
        brief,
        enhance,
        count: s.variations,
        images_per_prompt: per,
        models: s.image_models,
        references: refs.refs.map((r) => r.path),
      },
      { onSuccess: (j) => navigate({ search: { job: j.id } }) },
    )
  }

  return (
    <div className="mx-auto flex max-w-6xl flex-col gap-6 p-6">
      <Card>
        <CardContent className="flex flex-col gap-4">
          <Textarea
            value={brief}
            onChange={(e) => setBrief(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) submit()
            }}
            placeholder="What is the logo for? e.g. polytini, a small Python library for polygon geometry. Playful, two colors."
            rows={3}
            className="text-sm"
          />
          <ReferenceImages
            refs={refs.refs}
            uploading={refs.uploading}
            onAdd={refs.add}
            onRemove={(p) => refs.setRefs((r) => r.filter((x) => x.path !== p))}
          />
          <div className="grid gap-4 md:grid-cols-[minmax(0,2fr)_auto_auto_auto] md:items-end">
            <div className="flex min-w-0 flex-col gap-1.5">
              <Label>Image models</Label>
              <ImageModelPicker value={s?.image_models ?? []} onChange={(v) => update.mutate({ image_models: v })} />
            </div>
            <div className="flex flex-col gap-1.5">
              <Label>Directions</Label>
              <Select
                value={String(s?.variations ?? 4)}
                onValueChange={(v) => update.mutate({ variations: Number(v) })}
                disabled={!enhance}
              >
                <SelectTrigger className="w-24">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {[1, 2, 3, 4, 6, 8].map((n) => (
                    <SelectItem key={n} value={String(n)}>
                      {n}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <div className="flex flex-col gap-1.5">
              <Label>Images each</Label>
              <Select
                value={String(per)}
                onValueChange={(v) => update.mutate({ images_per_prompt: Number(v) })}
              >
                <SelectTrigger className="w-24" title={`Per direction and model. Capped at ${maxPer} by Settings, Limits.`}>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {Array.from({ length: Math.min(maxPer, 8) }, (_, i) => i + 1).map((n) => (
                    <SelectItem key={n} value={String(n)}>
                      {n}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <label className="flex h-8 items-center gap-2 text-xs">
              <Switch checked={enhance} onCheckedChange={setEnhance} />
              Write prompts with AI
            </label>
          </div>
          <div className="flex items-center justify-between gap-4">
            <p className="text-xs text-muted-foreground">
              {estimate.images} image{estimate.images === 1 ? "" : "s"}
              {estimate.cost !== null && ` · about ${formatUsd(estimate.cost)}`}
              {enhance && " · prompts written by the chat model set in Settings"}
            </p>
            <Button onClick={submit} disabled={!brief.trim() || start.isPending || running || !s?.image_models.length}>
              {start.isPending ? <Loader2Icon className="animate-spin" /> : <WandSparklesIcon />}
              Generate
              <kbd className="ml-1 text-[10px] opacity-60">Ctrl+Enter</kbd>
            </Button>
          </div>
        </CardContent>
      </Card>

      {job.data && (
        <section className="flex flex-col gap-4">
          <JobProgress job={job.data} />
          {job.data.result.prompts && job.data.result.prompts.length > 1 && (
            <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-4">
              {job.data.result.prompts.map((p, i) => (
                <div key={i} className="border p-3 text-xs">
                  <div className="font-medium">{p.concept}</div>
                  <div className="text-muted-foreground">{p.style}</div>
                </div>
              ))}
            </div>
          )}
          {job.data.result.rendered && <RenderedPrompts rendered={job.data.result.rendered} />}
          <ImageGrid
            images={jobImages.data ?? []}
            pending={pending}
            onOpen={(id) => navigate({ search: (prev) => ({ ...prev, image: id }) })}
          />
        </section>
      )}

      <section className="flex flex-col gap-3">
        <div className="flex items-baseline justify-between">
          <h2 className="font-heading text-sm font-semibold">Recent</h2>
          <Link to="/gallery" className="text-xs text-muted-foreground hover:text-foreground">
            Open gallery
          </Link>
        </div>
        {recent.data?.length ? (
          <ImageGrid
            images={recent.data.filter((r) => r.job_id !== jobId)}
            onOpen={(id) => navigate({ search: (prev) => ({ ...prev, image: id }) })}
          />
        ) : (
          <p className="text-xs text-muted-foreground">Nothing yet. Describe a logo above and press Generate.</p>
        )}
      </section>

      <ImageDialog
        imageId={image ?? null}
        onOpenChange={(open) => !open && navigate({ search: (prev) => ({ ...prev, image: undefined }) })}
        onSelect={(id) => navigate({ search: (prev) => ({ ...prev, image: id }) })}
      />
    </div>
  )
}
