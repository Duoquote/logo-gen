import { useState } from "react"
import { CopyIcon, RotateCcwIcon, WandSparklesIcon } from "lucide-react"
import { toast } from "sonner"

import { JobProgress } from "@/components/job-progress"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Textarea } from "@/components/ui/textarea"
import type { Brief, ImageDetail, RenderedPrompt } from "@/lib/api"
import { shortModel } from "@/lib/format"
import { useJob, useRerunPrompt } from "@/lib/queries"

/** The brief, as a short list of labeled lines. */
export function BriefView({ brief }: { brief: Brief }) {
  const rows: [string, string][] = [
    ["Mark", `${brief.mark_type}, ${brief.format}`],
    ["Subject", brief.subject],
    ["Composition", brief.composition],
    ["Shapes", brief.shape_language],
    ["Style", brief.style],
    ["Background", brief.background],
    ["Text", brief.text ? `"${brief.text}"` : "none"],
    ["Avoid", brief.avoid.join(", ")],
  ]
  return (
    <dl className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-0.5 text-xs">
      {rows
        .filter(([, v]) => v)
        .map(([k, v]) => (
          <div key={k} className="contents">
            <dt className="text-muted-foreground">{k}</dt>
            <dd>{v}</dd>
          </div>
        ))}
      {brief.palette.length > 0 && (
        <div className="contents">
          <dt className="text-muted-foreground">Colors</dt>
          <dd className="flex flex-wrap gap-2">
            {brief.palette.map((c) => (
              <span key={c.hex + c.name} className="flex items-center gap-1">
                <span className="size-3 border" style={{ background: c.hex || undefined }} />
                {c.name} {c.hex}
              </span>
            ))}
          </dd>
        </div>
      )}
    </dl>
  )
}

function ParamBadges({ params }: { params: Record<string, unknown> | null | undefined }) {
  const entries = Object.entries(params ?? {})
  if (!entries.length) return null
  return (
    <div className="flex flex-wrap gap-1">
      {entries.map(([k, v]) => (
        <Badge key={k} variant="outline" className="font-mono text-[10px]">
          {k}={String(v)}
        </Badge>
      ))}
    </div>
  )
}

/**
 * The prompt an image was made with, editable, with a button to run the edited version.
 * Mount it with key={img.id} so the edits reset when another image opens.
 */
export function PromptDetails({ img }: { img: ImageDetail }) {
  const [prompt, setPrompt] = useState(img.prompt)
  const [negative, setNegative] = useState(img.negative_prompt ?? "")
  const [count, setCount] = useState("2")
  const [jobId, setJobId] = useState<string | null>(null)
  const regenerate = useRerunPrompt()
  const job = useJob(jobId)

  if (!img.prompt) return null
  const edited = prompt.trim() !== img.prompt.trim() || negative.trim() !== (img.negative_prompt ?? "").trim()

  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-center gap-2 text-xs">
        <span className="font-medium">Prompt sent to {shortModel(img.model)}</span>
        {img.profile && (
          <Badge variant="secondary" className="text-[10px]" title="The prompt profile that wrote this prompt">
            {img.profile}
          </Badge>
        )}
        <Button
          size="icon-xs"
          variant="ghost"
          className="ml-auto"
          aria-label="Copy prompt"
          onClick={() => navigator.clipboard.writeText(prompt).then(() => toast.success("Prompt copied"))}
        >
          <CopyIcon />
        </Button>
      </div>
      <Textarea value={prompt} onChange={(e) => setPrompt(e.target.value)} rows={5} className="text-xs" />
      {(img.negative_prompt || negative) && (
        <>
          <span className="text-xs text-muted-foreground">Negative prompt</span>
          <Textarea value={negative} onChange={(e) => setNegative(e.target.value)} rows={2} className="text-xs" />
        </>
      )}
      <ParamBadges params={img.params} />
      <div className="flex items-center gap-2">
        <Select value={count} onValueChange={setCount}>
          <SelectTrigger className="w-24" size="sm">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {["1", "2", "4"].map((n) => (
              <SelectItem key={n} value={n}>
                {n} image{n === "1" ? "" : "s"}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
        <Button
          size="sm"
          variant={edited ? "default" : "outline"}
          disabled={!prompt.trim() || regenerate.isPending || job.data?.status === "running"}
          onClick={() =>
            regenerate.mutate(
              { id: img.id, prompt, negative_prompt: negative, count: Number(count) },
              { onSuccess: (j) => setJobId(j.id) },
            )
          }
        >
          <WandSparklesIcon />
          {edited ? "Run edited prompt" : "Run again"}
        </Button>
        {edited && (
          <Button
            size="icon-sm"
            variant="ghost"
            aria-label="Undo edits"
            onClick={() => {
              setPrompt(img.prompt)
              setNegative(img.negative_prompt ?? "")
            }}
          >
            <RotateCcwIcon />
          </Button>
        )}
      </div>
      {job.data && <JobProgress job={job.data} />}
      {img.brief && (
        <details className="text-xs">
          <summary className="cursor-pointer text-muted-foreground">Design brief behind this prompt</summary>
          <div className="mt-2">
            <BriefView brief={img.brief} />
          </div>
        </details>
      )}
    </div>
  )
}

/** What each model was sent in a job, grouped by direction. */
export function RenderedPrompts({ rendered }: { rendered: RenderedPrompt[] }) {
  if (!rendered.length) return null
  const byDirection = new Map<number, RenderedPrompt[]>()
  for (const r of rendered) byDirection.set(r.direction, [...(byDirection.get(r.direction) ?? []), r])
  return (
    <details className="text-xs">
      <summary className="cursor-pointer text-muted-foreground">
        Prompts sent ({rendered.length}, written per model from each direction's brief)
      </summary>
      <div className="mt-2 flex flex-col gap-3">
        {[...byDirection.entries()].map(([d, rs]) => (
          <div key={d} className="flex flex-col gap-2 border p-2">
            <span className="font-medium">{rs[0].concept || `Direction ${d + 1}`}</span>
            {rs.map((r) => (
              <div key={r.model} className="flex flex-col gap-1">
                <span className="text-muted-foreground">
                  {shortModel(r.model)} · {r.profile_name} ({r.style})
                </span>
                <p className="whitespace-pre-wrap">{r.prompt}</p>
                {r.negative_prompt && <p className="text-muted-foreground">Negative: {r.negative_prompt}</p>}
                <ParamBadges params={r.params} />
                {r.notes.map((n) => (
                  <p key={n} className="text-muted-foreground">
                    {n}
                  </p>
                ))}
              </div>
            ))}
          </div>
        ))}
      </div>
    </details>
  )
}
