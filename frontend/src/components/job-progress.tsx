import { Loader2Icon, XIcon } from "lucide-react"

import { Button } from "@/components/ui/button"
import { Progress } from "@/components/ui/progress"
import type { Job } from "@/lib/api"
import { formatUsd } from "@/lib/format"
import { useCancelJob } from "@/lib/queries"

/** "$0.028", "$0.003 + $0.001 BYOK", or "local, $0". */
export function jobCostLabel(job: Pick<Job, "cost" | "byok" | "local_images" | "kind">): string | null {
  if (job.kind === "calibrate") return "local, $0"
  const paid = job.cost + job.byok
  if (paid > 0) {
    return job.byok > 0 ? `${formatUsd(job.cost, 4)} + ${formatUsd(job.byok, 4)} BYOK` : formatUsd(job.cost, 4)
  }
  if (job.local_images > 0) return "local, $0"
  return null
}

export function JobProgress({ job, compact = false }: { job: Job; compact?: boolean }) {
  const cancel = useCancelJob()
  const running = job.status === "running"
  const pct = job.total ? Math.round((job.done / job.total) * 100) : 0
  const cost = jobCostLabel(job)

  return (
    <div className="flex flex-col gap-1.5">
      <div className="flex items-center gap-2 text-xs">
        {running && <Loader2Icon className="size-3.5 shrink-0 animate-spin text-muted-foreground" />}
        <span className="min-w-0 flex-1 truncate">
          {compact && <span className="font-medium">{job.title} · </span>}
          <span className="text-muted-foreground">{job.message || (running ? "Starting" : job.status)}</span>
        </span>
        {cost && <span className="text-muted-foreground tabular-nums" title="Spend for this job">{cost}</span>}
        {running && (
          <Button
            size="icon-xs"
            variant="ghost"
            aria-label="Cancel"
            disabled={cancel.isPending}
            onClick={() => cancel.mutate(job.id)}
          >
            <XIcon />
          </Button>
        )}
      </div>
      {running && <Progress value={job.total ? pct : null} className="h-1" />}
      {!compact && job.notes.length > 0 && (
        <ul className="flex flex-col gap-0.5 text-xs text-muted-foreground">
          {job.notes.map((n) => (
            <li key={n}>{n}</li>
          ))}
        </ul>
      )}
      {!compact && job.errors.length > 0 && (
        <ul className="flex flex-col gap-0.5 text-xs text-destructive">
          {[...new Set(job.errors)].slice(0, 4).map((e) => (
            <li key={e}>{e}</li>
          ))}
        </ul>
      )}
    </div>
  )
}
