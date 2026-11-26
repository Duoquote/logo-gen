import { Loader2Icon, XIcon } from "lucide-react"

import { Button } from "@/components/ui/button"
import { Progress } from "@/components/ui/progress"
import type { Job } from "@/lib/api"
import { formatUsd } from "@/lib/format"
import { useCancelJob } from "@/lib/queries"

export function JobProgress({ job, compact = false }: { job: Job; compact?: boolean }) {
  const cancel = useCancelJob()
  const running = job.status === "running"
  const pct = job.total ? Math.round((job.done / job.total) * 100) : 0

  return (
    <div className="flex flex-col gap-1.5">
      <div className="flex items-center gap-2 text-xs">
        {running && <Loader2Icon className="size-3.5 shrink-0 animate-spin text-muted-foreground" />}
        <span className="min-w-0 flex-1 truncate">
          {compact && <span className="font-medium">{job.title} · </span>}
          <span className="text-muted-foreground">{job.message || (running ? "Starting" : job.status)}</span>
        </span>
        {job.cost > 0 && <span className="text-muted-foreground tabular-nums">{formatUsd(job.cost)}</span>}
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
