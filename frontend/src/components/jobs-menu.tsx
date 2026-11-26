import { ActivityIcon, Loader2Icon } from "lucide-react"

import { JobProgress } from "@/components/job-progress"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover"
import { useJobs } from "@/lib/queries"

export function JobsMenu() {
  const { data: jobs = [] } = useJobs()
  const running = jobs.filter((j) => j.status === "running")
  const recent = jobs.slice(0, 8)

  return (
    <Popover>
      <PopoverTrigger asChild>
        <Button variant="ghost" size="sm" aria-label="Jobs">
          {running.length ? <Loader2Icon className="animate-spin" /> : <ActivityIcon />}
          Jobs
          {running.length > 0 && <Badge>{running.length}</Badge>}
        </Button>
      </PopoverTrigger>
      <PopoverContent align="end" className="w-96">
        {recent.length === 0 ? (
          <p className="text-xs text-muted-foreground">No jobs yet. Generations show up here with their progress.</p>
        ) : (
          <div className="flex flex-col gap-3">
            {recent.map((j) => (
              <JobProgress key={j.id} job={j} compact />
            ))}
          </div>
        )}
      </PopoverContent>
    </Popover>
  )
}
