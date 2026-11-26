import { Link } from "@tanstack/react-router"
import { CheckIcon, DownloadIcon, ExternalLinkIcon, GaugeIcon, Loader2Icon, PauseIcon, RefreshCwIcon, Trash2Icon } from "lucide-react"

import { JobProgress } from "@/components/job-progress"

import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from "@/components/ui/alert-dialog"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Progress } from "@/components/ui/progress"
import type { LocalModel, RuntimeStatus, Support, SystemInfo } from "@/lib/api"
import { formatBytes } from "@/lib/format"
import {
  useCalibrate,
  useCancelInstall,
  useCheckAccess,
  useInstallModel,
  useJob,
  useJobs,
  useLocalModels,
  useSettings,
  useUninstallModel,
  useUpdateSettings,
} from "@/lib/queries"

const DEVICE_LABEL = { cuda: "NVIDIA GPU", mps: "Apple Silicon", cpu: "CPU only" } as const

export function LocalModels() {
  const { data, isLoading, error } = useLocalModels()
  if (isLoading) return <Loader2Icon className="animate-spin text-muted-foreground" />
  if (error || !data) return <p className="text-sm text-destructive">{(error as Error)?.message}</p>
  return (
    <div className="flex flex-col gap-4">
      <MachineSummary system={data.system} runtime={data.runtime} />
      <div className="grid gap-4 lg:grid-cols-2">
        {data.models.map((m) => (
          <ModelCard key={m.id} m={m} />
        ))}
      </div>
    </div>
  )
}

function MachineSummary({ system, runtime }: { system: SystemInfo; runtime: RuntimeStatus }) {
  return (
    <div className="flex flex-col gap-3">
      <div className="flex flex-wrap items-center gap-x-6 gap-y-1 text-xs text-muted-foreground">
        <span>
          Runs on: <span className="text-foreground">{DEVICE_LABEL[system.device]}</span>
          {system.gpu && ` (${system.gpu.name}, ${system.gpu.vram_gb} GB)`}
          {system.forced_device && " · forced by LOGO_GEN_DEVICE / CUDA_VISIBLE_DEVICES"}
        </span>
        <span>
          Local model support:{" "}
          <span className="text-foreground">
            {runtime.installed ? `installed (torch ${runtime.torch})` : `not installed yet (PyTorch ${runtime.variant} on first install)`}
          </span>
        </span>
        <span>Free disk: {system.disk_free_gb} GB</span>
      </div>
      {runtime.running && (
        <Alert>
          <Loader2Icon className="animate-spin" />
          <AlertTitle>Installing local model support ({runtime.elapsed}s)</AlertTitle>
          <AlertDescription>
            <pre className="max-h-32 overflow-auto font-mono text-[11px] whitespace-pre-wrap">{runtime.log.join("\n")}</pre>
          </AlertDescription>
        </Alert>
      )}
      {runtime.error && !runtime.running && (
        <Alert variant="destructive">
          <AlertTitle>Installing local model support failed</AlertTitle>
          <AlertDescription>{runtime.error}</AlertDescription>
        </Alert>
      )}
      {system.device === "cpu" && (
        <p className="text-xs text-muted-foreground">
          No NVIDIA GPU found. Cloud models work fully. SD-Turbo runs on the CPU but slowly; the larger models need a GPU.
        </p>
      )}
    </div>
  )
}

function SupportBadge({ label, s }: { label: string; s: Support }) {
  const text = s === "slow" ? `${label} (slow)` : label
  return <Badge variant={s === "no" ? "outline" : "secondary"} className={s === "no" ? "line-through opacity-60" : ""} title={s === "no" ? `Does not run on ${label}` : undefined}>{text}</Badge>
}

function ModelCard({ m }: { m: LocalModel }) {
  const install = useInstallModel()
  const cancel = useCancelInstall()
  const uninstall = useUninstallModel()
  const check = useCheckAccess()
  const busy = m.state === "checking" || m.state === "downloading" || m.state === "runtime"
  const pct = m.size_bytes ? Math.min(100, (m.downloaded_bytes / m.size_bytes) * 100) : 0
  const blocked = m.access && (m.access.state === "gated" || m.access.state === "needs_token")

  const installButton = (
    <Button size="sm" disabled={install.isPending || m.runs_here === "no" || !!blocked} onClick={m.runs_here === "slow" ? undefined : () => install.mutate(m.id)}>
      <DownloadIcon />
      {m.state === "partial" ? "Resume" : "Install"} · {formatBytes(m.size_bytes)}
    </Button>
  )

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          {m.name}
          {m.installed && <Badge>Installed</Badge>}
          {m.gated && <Badge variant="outline">Gated</Badge>}
        </CardTitle>
        <CardDescription>{m.blurb}</CardDescription>
      </CardHeader>
      <CardContent className="flex flex-col gap-3 text-xs">
        <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1">
          <dt className="text-muted-foreground">Download</dt>
          <dd>{formatBytes(m.size_bytes)}</dd>
          <dt className="text-muted-foreground">VRAM</dt>
          <dd>about {m.vram_gb} GB on NVIDIA · {m.resolution}px · {m.steps} steps</dd>
          <dt className="text-muted-foreground">License</dt>
          <dd>
            <a href={m.license_url} target="_blank" rel="noreferrer" className="underline">
              {m.license}
            </a>{" "}
            ·{" "}
            <a href={m.page_url} target="_blank" rel="noreferrer" className="inline-flex items-center gap-0.5 underline">
              {m.repo} <ExternalLinkIcon className="size-3" />
            </a>
          </dd>
          <dt className="text-muted-foreground">Runs on</dt>
          <dd className="flex flex-wrap gap-1">
            <SupportBadge label="CUDA" s="ok" />
            <SupportBadge label="Apple Silicon" s={m.support.mps} />
            <SupportBadge label="CPU" s={m.support.cpu} />
          </dd>
        </dl>
        {m.device_note && <p className={m.runs_here === "no" ? "text-destructive" : "text-muted-foreground"}>{m.device_note}</p>}
        {blocked && (
          <Alert>
            <AlertTitle>{m.access?.state === "needs_token" ? "Needs a Hugging Face token" : "License not accepted yet"}</AlertTitle>
            <AlertDescription className="flex flex-col gap-2">
              <span>{m.access?.message}</span>
              <span className="flex gap-2">
                <Button asChild size="xs" variant="outline">
                  <a href={m.page_url} target="_blank" rel="noreferrer">
                    Open on Hugging Face
                  </a>
                </Button>
                {m.access?.state === "needs_token" && (
                  <Button asChild size="xs" variant="outline">
                    <Link to="/settings">Set token</Link>
                  </Button>
                )}
                <Button size="xs" variant="ghost" onClick={() => check.mutate(m.id)} disabled={check.isPending}>
                  <RefreshCwIcon /> Check again
                </Button>
              </span>
            </AlertDescription>
          </Alert>
        )}
        {busy && (
          <div className="flex flex-col gap-1">
            <Progress value={m.state === "downloading" ? pct : null} className="h-1.5" />
            <span className="text-muted-foreground tabular-nums">
              {m.state === "runtime"
                ? m.message || "Installing local model support"
                : m.state === "checking"
                  ? "Checking access and files"
                  : `${formatBytes(m.downloaded_bytes)} of ${formatBytes(m.size_bytes)} · ${formatBytes(m.speed_bps)}/s`}
            </span>
          </div>
        )}
        {m.state === "partial" && (
          <span className="text-muted-foreground">
            Paused at {formatBytes(m.downloaded_bytes)} of {formatBytes(m.size_bytes)}. Resume picks up where it stopped.
          </span>
        )}
        {m.state === "error" && <p className="text-destructive">{m.message}</p>}
        {m.installed && m.runs_here !== "no" && <SpeedTest m={m} />}
      </CardContent>
      <CardFooter className="gap-2">
        {busy ? (
          <Button size="sm" variant="outline" onClick={() => cancel.mutate(m.id)} disabled={cancel.isPending || m.state === "runtime"}>
            <PauseIcon />
            Cancel
          </Button>
        ) : m.installed ? null : m.runs_here === "slow" ? (
          <AlertDialog>
            <AlertDialogTrigger asChild>{installButton}</AlertDialogTrigger>
            <AlertDialogContent>
              <AlertDialogHeader>
                <AlertDialogTitle>This will be slow here</AlertDialogTitle>
                <AlertDialogDescription>{m.device_note} Install anyway?</AlertDialogDescription>
              </AlertDialogHeader>
              <AlertDialogFooter>
                <AlertDialogCancel>Cancel</AlertDialogCancel>
                <AlertDialogAction onClick={() => install.mutate(m.id)}>Install</AlertDialogAction>
              </AlertDialogFooter>
            </AlertDialogContent>
          </AlertDialog>
        ) : (
          installButton
        )}
        {(m.installed || m.state === "partial") && !busy && (
          <Button
            size="sm"
            variant="ghost"
            onClick={() => confirm(`Remove ${m.name} and free ${formatBytes(m.downloaded_bytes || m.size_bytes)}?`) && uninstall.mutate(m.id)}
          >
            <Trash2Icon />
            {m.installed ? "Uninstall" : "Discard download"}
          </Button>
        )}
      </CardFooter>
    </Card>
  )
}

function SpeedTest({ m }: { m: LocalModel }) {
  const jobs = useJobs()
  const start = useCalibrate()
  const settings = useSettings()
  const update = useUpdateSettings()
  const running = jobs.data?.find((j) => j.kind === "calibrate" && j.status === "running" && j.title.endsWith(m.name))
  const job = useJob(running?.id ?? start.data?.id)
  const cal = job.data?.status === "done" && job.data.result.results ? (job.data.result as LocalModel["calibration"]) : m.calibration
  const busy = job.data?.status === "running"
  const choose = (batch: number) =>
    update.mutate({ local_batch: { ...(settings.data?.settings.local_batch ?? {}), [m.id]: batch } })

  return (
    <div className="flex flex-col gap-2 border-t pt-3">
      <div className="flex items-center gap-2">
        <span className="flex-1">
          Batch size: <span className="font-medium">{m.batch}</span> at a time
          <span className="text-muted-foreground"> ({m.batch_source === "calibrated" ? "picked from a speed test" : "default"})</span>
        </span>
        <Button size="xs" variant="outline" disabled={busy || start.isPending} onClick={() => start.mutate(m.id)}>
          {busy ? <Loader2Icon className="animate-spin" /> : <GaugeIcon />}
          Speed test
        </Button>
      </div>
      {!cal && !busy && (
        <p className="text-muted-foreground">
          The speed test tries batch sizes on this {m.device === "cuda" ? "GPU" : "machine"}, times each one and
          records peak memory, then lets you pick. It takes a few minutes and costs nothing.
        </p>
      )}
      {job.data && busy && <JobProgress job={job.data} />}
      {cal && !busy && (
        <>
          <table className="w-full text-left whitespace-nowrap tabular-nums [&_td]:py-0.5 [&_td]:pr-4 [&_th]:pr-4">
            <thead className="text-muted-foreground">
              <tr>
                <th className="font-normal">Batch</th>
                <th className="font-normal">Time</th>
                <th className="font-normal">Per image</th>
                <th className="font-normal">Peak VRAM</th>
              </tr>
            </thead>
            <tbody>
              {cal.results.map((r) => (
                <tr key={r.batch} className={r.ok ? "" : "text-destructive"}>
                  <td>{r.batch}</td>
                  <td className={r.ok ? "" : "whitespace-normal"} colSpan={r.ok ? 1 : 3}>
                    {r.ok ? `${r.seconds} s` : r.error}
                  </td>
                  {r.ok && (
                    <>
                      <td>{r.per_image} s</td>
                      <td>
                        {r.peak_gb != null ? `${r.peak_gb} GB` : "n/a"}
                        {r.note && <span className="whitespace-normal text-muted-foreground"> ({r.note})</span>}
                      </td>
                    </>
                  )}
                </tr>
              ))}
            </tbody>
          </table>
          <span className="text-muted-foreground">
            {cal.vram_gb ? `${cal.vram_gb} GB card. ` : ""}Loading the model took {cal.load_seconds} s and isn't counted.
          </span>
          <div className="flex flex-wrap gap-2">
            {cal.options.map((o) => (
              <Button
                key={o.key}
                size="xs"
                variant={m.batch === o.batch && m.batch_source === "calibrated" ? "default" : "outline"}
                onClick={() => choose(o.batch)}
              >
                {m.batch === o.batch && m.batch_source === "calibrated" && <CheckIcon />}
                {o.label}: {o.batch} at a time, {o.per_image} s/image
              </Button>
            ))}
          </div>
        </>
      )}
    </div>
  )
}
