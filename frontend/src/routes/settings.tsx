import { useState } from "react"
import { createFileRoute, Link } from "@tanstack/react-router"
import { CheckIcon, ChevronsUpDownIcon, Loader2Icon } from "lucide-react"

import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Command, CommandEmpty, CommandInput, CommandItem, CommandList } from "@/components/ui/command"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Switch } from "@/components/ui/switch"
import type { ReasoningEffort, SecretStatus, Settings } from "@/lib/api"
import { SpendLine } from "@/components/spend"
import { formatUsd, perMillion } from "@/lib/format"
import {
  useCatalog,
  useClaudeCli,
  useKeyStatus,
  useLocalLlm,
  useLocalModels,
  useSpend,
  useResetSettings,
  useSetSecret,
  useSettings,
  useUpdateSettings,
} from "@/lib/queries"
import { cn } from "@/lib/utils"

export const Route = createFileRoute("/settings")({ component: SettingsPage })

function SettingsPage() {
  const { data } = useSettings()
  const reset = useResetSettings()
  if (!data) return <Loader2Icon className="m-6 animate-spin text-muted-foreground" />
  return (
    <div className="mx-auto flex max-w-3xl flex-col gap-6 p-6">
      <OpenRouterCard secret={data.secrets.OPENROUTER_KEY} />
      <ChatCard s={data.settings} />
      <ImageCard s={data.settings} />
      <LimitsCard s={data.settings} />
      <SpendCard />
      <HuggingFaceCard secret={data.secrets.HF_TOKEN} />
      <div>
        <Button
          variant="ghost"
          size="sm"
          onClick={() => confirm("Reset model choices and defaults? Keys stay as they are.") && reset.mutate()}
        >
          Reset to defaults
        </Button>
      </div>
    </div>
  )
}

function SecretField({
  name,
  secret,
  placeholder,
}: {
  name: "OPENROUTER_KEY" | "HF_TOKEN"
  secret: SecretStatus
  placeholder: string
}) {
  const [value, setValue] = useState("")
  const save = useSetSecret()
  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-center gap-2 text-xs">
        {secret.set ? (
          <>
            <Badge variant="secondary">{secret.masked}</Badge>
            <span className="text-muted-foreground">
              {secret.source === "env" ? "from an environment variable (it wins over the saved one)" : "saved in .env in the project folder"}
            </span>
          </>
        ) : (
          <span className="text-muted-foreground">Not set</span>
        )}
      </div>
      <form
        className="flex gap-2"
        onSubmit={(e) => {
          e.preventDefault()
          save.mutate({ name, value }, { onSuccess: () => setValue("") })
        }}
      >
        <Input
          type="password"
          autoComplete="off"
          placeholder={secret.set ? "Paste a new value to replace it" : placeholder}
          value={value}
          onChange={(e) => setValue(e.target.value)}
        />
        <Button type="submit" disabled={!value.trim() || save.isPending}>
          {save.isPending && <Loader2Icon className="animate-spin" />}
          {secret.set ? "Replace" : "Save"}
        </Button>
        {secret.set && secret.source === "file" && (
          <Button type="button" variant="ghost" onClick={() => confirm("Remove the saved value?") && save.mutate({ name, value: "" })}>
            Remove
          </Button>
        )}
      </form>
    </div>
  )
}

function OpenRouterCard({ secret }: { secret: SecretStatus }) {
  const key = useKeyStatus()
  return (
    <Card>
      <CardHeader>
        <CardTitle>OpenRouter</CardTitle>
        <CardDescription>
          Cloud chat and image models. Get a key at{" "}
          <a href="https://openrouter.ai/settings/keys" target="_blank" rel="noreferrer" className="underline">
            openrouter.ai/settings/keys
          </a>
          . It is stored only in this project's .env file, which git ignores.
        </CardDescription>
      </CardHeader>
      <CardContent className="flex flex-col gap-3">
        <SecretField name="OPENROUTER_KEY" secret={secret} placeholder="sk-or-v1-..." />
        {secret.set && key.data && (
          <p className={cn("text-xs", key.data.valid ? "text-muted-foreground" : "text-destructive")}>
            {key.data.valid
              ? `Key works. Spent ${formatUsd(key.data.usage ?? 0, 2)}${
                  key.data.limit != null
                    ? ` of its ${formatUsd(key.data.limit, 2)} limit (${formatUsd(key.data.limit_remaining ?? 0, 2)} left)`
                    : ", no spending limit on this key"
                }.`
              : key.data.error}
          </p>
        )}
      </CardContent>
    </Card>
  )
}

const EFFORTS: ReasoningEffort[] = ["none", "low", "medium", "high", "xhigh", "max"]

function ChatCard({ s }: { s: Settings }) {
  const update = useUpdateSettings()
  const localLlm = useLocalLlm(s.llm_backend === "local")
  const cli = useClaudeCli(s.llm_backend === "claude-cli")
  return (
    <Card>
      <CardHeader>
        <CardTitle>Chat and prompt writing</CardTitle>
        <CardDescription>The model that runs the designer chat and writes prompt directions.</CardDescription>
      </CardHeader>
      <CardContent className="flex flex-col gap-4">
        <div className="flex flex-col gap-1.5">
          <Label>Backend</Label>
          <Select value={s.llm_backend} onValueChange={(v) => update.mutate({ llm_backend: v as Settings["llm_backend"] })}>
            <SelectTrigger className="w-72">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="openrouter">OpenRouter</SelectItem>
              <SelectItem value="local">Local server (Ollama, LM Studio, vLLM)</SelectItem>
              <SelectItem value="claude-cli">Claude CLI (your Claude subscription)</SelectItem>
            </SelectContent>
          </Select>
        </div>
        {s.llm_backend === "openrouter" && (
          <div className="grid gap-4 sm:grid-cols-[1fr_auto]">
            <div className="flex flex-col gap-1.5">
              <Label>Model</Label>
              <ChatModelSelect value={s.chat_model} onChange={(id) => update.mutate({ chat_model: id })} />
            </div>
            <div className="flex flex-col gap-1.5">
              <Label>Reasoning effort</Label>
              <Select value={s.reasoning_effort} onValueChange={(v) => update.mutate({ reasoning_effort: v as ReasoningEffort })}>
                <SelectTrigger className="w-28">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {EFFORTS.map((e) => (
                    <SelectItem key={e} value={e}>
                      {e}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <p className="text-xs text-muted-foreground sm:col-span-2">
              Higher effort thinks longer before answering. Models without that level use the closest one they support.{" "}
              <Link to="/models" search={{ tab: "chat" }} className="underline">
                Browse chat models
              </Link>
            </p>
          </div>
        )}
        {s.llm_backend === "local" && (
          <LocalLlmFields s={s} models={localLlm.data?.models ?? []} reachable={localLlm.data?.reachable} />
        )}
        {s.llm_backend === "claude-cli" && (
          <p className="text-xs text-muted-foreground">
            {cli.isLoading
              ? "Looking for the claude command"
              : cli.data?.available
                ? "Found the claude command. Chats go through `claude -p` and use your subscription. Reference images are not sent."
                : "The claude command was not found on PATH. Install Claude Code and run `claude login`."}
          </p>
        )}
      </CardContent>
    </Card>
  )
}

function LocalLlmFields({ s, models, reachable }: { s: Settings; models: string[]; reachable?: boolean }) {
  const update = useUpdateSettings()
  const [url, setUrl] = useState(s.local_llm_base_url)
  const [model, setModel] = useState(s.local_llm_model)
  const [key, setKey] = useState(s.local_llm_api_key)
  return (
    <form
      className="grid gap-3 sm:grid-cols-2"
      onSubmit={(e) => {
        e.preventDefault()
        update.mutate({ local_llm_base_url: url, local_llm_model: model, local_llm_api_key: key })
      }}
    >
      <div className="flex flex-col gap-1.5 sm:col-span-2">
        <Label>Server URL</Label>
        <Input value={url} onChange={(e) => setUrl(e.target.value)} />
        <span className="text-xs text-muted-foreground">
          {reachable === undefined ? "" : reachable ? `Reachable, ${models.length} models` : "Not reachable right now"}
        </span>
      </div>
      <div className="flex flex-col gap-1.5">
        <Label>Model</Label>
        <Input value={model} onChange={(e) => setModel(e.target.value)} list="local-llm-models" />
        <datalist id="local-llm-models">
          {models.map((m) => (
            <option key={m} value={m} />
          ))}
        </datalist>
      </div>
      <div className="flex flex-col gap-1.5">
        <Label>API key (optional)</Label>
        <Input type="password" value={key} onChange={(e) => setKey(e.target.value)} />
      </div>
      <Button type="submit" size="sm" className="justify-self-start" disabled={update.isPending}>
        Save
      </Button>
    </form>
  )
}

function ChatModelSelect({ value, onChange }: { value: string; onChange: (id: string) => void }) {
  const [open, setOpen] = useState(false)
  const settings = useSettings()
  const catalog = useCatalog(settings.data?.secrets.OPENROUTER_KEY.set ?? false)
  const models = catalog.data?.chat ?? []
  const current = models.find((m) => m.id === value)
  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <Button variant="outline" className="justify-between">
          <span className="truncate">{current?.name ?? value}</span>
          <ChevronsUpDownIcon className="opacity-50" />
        </Button>
      </PopoverTrigger>
      <PopoverContent className="w-[480px] p-0" align="start">
        <Command>
          <CommandInput placeholder="Search chat models" />
          <CommandList className="max-h-80">
            <CommandEmpty>{catalog.isLoading ? "Loading" : "No match"}</CommandEmpty>
            {models.map((m) => (
              <CommandItem
                key={m.id}
                value={`${m.id} ${m.name}`}
                onSelect={() => {
                  onChange(m.id)
                  setOpen(false)
                }}
              >
                <CheckIcon className={cn(m.id === value ? "opacity-100" : "opacity-0")} />
                <span className="min-w-0 flex-1 truncate">{m.name}</span>
                <span className="text-xs text-muted-foreground tabular-nums">
                  {perMillion(m.prompt_price)} / {perMillion(m.completion_price)}
                </span>
              </CommandItem>
            ))}
          </CommandList>
        </Command>
      </PopoverContent>
    </Popover>
  )
}

const RATIOS = ["1:1", "4:3", "3:4", "3:2", "2:3", "16:9", "9:16"]

function ImageCard({ s }: { s: Settings }) {
  const update = useUpdateSettings()
  return (
    <Card>
      <CardHeader>
        <CardTitle>Image defaults</CardTitle>
        <CardDescription>
          Pick image models on the Generate page or under{" "}
          <Link to="/models" className="underline">
            Models
          </Link>
          . Options a model does not support are left out of its request.
        </CardDescription>
      </CardHeader>
      <CardContent className="flex flex-wrap items-end gap-6">
        <div className="flex flex-col gap-1.5">
          <Label>Aspect ratio</Label>
          <Select value={s.aspect_ratio} onValueChange={(v) => update.mutate({ aspect_ratio: v })}>
            <SelectTrigger className="w-28">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {RATIOS.map((r) => (
                <SelectItem key={r} value={r}>
                  {r}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
        <label className="flex h-8 items-center gap-2 text-xs">
          <Switch
            checked={s.transparent_background}
            onCheckedChange={(v) => update.mutate({ transparent_background: v })}
          />
          Ask for a transparent background (where supported)
        </label>
      </CardContent>
    </Card>
  )
}

function HuggingFaceCard({ secret }: { secret: SecretStatus }) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Hugging Face token</CardTitle>
        <CardDescription>
          Only needed for gated local models such as FLUX.1 schnell. Create a read token at{" "}
          <a href="https://huggingface.co/settings/tokens" target="_blank" rel="noreferrer" className="underline">
            huggingface.co/settings/tokens
          </a>{" "}
          and accept the model's terms on its page with the same account.
        </CardDescription>
      </CardHeader>
      <CardContent>
        <SecretField name="HF_TOKEN" secret={secret} placeholder="hf_..." />
      </CardContent>
    </Card>
  )
}

function NumberField({
  label,
  value,
  min,
  max,
  onSave,
  hint,
}: {
  label: string
  value: number
  min: number
  max: number
  onSave: (n: number) => void
  hint?: string
}) {
  const [draft, setDraft] = useState(String(value))
  const commit = () => {
    const n = Math.round(Number(draft))
    if (Number.isFinite(n) && n >= min && n <= max && n !== value) onSave(n)
    else setDraft(String(value))
  }
  return (
    <div className="flex flex-col gap-1.5">
      <Label>{label}</Label>
      <Input
        type="number"
        min={min}
        max={max}
        className="w-28"
        value={draft}
        onChange={(e) => setDraft(e.target.value)}
        onBlur={commit}
        onKeyDown={(e) => e.key === "Enter" && commit()}
      />
      {hint && <span className="text-xs text-muted-foreground">{hint}</span>}
    </div>
  )
}

function LimitsCard({ s }: { s: Settings }) {
  const update = useUpdateSettings()
  const local = useLocalModels()
  const [model, setModel] = useState("")
  const [cap, setCap] = useState("4")
  const overrides = Object.entries(s.model_caps)
  const installed = (local.data?.models ?? []).filter((m) => m.installed)

  return (
    <Card>
      <CardHeader>
        <CardTitle>Limits and throughput</CardTitle>
        <CardDescription>
          Caps on how many images one model makes in a single request. The server enforces them too, so a request over
          the cap is trimmed and the job says so.
        </CardDescription>
      </CardHeader>
      <CardContent className="flex flex-col gap-5">
        <div className="flex flex-wrap gap-6">
          <NumberField
            key={`c${s.max_images_cloud}`}
            label="OpenRouter, per model"
            value={s.max_images_cloud}
            min={1}
            max={100}
            onSave={(n) => update.mutate({ max_images_cloud: n })}
          />
          <NumberField
            key={`l${s.max_images_local}`}
            label="Local, per model"
            value={s.max_images_local}
            min={1}
            max={200}
            onSave={(n) => update.mutate({ max_images_local: n })}
          />
          <NumberField
            key={`p${s.cloud_concurrency}`}
            label="OpenRouter requests in parallel"
            value={s.cloud_concurrency}
            min={1}
            max={16}
            onSave={(n) => update.mutate({ cloud_concurrency: n })}
            hint="Halves itself on a rate limit, then climbs back."
          />
        </div>

        <div className="flex flex-col gap-2">
          <Label>Per-model caps</Label>
          {overrides.length === 0 && <span className="text-xs text-muted-foreground">None. Every model uses the caps above.</span>}
          {overrides.map(([id, n]) => (
            <div key={id} className="flex items-center gap-2 text-xs">
              <code className="min-w-0 flex-1 truncate">{id}</code>
              <span className="tabular-nums">{n} images</span>
              <Button
                size="xs"
                variant="ghost"
                onClick={() => {
                  const next = { ...s.model_caps }
                  delete next[id]
                  update.mutate({ model_caps: next })
                }}
              >
                Remove
              </Button>
            </div>
          ))}
          <form
            className="flex flex-wrap gap-2"
            onSubmit={(e) => {
              e.preventDefault()
              const n = Number(cap)
              if (!model.trim() || !(n >= 1)) return
              update.mutate({ model_caps: { ...s.model_caps, [model.trim()]: Math.round(n) } })
              setModel("")
            }}
          >
            <Input
              placeholder="Model id, e.g. openai/gpt-image-2 or local/zimage-turbo"
              value={model}
              onChange={(e) => setModel(e.target.value)}
              className="min-w-64 flex-1"
              list="cap-models"
            />
            <datalist id="cap-models">
              {[...s.image_models, ...installed.map((m) => m.ref)].map((m) => (
                <option key={m} value={m} />
              ))}
            </datalist>
            <Input type="number" min={1} max={200} value={cap} onChange={(e) => setCap(e.target.value)} className="w-24" />
            <Button type="submit" variant="outline" disabled={!model.trim()}>
              Add cap
            </Button>
          </form>
        </div>

        <div className="flex flex-col gap-2">
          <Label>Local batch sizes</Label>
          {installed.length === 0 ? (
            <span className="text-xs text-muted-foreground">No local models installed.</span>
          ) : (
            installed.map((m) => (
              <div key={m.id} className="flex items-center gap-2 text-xs">
                <span className="min-w-0 flex-1 truncate">{m.name}</span>
                <span className="tabular-nums">
                  {m.batch} at a time ({m.batch_source === "calibrated" ? "from a speed test" : "default"})
                </span>
                <Button asChild size="xs" variant="outline">
                  <Link to="/models" search={{ tab: "local" }}>
                    Speed test
                  </Link>
                </Button>
              </div>
            ))
          )}
        </div>
      </CardContent>
    </Card>
  )
}

function SpendCard() {
  const spend = useSpend()
  const kinds = Object.entries(spend.data?.by_kind ?? {})
  const label: Record<string, string> = { chat: "Designer chat", prompts: "Prompt writing", image: "Images" }
  return (
    <Card>
      <CardHeader>
        <CardTitle>Spend</CardTitle>
        <CardDescription>
          Everything logo-gen has spent from this folder, from OpenRouter's own usage numbers. Local models cost $0.
        </CardDescription>
      </CardHeader>
      <CardContent className="flex flex-col gap-2 text-xs">
        <SpendLine spend={spend.data} label="Total" />
        {kinds.map(([k, v]) => (
          <div key={k} className="flex gap-2">
            <span className="w-32 text-muted-foreground">{label[k] ?? k}</span>
            <span className="tabular-nums">
              {formatUsd(v.credits + v.byok, 4)}
              {v.byok > 0 && ` (${formatUsd(v.byok, 4)} BYOK)`} · {v.calls} call{v.calls === 1 ? "" : "s"}
            </span>
          </div>
        ))}
        <span className="text-muted-foreground">
          BYOK is what a provider billed on your own key through OpenRouter (for example an Anthropic key added to your
          OpenRouter account). OpenRouter itself charges no credits for those calls.
        </span>
      </CardContent>
    </Card>
  )
}
