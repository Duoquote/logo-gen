import { useState } from "react"
import { Link } from "@tanstack/react-router"
import {
  AlertCircleIcon,
  BrainIcon,
  CheckIcon,
  ChevronDownIcon,
  ChevronLeftIcon,
  ChevronRightIcon,
  CopyIcon,
  ExternalLinkIcon,
  Loader2Icon,
  PencilIcon,
  RefreshCwIcon,
  Trash2Icon,
  WandSparklesIcon,
} from "lucide-react"

import { Markdown } from "@/components/markdown"
import { Button } from "@/components/ui/button"
import { Checkbox } from "@/components/ui/checkbox"
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible"
import { Textarea } from "@/components/ui/textarea"
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip"
import type { LiveReply } from "@/hooks/use-live-reply"
import { fileUrl, type ChatMessage, type Prompt } from "@/lib/api"
import { maxPerDirection } from "@/lib/caps"
import { useCostEstimate } from "@/lib/cost"
import { formatUsd, shortModel } from "@/lib/format"
import { useSettings } from "@/lib/queries"
import { cn } from "@/lib/utils"

/** Hide the JSON block of directions while it streams in; it becomes the picker when done. */
const visibleText = (s: string) => s.replace(/```json[\s\S]*$/, "").trimEnd()

function IconAction({ label, onClick, children, disabled }: { label: string; onClick: () => void; children: React.ReactNode; disabled?: boolean }) {
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <Button size="icon-xs" variant="ghost" aria-label={label} onClick={onClick} disabled={disabled}>
          {children}
        </Button>
      </TooltipTrigger>
      <TooltipContent>{label}</TooltipContent>
    </Tooltip>
  )
}

function CopyAction({ text }: { text: string }) {
  const [done, setDone] = useState(false)
  return (
    <IconAction
      label={done ? "Copied" : "Copy"}
      onClick={() =>
        navigator.clipboard.writeText(text).then(() => {
          setDone(true)
          setTimeout(() => setDone(false), 1200)
        })
      }
    >
      {done ? <CheckIcon /> : <CopyIcon />}
    </IconAction>
  )
}

/** "< 2 / 3 >" between edits of a message or regenerations of a reply. */
function BranchNav({ msg, onSwitch, disabled }: { msg: ChatMessage; onSwitch: (id: number) => void; disabled: boolean }) {
  const i = msg.siblings.indexOf(msg.id)
  if (msg.siblings.length < 2 || i < 0) return null
  return (
    <span className="flex items-center text-[11px] text-muted-foreground tabular-nums">
      <Button size="icon-xs" variant="ghost" aria-label="Previous version" disabled={disabled || i === 0} onClick={() => onSwitch(msg.siblings[i - 1])}>
        <ChevronLeftIcon />
      </Button>
      {i + 1} / {msg.siblings.length}
      <Button
        size="icon-xs"
        variant="ghost"
        aria-label="Next version"
        disabled={disabled || i === msg.siblings.length - 1}
        onClick={() => onSwitch(msg.siblings[i + 1])}
      >
        <ChevronRightIcon />
      </Button>
    </span>
  )
}

export interface MessageHandlers {
  busy: boolean // a reply is running in this chat
  onEdit: (msg: ChatMessage, text: string) => void
  onRegenerate: (msg: ChatMessage) => void
  onDelete: (msg: ChatMessage) => void
  onSwitch: (id: number) => void
}

export function UserMessage({ msg, h }: { msg: ChatMessage; h: MessageHandlers }) {
  const [editing, setEditing] = useState(false)
  const [text, setText] = useState(msg.content)

  if (editing) {
    const submit = () => {
      if (!text.trim()) return
      setEditing(false)
      h.onEdit(msg, text)
    }
    return (
      <div className="ml-auto flex w-full max-w-[85%] flex-col gap-2">
        <Textarea
          autoFocus
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault()
              submit()
            }
            if (e.key === "Escape") setEditing(false)
          }}
          className="field-sizing-content max-h-72 min-h-16 text-sm"
          aria-label="Edit message"
        />
        <div className="flex items-center justify-end gap-2">
          <span className="mr-auto text-[11px] text-muted-foreground">Sends as a new branch. The old one stays.</span>
          <Button size="sm" variant="ghost" onClick={() => setEditing(false)}>
            Cancel
          </Button>
          <Button size="sm" onClick={submit} disabled={!text.trim() || h.busy}>
            Send
          </Button>
        </div>
      </div>
    )
  }

  return (
    <div className="group/msg ml-auto flex max-w-[85%] flex-col items-end gap-1">
      {msg.images.length > 0 && (
        <div className="flex flex-wrap justify-end gap-1">
          {msg.images.map((p) => (
            <img key={p} src={fileUrl(p)} alt="Reference" className="size-16 border object-cover" />
          ))}
        </div>
      )}
      <div className="bg-muted px-3 py-2 text-sm break-words whitespace-pre-wrap">{msg.content}</div>
      <div className="flex items-center gap-0.5 opacity-100 transition-opacity focus-within:opacity-100 md:opacity-0 md:group-hover/msg:opacity-100">
        <BranchNav msg={msg} onSwitch={h.onSwitch} disabled={h.busy} />
        <CopyAction text={msg.content} />
        <IconAction
          label="Edit"
          disabled={h.busy}
          onClick={() => {
            setText(msg.content)
            setEditing(true)
          }}
        >
          <PencilIcon />
        </IconAction>
        <IconAction label="Delete from here" onClick={() => h.onDelete(msg)}>
          <Trash2Icon />
        </IconAction>
      </div>
    </div>
  )
}

function Thinking({ text, active }: { text: string; active: boolean }) {
  const [open, setOpen] = useState(false)
  if (!text && !active) return null
  return (
    <Collapsible open={open} onOpenChange={setOpen}>
      <CollapsibleTrigger asChild>
        <button type="button" className="flex items-center gap-1.5 text-xs text-muted-foreground hover:text-foreground">
          <BrainIcon className={cn("size-3.5", active && "animate-pulse")} />
          {active ? "Thinking" : "Thought it through"}
          <ChevronDownIcon className={cn("size-3 transition-transform", open && "rotate-180")} />
        </button>
      </CollapsibleTrigger>
      <CollapsibleContent>
        <p className="mt-1.5 max-h-64 overflow-y-auto border-l-2 pl-3 text-xs whitespace-pre-wrap text-muted-foreground">
          {text || "..."}
        </p>
      </CollapsibleContent>
    </Collapsible>
  )
}

export function AssistantMessage({
  msg,
  live,
  isLast,
  h,
  directions,
  children,
}: {
  msg: ChatMessage
  live: LiveReply | null
  isLast: boolean
  h: MessageHandlers
  directions?: React.ReactNode
  children?: React.ReactNode
}) {
  const streaming = msg.status === "streaming"
  const content = live ? live.content : msg.content
  const reasoning = live ? live.reasoning : msg.reasoning
  const status = live?.status ?? msg.status
  const error = live?.error || msg.error
  const text = streaming ? visibleText(content) : content
  const cost = msg.cost ? msg.cost.credits + msg.cost.byok : null

  return (
    <div className="group/msg flex flex-col gap-2" aria-busy={streaming}>
      <Thinking text={reasoning} active={streaming && !text} />
      {text && <Markdown>{text}</Markdown>}
      {streaming && !text && !reasoning && (
        <span className="flex items-center gap-1.5 text-xs text-muted-foreground">
          <Loader2Icon className="size-3.5 animate-spin" />
          Starting
        </span>
      )}
      {streaming && content.includes("```json") && (
        <span className="flex items-center gap-1.5 text-xs text-muted-foreground">
          <Loader2Icon className="size-3 animate-spin" /> Writing directions
        </span>
      )}
      {status === "error" && (
        <div role="alert" className="flex flex-wrap items-center gap-2 border border-destructive/40 p-2 text-xs text-destructive">
          <AlertCircleIcon className="size-3.5" />
          <span className="min-w-0 flex-1">{error || "Something went wrong."}</span>
          <Button size="xs" variant="outline" onClick={() => h.onRegenerate(msg)} disabled={h.busy}>
            <RefreshCwIcon />
            Retry
          </Button>
        </div>
      )}
      {status === "stopped" && <span className="text-xs text-muted-foreground">Stopped.</span>}
      {directions}
      {children}
      {!streaming && (
        <div className="flex items-center gap-0.5 opacity-100 transition-opacity focus-within:opacity-100 md:opacity-0 md:group-hover/msg:opacity-100">
          <BranchNav msg={msg} onSwitch={h.onSwitch} disabled={h.busy} />
          {content && <CopyAction text={content} />}
          {isLast && (
            <IconAction label="Regenerate" onClick={() => h.onRegenerate(msg)} disabled={h.busy}>
              <RefreshCwIcon />
            </IconAction>
          )}
          <IconAction label="Delete from here" onClick={() => h.onDelete(msg)}>
            <Trash2Icon />
          </IconAction>
          <span className="ml-1 text-[11px] text-muted-foreground tabular-nums">
            {msg.model && shortModel(msg.model)}
            {cost !== null && ` · ${cost > 0 ? formatUsd(cost, 4) : "$0"}`}
            {msg.cost && msg.cost.byok > 0 && " BYOK"}
          </span>
        </div>
      )}
    </div>
  )
}

/** Tick the directions to render. Models and the count come from the composer toolbar. */
export function Directions({
  prompts,
  busy,
  rendered = false,
  onGenerate,
}: {
  prompts: Prompt[]
  busy: boolean
  rendered?: boolean // a round for these directions already ran (or is running)
  onGenerate: (prompts: Prompt[], per: number) => void
}) {
  const [selected, setSelected] = useState<Set<number>>(() => new Set(prompts.map((_, i) => i)))
  const settings = useSettings()
  const s = settings.data?.settings
  const models = s?.image_models ?? []
  const per = Math.min(s?.images_per_prompt ?? 1, s ? maxPerDirection(s, models, Math.max(1, selected.size)) : 1)
  const estimate = useCostEstimate(models, selected.size * per)
  const total = selected.size * per * models.length

  const toggle = (i: number) =>
    setSelected((prev) => {
      const next = new Set(prev)
      if (next.has(i)) next.delete(i)
      else next.add(i)
      return next
    })

  return (
    <div className="flex flex-col gap-2 border p-2 sm:p-3">
      <div className="grid gap-1">
        {prompts.map((p, i) => (
          <div key={i} className={cn("flex items-start gap-1 hover:bg-muted", selected.has(i) && "bg-muted/60")}>
            <label className="flex min-w-0 flex-1 cursor-pointer gap-3 p-2">
              <Checkbox checked={selected.has(i)} onCheckedChange={() => toggle(i)} className="mt-0.5" />
              <span className="flex min-w-0 flex-col gap-0.5">
                <span className="text-xs font-medium">
                  {p.concept}
                  {p.style && <span className="font-normal text-muted-foreground"> · {p.style}</span>}
                </span>
                <span className="text-xs break-words text-muted-foreground">{p.base_prompt || p.prompt}</span>
              </span>
            </label>
            <PromptActions prompt={p} />
          </div>
        ))}
      </div>
      <div className="flex flex-wrap items-center gap-2">
        <span className="min-w-0 flex-1 text-[11px] text-muted-foreground">
          {models.length === 0
            ? "Pick an image model below the message box."
            : `${per} per direction on ${models.length === 1 ? shortModel(models[0]) : `${models.length} models`}`}
          {estimate.cost !== null && estimate.cost > 0 && ` · about ${formatUsd(estimate.cost)}`}
          {models.some((m) => m.startsWith("local/")) && " · local images are free"}
        </span>
        <Button
          size="sm"
          disabled={!selected.size || !models.length || busy}
          onClick={() => onGenerate(prompts.filter((_, i) => selected.has(i)), per)}
        >
          {busy ? <Loader2Icon className="animate-spin" /> : <WandSparklesIcon />}
          {rendered ? "Generate more" : "Generate"} {total || ""}
        </Button>
      </div>
    </div>
  )
}

/** Copy one direction's prompt, or open it in the Generate page. Outside the row's label, so they don't tick it. */
function PromptActions({ prompt }: { prompt: Prompt }) {
  const [done, setDone] = useState(false)
  const text = prompt.base_prompt || prompt.prompt
  const label = done ? "Copied" : `Copy the prompt for ${prompt.concept}`
  return (
    <div className="flex shrink-0 gap-0.5 p-1.5">
      <Tooltip>
        <TooltipTrigger asChild>
          <Button
            size="icon-xs"
            variant="ghost"
            aria-label={label}
            onClick={() =>
              navigator.clipboard.writeText(text).then(() => {
                setDone(true)
                setTimeout(() => setDone(false), 1500)
              })
            }
          >
            {done ? <CheckIcon /> : <CopyIcon />}
          </Button>
        </TooltipTrigger>
        <TooltipContent className="max-w-64">
          {done ? "Copied" : "Copies the base prompt; each model gets its own phrasing when generated here"}
        </TooltipContent>
      </Tooltip>
      <Tooltip>
        <TooltipTrigger asChild>
          <Button size="icon-xs" variant="ghost" aria-label={`Open ${prompt.concept} in Generate`} asChild>
            <Link to="/" search={{ prompt: text }}>
              <ExternalLinkIcon />
            </Link>
          </Button>
        </TooltipTrigger>
        <TooltipContent className="max-w-64">Open in Generate with this prompt, sent as written</TooltipContent>
      </Tooltip>
    </div>
  )
}
