import { useEffect, useRef, useState } from "react"
import { createFileRoute } from "@tanstack/react-router"
import { useQueryClient } from "@tanstack/react-query"
import { BrainIcon, Loader2Icon, SendIcon, SquareIcon, WandSparklesIcon } from "lucide-react"
import { toast } from "sonner"

import { ImageDialog } from "@/components/image-dialog"
import { ImageGrid } from "@/components/image-grid"
import { JobProgress } from "@/components/job-progress"
import { Markdown } from "@/components/markdown"
import { ImageModelPicker } from "@/components/model-picker"
import { ReferenceImages, useReferenceUploads } from "@/components/reference-images"
import { Button } from "@/components/ui/button"
import { Checkbox } from "@/components/ui/checkbox"
import { Empty, EmptyDescription, EmptyHeader, EmptyTitle } from "@/components/ui/empty"
import { Textarea } from "@/components/ui/textarea"
import { api, fileUrl, streamChat, type Chat, type ChatMessage, type Prompt } from "@/lib/api"
import { useCostEstimate } from "@/lib/cost"
import { formatUsd } from "@/lib/format"
import { keys, useGenerate, useImages, useJob, useSettings, useUpdateSettings } from "@/lib/queries"
import { cn } from "@/lib/utils"
import { useQuery } from "@tanstack/react-query"

export const Route = createFileRoute("/designer/$chatId")({
  component: ChatPage,
})

interface Pending {
  user: { content: string; images: string[] }
  reply: string
  reasoning: string
}

const STARTERS = [
  "An icon for polytini, a small Python library for polygon geometry.",
  "A mark for a neighborhood coffee roaster. Warm, a bit retro.",
  "App icon for a privacy-first notes app. Calm, minimal.",
]

function ChatPage() {
  const { chatId } = Route.useParams()
  const navigate = Route.useNavigate()
  const qc = useQueryClient()
  const isNew = chatId === "new"
  const [pending, setPending] = useState<Pending | null>(null)
  const [input, setInput] = useState("")
  const [jobFor, setJobFor] = useState<{ chat: string; id: string } | null>(null)
  const jobId = jobFor?.chat === chatId ? jobFor.id : null
  const setJobId = (id: string) => setJobFor({ chat: chatId, id })
  const [openImage, setOpenImage] = useState<string | null>(null)
  const abort = useRef<AbortController | null>(null)
  const bottom = useRef<HTMLDivElement>(null)
  const refs = useReferenceUploads()

  const chat = useQuery({
    queryKey: keys.chat(chatId),
    queryFn: () => api.get<Chat>(`/chats/${chatId}`),
    enabled: !isNew && !pending,
  })
  const results = useImages({ chat_id: chatId }, !isNew)
  const job = useJob(jobId)

  useEffect(() => {
    bottom.current?.scrollIntoView({ block: "end" })
  }, [chat.data?.messages.length, pending?.reply, pending?.reasoning.length])

  const messages = chat.data?.messages ?? []
  const allRefs = messages.flatMap((m) => m.images)

  const send = async (text: string) => {
    const content = text.trim()
    if (!content || pending) return
    let id = chatId
    if (isNew) {
      const created = await api.post<Chat>("/chats", {})
      id = created.id
      qc.setQueryData(keys.chat(id), { ...created, messages: [] })
    }
    const images = refs.refs.map((r) => r.path)
    setInput("")
    refs.clear()
    setPending({ user: { content, images }, reply: "", reasoning: "" })
    if (isNew) navigate({ params: { chatId: id }, replace: true })

    const ctrl = new AbortController()
    abort.current = ctrl
    try {
      await streamChat(
        id,
        { content, images },
        (ev) => {
          if (ev.type === "token") setPending((p) => p && { ...p, reply: p.reply + ev.content })
          else if (ev.type === "reasoning") setPending((p) => p && { ...p, reasoning: p.reasoning + ev.content })
          else if (ev.type === "error") toast.error(ev.message)
        },
        ctrl.signal,
      )
    } catch (e) {
      if ((e as Error).name !== "AbortError") toast.error((e as Error).message)
    } finally {
      abort.current = null
      await qc.refetchQueries({ queryKey: keys.chat(id) })
      qc.invalidateQueries({ queryKey: keys.chats })
      setPending(null)
    }
  }

  return (
    <div className="grid min-h-0 grid-cols-[minmax(0,1fr)_minmax(280px,38%)]">
      <section className="flex min-h-0 flex-col border-r">
        <div className="min-h-0 flex-1 overflow-y-auto">
          <div className="mx-auto flex max-w-3xl flex-col gap-5 p-6">
            {messages.length === 0 && !pending && (
              <Empty>
                <EmptyHeader>
                  <EmptyTitle>Design a logo by talking it through</EmptyTitle>
                  <EmptyDescription>
                    Describe what it is for. The designer asks a few questions or goes straight to directions, and you
                    pick which ones to render.
                  </EmptyDescription>
                </EmptyHeader>
                <div className="flex flex-col gap-2">
                  {STARTERS.map((s) => (
                    <Button key={s} variant="outline" size="sm" className="h-auto py-2 whitespace-normal" onClick={() => send(s)}>
                      {s}
                    </Button>
                  ))}
                </div>
              </Empty>
            )}
            {messages.map((m) => (
              <Message
                key={m.id}
                msg={m}
                chatId={chatId}
                references={allRefs}
                onJob={setJobId}
                busy={job.data?.status === "running"}
              />
            ))}
            {pending && (
              <>
                <UserBubble content={pending.user.content} images={pending.user.images} />
                <div className="flex flex-col gap-2">
                  {pending.reasoning && !pending.reply && (
                    <details className="text-xs text-muted-foreground" open>
                      <summary className="flex cursor-pointer items-center gap-1.5">
                        <BrainIcon className="size-3.5 animate-pulse" /> Thinking
                      </summary>
                      <p className="mt-1 max-h-32 overflow-hidden whitespace-pre-wrap opacity-70">
                        {pending.reasoning.slice(-600)}
                      </p>
                    </details>
                  )}
                  {pending.reply ? (
                    <Markdown>{pending.reply.replace(/```json[\s\S]*$/, "")}</Markdown>
                  ) : (
                    !pending.reasoning && <Loader2Icon className="size-4 animate-spin text-muted-foreground" />
                  )}
                  {pending.reply.includes("```json") && (
                    <span className="flex items-center gap-1.5 text-xs text-muted-foreground">
                      <Loader2Icon className="size-3 animate-spin" /> Writing prompts
                    </span>
                  )}
                </div>
              </>
            )}
            <div ref={bottom} />
          </div>
        </div>
        <div className="border-t p-3">
          <div className="mx-auto flex max-w-3xl flex-col gap-2">
            <Textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault()
                  send(input)
                }
              }}
              onPaste={(e) => {
                const files = Array.from(e.clipboardData.files)
                if (files.length) refs.add(files)
              }}
              placeholder={messages.length ? "Feedback, or ask for new directions" : "Describe the brand or project"}
              rows={2}
              className="max-h-48 text-sm"
            />
            <div className="flex items-center justify-between">
              <ReferenceImages
                refs={refs.refs}
                uploading={refs.uploading}
                onAdd={refs.add}
                onRemove={(p) => refs.setRefs((r) => r.filter((x) => x.path !== p))}
              />
              {pending ? (
                <Button size="sm" variant="outline" onClick={() => abort.current?.abort()}>
                  <SquareIcon />
                  Stop
                </Button>
              ) : (
                <Button size="sm" onClick={() => send(input)} disabled={!input.trim()}>
                  <SendIcon />
                  Send
                </Button>
              )}
            </div>
          </div>
        </div>
      </section>

      <aside className="flex min-h-0 flex-col gap-4 overflow-y-auto p-4">
        <h2 className="font-heading text-sm font-semibold">Results</h2>
        {job.data && <JobProgress job={job.data} />}
        {results.data?.length || job.data?.status === "running" ? (
          <ImageGrid
            images={results.data ?? []}
            pending={job.data?.status === "running" ? Math.max(0, job.data.total - job.data.image_ids.length) : 0}
            onOpen={setOpenImage}
            className="grid-cols-[repeat(auto-fill,minmax(140px,1fr))]"
          />
        ) : (
          <p className="text-xs text-muted-foreground">
            When the designer proposes directions, tick the ones you like and press Generate. Images land here.
          </p>
        )}
      </aside>
      <ImageDialog imageId={openImage} onOpenChange={(o) => !o && setOpenImage(null)} onSelect={setOpenImage} />
    </div>
  )
}

function UserBubble({ content, images }: { content: string; images: string[] }) {
  return (
    <div className="ml-auto flex max-w-[85%] flex-col items-end gap-2">
      {images.length > 0 && (
        <div className="flex gap-1">
          {images.map((p) => (
            <img key={p} src={fileUrl(p)} alt="" className="size-14 border object-cover" />
          ))}
        </div>
      )}
      <div className="bg-muted px-3 py-2 text-sm whitespace-pre-wrap">{content}</div>
    </div>
  )
}

function Message({
  msg,
  chatId,
  references,
  onJob,
  busy,
}: {
  msg: ChatMessage
  chatId: string
  references: string[]
  onJob: (id: string) => void
  busy: boolean
}) {
  if (msg.role === "user") return <UserBubble content={msg.content} images={msg.images} />
  return (
    <div className="flex flex-col gap-3">
      <Markdown>{msg.content}</Markdown>
      {msg.prompts.length > 0 && (
        <PromptPicker prompts={msg.prompts} chatId={chatId} references={references} onJob={onJob} busy={busy} />
      )}
    </div>
  )
}

function PromptPicker({
  prompts,
  chatId,
  references,
  onJob,
  busy,
}: {
  prompts: Prompt[]
  chatId: string
  references: string[]
  onJob: (id: string) => void
  busy: boolean
}) {
  const [selected, setSelected] = useState<Set<number>>(() => new Set(prompts.map((_, i) => i)))
  const settings = useSettings()
  const update = useUpdateSettings()
  const generate = useGenerate()
  const s = settings.data?.settings
  const estimate = useCostEstimate(s?.image_models ?? [], selected.size * (s?.images_per_prompt ?? 1))

  const toggle = (i: number) =>
    setSelected((prev) => {
      const next = new Set(prev)
      if (next.has(i)) next.delete(i)
      else next.add(i)
      return next
    })

  return (
    <div className="flex flex-col gap-3 border p-3">
      <div className="grid gap-2">
        {prompts.map((p, i) => (
          <label
            key={i}
            className={cn("flex cursor-pointer gap-3 p-2 hover:bg-muted", selected.has(i) && "bg-muted/60")}
          >
            <Checkbox checked={selected.has(i)} onCheckedChange={() => toggle(i)} className="mt-0.5" />
            <div className="flex min-w-0 flex-col gap-0.5">
              <span className="text-xs font-medium">
                {p.concept}
                {p.style && <span className="font-normal text-muted-foreground"> · {p.style}</span>}
              </span>
              <span className="text-xs text-muted-foreground">{p.prompt}</span>
            </div>
          </label>
        ))}
      </div>
      <div className="flex flex-wrap items-center gap-2">
        <ImageModelPicker
          className="min-w-0 flex-1"
          value={s?.image_models ?? []}
          onChange={(v) => update.mutate({ image_models: v })}
        />
        <Button
          size="sm"
          disabled={!selected.size || generate.isPending || busy || !s?.image_models.length}
          onClick={() =>
            generate.mutate(
              {
                prompts: prompts.filter((_, i) => selected.has(i)),
                models: s?.image_models,
                images_per_prompt: s?.images_per_prompt,
                references,
                chat_id: chatId,
              },
              { onSuccess: (j) => onJob(j.id) },
            )
          }
        >
          {generate.isPending ? <Loader2Icon className="animate-spin" /> : <WandSparklesIcon />}
          Generate {estimate.images}
        </Button>
      </div>
      {estimate.cost !== null && estimate.cost > 0 && (
        <span className="text-[11px] text-muted-foreground">About {formatUsd(estimate.cost)}</span>
      )}
    </div>
  )
}
