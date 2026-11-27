import { useEffect, useRef, useState } from "react"
import { createFileRoute, Link } from "@tanstack/react-router"
import { ArrowDownIcon, MessageSquareDashedIcon, RefreshCwIcon, SlidersHorizontalIcon } from "lucide-react"
import { toast } from "sonner"

import { AssistantMessage, Directions, UserMessage, type MessageHandlers } from "@/components/chat/message"
import { Composer, type ComposerHandle } from "@/components/chat/composer"
import { Outputs } from "@/components/chat/outputs"
import { ImageDialog } from "@/components/image-dialog"
import { LearnedButton } from "@/components/learned-panel"
import { SpendLine } from "@/components/spend"
import { Button } from "@/components/ui/button"
import {
  DropdownMenu,
  DropdownMenuCheckboxItem,
  DropdownMenuContent,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import { Empty, EmptyContent, EmptyDescription, EmptyHeader, EmptyMedia, EmptyTitle } from "@/components/ui/empty"
import { Input } from "@/components/ui/input"
import { SidebarTrigger } from "@/components/ui/sidebar"
import { Skeleton } from "@/components/ui/skeleton"
import { useLiveReply } from "@/hooks/use-live-reply"
import { useStickToBottom } from "@/hooks/use-stick-to-bottom"
import { api, ApiError, fileUrl, type Chat, type ChatMessage, type ChatOutput, type Prompt } from "@/lib/api"
import {
  openChat,
  useChat,
  useDeleteMessage,
  useFeedback,
  useGenerate,
  useRegenerate,
  useRestoreMessage,
  useSendMessage,
  useSetBranch,
  useSettings,
  useStopReply,
  useUpdateChat,
} from "@/lib/queries"

export const Route = createFileRoute("/designer/$chatId")({
  component: ChatPage,
})

const EXAMPLES = [
  "An icon for polytini, a small Python library for polygon geometry.",
  "A mark for a neighborhood coffee roaster. Warm, a bit retro.",
  "App icon for a privacy-first notes app. Calm, minimal.",
  "A wordmark for Fieldnote, a tool for birdwatchers. No birds that look like Twitter.",
]

function ChatPage() {
  const { chatId } = Route.useParams()
  const navigate = Route.useNavigate()
  const isNew = chatId === "new"
  const chat = useChat(chatId, !isNew)
  const settings = useSettings()
  const send = useSendMessage()
  const regenerate = useRegenerate()
  const stop = useStopReply()
  const setBranch = useSetBranch()
  const delMessage = useDeleteMessage()
  const restoreMessage = useRestoreMessage()
  const generate = useGenerate()
  const feedback = useFeedback()
  const composer = useRef<ComposerHandle>(null)
  const [openImage, setOpenImage] = useState<string | null>(null)
  const { scrollRef, contentRef, atBottom, toBottom } = useStickToBottom<HTMLDivElement, HTMLDivElement>()

  const data = isNew ? undefined : chat.data
  const messages = data?.messages ?? []
  const streamingMsg = messages.find((m) => m.status === "streaming")
  const live = useLiveReply(chatId, streamingMsg)
  const busy = !!data?.running || !!streamingMsg
  const notFound = chat.error instanceof ApiError && chat.error.status === 404

  // Remember this chat, so /designer opens it next time (from any browser).
  const loaded = !!data
  useEffect(() => {
    if (!isNew && loaded) openChat(chatId)
  }, [chatId, isNew, loaded])

  // A new chat, or a different one: start at the bottom.
  useEffect(() => {
    toBottom()
  }, [chatId, loaded, toBottom])

  const onSend = async (text: string, images: string[]) => {
    try {
      let id = chatId
      if (isNew) id = (await api.post<Chat>("/chats", {})).id
      await send.mutateAsync({ chatId: id, content: text, images })
      if (isNew) navigate({ to: "/designer/$chatId", params: { chatId: id }, replace: true })
      toBottom()
      return true
    } catch {
      return false // the mutation already showed the error
    }
  }

  const h: MessageHandlers = {
    busy,
    onEdit: (msg, text) => send.mutate({ chatId, content: text, images: msg.images, edit_of: msg.id }),
    onRegenerate: (msg) => regenerate.mutate({ chatId, messageId: msg.id }),
    onSwitch: (id) => setBranch.mutate({ chatId, messageId: id }),
    onDelete: (msg) =>
      delMessage.mutate(
        { chatId, messageId: msg.id },
        {
          onSuccess: ({ stamp }) =>
            toast("Deleted from here", {
              description: "That message and everything after it, in every branch.",
              action: { label: "Undo", onClick: () => restoreMessage.mutate({ chatId, messageId: msg.id, stamp }) },
              duration: 8000,
            }),
        },
      ),
  }

  const refsInChat = messages.flatMap((m) => m.images)
  const runGenerate = (msg: ChatMessage, prompts: Prompt[], per: number) =>
    generate.mutate(
      {
        prompts,
        models: settings.data?.settings.image_models,
        images_per_prompt: per,
        references: refsInChat,
        chat_id: chatId,
        message_id: msg.id,
      },
      { onSuccess: () => chat.refetch() },
    )

  const useDirection = (img: ChatOutput) => {
    feedback.mutate({ kind: "pick", image_id: img.id, chat_id: chatId })
    composer.current?.addRefs([{ path: img.path, url: fileUrl(img.path), name: img.concept || "picked" }])
    composer.current?.setText(
      `Let's go with "${img.concept || "this one"}" (attached). Keep what works and give me a few tighter takes on it.`,
    )
  }

  const lastAssistant = [...messages].reverse().find((m) => m.role === "assistant")
  const orphanJobs = (data?.jobs ?? []).filter((j) => j.message_id == null || !messages.some((m) => m.id === j.message_id))
  const last = messages.at(-1)

  return (
    <div className="flex h-full min-h-0 flex-col">
      <header className="flex h-11 shrink-0 items-center gap-2 border-b px-2 sm:px-3">
        <SidebarTrigger aria-label="Toggle chat list" />
        {data ? <ChatTitle chat={data} /> : <span className="flex-1 text-xs text-muted-foreground">{isNew ? "New chat" : ""}</span>}
        {data && (
          <span className="hidden shrink-0 sm:inline">
            <SpendLine spend={data.spend} label="This chat" />
          </span>
        )}
        {data && <LearnedButton chatId={data.id} />}
        {data && <ChatOptions chat={data} />}
      </header>

      <div className="relative min-h-0 flex-1">
        <div ref={scrollRef} className="h-full overflow-y-auto">
          <div ref={contentRef} className="mx-auto flex max-w-3xl flex-col gap-6 px-3 py-5 sm:px-6">
            {!isNew && chat.isPending && <ThreadSkeleton />}
            {notFound && (
              <Empty>
                <EmptyHeader>
                  <EmptyMedia variant="icon">
                    <MessageSquareDashedIcon />
                  </EmptyMedia>
                  <EmptyTitle>This chat doesn't exist</EmptyTitle>
                  <EmptyDescription>It may have been deleted.</EmptyDescription>
                </EmptyHeader>
                <EmptyContent>
                  <Button asChild size="sm">
                    <Link to="/designer/$chatId" params={{ chatId: "new" }}>
                      Start a new chat
                    </Link>
                  </Button>
                </EmptyContent>
              </Empty>
            )}
            {chat.isError && !notFound && (
              <div role="alert" className="flex items-center gap-2 border p-3 text-xs">
                <span className="flex-1">Couldn't load this chat: {chat.error.message}</span>
                <Button size="xs" variant="outline" onClick={() => chat.refetch()}>
                  <RefreshCwIcon />
                  Retry
                </Button>
              </div>
            )}
            {(isNew || (data && messages.length === 0)) && (
              <Empty className="border-0">
                <EmptyHeader>
                  <EmptyTitle>Design a logo by talking it through</EmptyTitle>
                  <EmptyDescription>
                    Say what it's for. The designer asks a couple of questions or goes straight to directions. You tick the
                    ones you like and render them right here.
                  </EmptyDescription>
                </EmptyHeader>
                <EmptyContent className="grid gap-2 sm:grid-cols-2">
                  {EXAMPLES.map((ex) => (
                    <Button
                      key={ex}
                      variant="outline"
                      size="sm"
                      className="h-auto justify-start py-2 text-left whitespace-normal"
                      onClick={() => onSend(ex, [])}
                      disabled={send.isPending}
                    >
                      {ex}
                    </Button>
                  ))}
                </EmptyContent>
              </Empty>
            )}
            {messages.map((m, i) =>
              m.role === "user" ? (
                <UserMessage key={m.id} msg={m} h={h} />
              ) : (
                <AssistantMessage
                  key={m.id}
                  msg={m}
                  live={m.id === streamingMsg?.id ? live : null}
                  isLast={i === messages.length - 1}
                  h={h}
                  directions={
                    m.prompts.length > 0 && m.status !== "streaming" ? (
                      <Directions
                        prompts={m.prompts}
                        busy={generate.isPending}
                        rendered={(data?.jobs ?? []).some((j) => j.message_id === m.id)}
                        onGenerate={(p, per) => runGenerate(m, p, per)}
                      />
                    ) : null
                  }
                >
                  <Outputs
                    chatId={chatId}
                    messageId={m.id}
                    images={m.outputs}
                    jobs={[
                      ...(data?.jobs ?? []).filter((j) => j.message_id === m.id),
                      ...(m.id === lastAssistant?.id ? orphanJobs : []),
                    ]}
                    onOpen={setOpenImage}
                    onUseDirection={useDirection}
                  />
                </AssistantMessage>
              ),
            )}
            {last?.role === "user" && !busy && (
              <div role="alert" className="flex items-center gap-2 border p-2 text-xs text-muted-foreground">
                <span className="flex-1">No reply to this message.</span>
                <Button size="xs" variant="outline" onClick={() => h.onRegenerate(last)}>
                  <RefreshCwIcon />
                  Get a reply
                </Button>
              </div>
            )}
          </div>
        </div>
        {!atBottom && (
          <Button
            size="sm"
            variant="outline"
            className="absolute bottom-3 left-1/2 -translate-x-1/2 bg-background shadow-sm"
            onClick={() => toBottom(true)}
          >
            <ArrowDownIcon />
            Jump to latest
          </Button>
        )}
      </div>

      <div className="shrink-0 border-t p-2 sm:p-3">
        <div className="mx-auto max-w-3xl">
          <Composer
            ref={composer}
            placeholder={messages.length ? "Feedback, or ask for new directions" : "Describe the brand or project"}
            busy={busy}
            sending={send.isPending}
            stopping={stop.isPending}
            onSend={onSend}
            onStop={() => data?.running && stop.mutate({ chatId, messageId: data.running })}
          />
        </div>
      </div>
      <ImageDialog imageId={openImage} onOpenChange={(o) => !o && setOpenImage(null)} onSelect={setOpenImage} />
    </div>
  )
}

function ChatTitle({ chat }: { chat: Chat }) {
  const update = useUpdateChat()
  const [editing, setEditing] = useState(false)
  const [title, setTitle] = useState(chat.title)
  if (editing) {
    const save = () => {
      setEditing(false)
      if (title.trim() && title.trim() !== chat.title) update.mutate({ id: chat.id, title: title.trim() })
    }
    return (
      <Input
        autoFocus
        value={title}
        onChange={(e) => setTitle(e.target.value)}
        onBlur={save}
        onKeyDown={(e) => {
          if (e.key === "Enter") save()
          if (e.key === "Escape") setEditing(false)
        }}
        aria-label="Chat title"
        className="h-7 flex-1 text-xs"
      />
    )
  }
  return (
    <button
      type="button"
      title="Rename"
      onClick={() => {
        setTitle(chat.title)
        setEditing(true)
      }}
      className="min-w-0 flex-1 truncate text-left text-xs font-medium hover:underline"
    >
      {chat.title || "Untitled"}
    </button>
  )
}

/** Per-chat designer options. Each switch defaults to the setting of the same name. */
function ChatOptions({ chat }: { chat: Chat }) {
  const update = useUpdateChat()
  const settings = useSettings()
  const asks = chat.ask_first ?? settings.data?.settings.ask_before_generating ?? false
  const auto = chat.auto_generate ?? settings.data?.settings.generate_automatically ?? false
  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button size="icon-sm" variant="ghost" className="shrink-0" aria-label="Chat options" title="Chat options">
          <SlidersHorizontalIcon />
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="w-72 max-w-[calc(100vw-1rem)]">
        <DropdownMenuCheckboxItem checked={asks} onCheckedChange={(v) => update.mutate({ id: chat.id, ask_first: v === true })}>
          Ask questions before generating
        </DropdownMenuCheckboxItem>
        <p className="px-2 pb-1.5 text-[11px] text-muted-foreground">
          Off: the designer proposes directions right away, then asks.
        </p>
        <DropdownMenuCheckboxItem
          checked={auto}
          onCheckedChange={(v) => update.mutate({ id: chat.id, auto_generate: v === true })}
        >
          Generate automatically
        </DropdownMenuCheckboxItem>
        <p className="px-2 pb-1.5 text-[11px] text-muted-foreground">
          On: directions render as soon as a reply arrives. Off: you pick them and press Generate.
        </p>
        <p className="border-t px-2 pt-1.5 pb-1 text-[11px] text-muted-foreground">
          These apply to this chat only. The defaults are in Settings.
        </p>
      </DropdownMenuContent>
    </DropdownMenu>
  )
}

function ThreadSkeleton() {
  return (
    <div className="flex flex-col gap-6" aria-label="Loading chat">
      <Skeleton className="ml-auto h-10 w-2/3" />
      <div className="flex flex-col gap-2">
        <Skeleton className="h-4 w-full" />
        <Skeleton className="h-4 w-5/6" />
        <Skeleton className="h-4 w-2/3" />
      </div>
    </div>
  )
}
