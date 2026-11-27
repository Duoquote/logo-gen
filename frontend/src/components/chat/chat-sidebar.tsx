import { useMemo, useRef, useState } from "react"
import { Link, useNavigate, useParams } from "@tanstack/react-router"
import { ImageIcon, Loader2Icon, MoreHorizontalIcon, PencilIcon, PinIcon, PinOffIcon, PlusIcon, SearchIcon, Trash2Icon } from "lucide-react"
import { toast } from "sonner"

import { Button } from "@/components/ui/button"
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from "@/components/ui/dropdown-menu"
import { Input } from "@/components/ui/input"
import {
  Sidebar,
  SidebarContent,
  SidebarGroup,
  SidebarGroupLabel,
  SidebarHeader,
  SidebarMenu,
  SidebarMenuAction,
  SidebarMenuItem,
  SidebarMenuSkeleton,
  useSidebar,
} from "@/components/ui/sidebar"
import type { ChatSummary } from "@/lib/api"
import { formatUsd, timeAgo } from "@/lib/format"
import { useChats, useDeleteChat, useRestoreChat, useUpdateChat } from "@/lib/queries"
import { cn } from "@/lib/utils"

const DAY = 86400

function groupChats(chats: ChatSummary[]): [string, ChatSummary[]][] {
  const startOfToday = new Date().setHours(0, 0, 0, 0) / 1000
  const groups: Record<string, ChatSummary[]> = { Pinned: [], Today: [], "Previous 7 days": [], Older: [] }
  for (const c of chats) {
    if (c.pinned) groups.Pinned.push(c)
    else if (c.updated_at >= startOfToday) groups.Today.push(c)
    else if (c.updated_at >= startOfToday - 7 * DAY) groups["Previous 7 days"].push(c)
    else groups.Older.push(c)
  }
  return Object.entries(groups).filter(([, v]) => v.length)
}

export function ChatSidebar() {
  const [q, setQ] = useState("")
  const chats = useChats(q.trim())
  const params = useParams({ strict: false }) as { chatId?: string }
  const navigate = useNavigate()
  const del = useDeleteChat()
  const restore = useRestoreChat()
  const { isMobile, setOpenMobile } = useSidebar()
  const listRef = useRef<HTMLDivElement>(null)
  const groups = useMemo(() => groupChats(chats.data ?? []), [chats.data])

  const remove = (c: ChatSummary) =>
    del.mutate(c.id, {
      onSuccess: () => {
        if (params.chatId === c.id) navigate({ to: "/designer/$chatId", params: { chatId: "new" } })
        toast(`Deleted "${c.title || "Untitled"}"`, {
          action: { label: "Undo", onClick: () => restore.mutate(c.id) },
          duration: 8000,
        })
      },
    })

  // Arrow keys move between chats, like a list.
  const onKeyDown = (e: React.KeyboardEvent) => {
    if (e.key !== "ArrowDown" && e.key !== "ArrowUp") return
    const links = Array.from(listRef.current?.querySelectorAll<HTMLElement>("[data-chat-link]") ?? [])
    if (!links.length) return
    e.preventDefault()
    const i = links.indexOf(document.activeElement as HTMLElement)
    const next = e.key === "ArrowDown" ? Math.min(links.length - 1, i + 1) : Math.max(0, i < 0 ? 0 : i - 1)
    links[next]?.focus()
  }

  return (
    <Sidebar className="top-12 h-[calc(100svh-3rem)]">
      <SidebarHeader className="gap-2">
        <Button asChild variant="outline" size="sm" className="w-full">
          <Link to="/designer/$chatId" params={{ chatId: "new" }} onClick={() => isMobile && setOpenMobile(false)}>
            <PlusIcon />
            New chat
          </Link>
        </Button>
        <div className="relative">
          <SearchIcon className="pointer-events-none absolute top-1/2 left-2 size-3.5 -translate-y-1/2 text-muted-foreground" />
          <Input
            value={q}
            onChange={(e) => setQ(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "ArrowDown") {
                e.preventDefault()
                listRef.current?.querySelector<HTMLElement>("[data-chat-link]")?.focus()
              }
              if (e.key === "Escape") setQ("")
            }}
            placeholder="Search chats"
            aria-label="Search chats"
            className="h-8 pl-7 text-xs"
          />
        </div>
      </SidebarHeader>
      <SidebarContent ref={listRef} onKeyDown={onKeyDown}>
        {chats.isPending && (
          <SidebarGroup>
            {Array.from({ length: 5 }, (_, i) => (
              <SidebarMenuSkeleton key={i} showIcon />
            ))}
          </SidebarGroup>
        )}
        {chats.isError && (
          <div className="flex flex-col items-start gap-2 p-3 text-xs text-muted-foreground">
            Couldn't load chats.
            <Button size="xs" variant="outline" onClick={() => chats.refetch()}>
              Retry
            </Button>
          </div>
        )}
        {chats.data && chats.data.length === 0 && (
          <p className="px-4 py-2 text-xs text-muted-foreground">{q ? "No chats match." : "No chats yet."}</p>
        )}
        {groups.map(([label, items]) => (
          <SidebarGroup key={label} className="py-1">
            <SidebarGroupLabel>{label}</SidebarGroupLabel>
            <SidebarMenu>
              {items.map((c) => (
                <ChatItem
                  key={c.id}
                  chat={c}
                  active={params.chatId === c.id}
                  onDelete={() => remove(c)}
                  onNavigate={() => isMobile && setOpenMobile(false)}
                />
              ))}
            </SidebarMenu>
          </SidebarGroup>
        ))}
      </SidebarContent>
    </Sidebar>
  )
}

function ChatItem({
  chat,
  active,
  onDelete,
  onNavigate,
}: {
  chat: ChatSummary
  active: boolean
  onDelete: () => void
  onNavigate: () => void
}) {
  const update = useUpdateChat()
  const [editing, setEditing] = useState(false)
  const [title, setTitle] = useState(chat.title)

  const save = () => {
    setEditing(false)
    const t = title.trim()
    if (t && t !== chat.title) update.mutate({ id: chat.id, title: t })
    else setTitle(chat.title)
  }

  return (
    <SidebarMenuItem>
      {editing ? (
        <Input
          autoFocus
          value={title}
          onChange={(e) => setTitle(e.target.value)}
          onBlur={save}
          onKeyDown={(e) => {
            if (e.key === "Enter") save()
            if (e.key === "Escape") {
              setTitle(chat.title)
              setEditing(false)
            }
          }}
          aria-label="Chat title"
          className="h-8 text-xs"
        />
      ) : (
        <Link
          to="/designer/$chatId"
          params={{ chatId: chat.id }}
          data-chat-link
          data-active={active}
          onClick={onNavigate}
          onKeyDown={(e) => {
            if (e.key === "F2") {
              e.preventDefault()
              setEditing(true)
            }
            if (e.key === "Delete") {
              e.preventDefault()
              onDelete()
            }
          }}
          className={cn(
            "flex w-full items-center gap-2 p-1.5 pr-8 text-left outline-hidden ring-sidebar-ring hover:bg-sidebar-accent focus-visible:ring-2",
            active && "bg-sidebar-accent font-medium",
          )}
        >
          <span className="flex size-9 shrink-0 items-center justify-center overflow-hidden border bg-background">
            {chat.thumbnail ? (
              <img src={chat.thumbnail} alt="" className="size-full object-contain" loading="lazy" />
            ) : (
              <ImageIcon className="size-3.5 text-muted-foreground" />
            )}
          </span>
          <span className="flex min-w-0 flex-1 flex-col">
            <span className="truncate text-xs">{chat.title || "Untitled"}</span>
            <span className="flex items-center gap-1 text-[10px] text-muted-foreground tabular-nums">
              {chat.streaming && <Loader2Icon className="size-2.5 animate-spin" aria-label="Answering" />}
              {timeAgo(chat.updated_at)}
              {chat.spend > 0 && <span>· {formatUsd(chat.spend, 3)}</span>}
            </span>
          </span>
        </Link>
      )}
      {!editing && (
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <SidebarMenuAction showOnHover aria-label={`Actions for ${chat.title || "chat"}`} className="top-3">
              <MoreHorizontalIcon />
            </SidebarMenuAction>
          </DropdownMenuTrigger>
          <DropdownMenuContent side="right" align="start">
            <DropdownMenuItem onSelect={() => setEditing(true)}>
              <PencilIcon />
              Rename
            </DropdownMenuItem>
            <DropdownMenuItem onSelect={() => update.mutate({ id: chat.id, pinned: !chat.pinned })}>
              {chat.pinned ? <PinOffIcon /> : <PinIcon />}
              {chat.pinned ? "Unpin" : "Pin"}
            </DropdownMenuItem>
            <DropdownMenuItem variant="destructive" onSelect={onDelete}>
              <Trash2Icon />
              Delete
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      )}
    </SidebarMenuItem>
  )
}
