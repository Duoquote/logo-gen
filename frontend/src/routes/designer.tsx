import { createFileRoute, Link, Outlet } from "@tanstack/react-router"
import { PlusIcon, Trash2Icon } from "lucide-react"

import { Button } from "@/components/ui/button"
import { timeAgo } from "@/lib/format"
import { useChats, useDeleteChat } from "@/lib/queries"

export const Route = createFileRoute("/designer")({
  component: DesignerLayout,
})

function DesignerLayout() {
  const chats = useChats()
  const del = useDeleteChat()
  const navigate = Route.useNavigate()
  const list = (chats.data ?? []).filter((c) => c.message_count > 0)

  return (
    <div className="grid h-full grid-cols-[220px_minmax(0,1fr)]">
      <aside className="flex min-h-0 flex-col border-r">
        <div className="p-3">
          <Button asChild variant="outline" size="sm" className="w-full">
            <Link to="/designer/$chatId" params={{ chatId: "new" }}>
              <PlusIcon />
              New chat
            </Link>
          </Button>
        </div>
        <div className="min-h-0 flex-1 overflow-y-auto">
          <nav className="flex flex-col px-2 pb-3">
            {list.length === 0 && <p className="px-2 text-xs text-muted-foreground">No chats yet.</p>}
            {list.map((c) => (
              <div key={c.id} className="group flex items-center">
                <Link
                  to="/designer/$chatId"
                  params={{ chatId: c.id }}
                  className="min-w-0 flex-1 px-2 py-1.5 text-xs hover:bg-muted data-[status=active]:bg-muted"
                >
                  <div className="truncate">{c.title || "Untitled"}</div>
                  <div className="text-[10px] text-muted-foreground">{timeAgo(c.updated_at)}</div>
                </Link>
                <Button
                  variant="ghost"
                  size="icon-xs"
                  aria-label="Delete chat"
                  className="opacity-0 group-hover:opacity-100"
                  onClick={() =>
                    del.mutate(c.id, {
                      onSuccess: () => navigate({ to: "/designer/$chatId", params: { chatId: "new" } }),
                    })
                  }
                >
                  <Trash2Icon />
                </Button>
              </div>
            ))}
          </nav>
        </div>
      </aside>
      <Outlet />
    </div>
  )
}
