import { createFileRoute, Outlet } from "@tanstack/react-router"

import { ChatSidebar } from "@/components/chat/chat-sidebar"
import { SidebarInset, SidebarProvider } from "@/components/ui/sidebar"

export const Route = createFileRoute("/designer")({
  component: DesignerLayout,
})

const startOpen = () => !document.cookie.includes("sidebar_state=false")

function DesignerLayout() {
  return (
    <SidebarProvider defaultOpen={startOpen()} className="h-full min-h-0">
      <ChatSidebar />
      <SidebarInset className="min-h-0 min-w-0">
        <Outlet />
      </SidebarInset>
    </SidebarProvider>
  )
}
