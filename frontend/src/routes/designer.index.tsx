import { createFileRoute, redirect } from "@tanstack/react-router"

import { api } from "@/lib/api"

/** /designer opens the chat you had open last (remembered by the server, so it works from any browser). */
export const Route = createFileRoute("/designer/")({
  beforeLoad: async () => {
    const last = await api.get<{ chat_id: string | null }>("/designer/last").catch(() => ({ chat_id: null }))
    throw redirect({ to: "/designer/$chatId", params: { chatId: last.chat_id ?? "new" }, replace: true })
  },
})
