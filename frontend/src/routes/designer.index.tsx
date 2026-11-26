import { createFileRoute, redirect } from "@tanstack/react-router"

export const Route = createFileRoute("/designer/")({
  beforeLoad: () => {
    throw redirect({ to: "/designer/$chatId", params: { chatId: "new" } })
  },
})
