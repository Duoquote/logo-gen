import { useEffect, useState } from "react"
import { useQueryClient } from "@tanstack/react-query"

import { followReply, type ChatMessage, type MessageStatus } from "@/lib/api"
import { keys } from "@/lib/queries"

export interface LiveReply {
  id: number
  content: string
  reasoning: string
  status: MessageStatus
  error: string
}

/**
 * While a reply is streaming on the server, follow it. The server sends what's
 * there so far first, so this works the same on a fresh page, after
 * navigating away and back, or after a reload.
 */
export function useLiveReply(chatId: string, msg: ChatMessage | undefined): LiveReply | null {
  const qc = useQueryClient()
  const [live, setLive] = useState<LiveReply | null>(null)
  const id = msg?.id
  const streaming = msg?.status === "streaming"

  useEffect(() => {
    if (!streaming || id == null) return
    return followReply(chatId, id, (ev) => {
      switch (ev.type) {
        case "snapshot":
          setLive({ id, content: ev.content, reasoning: ev.reasoning, status: ev.status, error: ev.error })
          break
        case "token":
          setLive((l) => l && { ...l, content: l.content + ev.content })
          break
        case "reasoning":
          setLive((l) => l && { ...l, reasoning: l.reasoning + ev.content })
          break
        case "done":
          setLive((l) => l && { ...l, status: ev.status, content: ev.content })
          break
        case "error":
          setLive((l) => l && { ...l, status: "error", error: ev.message })
          break
        case "end":
          qc.invalidateQueries({ queryKey: keys.chat(chatId) })
          qc.invalidateQueries({ queryKey: keys.chats })
          qc.invalidateQueries({ queryKey: ["spend"] })
          break
      }
    })
  }, [chatId, id, streaming, qc])

  return streaming && live?.id === id ? live : null
}
