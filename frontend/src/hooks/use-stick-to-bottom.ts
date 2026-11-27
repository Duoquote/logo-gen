import { useCallback, useEffect, useRef, useState } from "react"

/**
 * Keep a scroll container pinned to the bottom while its content grows,
 * unless the user has scrolled up. Then `atBottom` is false and the caller
 * shows a "jump to latest" button.
 */
export function useStickToBottom<S extends HTMLElement, C extends HTMLElement>() {
  const scrollRef = useRef<S>(null)
  const contentRef = useRef<C>(null)
  const pinned = useRef(true)
  const [atBottom, setAtBottom] = useState(true)

  const toBottom = useCallback((smooth = false) => {
    const el = scrollRef.current
    if (!el) return
    pinned.current = true
    setAtBottom(true)
    el.scrollTo({ top: el.scrollHeight, behavior: smooth ? "smooth" : "auto" })
  }, [])

  useEffect(() => {
    const el = scrollRef.current
    const content = contentRef.current
    if (!el || !content) return
    let lastTop = el.scrollTop
    const onScroll = () => {
      const gap = el.scrollHeight - el.scrollTop - el.clientHeight
      const near = gap < 48
      // Only a scroll up by the user unpins; growth below us doesn't.
      if (el.scrollTop < lastTop - 2 && !near) pinned.current = false
      if (near) pinned.current = true
      lastTop = el.scrollTop
      setAtBottom(near)
    }
    const ro = new ResizeObserver(() => {
      if (pinned.current) el.scrollTop = el.scrollHeight
      else setAtBottom(el.scrollHeight - el.scrollTop - el.clientHeight < 48)
    })
    el.addEventListener("scroll", onScroll, { passive: true })
    ro.observe(content)
    return () => {
      el.removeEventListener("scroll", onScroll)
      ro.disconnect()
    }
  }, [])

  return { scrollRef, contentRef, atBottom, toBottom }
}
