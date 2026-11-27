import { forwardRef, useImperativeHandle, useRef, useState } from "react"
import { ArrowUpIcon, ImagePlusIcon, Loader2Icon, SquareIcon, XIcon } from "lucide-react"

import { ImageModelPicker } from "@/components/model-picker"
import { useReferenceUploads } from "@/components/reference-images"
import { Button } from "@/components/ui/button"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Textarea } from "@/components/ui/textarea"
import type { Upload } from "@/lib/api"
import { maxPerDirection } from "@/lib/caps"
import { useSettings, useUpdateSettings } from "@/lib/queries"

export interface ComposerHandle {
  focus: () => void
  setText: (t: string) => void
  addRefs: (refs: Upload[]) => void
}

/**
 * The message box. Enter sends, Shift+Enter is a new line, and pasted or
 * dropped images become references. Below it: the image models and how many
 * images each direction gets, which the Generate buttons in the thread use.
 */
export const Composer = forwardRef<
  ComposerHandle,
  {
    placeholder: string
    busy: boolean
    stopping?: boolean
    sending?: boolean
    onSend: (text: string, images: string[]) => Promise<boolean> | boolean
    onStop: () => void
  }
>(function Composer({ placeholder, busy, stopping, sending, onSend, onStop }, ref) {
  const [text, setText] = useState("")
  const refs = useReferenceUploads()
  const area = useRef<HTMLTextAreaElement>(null)
  const file = useRef<HTMLInputElement>(null)
  const settings = useSettings()
  const update = useUpdateSettings()
  const s = settings.data?.settings
  const models = s?.image_models ?? []
  const maxPer = s ? maxPerDirection(s, models, 1) : 1
  const per = Math.min(s?.images_per_prompt ?? 1, maxPer)

  useImperativeHandle(ref, () => ({
    focus: () => area.current?.focus(),
    setText: (t) => {
      setText(t)
      area.current?.focus()
    },
    addRefs: (r) => refs.setRefs((cur) => [...cur, ...r.filter((x) => !cur.some((c) => c.path === x.path))]),
  }))

  const send = async () => {
    const content = text.trim()
    if (!content || busy || sending || refs.uploading) return
    const ok = await onSend(content, refs.refs.map((r) => r.path))
    if (ok) {
      setText("")
      refs.clear()
    }
  }

  return (
    <div
      className="flex flex-col gap-2"
      onDragOver={(e) => e.preventDefault()}
      onDrop={(e) => {
        e.preventDefault()
        refs.add(Array.from(e.dataTransfer.files))
      }}
    >
      <div className="flex flex-col border bg-background focus-within:ring-2 focus-within:ring-ring/40">
        {(refs.refs.length > 0 || refs.uploading) && (
          <div className="flex flex-wrap gap-1.5 p-2 pb-0">
            {refs.refs.map((r) => (
              <span key={r.path} className="relative">
                <img src={r.url} alt={r.name} className="size-12 border object-cover" />
                <button
                  type="button"
                  aria-label={`Remove ${r.name}`}
                  onClick={() => refs.setRefs((cur) => cur.filter((x) => x.path !== r.path))}
                  className="absolute -top-1.5 -right-1.5 bg-foreground p-0.5 text-background"
                >
                  <XIcon className="size-2.5" />
                </button>
              </span>
            ))}
            {refs.uploading && <Loader2Icon className="size-4 animate-spin self-center text-muted-foreground" />}
          </div>
        )}
        <Textarea
          ref={area}
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
              e.preventDefault()
              send()
            }
          }}
          onPaste={(e) => {
            const files = Array.from(e.clipboardData.files)
            if (files.length) refs.add(files)
          }}
          placeholder={placeholder}
          aria-label="Message"
          rows={1}
          className="field-sizing-content max-h-60 min-h-11 resize-none border-0 text-sm shadow-none focus-visible:ring-0"
        />
        <div className="flex items-center gap-1 p-1.5 pt-0">
          <input
            ref={file}
            type="file"
            accept="image/*"
            multiple
            hidden
            onChange={(e) => {
              refs.add(Array.from(e.target.files ?? []))
              e.target.value = ""
            }}
          />
          <Button size="icon-sm" variant="ghost" aria-label="Attach reference images" onClick={() => file.current?.click()}>
            <ImagePlusIcon />
          </Button>
          <span className="hidden text-[11px] text-muted-foreground sm:inline">Enter to send, Shift+Enter for a new line</span>
          {busy ? (
            <Button size="sm" variant="outline" className="ml-auto" onClick={onStop} disabled={stopping}>
              <SquareIcon className="fill-current" />
              Stop
            </Button>
          ) : (
            <Button size="icon-sm" className="ml-auto" aria-label="Send" onClick={send} disabled={!text.trim() || sending || refs.uploading}>
              {sending ? <Loader2Icon className="animate-spin" /> : <ArrowUpIcon />}
            </Button>
          )}
        </div>
      </div>
      <div className="flex flex-wrap items-center gap-2">
        <ImageModelPicker
          className="min-w-0 flex-1"
          value={models}
          onChange={(v) => update.mutate({ image_models: v })}
          compact
        />
        <Select value={String(per)} onValueChange={(v) => update.mutate({ images_per_prompt: Number(v) })}>
          <SelectTrigger
            size="sm"
            className="w-auto"
            aria-label="Images per direction"
            title={`Images per direction for each model. Capped at ${maxPer} in Settings, Limits.`}
          >
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {Array.from({ length: Math.min(maxPer, 8) }, (_, i) => i + 1).map((n) => (
              <SelectItem key={n} value={String(n)}>
                {n} per direction
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>
    </div>
  )
})
