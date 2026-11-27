import { BrainIcon, Trash2Icon } from "lucide-react"

import { Button } from "@/components/ui/button"
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover"
import type { TasteProfile } from "@/lib/api"
import { shortModel } from "@/lib/format"
import { useClearPreferences, usePreferences } from "@/lib/queries"
import { cn } from "@/lib/utils"

const LABELS: Record<string, string> = { mark_type: "Mark types", style: "Styles", shape: "Shapes", color: "Colors" }

function TraitRows({ title, traits }: { title: string; traits: Record<string, string[]> }) {
  const rows = Object.entries(traits)
  if (!rows.length) return null
  return (
    <div className="flex flex-col gap-0.5">
      <span className="font-medium">{title}</span>
      {rows.map(([k, v]) => (
        <span key={k} className="break-words">
          <span className="text-muted-foreground">{LABELS[k] ?? k}: </span>
          {v.join(", ")}
        </span>
      ))}
    </div>
  )
}

export function TasteSummary({ profile }: { profile: TasteProfile }) {
  if (profile.empty) {
    return (
      <p className="text-xs text-muted-foreground">
        Nothing yet. Thumbs up, thumbs down and "Use this direction" on the images teach it, and so does feedback you
        type in the chat.
      </p>
    )
  }
  const { like, dislike, pick } = profile.counts
  const worked = profile.models.filter((m) => m.score > 0).slice(0, 3)
  return (
    <div className="flex flex-col gap-2 text-xs">
      <span className="text-muted-foreground">
        From {pick} pick{pick === 1 ? "" : "s"}, {like} like{like === 1 ? "" : "s"} and {dislike} dislike
        {dislike === 1 ? "" : "s"}
        {profile.counts.notes > 0 && `, plus ${profile.counts.notes} thing${profile.counts.notes === 1 ? "" : "s"} you said`}
      </span>
      <TraitRows title="Leans toward" traits={profile.liked} />
      <TraitRows title="Steers away from" traits={profile.disliked} />
      {profile.more.length > 0 && (
        <span className="break-words">
          <span className="text-muted-foreground">You asked for: </span>
          {profile.more.join("; ")}
        </span>
      )}
      {profile.less.length > 0 && (
        <span className="break-words">
          <span className="text-muted-foreground">You asked to avoid: </span>
          {profile.less.join("; ")}
        </span>
      )}
      {worked.length > 0 && (
        <span className="break-words">
          <span className="text-muted-foreground">Models behind the results you liked: </span>
          {worked.map((m) => shortModel(m.model)).join(", ")}
        </span>
      )}
    </div>
  )
}

/**
 * What the designer has learned, in one chat (chatId) or across chats (no chatId), with a button to forget it.
 * `bare` drops the border, for use inside a popover.
 */
export function LearnedPanel({ chatId, bare = false }: { chatId?: string | null; bare?: boolean }) {
  const prefs = usePreferences(chatId)
  const clear = useClearPreferences()
  const profile = chatId ? prefs.data?.chat : prefs.data?.global
  if (!profile) return null
  return (
    <div className={cn("flex flex-col gap-2", !bare && "border p-3")}>
      <div className="flex items-center gap-2">
        <BrainIcon className="size-3.5 shrink-0 text-muted-foreground" />
        <span className="text-xs font-medium">What the designer has learned{chatId ? " in this chat" : ""}</span>
        {!profile.empty && (
          <Button
            size="icon-xs"
            variant="ghost"
            className="ml-auto"
            aria-label="Forget what it learned"
            title="Forget what it learned"
            disabled={clear.isPending}
            onClick={() => clear.mutate(chatId ? { chatId } : { global: true })}
          >
            <Trash2Icon />
          </Button>
        )}
      </div>
      <TasteSummary profile={profile} />
    </div>
  )
}

/** The chat header's button: a brain that opens LearnedPanel, with a dot once there's something learned. */
export function LearnedButton({ chatId }: { chatId: string }) {
  const prefs = usePreferences(chatId)
  const learned = prefs.data?.chat && !prefs.data.chat.empty
  return (
    <Popover>
      <PopoverTrigger asChild>
        <Button
          size="icon-sm"
          variant="ghost"
          className="relative shrink-0"
          aria-label="What the designer has learned in this chat"
          title="What the designer has learned"
        >
          <BrainIcon />
          {learned && <span className="absolute top-1 right-1 size-1.5 rounded-full bg-primary" aria-hidden />}
        </Button>
      </PopoverTrigger>
      <PopoverContent align="end" className="w-80 max-w-[calc(100vw-1rem)]">
        <LearnedPanel chatId={chatId} bare />
      </PopoverContent>
    </Popover>
  )
}
