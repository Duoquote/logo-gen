import { BrainIcon, Trash2Icon } from "lucide-react"

import { Button } from "@/components/ui/button"
import type { TasteProfile } from "@/lib/api"
import { shortModel } from "@/lib/format"
import { useClearPreferences, usePreferences } from "@/lib/queries"

const LABELS: Record<string, string> = { mark_type: "Mark types", style: "Styles", shape: "Shapes", color: "Colors" }

function TraitRows({ title, traits }: { title: string; traits: Record<string, string[]> }) {
  const rows = Object.entries(traits)
  if (!rows.length) return null
  return (
    <div className="flex flex-col gap-0.5">
      <span className="font-medium">{title}</span>
      {rows.map(([k, v]) => (
        <span key={k}>
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
        Nothing yet. Likes, dislikes, picks and favorites teach it, and so does feedback you type in the chat.
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
        <span>
          <span className="text-muted-foreground">You asked for: </span>
          {profile.more.join("; ")}
        </span>
      )}
      {profile.less.length > 0 && (
        <span>
          <span className="text-muted-foreground">You asked to avoid: </span>
          {profile.less.join("; ")}
        </span>
      )}
      {worked.length > 0 && (
        <span>
          <span className="text-muted-foreground">Models behind the results you liked: </span>
          {worked.map((m) => shortModel(m.model)).join(", ")}
        </span>
      )}
    </div>
  )
}

/**
 * What the designer has learned in one chat, with a button to forget it.
 * Meant for the designer page (the chat UI mounts it); also works with no chat for the global profile.
 */
export function LearnedPanel({ chatId }: { chatId?: string | null }) {
  const prefs = usePreferences(chatId)
  const clear = useClearPreferences()
  const profile = chatId ? prefs.data?.chat : prefs.data?.global
  if (!profile) return null
  return (
    <div className="flex flex-col gap-2 border p-3">
      <div className="flex items-center gap-2">
        <BrainIcon className="size-3.5 text-muted-foreground" />
        <span className="text-xs font-medium">What the designer has learned{chatId ? " in this chat" : ""}</span>
        {!profile.empty && (
          <Button
            size="icon-xs"
            variant="ghost"
            className="ml-auto"
            aria-label="Forget this"
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
