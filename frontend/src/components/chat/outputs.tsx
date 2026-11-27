import {
  CheckCircle2Icon,
  DownloadIcon,
  EraserIcon,
  Maximize2Icon,
  MoreHorizontalIcon,
  ShuffleIcon,
  ThumbsDownIcon,
  ThumbsUpIcon,
} from "lucide-react"
import { toast } from "sonner"

import { JobProgress } from "@/components/job-progress"
import { Button } from "@/components/ui/button"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import { Skeleton } from "@/components/ui/skeleton"
import type { ChatOutput, Job } from "@/lib/api"
import { shortModel } from "@/lib/format"
import { useFeedback, useRemoveBackground, useUpscale, useVariations } from "@/lib/queries"
import { cn } from "@/lib/utils"

/** Images made from one designer reply, with progress for the ones still coming. */
export function Outputs({
  chatId,
  messageId,
  images,
  jobs,
  onOpen,
  onUseDirection,
}: {
  chatId: string
  messageId: number
  images: ChatOutput[]
  jobs: Job[]
  onOpen: (id: string) => void
  onUseDirection: (img: ChatOutput) => void
}) {
  const running = jobs.filter((j) => j.status === "running")
  const failed = jobs.filter((j) => j.status === "error")
  const pending = running.reduce((n, j) => n + Math.max(0, j.total - j.image_ids.length), 0)
  if (!images.length && !running.length && !failed.length) return null

  return (
    <div className="flex flex-col gap-2">
      {running.map((j) => (
        <JobProgress key={j.id} job={j} />
      ))}
      {failed.map((j) => (
        <JobProgress key={j.id} job={j} />
      ))}
      <div className="grid grid-cols-2 gap-2 sm:grid-cols-3 lg:grid-cols-4">
        {images.map((img) => (
          <OutputTile
            key={img.id}
            img={img}
            chatId={chatId}
            messageId={messageId}
            onOpen={() => onOpen(img.id)}
            onUseDirection={() => onUseDirection(img)}
          />
        ))}
        {Array.from({ length: Math.min(pending, 12) }, (_, i) => (
          <Skeleton key={`p${i}`} className="aspect-square w-full" />
        ))}
      </div>
    </div>
  )
}

function OutputTile({
  img,
  chatId,
  messageId,
  onOpen,
  onUseDirection,
}: {
  img: ChatOutput
  chatId: string
  messageId: number
  onOpen: () => void
  onUseDirection: () => void
}) {
  const feedback = useFeedback()
  const variations = useVariations()
  const upscale = useUpscale()
  const background = useRemoveBackground()
  const shown = img.primary ?? img

  const rate = (kind: "like" | "dislike") =>
    feedback.mutate({ kind: img.rating === kind ? "unrate" : kind, image_id: img.id, chat_id: chatId, message_id: messageId })

  const openToast = (text: string) =>
    toast.success(text, { action: { label: "Compare", onClick: onOpen }, description: "Kept next to the original." })

  return (
    <figure className="group/tile flex min-w-0 flex-col border bg-card">
      <button
        type="button"
        onClick={onOpen}
        className="relative block focus-visible:ring-2 focus-visible:ring-ring focus-visible:outline-none"
        aria-label={`Open ${img.concept || "image"}`}
      >
        <img
          src={shown.url}
          alt={img.concept || img.prompt}
          loading="lazy"
          className={cn("aspect-square w-full object-contain", shown.kind === "cleaned" && "checker")}
        />
        {img.rating && (
          <span className="absolute top-1 left-1 bg-background/90 p-0.5" title={img.rating === "like" ? "Liked" : "Disliked"}>
            {img.rating === "like" ? <ThumbsUpIcon className="size-3 fill-current text-primary" /> : <ThumbsDownIcon className="size-3" />}
          </span>
        )}
      </button>
      <figcaption className="flex items-center gap-0.5 border-t px-1 py-0.5">
        <span className="min-w-0 flex-1 truncate pl-0.5 text-[10px] text-muted-foreground" title={`${img.concept} · ${img.model}`}>
          {shortModel(img.model)}
        </span>
        <Button
          size="icon-xs"
          variant="ghost"
          aria-label={img.rating === "like" ? "Remove like" : "Like"}
          aria-pressed={img.rating === "like"}
          onClick={() => rate("like")}
        >
          <ThumbsUpIcon className={cn(img.rating === "like" && "fill-current text-primary")} />
        </Button>
        <Button
          size="icon-xs"
          variant="ghost"
          aria-label={img.rating === "dislike" ? "Remove dislike" : "Dislike"}
          aria-pressed={img.rating === "dislike"}
          onClick={() => rate("dislike")}
        >
          <ThumbsDownIcon className={cn(img.rating === "dislike" && "fill-current")} />
        </Button>
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button size="icon-xs" variant="ghost" aria-label="More actions">
              <MoreHorizontalIcon />
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end">
            <DropdownMenuItem onSelect={onUseDirection}>
              <CheckCircle2Icon />
              Use this direction
            </DropdownMenuItem>
            <DropdownMenuItem onSelect={onOpen}>
              <Maximize2Icon />
              Open
            </DropdownMenuItem>
            <DropdownMenuSeparator />
            <DropdownMenuItem
              disabled={!img.prompt}
              onSelect={() => variations.mutate({ id: img.id, count: 4, mode: "remix" }, { onSuccess: () => toast("Making 4 variations") })}
            >
              <ShuffleIcon />
              4 variations
            </DropdownMenuItem>
            <DropdownMenuItem
              disabled={img.vector}
              onSelect={() =>
                upscale.mutate({ id: img.id, method: "lanczos", scale: 2 }, { onSuccess: () => openToast("Upscaled 2x (Lanczos)") })
              }
            >
              <Maximize2Icon />
              Upscale 2x
            </DropdownMenuItem>
            <DropdownMenuItem
              disabled={img.vector}
              onSelect={() =>
                background.mutate({ id: img.id, method: "ai" }, { onSuccess: () => openToast("Background removed") })
              }
            >
              <EraserIcon />
              Remove background
            </DropdownMenuItem>
            <DropdownMenuItem asChild>
              <a href={`${shown.url}?download=1`} download>
                <DownloadIcon />
                Download
              </a>
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      </figcaption>
    </figure>
  )
}
