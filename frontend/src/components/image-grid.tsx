import { HeartIcon } from "lucide-react"

import { Skeleton } from "@/components/ui/skeleton"
import type { ImageItem } from "@/lib/api"
import { shortModel } from "@/lib/format"
import { useToggleFavorite } from "@/lib/queries"
import { cn } from "@/lib/utils"

export function ImageGrid({
  images,
  onOpen,
  pending = 0,
  className,
}: {
  images: ImageItem[]
  onOpen: (id: string) => void
  pending?: number
  className?: string
}) {
  const fav = useToggleFavorite()
  return (
    <div className={cn("grid grid-cols-[repeat(auto-fill,minmax(180px,1fr))] gap-3", className)}>
      {Array.from({ length: pending }).map((_, i) => (
        <Skeleton key={`p${i}`} className="aspect-square" />
      ))}
      {images.map((img) => (
        <figure key={img.id} className="group relative border bg-card">
          <button type="button" onClick={() => onOpen(img.id)} className="block w-full">
            <img
              src={(img.primary ?? img).url}
              alt={img.concept || img.prompt}
              loading="lazy"
              className={cn("aspect-square w-full object-contain", (img.primary ?? img).kind === "cleaned" && "checker")}
            />
          </button>
          <figcaption className="flex items-center gap-1 border-t px-2 py-1 text-[11px] text-muted-foreground">
            <span className="min-w-0 flex-1 truncate" title={img.prompt}>
              {img.concept || shortModel(img.model)}
            </span>
            {img.kind !== "generated" && <span className="uppercase">{img.kind === "cleaned" ? "bg" : "up"}</span>}
            {img.primary && <span title={`Showing the version you picked: ${img.primary.note}`}>edited</span>}
            {img.vector && <span>SVG</span>}
            <button
              type="button"
              aria-label={img.favorite ? "Remove from favorites" : "Add to favorites"}
              onClick={() => fav.mutate({ id: img.id, favorite: !img.favorite })}
              className={cn("p-0.5", img.favorite ? "text-primary" : "opacity-0 group-hover:opacity-100")}
            >
              <HeartIcon className={cn("size-3.5", img.favorite && "fill-current")} />
            </button>
          </figcaption>
        </figure>
      ))}
    </div>
  )
}
