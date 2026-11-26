import { useEffect, useState } from "react"
import { createFileRoute } from "@tanstack/react-router"
import { HeartIcon, SearchIcon } from "lucide-react"

import { ImageDialog } from "@/components/image-dialog"
import { ImageGrid } from "@/components/image-grid"
import { Empty, EmptyDescription, EmptyHeader, EmptyTitle } from "@/components/ui/empty"
import { InputGroup, InputGroupAddon, InputGroupInput } from "@/components/ui/input-group"
import { Toggle } from "@/components/ui/toggle"
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group"
import { formatUsd } from "@/lib/format"
import { useImages, useSystem } from "@/lib/queries"

type Kind = "all" | "generated" | "upscaled" | "cleaned"

export const Route = createFileRoute("/gallery")({
  validateSearch: (s: Record<string, unknown>): { q?: string; kind?: Kind; fav?: boolean; image?: string } => ({
    q: typeof s.q === "string" ? s.q : undefined,
    kind: (["generated", "upscaled", "cleaned"].includes(s.kind as string) ? s.kind : undefined) as Kind | undefined,
    fav: s.fav === true || s.fav === "true" ? true : undefined,
    image: typeof s.image === "string" ? s.image : undefined,
  }),
  component: GalleryPage,
})

function GalleryPage() {
  const search = Route.useSearch()
  const navigate = Route.useNavigate()
  const [q, setQ] = useState(search.q ?? "")
  const system = useSystem()

  // Debounce the search box into the URL.
  useEffect(() => {
    const t = setTimeout(() => navigate({ search: (p) => ({ ...p, q: q || undefined }), replace: true }), 250)
    return () => clearTimeout(t)
  }, [q, navigate])

  const images = useImages({ q: search.q, kind: search.kind, favorite: search.fav, limit: 500 })

  return (
    <div className="mx-auto flex max-w-7xl flex-col gap-4 p-6">
      <div className="flex flex-wrap items-center gap-3">
        <InputGroup className="w-72">
          <InputGroupAddon>
            <SearchIcon />
          </InputGroupAddon>
          <InputGroupInput placeholder="Search prompts, concepts, models" value={q} onChange={(e) => setQ(e.target.value)} />
        </InputGroup>
        <ToggleGroup
          type="single"
          variant="outline"
          size="sm"
          value={search.kind ?? "all"}
          onValueChange={(v) =>
            v && navigate({ search: (p) => ({ ...p, kind: v === "all" ? undefined : (v as Kind) }) })
          }
        >
          <ToggleGroupItem value="all">All</ToggleGroupItem>
          <ToggleGroupItem value="generated">Generated</ToggleGroupItem>
          <ToggleGroupItem value="upscaled">Upscaled</ToggleGroupItem>
          <ToggleGroupItem value="cleaned">No background</ToggleGroupItem>
        </ToggleGroup>
        <Toggle
          variant="outline"
          size="sm"
          pressed={!!search.fav}
          onPressedChange={(v) => navigate({ search: (p) => ({ ...p, fav: v || undefined }) })}
        >
          <HeartIcon />
          Favorites
        </Toggle>
        {system.data?.library && (
          <span className="ml-auto text-xs text-muted-foreground">
            {system.data.library.images} images · {formatUsd(system.data.library.cost, 2)} spent on OpenRouter images
          </span>
        )}
      </div>

      {images.data && images.data.length === 0 ? (
        <Empty>
          <EmptyHeader>
            <EmptyTitle>No images here</EmptyTitle>
            <EmptyDescription>
              {search.q || search.kind || search.fav ? "Nothing matches these filters." : "Generated logos are kept here."}
            </EmptyDescription>
          </EmptyHeader>
        </Empty>
      ) : (
        <ImageGrid images={images.data ?? []} onOpen={(id) => navigate({ search: (p) => ({ ...p, image: id }) })} />
      )}

      <ImageDialog
        imageId={search.image ?? null}
        onOpenChange={(open) => !open && navigate({ search: (p) => ({ ...p, image: undefined }) })}
        onSelect={(id) => navigate({ search: (p) => ({ ...p, image: id }) })}
      />
    </div>
  )
}
