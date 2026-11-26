import { useRef, useState } from "react"
import { ImagePlusIcon, Loader2Icon, XIcon } from "lucide-react"
import { toast } from "sonner"

import { Button } from "@/components/ui/button"
import { uploadImages, type Upload } from "@/lib/api"

/** Upload reference images. They go to the chat model and to image models that accept references. */
export function useReferenceUploads() {
  const [refs, setRefs] = useState<Upload[]>([])
  const [uploading, setUploading] = useState(false)

  const add = async (files: File[]) => {
    const images = files.filter((f) => f.type.startsWith("image/"))
    if (!images.length) return
    setUploading(true)
    try {
      const saved = await uploadImages(images)
      setRefs((r) => [...r, ...saved])
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Upload failed")
    } finally {
      setUploading(false)
    }
  }
  return { refs, setRefs, add, uploading, clear: () => setRefs([]) }
}

export function ReferenceImages({
  refs,
  onRemove,
  onAdd,
  uploading,
}: {
  refs: Upload[]
  onRemove: (path: string) => void
  onAdd: (files: File[]) => void
  uploading: boolean
}) {
  const input = useRef<HTMLInputElement>(null)
  return (
    <div className="flex flex-wrap items-center gap-2">
      {refs.map((r) => (
        <div key={r.path} className="group relative size-12 border">
          <img src={r.url} alt={r.name} className="size-full object-cover" />
          <button
            type="button"
            onClick={() => onRemove(r.path)}
            className="absolute -top-1.5 -right-1.5 hidden size-4 items-center justify-center bg-foreground text-background group-hover:flex"
            aria-label="Remove reference"
          >
            <XIcon className="size-3" />
          </button>
        </div>
      ))}
      <input
        ref={input}
        type="file"
        accept="image/png,image/jpeg,image/webp,image/gif"
        multiple
        hidden
        onChange={(e) => {
          onAdd(Array.from(e.target.files ?? []))
          e.target.value = ""
        }}
      />
      <Button type="button" variant="ghost" size="sm" onClick={() => input.current?.click()} disabled={uploading}>
        {uploading ? <Loader2Icon className="animate-spin" /> : <ImagePlusIcon />}
        Reference
      </Button>
    </div>
  )
}
