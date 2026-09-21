import { useEffect, useState } from "react";
import { X, Download, ZoomIn } from "lucide-react";
import { imageUrl } from "../api";
import type { ImageInfo } from "../types";

interface Props {
  images: ImageInfo[];
  columns?: number;
  selectedIndex?: number;
  onSelect?: (index: number, image: ImageInfo) => void;
  emptyText?: string;
  showChecker?: boolean;
}

export default function ImageGallery({
  images,
  columns = 4,
  selectedIndex,
  onSelect,
  emptyText = "No images yet",
  showChecker = false,
}: Props) {
  const [lightbox, setLightbox] = useState<number | null>(null);

  // Lock scroll when lightbox is open.
  useEffect(() => {
    if (lightbox === null) return;
    const prev = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => { document.body.style.overflow = prev; };
  }, [lightbox]);

  if (images.length === 0) {
    return (
      <div className="flex items-center justify-center h-48 rounded-panel border border-dashed border-glass-border-strong text-surface-500 text-sm">
        {emptyText}
      </div>
    );
  }

  const gridCols = {
    2: "grid-cols-2",
    3: "grid-cols-2 sm:grid-cols-3",
    4: "grid-cols-2 sm:grid-cols-3 lg:grid-cols-4",
    5: "grid-cols-2 sm:grid-cols-3 lg:grid-cols-5",
    6: "grid-cols-3 sm:grid-cols-4 lg:grid-cols-6",
  }[columns] ?? "grid-cols-2 sm:grid-cols-3 lg:grid-cols-4";

  return (
    <>
      <div className={`grid ${gridCols} gap-3`}>
        {images.map((img, i) => (
          <div
            key={img.path}
            className={`
              group relative rounded-panel overflow-hidden cursor-pointer
              border transition-all duration-150
              ${showChecker ? "checker-bg" : ""}
              ${
                selectedIndex === i
                  ? "border-accent shadow-glow"
                  : "border-glass-border hover:border-glass-border-strong"
              }
            `}
            onClick={() => onSelect?.(i, img)}
          >
            <img
              src={imageUrl(img.path)}
              alt={img.name}
              className="w-full aspect-square object-cover"
              loading="lazy"
            />
            {/* Hover overlay */}
            <div
              className="absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity duration-200 flex items-center justify-center bg-glass-strong"
              style={{ backdropFilter: "blur(4px)", WebkitBackdropFilter: "blur(4px)" }}
            >
              <button
                className="p-2.5 rounded-full bg-surface-1000/70 border border-glass-border-strong text-surface-100 hover:bg-surface-1000/90 transition-colors"
                onClick={(e) => {
                  e.stopPropagation();
                  setLightbox(i);
                }}
                aria-label="Zoom in"
              >
                <ZoomIn className="w-5 h-5" />
              </button>
            </div>
            <div className="absolute bottom-0 inset-x-0 bg-gradient-to-t from-black/70 to-transparent px-2 py-1.5 opacity-0 group-hover:opacity-100 transition-opacity">
              <p className="text-[11px] text-surface-100 truncate">{img.name}</p>
            </div>
          </div>
        ))}
      </div>

      {lightbox !== null && images[lightbox] && (
        <div
          className="fixed inset-0 z-[200] flex items-center justify-center motion-safe:animate-fade-in p-6"
          onClick={() => setLightbox(null)}
        >
          <div
            aria-hidden
            className="absolute inset-0 bg-surface-1000/80"
            style={{ backdropFilter: "blur(28px)", WebkitBackdropFilter: "blur(28px)" }}
          />
          <div className="relative max-w-[92vw] max-h-[92vh]" onClick={(e) => e.stopPropagation()}>
            <img
              src={imageUrl(images[lightbox].path)}
              alt={images[lightbox].name}
              className={`max-w-full max-h-[85vh] object-contain rounded-panel shadow-glass ${showChecker ? "checker-bg" : ""}`}
            />
            <div className="absolute top-3 right-3 flex gap-2">
              <a
                href={imageUrl(images[lightbox].path)}
                download={images[lightbox].name}
                onClick={(e) => e.stopPropagation()}
                className="p-2 rounded-full bg-surface-1000/70 border border-glass-border-strong text-surface-100 hover:bg-surface-1000/90 transition-colors"
                title="Download"
              >
                <Download className="w-5 h-5" />
              </a>
              <button
                onClick={() => setLightbox(null)}
                className="p-2 rounded-full bg-surface-1000/70 border border-glass-border-strong text-surface-100 hover:bg-surface-1000/90 transition-colors"
                aria-label="Close"
              >
                <X className="w-5 h-5" />
              </button>
            </div>
            <p className="text-center text-surface-300 text-xs mt-3 font-mono">{images[lightbox].name}</p>
          </div>
        </div>
      )}
    </>
  );
}
