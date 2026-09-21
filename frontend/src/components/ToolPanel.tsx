import type { ReactNode } from "react";
import GlassPanel from "./ui/GlassPanel";
import ImageGallery from "./ImageGallery";
import ProgressLine from "./ui/ProgressLine";
import { imageUrl } from "../api";
import type { ImageInfo } from "../types";

export interface ToolGallerySpec {
  title: string;
  images: ImageInfo[];
  selectedIndex?: number;
  onSelect?: (idx: number) => void;
  showChecker?: boolean;
  emptyText?: string;
}

interface Props {
  toolbar: ReactNode;
  /** Optional second toolbar row (mode-specific options). */
  toolbarExtras?: ReactNode;
  selected?: ImageInfo | null;
  galleries: [ToolGallerySpec, ToolGallerySpec];
  progress?: { current: number; total: number; message: string } | null;
  error?: string;
}

export default function ToolPanel({
  toolbar,
  toolbarExtras,
  selected,
  galleries,
  progress,
  error,
}: Props) {
  return (
    <div className="space-y-4">
      <GlassPanel className="overflow-hidden">
        <div className="flex flex-wrap items-center gap-3 px-4 py-3 border-b border-glass-border">
          {toolbar}
        </div>
        {toolbarExtras && (
          <div className="flex flex-wrap items-center gap-4 px-4 py-3">
            {toolbarExtras}
          </div>
        )}
      </GlassPanel>

      {progress && progress.total > 0 && <ProgressLine {...progress} />}

      {error && (
        <p className="text-sm text-red-400 bg-red-400/10 border border-red-400/20 rounded-field px-3 py-2">
          {error}
        </p>
      )}

      {selected && (
        <GlassPanel className="p-4 flex items-start gap-4 motion-safe:animate-fade-in">
          <div>
            <p className="text-[11px] uppercase tracking-wider text-surface-500 mb-2">Selected</p>
            <img
              src={imageUrl(selected.path)}
              alt={selected.name}
              className="w-40 h-40 object-cover rounded-field bg-surface-850 border border-glass-border"
            />
            <p className="text-xs text-surface-400 mt-2 truncate max-w-[10rem]">{selected.name}</p>
          </div>
        </GlassPanel>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {galleries.map((g, i) => (
          <GallerySection key={i} {...g} />
        ))}
      </div>
    </div>
  );
}

function GallerySection({
  title,
  images,
  selectedIndex,
  onSelect,
  showChecker,
  emptyText,
}: ToolGallerySpec) {
  return (
    <GlassPanel tone="soft" className="p-4 flex flex-col min-h-0">
      <h3 className="font-display text-[11px] font-semibold tracking-wider uppercase text-surface-300 mb-3">
        {title}
        <span className="ml-2 text-surface-500 normal-case font-normal">{images.length}</span>
      </h3>
      <ImageGallery
        images={images}
        columns={3}
        selectedIndex={selectedIndex}
        onSelect={onSelect}
        showChecker={showChecker}
        emptyText={emptyText}
      />
    </GlassPanel>
  );
}
