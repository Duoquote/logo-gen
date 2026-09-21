import { useEffect, useRef, useState } from "react";
import { Zap, Loader2, ImagePlus, X } from "lucide-react";
import { streamPost, uploadImages } from "../api";
import type { GeneratedLogo, PromptVariation } from "../types";
import ImageGallery from "./ImageGallery";
import GlassPanel from "./ui/GlassPanel";
import Button from "./ui/Button";
import ProgressLine from "./ui/ProgressLine";
import { onCommand } from "../shell/commands";

export default function QuickGenerate() {
  const [concept, setConcept] = useState("");
  const [isGenerating, setIsGenerating] = useState(false);
  const [prompts, setPrompts] = useState<PromptVariation[]>([]);
  const [logos, setLogos] = useState<GeneratedLogo[]>([]);
  const [progress, setProgress] = useState({ current: 0, total: 0, message: "" });
  const [error, setError] = useState("");
  const [attachedImages, setAttachedImages] = useState<{ file: File; preview: string }[]>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFiles = (files: FileList | File[]) => {
    const accepted = Array.from(files).filter((f) =>
      /\.(png|jpe?g|webp|gif)$/i.test(f.name),
    );
    setAttachedImages((prev) => [
      ...prev,
      ...accepted.map((file) => ({ file, preview: URL.createObjectURL(file) })),
    ]);
  };

  const removeAttached = (idx: number) => {
    setAttachedImages((prev) => {
      URL.revokeObjectURL(prev[idx]!.preview);
      return prev.filter((_, i) => i !== idx);
    });
  };

  const handleGenerate = async () => {
    if (!concept.trim() || isGenerating) return;
    setIsGenerating(true);
    setError("");
    setLogos([]);
    setPrompts([]);
    setProgress({ current: 0, total: 0, message: "Generating prompt variations..." });

    let imgPaths: string[] = [];
    if (attachedImages.length > 0) {
      try {
        const uploaded = await uploadImages(attachedImages.map((a) => a.file));
        imgPaths = uploaded.map((u) => u.path);
      } catch {
        setError("Failed to upload images");
        setIsGenerating(false);
        return;
      }
    }

    const body: Record<string, unknown> = { concept: concept.trim() };
    if (imgPaths.length > 0) body.images = imgPaths;

    await streamPost("/quick-generate", body, {
      onPrompts(p) { setPrompts(p); },
      onProgress(current, total, message) {
        setProgress({ current, total, message });
      },
      onResult(data) {
        setLogos((data.logos as GeneratedLogo[]) || []);
        const totalTime = (data.total_time as number) || 0;
        const errCount = (data.errors as string[])?.length || 0;
        setProgress({
          current: 0, total: 0,
          message: `Done in ${totalTime.toFixed(1)}s${errCount > 0 ? ` (${errCount} errors)` : ""}`,
        });
      },
      onError(msg) { setError(msg); },
    });

    setIsGenerating(false);
  };

  const handleClear = () => {
    setLogos([]);
    setPrompts([]);
    setProgress({ current: 0, total: 0, message: "" });
    setError("");
  };

  // Latest-handler pattern so command subs stay stable across renders.
  const latest = useRef({ handleGenerate, handleClear });
  latest.current = { handleGenerate, handleClear };

  useEffect(() => {
    const unsubs = [
      onCommand("quick.generate", () => latest.current.handleGenerate()),
      onCommand("quick.clear", () => latest.current.handleClear()),
    ];
    return () => unsubs.forEach((u) => u());
  }, []);

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    if (e.dataTransfer.files.length > 0) handleFiles(e.dataTransfer.files);
  };

  return (
    <div className="space-y-5">
      <div>
        <h1 className="font-display text-[28px] font-semibold tracking-[-0.02em] text-surface-100">
          Quick generate
        </h1>
        <p className="text-sm text-surface-400 mt-1">
          One-shot: describe the brand and we'll fan out prompt variations across every configured model.
        </p>
      </div>

      <GlassPanel
        className="p-5 relative"
        onDragOver={(e) => e.preventDefault()}
        onDrop={handleDrop}
      >
        <textarea
          value={concept}
          onChange={(e) => setConcept(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              handleGenerate();
            }
          }}
          placeholder="e.g. a modern coffee shop called 'Brew & Co', minimalist and warm"
          className="input-field w-full resize-none text-[15px] leading-relaxed"
          rows={3}
          disabled={isGenerating}
        />

        {attachedImages.length > 0 && (
          <div className="flex gap-2 flex-wrap mt-3">
            {attachedImages.map((img, i) => (
              <div key={i} className="relative group">
                <img
                  src={img.preview}
                  alt={img.file.name}
                  className="w-16 h-16 object-cover rounded-field border border-glass-border-strong"
                />
                <button
                  onClick={() => removeAttached(i)}
                  className="absolute -top-1.5 -right-1.5 w-5 h-5 bg-surface-800 rounded-full flex items-center justify-center
                             opacity-0 group-hover:opacity-100 transition-opacity hover:bg-red-500"
                >
                  <X className="w-3 h-3 text-surface-100" />
                </button>
              </div>
            ))}
          </div>
        )}

        <div className="flex items-center justify-between gap-3 mt-3">
          <input
            ref={fileInputRef}
            type="file"
            accept="image/png,image/jpeg,image/webp,image/gif"
            multiple
            className="hidden"
            onChange={(e) => {
              if (e.target.files) handleFiles(e.target.files);
              e.target.value = "";
            }}
          />
          <Button
            variant="ghost"
            size="sm"
            onClick={() => fileInputRef.current?.click()}
            disabled={isGenerating}
          >
            <ImagePlus className="w-4 h-4" />
            Reference images
          </Button>

          <Button
            onClick={handleGenerate}
            disabled={isGenerating || !concept.trim()}
            size="lg"
          >
            {isGenerating ? <Loader2 className="w-4 h-4 animate-spin" /> : <Zap className="w-4 h-4" />}
            Generate
          </Button>
        </div>
      </GlassPanel>

      {prompts.length > 0 && (
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 motion-safe:animate-fade-in">
          {prompts.map((p, i) => (
            <GlassPanel key={i} tone="soft" className="p-3.5 space-y-1">
              <div className="flex items-center gap-2">
                <span className="text-[11px] font-mono text-accent">#{i + 1}</span>
                <span className="text-sm font-semibold text-surface-100">{p.concept}</span>
              </div>
              <p className="text-xs text-surface-400 italic">{p.style}</p>
            </GlassPanel>
          ))}
        </div>
      )}

      {isGenerating && (
        <div className="flex items-center gap-3">
          <Loader2 className="w-3.5 h-3.5 animate-spin text-accent" />
          <ProgressLine
            current={progress.current}
            total={progress.total}
            message={progress.message || "Working…"}
            className="flex-1"
          />
        </div>
      )}

      {!isGenerating && progress.message && (
        <p className="text-xs text-surface-400">{progress.message}</p>
      )}

      {error && (
        <p className="text-sm text-red-400 bg-red-400/10 border border-red-400/20 rounded-field px-3 py-2">{error}</p>
      )}

      {logos.length > 0 && (
        <div className="motion-safe:animate-slide-up">
          <ImageGallery images={logos} columns={4} />
        </div>
      )}
    </div>
  );
}
