import { useEffect, useRef, useState, useCallback } from "react";
import { RefreshCw, Maximize2, Loader2 } from "lucide-react";
import {
  fetchGenerated,
  fetchUpscaled,
  fetchUpscaleOptions,
  upscaleOne,
  streamPost,
} from "../api";
import type { ImageInfo } from "../types";
import ToolPanel from "./ToolPanel";
import Field from "./ui/Field";
import Button from "./ui/Button";
import { onCommand } from "../shell/commands";

export default function Upscale() {
  const [sources, setSources] = useState<ImageInfo[]>([]);
  const [upscaled, setUpscaled] = useState<ImageInfo[]>([]);
  const [selectedIdx, setSelectedIdx] = useState<number | null>(null);
  const [methods, setMethods] = useState<Record<string, string>>({});
  const [scales, setScales] = useState<Record<string, number>>({});
  const [method, setMethod] = useState("lanczos");
  const [scale, setScale] = useState("4x");
  const [processing, setProcessing] = useState(false);
  const [batchProcessing, setBatchProcessing] = useState(false);
  const [progress, setProgress] = useState({ current: 0, total: 0, message: "" });
  const [error, setError] = useState("");

  const refresh = useCallback(async () => {
    const [gen, up, opts] = await Promise.all([
      fetchGenerated(),
      fetchUpscaled(),
      fetchUpscaleOptions(),
    ]);
    setSources(gen);
    setUpscaled(up);
    setMethods(opts.methods);
    setScales(opts.scales);
    if (!Object.keys(opts.methods).includes(method)) {
      setMethod(Object.keys(opts.methods)[0] || "lanczos");
    }
  }, [method]);

  useEffect(() => { refresh(); }, []);  // eslint-disable-line react-hooks/exhaustive-deps

  const selected = selectedIdx !== null ? sources[selectedIdx] ?? null : null;

  const handleUpscaleOne = async () => {
    if (!selected || processing) return;
    setProcessing(true);
    setError("");
    try {
      await upscaleOne(selected.path, method, scales[scale] || 4);
      const up = await fetchUpscaled();
      setUpscaled(up);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Upscale failed");
    }
    setProcessing(false);
  };

  const handleUpscaleAll = async () => {
    if (sources.length === 0 || batchProcessing) return;
    setBatchProcessing(true);
    setError("");
    setProgress({ current: 0, total: 0, message: "Starting batch upscale..." });

    await streamPost("/upscale/batch", { method, scale: scales[scale] || 4 }, {
      onProgress(cur, tot, msg) { setProgress({ current: cur, total: tot, message: msg }); },
      onResult() {
        fetchUpscaled().then(setUpscaled);
        setProgress({ current: 0, total: 0, message: "Batch complete" });
      },
      onError(msg) { setError(msg); },
    });

    setBatchProcessing(false);
  };

  const latest = useRef({ handleUpscaleOne, handleUpscaleAll, refresh });
  latest.current = { handleUpscaleOne, handleUpscaleAll, refresh };
  useEffect(() => {
    const unsubs = [
      onCommand("upscale.selected", () => latest.current.handleUpscaleOne()),
      onCommand("upscale.all", () => latest.current.handleUpscaleAll()),
      onCommand("upscale.refresh", () => latest.current.refresh()),
    ];
    return () => unsubs.forEach((u) => u());
  }, []);

  const toolbar = (
    <>
      <Field label="Method" className="min-w-[200px]">
        <select
          value={method}
          onChange={(e) => setMethod(e.target.value)}
          className="input-field text-sm w-full"
        >
          {Object.entries(methods).map(([k, v]) => (
            <option key={k} value={k}>{v}</option>
          ))}
        </select>
      </Field>
      <Field label="Scale">
        <select
          value={scale}
          onChange={(e) => setScale(e.target.value)}
          className="input-field text-sm"
        >
          {Object.keys(scales).map((k) => (
            <option key={k} value={k}>{k}</option>
          ))}
        </select>
      </Field>

      <div className="flex gap-2 ml-auto self-end">
        <Button variant="ghost" size="md" onClick={refresh} aria-label="Refresh">
          <RefreshCw className="w-4 h-4" />
        </Button>
        <Button onClick={handleUpscaleOne} disabled={!selected || processing}>
          {processing ? <Loader2 className="w-4 h-4 animate-spin" /> : <Maximize2 className="w-4 h-4" />}
          Upscale selected
        </Button>
        <Button variant="secondary" onClick={handleUpscaleAll} disabled={sources.length === 0 || batchProcessing}>
          {batchProcessing && <Loader2 className="w-4 h-4 animate-spin" />}
          Upscale all
        </Button>
      </div>
    </>
  );

  return (
    <div className="space-y-5">
      <div>
        <h1 className="font-display text-[28px] font-semibold tracking-[-0.02em] text-surface-100">
          Upscale
        </h1>
        <p className="text-sm text-surface-400 mt-1">
          Pull generated logos into higher resolution. Real-ESRGAN for quality, Lanczos for speed.
        </p>
      </div>

      <ToolPanel
        toolbar={toolbar}
        selected={selected}
        progress={batchProcessing ? progress : null}
        error={error}
        galleries={[
          {
            title: "Source",
            images: sources,
            selectedIndex: selectedIdx ?? undefined,
            onSelect: (i) => setSelectedIdx(i),
            emptyText: "No generated images found",
          },
          {
            title: "Upscaled",
            images: upscaled,
            emptyText: "No upscaled images yet",
          },
        ]}
      />
    </div>
  );
}
