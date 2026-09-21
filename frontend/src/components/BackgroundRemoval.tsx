import { useEffect, useRef, useState, useCallback } from "react";
import { RefreshCw, Eraser, Loader2 } from "lucide-react";
import {
  fetchAllSources,
  fetchCleaned,
  fetchBgOptions,
  bgRemoveOne,
  streamPost,
} from "../api";
import type { ImageInfo } from "../types";
import ToolPanel from "./ToolPanel";
import Field from "./ui/Field";
import Button from "./ui/Button";
import SegmentedControl from "./ui/SegmentedControl";
import { onCommand } from "../shell/commands";

type Method = "ai" | "color";

export default function BackgroundRemoval() {
  const [sources, setSources] = useState<ImageInfo[]>([]);
  const [cleaned, setCleaned] = useState<ImageInfo[]>([]);
  const [selectedIdx, setSelectedIdx] = useState<number | null>(null);
  const [method, setMethod] = useState<Method>("ai");
  const [models, setModels] = useState<Record<string, string>>({});
  const [model, setModel] = useState("");
  const [alphaMatting, setAlphaMatting] = useState(false);
  const [erodePixels, setErodePixels] = useState(0);
  const [tolerance, setTolerance] = useState(30);
  const [autoCorners, setAutoCorners] = useState(true);
  const [color, setColor] = useState("#ffffff");
  const [processing, setProcessing] = useState(false);
  const [batchProcessing, setBatchProcessing] = useState(false);
  const [progress, setProgress] = useState({ current: 0, total: 0, message: "" });
  const [error, setError] = useState("");

  const refresh = useCallback(async () => {
    const [src, cl, opts] = await Promise.all([
      fetchAllSources(),
      fetchCleaned(),
      fetchBgOptions(),
    ]);
    setSources(src);
    setCleaned(cl);
    setModels(opts.models);
    if (!model || !Object.keys(opts.models).includes(model)) {
      setModel(opts.default);
    }
  }, [model]);

  useEffect(() => { refresh(); }, []);  // eslint-disable-line react-hooks/exhaustive-deps

  const selected = selectedIdx !== null ? sources[selectedIdx] ?? null : null;

  const buildParams = (path: string) => ({
    path,
    method,
    erode_pixels: erodePixels,
    ...(method === "ai"
      ? { model, alpha_matting: alphaMatting }
      : { tolerance, auto_corners: autoCorners, color }),
  });

  const handleCleanOne = async () => {
    if (!selected || processing) return;
    setProcessing(true);
    setError("");
    try {
      await bgRemoveOne(buildParams(selected.path));
      const cl = await fetchCleaned();
      setCleaned(cl);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Background removal failed");
    }
    setProcessing(false);
  };

  const handleCleanAll = async () => {
    if (sources.length === 0 || batchProcessing) return;
    setBatchProcessing(true);
    setError("");
    setProgress({ current: 0, total: 0, message: "Starting batch processing..." });

    const body =
      method === "ai"
        ? { method, model, alpha_matting: alphaMatting, erode_pixels: erodePixels }
        : { method, tolerance, auto_corners: autoCorners, color, erode_pixels: erodePixels };

    await streamPost("/bg-remove/batch", body, {
      onProgress(cur, tot, msg) { setProgress({ current: cur, total: tot, message: msg }); },
      onResult() {
        fetchCleaned().then(setCleaned);
        setProgress({ current: 0, total: 0, message: "Batch complete" });
      },
      onError(msg) { setError(msg); },
    });

    setBatchProcessing(false);
  };

  const latest = useRef({ handleCleanOne, handleCleanAll, refresh });
  latest.current = { handleCleanOne, handleCleanAll, refresh };
  useEffect(() => {
    const unsubs = [
      onCommand("clean.selected", () => latest.current.handleCleanOne()),
      onCommand("clean.all", () => latest.current.handleCleanAll()),
      onCommand("clean.refresh", () => latest.current.refresh()),
    ];
    return () => unsubs.forEach((u) => u());
  }, []);

  const toolbar = (
    <>
      <SegmentedControl<Method>
        value={method}
        onChange={setMethod}
        options={[
          { value: "ai", label: "AI" },
          { value: "color", label: "Color" },
        ]}
      />
      <div className="flex gap-2 ml-auto self-end">
        <Button variant="ghost" onClick={refresh} aria-label="Refresh">
          <RefreshCw className="w-4 h-4" />
        </Button>
        <Button onClick={handleCleanOne} disabled={!selected || processing}>
          {processing ? <Loader2 className="w-4 h-4 animate-spin" /> : <Eraser className="w-4 h-4" />}
          Clean selected
        </Button>
        <Button variant="secondary" onClick={handleCleanAll} disabled={sources.length === 0 || batchProcessing}>
          {batchProcessing && <Loader2 className="w-4 h-4 animate-spin" />}
          Clean all
        </Button>
      </div>
    </>
  );

  const toolbarExtras = (
    <>
      {method === "ai" ? (
        <>
          <Field label="Model" className="min-w-[220px]">
            <select
              value={model}
              onChange={(e) => setModel(e.target.value)}
              className="input-field text-sm w-full"
            >
              {Object.entries(models).map(([k, v]) => (
                <option key={k} value={k}>{v}</option>
              ))}
            </select>
          </Field>
          <label className="flex items-center gap-2 cursor-pointer self-end pb-2.5">
            <input
              type="checkbox"
              checked={alphaMatting}
              onChange={(e) => setAlphaMatting(e.target.checked)}
              className="rounded accent-accent"
            />
            <span className="text-sm text-surface-300">Alpha matting</span>
          </label>
        </>
      ) : (
        <>
          <Field label="Tolerance" trailing={<span className="font-mono text-accent">{tolerance}</span>}>
            <input
              type="range"
              min={0}
              max={100}
              value={tolerance}
              onChange={(e) => setTolerance(Number(e.target.value))}
              className="w-40"
            />
          </Field>
          <label className="flex items-center gap-2 cursor-pointer self-end pb-2.5">
            <input
              type="checkbox"
              checked={autoCorners}
              onChange={(e) => setAutoCorners(e.target.checked)}
              className="rounded accent-accent"
            />
            <span className="text-sm text-surface-300">Auto-detect corners</span>
          </label>
          {!autoCorners && (
            <Field label="Color">
              <div className="flex items-center gap-2">
                <input
                  type="color"
                  value={color}
                  onChange={(e) => setColor(e.target.value)}
                  className="h-9 w-10 rounded-field cursor-pointer bg-transparent border border-glass-border-strong"
                />
                <input
                  type="text"
                  value={color}
                  onChange={(e) => setColor(e.target.value)}
                  className="input-field text-sm w-28 font-mono"
                  placeholder="#ffffff"
                />
              </div>
            </Field>
          )}
        </>
      )}
      <Field label="Erode" trailing={<span className="font-mono text-accent">{erodePixels}px</span>}>
        <input
          type="range"
          min={0}
          max={20}
          value={erodePixels}
          onChange={(e) => setErodePixels(Number(e.target.value))}
          className="w-32"
        />
      </Field>
    </>
  );

  return (
    <div className="space-y-5">
      <div>
        <h1 className="font-display text-[28px] font-semibold tracking-[-0.02em] text-surface-100">
          Clean background
        </h1>
        <p className="text-sm text-surface-400 mt-1">
          Knock out backgrounds with a neural model or by sampling a single color.
        </p>
      </div>

      <ToolPanel
        toolbar={toolbar}
        toolbarExtras={toolbarExtras}
        selected={selected}
        progress={batchProcessing ? progress : null}
        error={error}
        galleries={[
          {
            title: "Source",
            images: sources,
            selectedIndex: selectedIdx ?? undefined,
            onSelect: (i) => setSelectedIdx(i),
            emptyText: "No source images found",
          },
          {
            title: "Cleaned",
            images: cleaned,
            showChecker: true,
            emptyText: "No cleaned images yet",
          },
        ]}
      />
    </div>
  );
}
