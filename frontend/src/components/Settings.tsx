import { useEffect, useRef, useState } from "react";
import { Save, Loader2, Check, RotateCcw, Terminal, ChevronDown } from "lucide-react";
import { fetchSettings, fetchDefaultSettings, updateSettings, fetchClaudeCliStatus } from "../api";
import { loadLocalSettings, saveLocalSettings } from "../db";
import GlassPanel from "./ui/GlassPanel";
import Button from "./ui/Button";
import Field from "./ui/Field";
import { onCommand } from "../shell/commands";

export default function Settings() {
  const [llmModel, setLlmModel] = useState("");
  const [imageModels, setImageModels] = useState("");
  const [imagesPerModel, setImagesPerModel] = useState(2);
  const [useClaudeCli, setUseClaudeCli] = useState(false);
  const [cliAvailable, setCliAvailable] = useState<boolean | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  const [resetting, setResetting] = useState(false);
  const [error, setError] = useState("");
  const [refOpen, setRefOpen] = useState(false);

  useEffect(() => {
    (async () => {
      const local = await loadLocalSettings();
      if (local) {
        setLlmModel(local.llmModel);
        setImageModels(local.imageModels.join("\n"));
        setImagesPerModel(local.imagesPerModel);
        setUseClaudeCli(!!local.useClaudeCli);
        setLoading(false);
        try {
          await updateSettings({
            llm_model: local.llmModel,
            image_models: local.imageModels,
            images_per_model: local.imagesPerModel,
            use_claude_cli: !!local.useClaudeCli,
          });
        } catch {
          // backend not ready, ignore
        }
      } else {
        try {
          const s = await fetchSettings();
          setLlmModel(s.llm_model);
          setImageModels(s.image_models.join("\n"));
          setImagesPerModel(s.images_per_model);
          setUseClaudeCli(!!s.use_claude_cli);
          await saveLocalSettings({
            llmModel: s.llm_model,
            imageModels: s.image_models,
            imagesPerModel: s.images_per_model,
            useClaudeCli: !!s.use_claude_cli,
          });
        } catch (e) {
          setError(e instanceof Error ? e.message : "Failed to load settings");
        }
        setLoading(false);
      }

      try {
        const status = await fetchClaudeCliStatus();
        setCliAvailable(status.available);
      } catch {
        setCliAvailable(false);
      }
    })();
  }, []);

  const handleSave = async () => {
    setSaving(true);
    setSaved(false);
    setError("");
    try {
      const models = imageModels.split("\n").map((s) => s.trim()).filter(Boolean);
      await saveLocalSettings({
        llmModel, imageModels: models, imagesPerModel, useClaudeCli,
      });
      await updateSettings({
        llm_model: llmModel,
        image_models: models,
        images_per_model: imagesPerModel,
        use_claude_cli: useClaudeCli,
      });
      setSaved(true);
      setTimeout(() => setSaved(false), 2000);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Save failed");
    }
    setSaving(false);
  };

  const handleReset = async () => {
    if (!confirm("Reset all settings to defaults? This will overwrite your current settings.")) return;
    setResetting(true);
    setSaved(false);
    setError("");
    try {
      const defaults = await fetchDefaultSettings();
      setLlmModel(defaults.llm_model);
      setImageModels(defaults.image_models.join("\n"));
      setImagesPerModel(defaults.images_per_model);
      setUseClaudeCli(!!defaults.use_claude_cli);
      await saveLocalSettings({
        llmModel: defaults.llm_model,
        imageModels: defaults.image_models,
        imagesPerModel: defaults.images_per_model,
        useClaudeCli: !!defaults.use_claude_cli,
      });
      await updateSettings({
        llm_model: defaults.llm_model,
        image_models: defaults.image_models,
        images_per_model: defaults.images_per_model,
        use_claude_cli: !!defaults.use_claude_cli,
      });
    } catch (e) {
      setError(e instanceof Error ? e.message : "Reset failed");
    }
    setResetting(false);
  };

  const latest = useRef({ handleSave, handleReset });
  latest.current = { handleSave, handleReset };

  useEffect(() => {
    const unsubs = [
      onCommand("settings.save", () => latest.current.handleSave()),
      onCommand("settings.reset", () => latest.current.handleReset()),
    ];
    return () => unsubs.forEach((u) => u());
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64 text-surface-500">
        <Loader2 className="w-6 h-6 animate-spin" />
      </div>
    );
  }

  return (
    <div className="max-w-2xl mx-auto space-y-5">
      <div>
        <h1 className="font-display text-[28px] font-semibold tracking-[-0.02em] text-surface-100">
          Settings
        </h1>
        <p className="text-sm text-surface-400 mt-1">
          Live configuration. Changes persist locally and sync to the backend.
        </p>
      </div>

      <GlassPanel className="overflow-hidden">
        <div className="p-6 space-y-6">
          {/* Claude CLI sub-panel */}
          <div className="rounded-field border border-glass-border-strong bg-surface-900/40 p-4">
            <label className="flex items-start gap-3 cursor-pointer">
              <input
                type="checkbox"
                checked={useClaudeCli}
                onChange={(e) => setUseClaudeCli(e.target.checked)}
                className="mt-0.5 w-4 h-4 accent-accent shrink-0"
              />
              <div className="flex-1">
                <div className="flex items-center gap-2 text-sm font-medium text-surface-100">
                  <Terminal className="w-4 h-4" />
                  Use local Claude CLI
                  {cliAvailable === true && (
                    <span className="text-[10px] uppercase tracking-wider px-1.5 py-0.5 rounded bg-emerald-500/15 text-emerald-300 border border-emerald-500/20">
                      detected
                    </span>
                  )}
                  {cliAvailable === false && (
                    <span className="text-[10px] uppercase tracking-wider px-1.5 py-0.5 rounded bg-red-500/15 text-red-300 border border-red-500/20">
                      not found
                    </span>
                  )}
                </div>
                <p className="text-xs text-surface-400 mt-1.5 leading-relaxed">
                  Routes chat through <code className="text-surface-200 font-mono text-[11px]">claude -p</code> using your existing
                  subscription. Run <code className="text-surface-200 font-mono text-[11px]">claude login</code> first. The model below is ignored when this is on.
                </p>
              </div>
            </label>
          </div>

          <div className={useClaudeCli ? "opacity-50 pointer-events-none space-y-6" : "space-y-6"}>
            <Field
              label="LLM model"
              hint={'Used for prompt enhancement. Prefix with "local/" for local LLM (e.g. local/qwen3.5:9b).'}
            >
              <input
                type="text"
                value={llmModel}
                onChange={(e) => setLlmModel(e.target.value)}
                className="input-field w-full"
                placeholder="anthropic/claude-sonnet-4"
                disabled={useClaudeCli}
              />
            </Field>
          </div>

          <Field
            label="Image models"
            hint={
              <>
                One per line. Cloud uses OpenRouter. Local:{" "}
                <code className="font-mono">flux1-schnell</code>,{" "}
                <code className="font-mono">zimage-turbo</code>. Append{" "}
                <code className="font-mono">:N</code> to a local model to override its
                batch size — e.g. <code className="font-mono">zimage-turbo:8</code>{" "}
                generates 8 images per pipeline call.
              </>
            }
          >
            <textarea
              value={imageModels}
              onChange={(e) => setImageModels(e.target.value)}
              className="input-field w-full font-mono text-[13px] leading-relaxed"
              rows={5}
              placeholder="One model per line"
            />
          </Field>

          <Field
            label="Images per model"
            trailing={<span className="font-mono text-accent">{imagesPerModel}</span>}
          >
            <input
              type="range"
              min={1}
              max={100}
              value={imagesPerModel}
              onChange={(e) => setImagesPerModel(Number(e.target.value))}
              className="w-full"
            />
            <div className="flex justify-between text-[10px] text-surface-500 mt-1">
              <span>1</span>
              <span>100</span>
            </div>
          </Field>

          {error && (
            <p className="text-sm text-red-400 bg-red-400/10 border border-red-400/20 rounded-field px-3 py-2">{error}</p>
          )}
        </div>

        {/* Sticky footer */}
        <div className="flex items-center gap-2 px-6 py-4 border-t border-glass-border bg-surface-900/40">
          <Button onClick={handleSave} disabled={saving || resetting}>
            {saving ? <Loader2 className="w-4 h-4 animate-spin" /> : saved ? <Check className="w-4 h-4" /> : <Save className="w-4 h-4" />}
            {saved ? "Saved" : "Save settings"}
          </Button>
          <Button variant="secondary" onClick={handleReset} disabled={saving || resetting}>
            {resetting ? <Loader2 className="w-4 h-4 animate-spin" /> : <RotateCcw className="w-4 h-4" />}
            Reset to defaults
          </Button>
        </div>
      </GlassPanel>

      {/* Model reference disclosure */}
      <GlassPanel tone="soft" className="overflow-hidden">
        <button
          type="button"
          onClick={() => setRefOpen((o) => !o)}
          className="w-full flex items-center justify-between px-5 py-3 text-left hover:bg-glass transition-colors"
        >
          <span className="font-display text-[12px] font-semibold tracking-wider uppercase text-surface-300">
            Model reference
          </span>
          <ChevronDown className={`w-4 h-4 text-surface-400 transition-transform ${refOpen ? "rotate-180" : ""}`} />
        </button>
        {refOpen && (
          <div className="px-5 pb-5 grid sm:grid-cols-3 gap-5 text-sm text-surface-400 motion-safe:animate-fade-in">
            <div>
              <h4 className="text-surface-200 font-semibold mb-1.5 text-[12px] uppercase tracking-wider">Cloud images</h4>
              <ul className="space-y-1 text-xs font-mono">
                <li>openai/gpt-5-image-mini</li>
                <li>google/gemini-2.5-flash-image</li>
                <li>google/gemini-3.1-flash-image-preview</li>
              </ul>
            </div>
            <div>
              <h4 className="text-surface-200 font-semibold mb-1.5 text-[12px] uppercase tracking-wider">Local images</h4>
              <ul className="space-y-1 text-xs font-mono">
                <li>flux1-schnell</li>
                <li>zimage-turbo</li>
              </ul>
            </div>
            <div>
              <h4 className="text-surface-200 font-semibold mb-1.5 text-[12px] uppercase tracking-wider">LLMs</h4>
              <ul className="space-y-1 text-xs font-mono">
                <li>anthropic/claude-sonnet-4</li>
                <li>local/qwen3.5:9b</li>
                <li>local/gemma3:12b</li>
              </ul>
            </div>
          </div>
        )}
      </GlassPanel>
    </div>
  );
}
