import { useState, useRef, useEffect, useCallback, memo } from "react";
import { Virtuoso, type VirtuosoHandle } from "react-virtuoso";
import { VList } from "virtua";
import {
  Send, RotateCcw, Sparkles, Loader2, ImagePlus, X, Square,
  ChevronDown, ArrowDown, Check, Image as ImageIcon, Minus, Plus,
  History, Trash2, MessageSquare,
} from "lucide-react";
import { streamPost, uploadImages, imageUrl, fetchSettings } from "../api";
import type { ChatMessage, PromptVariation, GeneratedLogo, PromptBatch } from "../types";
import { chatReset } from "../api";
import ImageGallery from "./ImageGallery";
import Markdown from "./Markdown";
import GlassPanel from "./ui/GlassPanel";
import Button from "./ui/Button";
import ProgressLine from "./ui/ProgressLine";
import { onCommand } from "../shell/commands";
import { useRafBuffer } from "../hooks/useRafBuffer";
import {
  getCurrentSessionId,
  resetCurrentSession,
  setCurrentSessionId,
  loadMessages,
  addMessage,
  updateLastAssistantMessage,
  saveSessionMeta,
  loadSessionMeta,
  loadLocalSettings,
  listSessions,
  deleteSession,
  touchSession,
  type SessionSummary,
} from "../db";

function relativeTime(ts: number): string {
  const diff = Date.now() - ts;
  const min = 60_000, hour = 60 * min, day = 24 * hour;
  if (diff < min) return "just now";
  if (diff < hour) return `${Math.floor(diff / min)}m ago`;
  if (diff < day) return `${Math.floor(diff / hour)}h ago`;
  if (diff < 7 * day) return `${Math.floor(diff / day)}d ago`;
  return new Date(ts).toLocaleDateString();
}

export default function ChatDesigner() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const [batches, setBatches] = useState<PromptBatch[]>([]);
  // Selection is keyed as `${batchId}:${idx}` so it survives re-keying
  // and lets a single Set drive selection across all batches.
  const [selection, setSelection] = useState<Set<string>>(new Set());
  const [expandedBatches, setExpandedBatches] = useState<Set<string>>(new Set());
  const [logos, setLogos] = useState<GeneratedLogo[]>([]);
  const [isGenerating, setIsGenerating] = useState(false);
  const [progress, setProgress] = useState({ current: 0, total: 0, message: "" });
  const [error, setError] = useState("");
  const [attachedImages, setAttachedImages] = useState<{ file: File; preview: string }[]>([]);
  const [uploading, setUploading] = useState(false);
  // Output panel below the chat. Defaults to "prompts" so the section
  // is open as soon as there's anything to show. null = collapsed.
  const [outputTab, setOutputTab] = useState<"prompts" | "logos" | null>("prompts");
  const [dragging, setDragging] = useState(false);
  const [sessionId, setSessionId] = useState(getCurrentSessionId);
  const [imagesPerModel, setImagesPerModel] = useState(2);
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [sessionsOpen, setSessionsOpen] = useState(false);

  const [refImages, setRefImages] = useState<string[]>([]);
  const [selectedRefImages, setSelectedRefImages] = useState<string[]>([]);

  const [atBottom, setAtBottom] = useState(true);
  const virtuosoRef = useRef<VirtuosoHandle>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const dragCounter = useRef(0);
  const chatAbortRef = useRef<AbortController | null>(null);
  const genAbortRef = useRef<AbortController | null>(null);
  const sessionsPopoverRef = useRef<HTMLDivElement>(null);

  // Auto-resize textarea to fit content
  useEffect(() => {
    const el = inputRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${el.scrollHeight}px`;
  }, [input]);

  // Load persisted state for the current session
  useEffect(() => {
    (async () => {
      const stored = await loadMessages(sessionId);
      if (stored.length > 0) {
        setMessages(stored.map((m) => ({
          role: m.role,
          content: m.content,
          images: m.images,
        })));
      }
      const meta = await loadSessionMeta(sessionId);
      if (meta) {
        const loadedBatches = meta.batches as PromptBatch[];
        setBatches(loadedBatches);
        // No auto-selection on load. Restore prior selection from disk
        // if there is one; otherwise start empty.
        setSelection(new Set(meta.selection || []));
        // Default expansion: only the most recent batch is open. Earlier
        // batches stay collapsed so older sessions don't dump 200 prompts
        // into view.
        if (loadedBatches.length > 0) {
          setExpandedBatches(new Set([loadedBatches[loadedBatches.length - 1]!.id]));
        } else {
          setExpandedBatches(new Set());
        }
        setLogos(meta.logos as GeneratedLogo[]);
        setRefImages(meta.referenceImages);
        setSelectedRefImages(meta.selectedImages);
      }
    })();
  }, [sessionId]);

  // Load images-per-model default
  useEffect(() => {
    (async () => {
      const local = await loadLocalSettings();
      if (local) {
        setImagesPerModel(local.imagesPerModel);
        return;
      }
      try {
        const s = await fetchSettings();
        setImagesPerModel(s.images_per_model);
      } catch {
        // keep default
      }
    })();
  }, []);

  const refreshSessions = useCallback(async () => {
    const list = await listSessions();
    setSessions(list);
  }, []);

  useEffect(() => {
    if (sessionsOpen) refreshSessions();
  }, [sessionsOpen, refreshSessions]);

  useEffect(() => {
    if (!sessionsOpen) return;
    const onClick = (e: MouseEvent) => {
      if (!sessionsPopoverRef.current?.contains(e.target as Node)) {
        setSessionsOpen(false);
      }
    };
    window.addEventListener("mousedown", onClick);
    return () => window.removeEventListener("mousedown", onClick);
  }, [sessionsOpen]);

  const scrollToBottom = useCallback(() => {
    virtuosoRef.current?.scrollToIndex({
      index: "LAST",
      align: "end",
      behavior: "smooth",
    });
  }, []);

  // RAF-batched assistant content commit. Streaming tokens arrive far
  // faster than React can usefully render — coalesce to one update per
  // animation frame.
  const flushAssistant = useRafBuffer<string>((latest) => {
    setMessages((prev) => {
      if (prev.length === 0) return prev;
      const next = prev.slice();
      next[next.length - 1] = { role: "assistant", content: latest };
      return next;
    });
  });

  useEffect(() => {
    saveSessionMeta(sessionId, {
      batches,
      selection: Array.from(selection),
      logos,
      referenceImages: refImages,
      selectedImages: selectedRefImages,
    });
  }, [sessionId, batches, selection, logos, refImages, selectedRefImages]);

  // ── File handling ─────────────────────────────────────────────

  const handleFiles = (files: FileList | File[]) => {
    const accepted = Array.from(files).filter((f) =>
      /\.(png|jpe?g|webp|gif)$/i.test(f.name),
    );
    if (accepted.length === 0) return;
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

  // ── Drag and drop ─────────────────────────────────────────────

  const onDragEnter = (e: React.DragEvent) => {
    e.preventDefault();
    dragCounter.current++;
    if (e.dataTransfer.types.includes("Files")) setDragging(true);
  };
  const onDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    dragCounter.current--;
    if (dragCounter.current === 0) setDragging(false);
  };
  const onDragOver = (e: React.DragEvent) => e.preventDefault();
  const onDrop = (e: React.DragEvent) => {
    e.preventDefault();
    dragCounter.current = 0;
    setDragging(false);
    if (e.dataTransfer.files.length > 0) handleFiles(e.dataTransfer.files);
  };

  const toggleRefImage = (path: string) => {
    setSelectedRefImages((prev) =>
      prev.includes(path) ? prev.filter((p) => p !== path) : [...prev, path],
    );
  };
  const selectAllRef = () => setSelectedRefImages([...refImages]);
  const deselectAllRef = () => setSelectedRefImages([]);

  // ── Send ──────────────────────────────────────────────────────

  const handleSend = async () => {
    const msg = input.trim();
    if (!msg || isStreaming) return;

    setInput("");
    setError("");

    let newPaths: string[] = [];
    if (attachedImages.length > 0) {
      setUploading(true);
      try {
        const uploaded = await uploadImages(attachedImages.map((a) => a.file));
        newPaths = uploaded.map((u) => u.path);
      } catch {
        setError("Failed to upload images");
        setUploading(false);
        return;
      }
      setUploading(false);
    }

    if (newPaths.length > 0) {
      setRefImages((prev) => [...prev, ...newPaths]);
      setSelectedRefImages((prev) => [...prev, ...newPaths]);
    }

    const userMsg: ChatMessage = {
      role: "user",
      content: msg,
      images: newPaths.length > 0 ? newPaths : undefined,
    };
    setMessages((prev) => [...prev, userMsg]);
    await addMessage(sessionId, "user", msg, newPaths.length > 0 ? newPaths : undefined);

    attachedImages.forEach((a) => URL.revokeObjectURL(a.preview));
    setAttachedImages([]);
    setIsStreaming(true);

    let assistantContent = "";
    setMessages((prev) => [...prev, { role: "assistant", content: "" }]);
    await addMessage(sessionId, "assistant", "");

    const historyPayload = messages.map((m) => ({
      role: m.role, content: m.content, images: m.images,
    }));

    const body: Record<string, unknown> = { message: msg, history: historyPayload };
    if (newPaths.length > 0) body.images = newPaths;

    const abort = new AbortController();
    chatAbortRef.current = abort;

    await streamPost("/chat/send", body, {
      onToken(token) {
        assistantContent += token;
        flushAssistant(assistantContent);
      },
      onResult(data) {
        if (data.has_prompts && data.prompts) {
          const incoming = data.prompts as PromptVariation[];
          // Anchor this batch to the assistant message we just streamed
          // (the last message in the messages array is that assistant
          // turn — its index is messages.length here, since at this
          // point we've appended both the user msg and the assistant
          // placeholder).
          const newBatch: PromptBatch = {
            id: crypto.randomUUID(),
            messageIdx: messages.length + 1, // user + assistant just appended
            prompts: incoming,
            createdAt: Date.now(),
          };
          setBatches((prev) => [...prev, newBatch]);
          // Live UX: auto-select the freshly arrived batch (the user is
          // about to act on it). Loaded batches stay deselected.
          setSelection((prev) => {
            const next = new Set(prev);
            incoming.forEach((_, i) => next.add(`${newBatch.id}:${i}`));
            return next;
          });
          setExpandedBatches((prev) => new Set(prev).add(newBatch.id));
        }
      },
      onError(errMsg) { setError(errMsg); },
    }, { signal: abort.signal });

    await updateLastAssistantMessage(sessionId, assistantContent);
    chatAbortRef.current = null;
    setIsStreaming(false);
  };

  const handleCancelChat = () => {
    chatAbortRef.current?.abort();
    chatAbortRef.current = null;
    setIsStreaming(false);
  };

  const handleStopAny = () => {
    chatAbortRef.current?.abort();
    genAbortRef.current?.abort();
    chatAbortRef.current = null;
    genAbortRef.current = null;
    setIsStreaming(false);
    setIsGenerating(false);
  };

  const handleCancelGenerate = () => {
    genAbortRef.current?.abort();
    genAbortRef.current = null;
    setIsGenerating(false);
    setProgress({ current: 0, total: 0, message: "Cancelled" });
  };

  // ── Batch selection helpers ──────────────────────────────────
  const toggleSelection = useCallback((batchId: string, idx: number) => {
    setSelection((prev) => {
      const key = `${batchId}:${idx}`;
      const next = new Set(prev);
      if (next.has(key)) next.delete(key); else next.add(key);
      return next;
    });
  }, []);

  const selectAllInBatch = useCallback((batch: PromptBatch) => {
    setSelection((prev) => {
      const next = new Set(prev);
      batch.prompts.forEach((_, i) => next.add(`${batch.id}:${i}`));
      return next;
    });
  }, []);

  const deselectAllInBatch = useCallback((batch: PromptBatch) => {
    setSelection((prev) => {
      const next = new Set(prev);
      batch.prompts.forEach((_, i) => next.delete(`${batch.id}:${i}`));
      return next;
    });
  }, []);

  // Shift-click range select: set every index in [start, end] to
  // `value`. Used by BatchGroup to mirror standard list-selection UX.
  const setRangeInBatch = useCallback(
    (batchId: string, start: number, end: number, value: boolean) => {
      const lo = Math.min(start, end);
      const hi = Math.max(start, end);
      setSelection((prev) => {
        const next = new Set(prev);
        for (let i = lo; i <= hi; i++) {
          const key = `${batchId}:${i}`;
          if (value) next.add(key); else next.delete(key);
        }
        return next;
      });
    },
    [],
  );

  const toggleBatchExpanded = useCallback((batchId: string) => {
    setExpandedBatches((prev) => {
      const next = new Set(prev);
      if (next.has(batchId)) next.delete(batchId); else next.add(batchId);
      return next;
    });
  }, []);

  // Run a generation for the prompts currently selected within a single
  // batch. Reuses the same /generate stream as the previous all-prompts
  // flow.
  const handleGenerateBatch = async (batchId: string) => {
    const batch = batches.find((b) => b.id === batchId);
    if (!batch || isGenerating) return;
    const selected = batch.prompts.filter((_, i) => selection.has(`${batchId}:${i}`));
    if (selected.length === 0) {
      setError("Select at least one prompt to generate.");
      return;
    }

    setIsGenerating(true);
    setError("");
    setProgress({ current: 0, total: 0, message: "Starting generation..." });

    const body: Record<string, unknown> = {
      prompts: selected,
      images_per_model: imagesPerModel,
    };
    if (selectedRefImages.length > 0) body.images = selectedRefImages;

    const abort = new AbortController();
    genAbortRef.current = abort;

    await streamPost("/generate", body, {
      onProgress(current, total, message) {
        setProgress({ current, total, message });
      },
      onResult(data) {
        const incoming = (data.logos as GeneratedLogo[]) || [];
        setLogos((prev) => {
          const seen = new Set(prev.map((l) => l.path));
          const unique = incoming.filter((l) => !seen.has(l.path));
          return [...prev, ...unique];
        });
        const totalTime = (data.total_time as number) || 0;
        const errorCount = (data.errors as string[])?.length || 0;
        setProgress({
          current: 0, total: 0,
          message: `Done in ${totalTime.toFixed(1)}s${errorCount > 0 ? ` (${errorCount} errors)` : ""}`,
        });
      },
      onError(errMsg) { setError(errMsg); },
    }, { signal: abort.signal });

    genAbortRef.current = null;
    setIsGenerating(false);
  };

  // ── Session management ───────────────────────────────────────

  const clearLocalState = () => {
    chatAbortRef.current?.abort();
    genAbortRef.current?.abort();
    chatAbortRef.current = null;
    genAbortRef.current = null;
    setMessages([]);
    setBatches([]);
    setSelection(new Set());
    setExpandedBatches(new Set());
    setLogos([]);
    setRefImages([]);
    setSelectedRefImages([]);
    setError("");
    setProgress({ current: 0, total: 0, message: "" });
    setIsStreaming(false);
    setIsGenerating(false);
    attachedImages.forEach((a) => URL.revokeObjectURL(a.preview));
    setAttachedImages([]);
  };

  const handleReset = async () => {
    await chatReset();
    clearLocalState();
    const newId = resetCurrentSession();
    setSessionId(newId);
  };

  const handleNewSession = async () => {
    setSessionsOpen(false);
    await handleReset();
  };

  const handleSwitchSession = async (id: string) => {
    if (id === sessionId) {
      setSessionsOpen(false);
      return;
    }
    clearLocalState();
    setCurrentSessionId(id);
    setSessionId(id);
    setSessionsOpen(false);
  };

  const handleDeleteSession = async (id: string) => {
    await deleteSession(id);
    if (id === sessionId) {
      clearLocalState();
      const newId = resetCurrentSession();
      setSessionId(newId);
    }
    await refreshSessions();
  };

  useEffect(() => {
    if (messages.length > 0) touchSession(sessionId);
  }, [sessionId, messages.length]);

  // ── Command subscriptions ────────────────────────────────────
  const latest = useRef({ handleNewSession, handleReset, handleStopAny });
  latest.current = { handleNewSession, handleReset, handleStopAny };
  useEffect(() => {
    const unsubs = [
      onCommand("chat.new", () => latest.current.handleNewSession()),
      onCommand("chat.reset", () => latest.current.handleReset()),
      onCommand("chat.stop", () => latest.current.handleStopAny()),
    ];
    return () => unsubs.forEach((u) => u());
  }, []);

  const hasRightPanel = refImages.length > 0;
  const promptCount = batches.reduce((acc, b) => acc + b.prompts.length, 0);

  const firstUserMsg = messages.find((m) => m.role === "user");
  const rawTitle = (firstUserMsg?.content || "").replace(/\s+/g, " ").trim();
  const currentTitle = rawTitle
    ? rawTitle.slice(0, 50) + (rawTitle.length > 50 ? "…" : "")
    : "New chat";

  const toggleOutputTab = (tab: "prompts" | "logos") => {
    setOutputTab((prev) => (prev === tab ? null : tab));
  };

  return (
    // Outer is NOT h-full — children stack to natural height and the
    // tab wrapper handles vertical scroll. The chat row below is sized
    // to the full visible viewport area so it never shrinks regardless
    // of what the output panel below it shows.
    <div className="flex flex-col gap-2">
      {/* Chat row pinned to full visible height (viewport minus topbar
          48px + workspace py-4 32px = 80px = 5rem). */}
      <div className="flex gap-3 min-h-0 h-[calc(100vh-5rem)]">
        {/* Chat panel */}
        <GlassPanel
          className={`flex flex-col flex-1 min-w-0 min-h-0 relative transition-colors duration-150 ${
            dragging ? "border-accent/60 ring-2 ring-accent/30" : ""
          }`}
          onDragEnter={onDragEnter}
          onDragLeave={onDragLeave}
          onDragOver={onDragOver}
          onDrop={onDrop}
        >
          {dragging && (
            <div className="absolute inset-0 z-10 flex items-center justify-center bg-surface-1000/60 rounded-panel pointer-events-none backdrop-blur-sm">
              <div className="flex flex-col items-center gap-2 text-accent">
                <ImagePlus className="w-10 h-10" />
                <p className="text-sm font-medium">Drop reference images here</p>
              </div>
            </div>
          )}

          {/* Session header */}
          <div className="px-4 py-2.5 border-b border-glass-border flex items-center justify-between shrink-0">
            <div className="relative" ref={sessionsPopoverRef}>
              <button
                onClick={() => setSessionsOpen((o) => !o)}
                className="flex items-center gap-1.5 text-sm text-surface-300 hover:text-surface-100 px-2 py-1 rounded-field hover:bg-glass transition-colors"
                title="Previous sessions"
              >
                <History className="w-3.5 h-3.5" />
                <span className="max-w-[240px] truncate">{currentTitle}</span>
                <ChevronDown className="w-3 h-3 opacity-60" />
              </button>

              {sessionsOpen && (
                <div className="absolute top-full left-0 mt-2 w-80 max-h-96 overflow-y-auto glass-panel-strong rounded-panel z-20 py-1">
                  {sessions.length === 0 && (
                    <div className="px-3 py-6 text-xs text-surface-500 text-center">No saved sessions yet.</div>
                  )}
                  {sessions.map((s) => {
                    const isActive = s.id === sessionId;
                    return (
                      <div
                        key={s.id}
                        className={`group flex items-start gap-2 px-3 py-2 cursor-pointer transition-colors ${
                          isActive ? "bg-accent/10" : "hover:bg-glass"
                        }`}
                        onClick={() => handleSwitchSession(s.id)}
                      >
                        <MessageSquare className={`w-3.5 h-3.5 mt-0.5 shrink-0 ${isActive ? "text-accent" : "text-surface-500"}`} />
                        <div className="flex-1 min-w-0">
                          <p className={`text-sm truncate ${isActive ? "text-surface-100 font-medium" : "text-surface-200"}`}>
                            {s.title}
                          </p>
                          <p className="text-[10px] text-surface-500 mt-0.5">
                            {relativeTime(s.lastActivityAt)} · {s.messageCount} msg
                          </p>
                        </div>
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            if (confirm("Delete this session? This cannot be undone.")) {
                              handleDeleteSession(s.id);
                            }
                          }}
                          className="opacity-0 group-hover:opacity-100 text-surface-500 hover:text-red-400 transition-all p-1"
                          title="Delete session"
                        >
                          <Trash2 className="w-3.5 h-3.5" />
                        </button>
                      </div>
                    );
                  })}
                </div>
              )}
            </div>

            <div className="flex items-center gap-1.5">
              {isGenerating && (
                <button
                  onClick={handleCancelGenerate}
                  className="flex items-center gap-1.5 text-xs text-red-300 hover:text-red-200 px-2 py-1 rounded-field hover:bg-red-500/10 transition-colors"
                  title="Stop current generation"
                >
                  <Square className="w-3 h-3" fill="currentColor" />
                  Stop
                </button>
              )}
              <button
                onClick={handleNewSession}
                className="flex items-center gap-1.5 text-xs text-surface-400 hover:text-accent px-2 py-1 rounded-field hover:bg-glass transition-colors"
                title="Start a new session"
              >
                <Plus className="w-3.5 h-3.5" />
                New
              </button>
            </div>
          </div>

          {/* Messages — virtualized with smart autoscroll */}
          <div className="relative flex-1 min-h-0">
            {messages.length === 0 ? (
              <div className="flex flex-col items-center justify-center h-full text-center text-surface-500 gap-3 p-5">
                <Sparkles className="w-9 h-9 text-surface-600" />
                <p className="text-sm max-w-sm leading-relaxed">
                  Describe your brand, product, or idea. Drag & drop reference images
                  or use the attach button. The AI designer will help you craft the perfect logo.
                </p>
              </div>
            ) : (
              <Virtuoso
                ref={virtuosoRef}
                data={messages}
                className="h-full"
                followOutput={(isAtBottom) => (isAtBottom ? "smooth" : false)}
                atBottomStateChange={setAtBottom}
                atBottomThreshold={80}
                initialTopMostItemIndex={Math.max(messages.length - 1, 0)}
                increaseViewportBy={{ top: 200, bottom: 400 }}
                itemContent={(i, msg) => (
                  <MessageRow
                    msg={msg}
                    isLast={i === messages.length - 1}
                    isStreaming={isStreaming}
                  />
                )}
                computeItemKey={(i, msg) => `${i}-${msg.role}`}
              />
            )}

            {/* Scroll-to-bottom pill */}
            {!atBottom && messages.length > 0 && (
              <button
                onClick={scrollToBottom}
                className="absolute bottom-3 left-1/2 -translate-x-1/2 z-10
                           flex items-center gap-1.5 px-3 py-1.5 rounded-pill
                           bg-glass-strong border border-glass-border-strong
                           text-surface-100 text-xs shadow-glass
                           hover:bg-surface-800 transition-colors
                           motion-safe:animate-fade-in"
                style={{ backdropFilter: "blur(12px)", WebkitBackdropFilter: "blur(12px)" }}
                aria-label="Scroll to latest"
              >
                <ArrowDown className="w-3.5 h-3.5" />
                Latest
              </button>
            )}
          </div>

          {/* Pending attachments */}
          {attachedImages.length > 0 && (
            <div className="px-4 pt-3 flex gap-2 flex-wrap shrink-0">
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

          {/* Input */}
          <div className="border-t border-glass-border p-3 flex gap-2 items-end shrink-0">
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
              size="md"
              onClick={() => fileInputRef.current?.click()}
              title="Attach reference images"
              disabled={isStreaming}
            >
              <ImagePlus className="w-4 h-4" />
            </Button>
            <textarea
              ref={inputRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  handleSend();
                }
              }}
              placeholder="Describe your logo idea… (Shift+Enter for newline)"
              className="input-field flex-1 resize-none max-h-40"
              rows={1}
              disabled={isStreaming}
            />
            {isStreaming ? (
              <Button variant="danger" onClick={handleCancelChat} title="Stop generating">
                <Square className="w-4 h-4" fill="currentColor" />
              </Button>
            ) : (
              <Button
                onClick={handleSend}
                disabled={uploading || !input.trim()}
              >
                {uploading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Send className="w-4 h-4" />}
              </Button>
            )}
            <Button variant="ghost" onClick={handleReset} title="New session">
              <RotateCcw className="w-4 h-4" />
            </Button>
          </div>
        </GlassPanel>

        {/* Right panel */}
        {hasRightPanel && (
          <GlassPanel className="w-80 shrink-0 flex flex-col motion-safe:animate-fade-in min-h-0">
            {refImages.length > 0 && (
              <div className="shrink-0 border-b border-glass-border">
                <div className="px-4 py-3 flex items-center justify-between">
                  <h3 className="font-display text-[11px] font-semibold uppercase tracking-wider text-surface-300 flex items-center gap-2">
                    <ImageIcon className="w-3.5 h-3.5" />
                    References
                    <span className="text-[10px] text-surface-500 font-body normal-case font-normal">
                      {selectedRefImages.length}/{refImages.length}
                    </span>
                  </h3>
                  <div className="flex gap-1">
                    <button onClick={selectAllRef} className="text-[10px] text-surface-400 hover:text-accent px-1.5 py-0.5 rounded hover:bg-glass transition-colors">
                      All
                    </button>
                    <button onClick={deselectAllRef} className="text-[10px] text-surface-400 hover:text-accent px-1.5 py-0.5 rounded hover:bg-glass transition-colors">
                      None
                    </button>
                  </div>
                </div>
                <div className="px-4 pb-3 flex gap-2 flex-wrap max-h-36 overflow-y-auto">
                  {refImages.map((path) => {
                    const isSelected = selectedRefImages.includes(path);
                    return (
                      <button
                        key={path}
                        onClick={() => toggleRefImage(path)}
                        className={`relative group w-14 h-14 rounded-field overflow-hidden border-2 transition-all duration-150 ${
                          isSelected
                            ? "border-accent shadow-glow"
                            : "border-glass-border-strong opacity-50 hover:opacity-90"
                        }`}
                      >
                        <img src={imageUrl(path)} alt="ref" className="w-full h-full object-cover" />
                        {isSelected && (
                          <div className="absolute top-0.5 right-0.5 w-4 h-4 bg-accent rounded-full flex items-center justify-center">
                            <Check className="w-2.5 h-2.5 text-accent-ink" strokeWidth={3} />
                          </div>
                        )}
                      </button>
                    );
                  })}
                </div>
              </div>
            )}

          </GlassPanel>
        )}
      </div>

      {/* ── Output panel (after the chat) ─────────────────────
          Tab pills sit on a slim always-visible header bar so the
          chat keeps the rest of the column. Click a pill to expand
          its tab; click it again to collapse. Body height is clamped
          so the chat is never crushed. */}
      {(batches.length > 0 || logos.length > 0 || isGenerating || error) && (
        <div className="shrink-0 flex flex-col">
          <GlassPanel tone="soft" className="flex flex-col overflow-hidden">
            {/* Header bar */}
            <div className="flex items-center gap-2 px-3 py-2 border-b border-glass-border">
              <div className="flex items-center gap-1.5">
                <TabPill
                  active={outputTab === "prompts"}
                  onClick={() => toggleOutputTab("prompts")}
                  disabled={batches.length === 0}
                  count={promptCount}
                  selected={selection.size}
                  label="Prompts"
                />
                <TabPill
                  active={outputTab === "logos"}
                  onClick={() => toggleOutputTab("logos")}
                  disabled={logos.length === 0 && !isGenerating}
                  count={logos.length}
                  label="Logos"
                />
                {isGenerating && progress.total > 0 && (
                  <span className="ml-2 text-[11px] text-surface-400 font-mono">
                    {progress.current}/{progress.total}
                  </span>
                )}
              </div>

              <div className="ml-auto flex items-center gap-2">
                <div className="flex items-center gap-1" title="Images per model (1-100)">
                  <span className="text-[10px] uppercase tracking-wider text-surface-500 mr-1">imgs/model</span>
                  <button
                    onClick={() => setImagesPerModel((v) => Math.max(1, v - 1))}
                    disabled={isGenerating || imagesPerModel <= 1}
                    className="w-5 h-5 rounded-field bg-glass hover:bg-glass-strong disabled:opacity-40 disabled:cursor-not-allowed flex items-center justify-center"
                  >
                    <Minus className="w-3 h-3" />
                  </button>
                  <input
                    type="number"
                    min={1}
                    max={100}
                    value={imagesPerModel}
                    disabled={isGenerating}
                    onChange={(e) => {
                      const raw = parseInt(e.target.value, 10);
                      if (!Number.isFinite(raw)) return;
                      setImagesPerModel(Math.max(1, Math.min(100, raw)));
                    }}
                    onBlur={(e) => {
                      // Snap to a valid value if user cleared the field.
                      const raw = parseInt(e.target.value, 10);
                      if (!Number.isFinite(raw)) setImagesPerModel(1);
                    }}
                    onFocus={(e) => e.currentTarget.select()}
                    className="w-9 h-5 px-1 text-center text-xs font-mono text-surface-100 tabular-nums
                               bg-surface-900/60 border border-glass-border rounded
                               focus:outline-none focus:border-accent/60 focus:ring-1 focus:ring-accent/30
                               disabled:opacity-50
                               [appearance:textfield]
                               [&::-webkit-outer-spin-button]:appearance-none
                               [&::-webkit-inner-spin-button]:appearance-none"
                  />
                  <button
                    onClick={() => setImagesPerModel((v) => Math.min(100, v + 1))}
                    disabled={isGenerating || imagesPerModel >= 100}
                    className="w-5 h-5 rounded-field bg-glass hover:bg-glass-strong disabled:opacity-40 disabled:cursor-not-allowed flex items-center justify-center"
                  >
                    <Plus className="w-3 h-3" />
                  </button>
                </div>
                {isGenerating && (
                  <Button variant="danger" size="sm" onClick={handleCancelGenerate}>
                    <Square className="w-3 h-3" fill="currentColor" />
                    Stop
                  </Button>
                )}
              </div>
            </div>

            {/* Progress strip — visible when generating, regardless of
                whether the body is expanded. */}
            {isGenerating && progress.total > 0 && (
              <div className="px-3 py-2 border-b border-glass-border">
                <ProgressLine current={progress.current} total={progress.total} message={progress.message} />
              </div>
            )}

            {error && outputTab === null && (
              <p className="m-3 text-sm text-red-400 bg-red-400/10 border border-red-400/20 rounded-field px-3 py-2">
                {error}
              </p>
            )}

            {/* Body sizes to its content. Chat row above is pinned at
                full visible height, so the output panel sits below the
                fold; the tab wrapper scrolls to bring it into view. No
                cap needed — output never crushes chat. */}
            {outputTab !== null && (
              <div className="motion-safe:animate-fade-in">
                {outputTab === "prompts" && (
                  <PromptsTab
                    batches={batches}
                    messages={messages}
                    expanded={expandedBatches}
                    selection={selection}
                    isGenerating={isGenerating}
                    refsCount={selectedRefImages.length}
                    onToggleExpand={toggleBatchExpanded}
                    onToggleSelect={toggleSelection}
                    onSelectRange={setRangeInBatch}
                    onSelectAll={selectAllInBatch}
                    onSelectNone={deselectAllInBatch}
                    onGenerateBatch={handleGenerateBatch}
                  />
                )}
                {outputTab === "logos" && (
                  <div className="p-4">
                    {error && (
                      <p className="mb-3 text-sm text-red-400 bg-red-400/10 border border-red-400/20 rounded-field px-3 py-2">
                        {error}
                      </p>
                    )}
                    {logos.length > 0 ? (
                      <ImageGallery images={logos} columns={5} emptyText="No logos generated yet" />
                    ) : (
                      <p className="text-sm text-surface-500 text-center py-8">
                        {isGenerating ? "Generating…" : "No logos yet."}
                      </p>
                    )}
                  </div>
                )}
              </div>
            )}
          </GlassPanel>
        </div>
      )}
    </div>
  );
}

// ── Tab pill ──────────────────────────────────────────────────

function TabPill({
  active,
  onClick,
  disabled,
  count,
  selected,
  label,
}: {
  active: boolean;
  onClick: () => void;
  disabled?: boolean;
  count: number;
  /** When provided, the badge reads "selected / count" so the selection
      total stays visible regardless of which tab is open. */
  selected?: number;
  label: string;
}) {
  const showSelected = selected !== undefined;
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      className={`
        inline-flex items-center gap-1.5 px-2.5 h-7 rounded-pill text-[12px] font-medium
        transition-all duration-150
        disabled:opacity-40 disabled:cursor-not-allowed
        ${
          active
            ? "bg-accent text-accent-ink shadow-glass-sm"
            : "text-surface-300 hover:text-surface-100 hover:bg-glass"
        }
      `}
    >
      {label}
      {showSelected ? (
        <span
          className={`inline-flex items-center text-[10px] font-mono px-1 rounded-sm ${
            active ? "bg-accent-ink/10" : "bg-surface-800/60"
          }`}
        >
          <span
            className={`font-semibold ${
              active
                ? "text-accent-ink"
                : selected! > 0
                ? "text-accent"
                : "text-surface-400"
            }`}
          >
            {selected}
          </span>
          <span className={active ? "text-accent-ink/60" : "text-surface-500"}>
            /{count}
          </span>
        </span>
      ) : (
        <span
          className={`text-[10px] font-mono px-1 rounded-sm ${
            active ? "bg-accent-ink/15 text-accent-ink" : "text-surface-500"
          }`}
        >
          {count}
        </span>
      )}
    </button>
  );
}

// ── Prompts tab body ──────────────────────────────────────────
// All batches grouped by source. Each batch is a labeled section.
// Within a batch, prompts render as a compact 2-column grid where
// clicking a card toggles selection.

interface PromptsTabProps {
  batches: PromptBatch[];
  messages: ChatMessage[];
  expanded: Set<string>;
  selection: Set<string>;
  isGenerating: boolean;
  refsCount: number;
  onToggleExpand: (batchId: string) => void;
  onToggleSelect: (batchId: string, idx: number) => void;
  onSelectRange: (batchId: string, start: number, end: number, value: boolean) => void;
  onSelectAll: (batch: PromptBatch) => void;
  onSelectNone: (batch: PromptBatch) => void;
  onGenerateBatch: (batchId: string) => void;
}

// Look up the user message that produced a batch — the prompts come
// from the assistant turn at messageIdx, and the user prompt that
// triggered it sits at messageIdx - 1.
function userPromptFor(batch: PromptBatch, messages: ChatMessage[]): string | null {
  if (batch.messageIdx <= 0) return null;
  const userMsg = messages[batch.messageIdx - 1];
  if (!userMsg || userMsg.role !== "user") return null;
  const cleaned = userMsg.content.replace(/\s+/g, " ").trim();
  if (!cleaned) return null;
  return cleaned.length > 60 ? cleaned.slice(0, 60) + "…" : cleaned;
}

function PromptsTab({
  batches, messages, expanded, selection, isGenerating, refsCount,
  onToggleExpand, onToggleSelect, onSelectRange,
  onSelectAll, onSelectNone, onGenerateBatch,
}: PromptsTabProps) {
  // Show newest batch first.
  const ordered = [...batches].sort((a, b) => b.createdAt - a.createdAt);
  return (
    <div className="p-3 space-y-3">
      {ordered.map((batch) => (
        <BatchGroup
          key={batch.id}
          batch={batch}
          sourceLabel={userPromptFor(batch, messages)}
          expanded={expanded.has(batch.id)}
          selection={selection}
          isGenerating={isGenerating}
          refsCount={refsCount}
          onToggleExpand={() => onToggleExpand(batch.id)}
          onToggleSelect={(idx) => onToggleSelect(batch.id, idx)}
          onSelectRange={(start, end, value) => onSelectRange(batch.id, start, end, value)}
          onSelectAll={() => onSelectAll(batch)}
          onSelectNone={() => onSelectNone(batch)}
          onGenerate={() => onGenerateBatch(batch.id)}
        />
      ))}
    </div>
  );
}

// ── Message row ──────────────────────────────────────────────
// Content lives in a centered column so the chat doesn't read as
// wide-and-sprawling on big screens. The actual heavy work (markdown
// parse) happens in the Markdown component which uses block-level
// memoization, so an unmemoized row here is fine — only visible rows
// render anyway thanks to Virtuoso.

interface MessageRowProps {
  msg: ChatMessage;
  isLast: boolean;
  isStreaming: boolean;
}

function MessageRow({ msg, isLast, isStreaming }: MessageRowProps) {
  const showCursor = isLast && isStreaming && msg.role === "assistant";
  return (
    <div className="px-5 py-2.5">
      <div className="max-w-[760px] mx-auto">
        <div className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}>
          {msg.role === "user" ? (
            <div className="max-w-[85%] rounded-panel rounded-br-md px-4 py-2.5 text-sm leading-relaxed bg-accent/12 text-surface-50 border border-accent/20">
              <p className="whitespace-pre-wrap">{msg.content}</p>
              {msg.images && msg.images.length > 0 && (
                <div className="flex gap-1.5 mt-2 flex-wrap">
                  {msg.images.map((path, j) => (
                    <img
                      key={j}
                      src={imageUrl(path)}
                      alt="attached"
                      className="w-14 h-14 object-cover rounded-field border border-accent/30"
                    />
                  ))}
                </div>
              )}
            </div>
          ) : (
            <div className="w-full border-l-2 border-accent/30 pl-4 text-[14px] leading-relaxed text-surface-200">
              <Markdown content={msg.content} />
              {showCursor && (
                <span className="inline-block w-1.5 h-4 bg-accent ml-1 animate-pulse rounded-sm align-text-bottom" />
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// ── Batch group ─────────────────────────────────────────────
// One batch rendered as its own labeled group. Header shows what
// produced it (assistant turn or "loaded from earlier session"),
// selection count, and inline All/None + Generate. Body is a tight
// 2-column grid of compact prompt tiles — clicking a tile toggles
// selection, hovering reveals the full prompt as a tooltip.

interface BatchGroupProps {
  batch: PromptBatch;
  /** Snippet of the user prompt that produced this batch, or null if
      the batch is orphaned (legacy session) or the source can't be
      resolved. Stays referentially stable across streaming updates so
      the memo'd group skips re-renders. */
  sourceLabel: string | null;
  expanded: boolean;
  selection: Set<string>;
  isGenerating: boolean;
  refsCount: number;
  onToggleExpand: () => void;
  onToggleSelect: (idx: number) => void;
  onSelectRange: (start: number, end: number, value: boolean) => void;
  onSelectAll: () => void;
  onSelectNone: () => void;
  onGenerate: () => void;
}

function BatchGroupInner({
  batch, sourceLabel, expanded, selection, isGenerating, refsCount,
  onToggleExpand, onToggleSelect, onSelectRange,
  onSelectAll, onSelectNone, onGenerate,
}: BatchGroupProps) {
  // Anchor for shift-click range select. Set to the last index the
  // user single-clicked. Reset when the batch identity changes.
  const anchorRef = useRef<number | null>(null);
  useEffect(() => {
    anchorRef.current = null;
  }, [batch.id]);

  const handleClick = (idx: number, e: React.MouseEvent) => {
    if (e.shiftKey && anchorRef.current !== null) {
      // Use the anchor's selected state to decide whether the range
      // becomes selected or deselected — mirrors finder/file-manager UX.
      const anchorKey = `${batch.id}:${anchorRef.current}`;
      const value = selection.has(anchorKey);
      onSelectRange(anchorRef.current, idx, value);
    } else {
      onToggleSelect(idx);
      anchorRef.current = idx;
    }
  };
  const total = batch.prompts.length;
  let selectedCount = 0;
  for (let i = 0; i < total; i++) {
    if (selection.has(`${batch.id}:${i}`)) selectedCount++;
  }
  const time = new Date(batch.createdAt).toLocaleTimeString([], {
    hour: "numeric", minute: "2-digit",
  });
  const isOrphan = batch.messageIdx < 0;
  const turnLabel = isOrphan
    ? "Earlier prompts"
    : `Turn ${Math.floor(batch.messageIdx / 2) + 1}`;
  // Prefer showing a snippet of the user prompt that produced this
  // batch — much clearer than a bare turn number.
  const subtitle = sourceLabel ?? turnLabel;
  const showTurnHint = !isOrphan && sourceLabel !== null;

  return (
    <section className="rounded-panel border border-glass-border bg-surface-900/30 overflow-hidden">
      <header className="flex items-center gap-2 px-3 py-2 border-b border-glass-border">
        <button
          type="button"
          onClick={onToggleExpand}
          className="flex items-center gap-2 min-w-0 -ml-1 px-1 py-0.5 rounded hover:bg-glass transition-colors text-left"
        >
          <ChevronDown className={`w-3.5 h-3.5 text-surface-400 shrink-0 transition-transform ${expanded ? "" : "-rotate-90"}`} />
          {showTurnHint && (
            <span className="text-[10px] font-mono uppercase tracking-wider text-surface-500 shrink-0">
              {turnLabel}
            </span>
          )}
          <span className="text-[12px] font-semibold text-surface-100 truncate" title={subtitle}>
            {subtitle}
          </span>
          <span className="text-[11px] text-surface-500 font-mono shrink-0">{selectedCount}/{total}</span>
          <span className="text-[10px] text-surface-500 font-mono shrink-0">· {time}</span>
        </button>
        <div className="ml-auto flex items-center gap-1">
          <button onClick={onSelectAll} className="text-[10px] text-surface-400 hover:text-accent px-1.5 py-0.5 rounded hover:bg-glass transition-colors">
            All
          </button>
          <button onClick={onSelectNone} className="text-[10px] text-surface-400 hover:text-accent px-1.5 py-0.5 rounded hover:bg-glass transition-colors">
            None
          </button>
          <Button
            size="sm"
            onClick={onGenerate}
            disabled={isGenerating || selectedCount === 0}
            className="ml-1.5"
          >
            <Sparkles className="w-3.5 h-3.5" />
            Generate {selectedCount}
            {refsCount > 0 ? ` · ${refsCount} refs` : ""}
          </Button>
        </div>
      </header>

      {expanded && (
        // Contained virtualized list. Height grows with content but is
        // clamped to min(70vh, 720px) so a huge batch can't blow the
        // page open while still giving lots of visible cards on tall
        // displays. virtua handles dynamic-size measurement itself.
        <div className="p-2 select-none">
          <VList
            style={{
              height: `min(${batch.prompts.length * 90 + 12}px, 70vh, 720px)`,
            }}
          >
            {batch.prompts.map((p, i) => {
              const key = `${batch.id}:${i}`;
              const isSelected = selection.has(key);
              return (
                <div key={i} className="pb-1.5">
                  <button
                    type="button"
                    onClick={(e) => handleClick(i, e)}
                    onMouseDown={(e) => {
                      // Prevent text selection when shift-clicking.
                      if (e.shiftKey) e.preventDefault();
                    }}
                    title={`${p.prompt}\n\nTip: shift-click to select a range.`}
                    className={`group w-full text-left rounded-field p-2.5 transition-all border ${
                      isSelected
                        ? "bg-accent/10 border-accent/50"
                        : "bg-surface-900/30 border-glass-border hover:border-glass-border-strong"
                    }`}
                  >
                <div className="flex items-start gap-2">
                  <span
                    aria-hidden
                    className={`mt-0.5 w-3.5 h-3.5 rounded-sm border shrink-0 flex items-center justify-center ${
                      isSelected
                        ? "bg-accent border-accent"
                        : "border-glass-border-strong group-hover:border-surface-400"
                    }`}
                  >
                    {isSelected && <Check className="w-2.5 h-2.5 text-accent-ink" strokeWidth={3} />}
                  </span>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-1.5">
                      <span className="text-[10px] font-mono text-accent shrink-0">#{i + 1}</span>
                      <span className="text-[13px] font-semibold text-surface-100 truncate">
                        {p.concept}
                      </span>
                    </div>
                    {p.style && (
                      <span className="inline-block mt-1 text-[10px] uppercase tracking-wider text-surface-400 bg-surface-800/60 border border-glass-border rounded-pill px-1.5 py-0.5">
                        {p.style}
                      </span>
                    )}
                    <p className="text-[11px] text-surface-400 leading-relaxed mt-1.5 line-clamp-2">
                      {p.prompt}
                    </p>
                  </div>
                  </div>
                </button>
              </div>
              );
            })}
          </VList>
        </div>
      )}
    </section>
  );
}

const BatchGroup = memo(BatchGroupInner);
