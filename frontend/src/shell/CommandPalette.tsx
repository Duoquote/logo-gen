import { useEffect } from "react";
import { Command } from "cmdk";
import {
  MessageSquare,
  Zap,
  Maximize2,
  Eraser,
  Settings as SettingsIcon,
  Plus,
  RotateCcw,
  Square,
  Play,
  Trash2,
  Save,
  RefreshCw,
} from "lucide-react";
import type { TabId } from "../types";
import { emitCommand } from "./commands";

interface Props {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  tab: TabId;
  setTab: (id: TabId) => void;
}

export default function CommandPalette({ open, onOpenChange, tab, setTab }: Props) {
  // Lock body scroll while open.
  useEffect(() => {
    if (!open) return;
    const prev = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      document.body.style.overflow = prev;
    };
  }, [open]);

  if (!open) return null;

  const close = () => onOpenChange(false);
  const go = (id: TabId) => {
    setTab(id);
    close();
  };
  const run = (fn: () => void) => {
    fn();
    close();
  };

  return (
    <div
      className="fixed inset-0 z-50 flex items-start justify-center pt-[18vh] motion-safe:animate-fade-in"
      onClick={close}
    >
      <div
        aria-hidden
        className="absolute inset-0 bg-surface-1000/60"
        style={{ backdropFilter: "blur(10px)", WebkitBackdropFilter: "blur(10px)" }}
      />
      <Command
        label="Command palette"
        loop
        className="relative glass-panel-strong w-[92vw] max-w-xl overflow-hidden motion-safe:animate-palette-in"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="border-b border-glass-border px-4 py-3">
          <Command.Input
            autoFocus
            placeholder="Search commands & navigate…"
            className="w-full bg-transparent text-surface-100 placeholder-surface-500
                       text-[15px] outline-none border-none"
          />
        </div>
        <Command.List className="max-h-[55vh] overflow-y-auto p-2">
          <Command.Empty className="px-3 py-6 text-center text-sm text-surface-400">
            No commands match.
          </Command.Empty>

          <Group heading="Navigate">
            <Item icon={MessageSquare} label="Go to chat" hint={tab === "chat" ? "current" : undefined} onSelect={() => go("chat")} />
            <Item icon={Zap} label="Go to quick generate" hint={tab === "quick" ? "current" : undefined} onSelect={() => go("quick")} />
            <Item icon={Maximize2} label="Go to upscale" hint={tab === "upscale" ? "current" : undefined} onSelect={() => go("upscale")} />
            <Item icon={Eraser} label="Go to clean background" hint={tab === "clean" ? "current" : undefined} onSelect={() => go("clean")} />
            <Item icon={SettingsIcon} label="Open settings" hint={tab === "settings" ? "current" : undefined} onSelect={() => go("settings")} />
          </Group>

          <Group heading="Chat">
            <Item icon={Plus} label="New chat session" onSelect={() => run(() => { setTab("chat"); emitCommand("chat.new"); })} />
            <Item icon={RotateCcw} label="Reset current chat" onSelect={() => run(() => { setTab("chat"); emitCommand("chat.reset"); })} />
            <Item icon={Square} label="Stop streaming" onSelect={() => run(() => emitCommand("chat.stop"))} />
          </Group>

          <Group heading="Quick generate">
            <Item icon={Play} label="Generate now" onSelect={() => run(() => { setTab("quick"); emitCommand("quick.generate"); })} />
            <Item icon={Trash2} label="Clear results" onSelect={() => run(() => emitCommand("quick.clear"))} />
          </Group>

          <Group heading="Upscale">
            <Item icon={Maximize2} label="Upscale selected" onSelect={() => run(() => { setTab("upscale"); emitCommand("upscale.selected"); })} />
            <Item icon={Maximize2} label="Upscale all" onSelect={() => run(() => { setTab("upscale"); emitCommand("upscale.all"); })} />
            <Item icon={RefreshCw} label="Refresh upscale galleries" onSelect={() => run(() => { setTab("upscale"); emitCommand("upscale.refresh"); })} />
          </Group>

          <Group heading="Background">
            <Item icon={Eraser} label="Clean selected" onSelect={() => run(() => { setTab("clean"); emitCommand("clean.selected"); })} />
            <Item icon={Eraser} label="Clean all" onSelect={() => run(() => { setTab("clean"); emitCommand("clean.all"); })} />
            <Item icon={RefreshCw} label="Refresh clean galleries" onSelect={() => run(() => { setTab("clean"); emitCommand("clean.refresh"); })} />
          </Group>

          <Group heading="Settings">
            <Item icon={Save} label="Save settings" onSelect={() => run(() => { setTab("settings"); emitCommand("settings.save"); })} />
            <Item icon={RotateCcw} label="Reset settings to defaults" onSelect={() => run(() => { setTab("settings"); emitCommand("settings.reset"); })} />
          </Group>
        </Command.List>

        <div className="border-t border-glass-border px-4 py-2 flex items-center justify-between text-[11px] text-surface-500">
          <div className="flex items-center gap-3">
            <span className="flex items-center gap-1.5"><span className="kbd">↵</span> select</span>
            <span className="flex items-center gap-1.5"><span className="kbd">↑↓</span> move</span>
          </div>
          <span className="flex items-center gap-1.5"><span className="kbd">esc</span> close</span>
        </div>
      </Command>
    </div>
  );
}

function Group({ heading, children }: { heading: string; children: React.ReactNode }) {
  return (
    <Command.Group
      heading={heading}
      className="[&_[cmdk-group-heading]]:px-2 [&_[cmdk-group-heading]]:py-1.5
                 [&_[cmdk-group-heading]]:text-[10px] [&_[cmdk-group-heading]]:uppercase
                 [&_[cmdk-group-heading]]:tracking-wider
                 [&_[cmdk-group-heading]]:text-surface-500"
    >
      {children}
    </Command.Group>
  );
}

function Item({
  icon: Icon,
  label,
  hint,
  onSelect,
}: {
  icon: typeof MessageSquare;
  label: string;
  hint?: string;
  onSelect: () => void;
}) {
  return (
    <Command.Item
      onSelect={onSelect}
      className="group flex items-center gap-3 px-3 py-2 rounded-field cursor-pointer
                 text-[14px] text-surface-200
                 data-[selected=true]:bg-glass-strong data-[selected=true]:text-surface-50"
    >
      <Icon className="w-[15px] h-[15px] text-surface-400 group-data-[selected=true]:text-accent" strokeWidth={2} />
      <span className="flex-1">{label}</span>
      {hint && <span className="text-[11px] text-surface-500">{hint}</span>}
    </Command.Item>
  );
}
