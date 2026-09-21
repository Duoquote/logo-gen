import {
  MessageSquare,
  Zap,
  Maximize2,
  Eraser,
  Settings as SettingsIcon,
} from "lucide-react";
import type { TabId } from "../types";

const TOP: { id: TabId; label: string; icon: typeof MessageSquare }[] = [
  { id: "chat", label: "Chat", icon: MessageSquare },
  { id: "quick", label: "Quick generate", icon: Zap },
  { id: "upscale", label: "Upscale", icon: Maximize2 },
  { id: "clean", label: "Clean background", icon: Eraser },
];

const BOTTOM: { id: TabId; label: string; icon: typeof MessageSquare }[] = [
  { id: "settings", label: "Settings", icon: SettingsIcon },
];

interface Props {
  tab: TabId;
  onTab: (id: TabId) => void;
}

export default function IconRail({ tab, onTab }: Props) {
  return (
    <aside
      className="relative z-30 w-[60px] shrink-0 flex flex-col items-center
                 py-3 bg-glass-soft border-r border-glass-border"
      style={{
        backdropFilter: "blur(24px)",
        WebkitBackdropFilter: "blur(24px)",
        willChange: "backdrop-filter",
      }}
    >
      <div className="flex flex-col items-center gap-1 flex-1">
        {TOP.map((item) => (
          <RailItem
            key={item.id}
            {...item}
            active={tab === item.id}
            onClick={() => onTab(item.id)}
          />
        ))}
      </div>
      <div className="flex flex-col items-center gap-1">
        {BOTTOM.map((item) => (
          <RailItem
            key={item.id}
            {...item}
            active={tab === item.id}
            onClick={() => onTab(item.id)}
          />
        ))}
      </div>
    </aside>
  );
}

function RailItem({
  label,
  icon: Icon,
  active,
  onClick,
}: {
  label: string;
  icon: typeof MessageSquare;
  active: boolean;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      title={label}
      aria-label={label}
      className={`
        relative flex items-center justify-center
        w-10 h-10 rounded-field transition-all duration-150
        ${
          active
            ? "text-accent bg-accent/10"
            : "text-surface-400 hover:text-surface-100 hover:bg-glass"
        }
      `}
    >
      {active && (
        <span
          aria-hidden
          className="absolute left-[-9px] top-1/2 -translate-y-1/2 w-[3px] h-5 rounded-full bg-accent"
          style={{ boxShadow: "0 0 12px rgba(124,199,255,0.6)" }}
        />
      )}
      <Icon className="w-[18px] h-[18px]" strokeWidth={2} />
    </button>
  );
}
