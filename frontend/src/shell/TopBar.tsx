interface Props {
  onOpenPalette: () => void;
}

export default function TopBar({ onOpenPalette }: Props) {
  return (
    <header
      className="relative z-30 h-12 shrink-0 flex items-center justify-between
                 px-5 bg-glass-soft border-b border-glass-border"
      style={{
        backdropFilter: "blur(24px)",
        WebkitBackdropFilter: "blur(24px)",
        willChange: "backdrop-filter",
      }}
    >
      <span className="font-display font-semibold tracking-[-0.02em] text-[15px] text-surface-100 lowercase">
        logogen
      </span>
      <button
        type="button"
        onClick={onOpenPalette}
        className="group flex items-center gap-2 text-xs text-surface-400 hover:text-surface-200
                   transition-colors duration-150 px-2 py-1 rounded-field hover:bg-glass"
        aria-label="Open command palette"
      >
        <span className="hidden sm:inline">Search & run</span>
        <span className="kbd">⌘K</span>
      </button>
    </header>
  );
}
