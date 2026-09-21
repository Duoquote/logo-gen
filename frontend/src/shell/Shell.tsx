import { useEffect, type ReactNode } from "react";
import IconRail from "./IconRail";
import TopBar from "./TopBar";
import Workspace from "./Workspace";
import type { TabId } from "../types";

interface Props {
  tab: TabId;
  onTab: (id: TabId) => void;
  paletteOpen: boolean;
  onOpenPalette: () => void;
  onClosePalette: () => void;
  children: ReactNode;
}

export default function Shell({
  tab,
  onTab,
  paletteOpen,
  onOpenPalette,
  onClosePalette,
  children,
}: Props) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const isK = e.key === "k" || e.key === "K";
      if ((e.metaKey || e.ctrlKey) && isK) {
        e.preventDefault();
        if (paletteOpen) onClosePalette();
        else onOpenPalette();
        return;
      }
      if (e.key === "Escape" && paletteOpen) {
        e.preventDefault();
        onClosePalette();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [paletteOpen, onOpenPalette, onClosePalette]);

  return (
    <>
      <div id="backdrop" />
      <div id="app" className="relative z-10 h-screen flex">
        <IconRail tab={tab} onTab={onTab} />
        <main className="flex-1 flex flex-col min-w-0">
          <TopBar onOpenPalette={onOpenPalette} />
          <Workspace>{children}</Workspace>
        </main>
      </div>
    </>
  );
}
