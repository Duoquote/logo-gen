import { useState } from "react";
import type { TabId } from "./types";
import Shell from "./shell/Shell";
import CommandPalette from "./shell/CommandPalette";
import ChatDesigner from "./components/ChatDesigner";
import QuickGenerate from "./components/QuickGenerate";
import Upscale from "./components/Upscale";
import BackgroundRemoval from "./components/BackgroundRemoval";
import Settings from "./components/Settings";

export default function App() {
  const [tab, setTab] = useState<TabId>("chat");
  const [paletteOpen, setPaletteOpen] = useState(false);

  return (
    <>
      <Shell
        tab={tab}
        onTab={setTab}
        paletteOpen={paletteOpen}
        onOpenPalette={() => setPaletteOpen(true)}
        onClosePalette={() => setPaletteOpen(false)}
      >
        {/* All tabs stay mounted; hidden via CSS to preserve in-flight
            state. Workspace owns the page scroll, so tab wrappers just
            stack — no internal overflow. */}
        <div className={tab === "chat" ? "" : "hidden"}><ChatDesigner /></div>
        <div className={tab === "quick" ? "" : "hidden"}><QuickGenerate /></div>
        <div className={tab === "upscale" ? "" : "hidden"}><Upscale /></div>
        <div className={tab === "clean" ? "" : "hidden"}><BackgroundRemoval /></div>
        <div className={tab === "settings" ? "" : "hidden"}><Settings /></div>
      </Shell>
      <CommandPalette
        open={paletteOpen}
        onOpenChange={setPaletteOpen}
        tab={tab}
        setTab={setTab}
      />
    </>
  );
}
