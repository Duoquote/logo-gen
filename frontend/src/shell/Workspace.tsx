import type { ReactNode } from "react";

// Pass-through layout shell. We set a definite height down the chain so
// tabs (especially Chat) can use `h-full` for bounded internal scrolling
// instead of pushing the page to scroll. Each tab decides whether to
// scroll its own content or stay full-bleed (see App.tsx wrappers).
export default function Workspace({ children }: { children: ReactNode }) {
  return (
    // Workspace itself owns the page scroll so the scrollbar sticks to
    // the viewport's right edge (outside the content's max-width
    // column). `min-h-full` keeps the inner column at least as tall as
    // the visible area so backgrounds fill correctly when content is
    // short.
    <div className="flex-1 min-h-0 overflow-y-auto">
      <div className="max-w-[1600px] mx-auto px-6 py-4 min-h-full">{children}</div>
    </div>
  );
}
