import type { ReactNode } from "react";

export default function Kbd({ children }: { children: ReactNode }) {
  return <span className="kbd">{children}</span>;
}
