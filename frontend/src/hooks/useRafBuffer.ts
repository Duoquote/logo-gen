import { useCallback, useEffect, useRef } from "react";

/**
 * Coalesce high-frequency updates into one commit per animation frame.
 * Token streams arrive far faster than React can usefully render — this
 * lets the upstream caller fire freely while we drain at most once per
 * frame.
 *
 * Usage:
 *   const flush = useRafBuffer<string>((latest) => setContent(latest));
 *   onToken(t => { buffer.current += t; flush(buffer.current); });
 *
 * The committer is invoked on the rAF callback with the latest value seen
 * since the previous commit. Trailing-edge: a final pending value is
 * always committed, even after rapid bursts stop.
 */
export function useRafBuffer<T>(
  commit: (latest: T) => void,
): (next: T) => void {
  const pending = useRef<{ value: T } | null>(null);
  const rafId = useRef<number | null>(null);
  const commitRef = useRef(commit);
  commitRef.current = commit;

  const drain = useCallback(() => {
    rafId.current = null;
    const p = pending.current;
    if (!p) return;
    pending.current = null;
    commitRef.current(p.value);
  }, []);

  const schedule = useCallback(
    (next: T) => {
      pending.current = { value: next };
      if (rafId.current === null) {
        rafId.current = requestAnimationFrame(drain);
      }
    },
    [drain],
  );

  // Cancel any pending frame on unmount.
  useEffect(() => {
    return () => {
      if (rafId.current !== null) cancelAnimationFrame(rafId.current);
    };
  }, []);

  return schedule;
}
