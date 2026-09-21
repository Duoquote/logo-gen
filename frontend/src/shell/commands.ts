// Tiny event bus the command palette uses to fire actions inside tabs
// without importing them. Each tab subscribes via onCommand() in a useEffect.

export type CommandId =
  | "chat.new"
  | "chat.reset"
  | "chat.stop"
  | "quick.generate"
  | "quick.clear"
  | "upscale.selected"
  | "upscale.all"
  | "upscale.refresh"
  | "clean.selected"
  | "clean.all"
  | "clean.refresh"
  | "settings.save"
  | "settings.reset";

const listeners = new Map<CommandId, Set<() => void>>();

export function onCommand(id: CommandId, fn: () => void): () => void {
  let set = listeners.get(id);
  if (!set) {
    set = new Set();
    listeners.set(id, set);
  }
  set.add(fn);
  return () => {
    set!.delete(fn);
  };
}

export function emitCommand(id: CommandId): void {
  const set = listeners.get(id);
  if (!set) return;
  for (const fn of set) {
    try {
      fn();
    } catch {
      // swallow — one bad subscriber shouldn't break the bus
    }
  }
}
