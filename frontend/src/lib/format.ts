export function formatBytes(n: number): string {
  if (!n) return "0 B"
  const units = ["B", "KB", "MB", "GB", "TB"]
  const i = Math.min(units.length - 1, Math.floor(Math.log(n) / Math.log(1000)))
  return `${(n / 1000 ** i).toFixed(i >= 3 ? 1 : 0)} ${units[i]}`
}

export function formatUsd(n: number | null | undefined, digits = 3): string {
  if (n === null || n === undefined) return "n/a"
  if (n === 0) return "free"
  if (n < 0.001) return `$${n.toFixed(5)}`
  return `$${n.toFixed(digits)}`
}

/** OpenRouter quotes chat prices per token; show them per million tokens. */
export function perMillion(n: number | null | undefined): string {
  if (n === null || n === undefined) return "n/a"
  if (n < 0) return "varies"
  if (n === 0) return "free"
  const m = n * 1_000_000
  return `$${m < 1 ? m.toFixed(2) : m.toFixed(m < 10 ? 2 : 0)}`
}

export function formatContext(n: number | null | undefined): string {
  if (!n) return "n/a"
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(n % 1_000_000 ? 1 : 0)}M`
  return `${Math.round(n / 1000)}K`
}

export function timeAgo(ts: number): string {
  const s = Date.now() / 1000 - ts
  if (s < 60) return "just now"
  if (s < 3600) return `${Math.floor(s / 60)}m ago`
  if (s < 86400) return `${Math.floor(s / 3600)}h ago`
  if (s < 7 * 86400) return `${Math.floor(s / 86400)}d ago`
  return new Date(ts * 1000).toLocaleDateString()
}

export function shortModel(id: string): string {
  if (id.startsWith("local/")) return id.slice(6)
  return id.split("/").pop() ?? id
}
