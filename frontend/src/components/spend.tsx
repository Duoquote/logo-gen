import type { Spend } from "@/lib/api"
import { formatUsd } from "@/lib/format"

/** One line of spend: what OpenRouter charged, what a provider billed on your own key, and free local images. */
export function SpendLine({ spend, label = "Spent" }: { spend: Spend | undefined; label?: string }) {
  if (!spend) return null
  const parts: string[] = []
  if (spend.chat_calls) parts.push(`${spend.chat_calls} chat call${spend.chat_calls === 1 ? "" : "s"}`)
  if (spend.cloud_images) parts.push(`${spend.cloud_images} cloud image${spend.cloud_images === 1 ? "" : "s"}`)
  if (spend.local_images) parts.push(`${spend.local_images} local (free)`)
  return (
    <span className="text-xs text-muted-foreground tabular-nums">
      {label}: <span className="text-foreground">{spend.total ? formatUsd(spend.total, 4) : "$0"}</span>
      {spend.byok > 0 && (
        <span title="Billed by the provider on your own key (BYOK on OpenRouter), not from OpenRouter credits">
          {" "}
          ({formatUsd(spend.credits, 4)} OpenRouter, {formatUsd(spend.byok, 4)} BYOK)
        </span>
      )}
      {parts.length > 0 && ` · ${parts.join(", ")}`}
    </span>
  )
}
