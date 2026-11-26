import { useCatalog, useSettings } from "@/lib/queries"

/** Rough cost of a run: per-image price of each model times images per model. Local models are free. */
export function useCostEstimate(models: string[], imagesPerModel: number) {
  const settings = useSettings()
  const catalog = useCatalog(settings.data?.secrets.OPENROUTER_KEY.set ?? false)
  let cost: number | null = 0
  for (const m of models) {
    if (m.startsWith("local/")) continue
    const info = catalog.data?.image.find((x) => x.id === m)
    if (!info || info.price_per_image === null) {
      cost = null
      break
    }
    cost += info.price_per_image * imagesPerModel
  }
  return { images: models.length * imagesPerModel, cost }
}
