import type { Settings } from "@/lib/api"

/** Max images one model may make per request (mirrors Settings.cap_for on the server). */
export function capFor(s: Settings, model: string): number {
  if (model in s.model_caps) return Math.max(1, s.model_caps[model])
  return model.startsWith("local/") ? s.max_images_local : s.max_images_cloud
}

/** Largest "images per direction" every selected model allows for this many directions. */
export function maxPerDirection(s: Settings, models: string[], directions: number): number {
  if (!models.length || directions < 1) return 1
  return Math.max(1, Math.min(...models.map((m) => Math.floor(capFor(s, m) / directions))))
}
