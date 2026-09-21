export interface ImageInfo {
  path: string;
  name: string;
}

export interface GeneratedLogo extends ImageInfo {
  prompt: string;
  concept: string;
  model: string;
  seed: number | null;
  generation_time: number;
}

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  images?: string[];  // uploaded image paths (relative to output/)
}

export interface PromptVariation {
  prompt: string;
  concept: string;
  style: string;
}

export interface AppSettings {
  llm_model: string;
  image_models: string[];
  images_per_model: number;
  image_size: string;
  use_claude_cli: boolean;
}

export type TabId = "chat" | "quick" | "upscale" | "clean" | "settings";

export interface PromptBatch {
  id: string;
  // Index in the messages array of the assistant message that produced
  // this batch. -1 means the batch is "orphaned" — produced by a
  // session loaded from a legacy schema where we don't know the
  // anchor; render at the end of the message list.
  messageIdx: number;
  prompts: PromptVariation[];
  createdAt: number;
}
