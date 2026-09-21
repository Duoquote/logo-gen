import Dexie, { type EntityTable } from "dexie";

// ── Chat history ─────────────────────────────────────────────────

export interface StoredMessage {
  id?: number;
  sessionId: string;
  role: "user" | "assistant";
  content: string;
  images?: string[];  // uploaded image paths for this message
  timestamp: number;
}

export interface StoredSession {
  id?: string;
  createdAt: number;
  lastActivityAt: number;
  prompts: string;          // legacy: PromptVariation[] JSON (kept for backward compat)
  batches?: string;         // PromptBatch[] JSON (v4+)
  selection?: string;       // string[] JSON of "batchId:idx" entries (v4+)
  logos: string;             // JSON-serialised GeneratedLogo[]
  referenceImages: string;   // JSON-serialised string[] of all ref image paths
  selectedImages: string;    // JSON-serialised string[] of selected ref image paths
}

export interface SessionSummary {
  id: string;
  createdAt: number;
  lastActivityAt: number;
  title: string;
  messageCount: number;
}

// ── Settings ─────────────────────────────────────────────────────

export interface StoredSettings {
  key: string;
  llmModel: string;
  imageModels: string[];
  imagesPerModel: number;
  useClaudeCli?: boolean;
}

// ── Database ─────────────────────────────────────────────────────

const db = new Dexie("LogoForge") as Dexie & {
  messages: EntityTable<StoredMessage, "id">;
  sessions: EntityTable<StoredSession, "id">;
  settings: EntityTable<StoredSettings, "key">;
};

db.version(1).stores({
  messages: "++id, sessionId",
  sessions: "id",
  settings: "key",
});

// v2: add images to messages, referenceImages/selectedImages to sessions
db.version(2).stores({
  messages: "++id, sessionId",
  sessions: "id",
  settings: "key",
}).upgrade((tx) => {
  return tx.table("sessions").toCollection().modify((session) => {
    if (!session.referenceImages) session.referenceImages = "[]";
    if (!session.selectedImages) session.selectedImages = "[]";
  });
});

// v3: add lastActivityAt to sessions, index for sort
db.version(3).stores({
  messages: "++id, sessionId",
  sessions: "id, lastActivityAt",
  settings: "key",
}).upgrade((tx) => {
  return tx.table("sessions").toCollection().modify((session) => {
    if (!session.lastActivityAt) {
      session.lastActivityAt = session.createdAt || Date.now();
    }
  });
});

// v4: add batches + selection to sessions (legacy `prompts` field stays
// for backward compat; loadSessionMeta synthesises a single orphan
// batch from it on read so old conversations still show their prompts).
db.version(4).stores({
  messages: "++id, sessionId",
  sessions: "id, lastActivityAt",
  settings: "key",
}).upgrade((tx) => {
  return tx.table("sessions").toCollection().modify((session) => {
    if (typeof session.batches !== "string") session.batches = "[]";
    if (typeof session.selection !== "string") session.selection = "[]";
  });
});

export default db;

// ── Helpers ──────────────────────────────────────────────────────

const CURRENT_SESSION_KEY = "logoforge_current_session";

export function getCurrentSessionId(): string {
  let id = localStorage.getItem(CURRENT_SESSION_KEY);
  if (!id) {
    // Migrate from sessionStorage (older builds) if present
    const legacy = sessionStorage.getItem(CURRENT_SESSION_KEY);
    id = legacy || crypto.randomUUID();
    localStorage.setItem(CURRENT_SESSION_KEY, id);
    if (legacy) sessionStorage.removeItem(CURRENT_SESSION_KEY);
  }
  return id;
}

export function resetCurrentSession(): string {
  const id = crypto.randomUUID();
  localStorage.setItem(CURRENT_SESSION_KEY, id);
  return id;
}

export function setCurrentSessionId(id: string): void {
  localStorage.setItem(CURRENT_SESSION_KEY, id);
}

// Chat persistence

export async function loadMessages(sessionId: string): Promise<StoredMessage[]> {
  return db.messages.where("sessionId").equals(sessionId).sortBy("timestamp");
}

export async function addMessage(
  sessionId: string,
  role: "user" | "assistant",
  content: string,
  images?: string[],
): Promise<void> {
  await db.messages.add({
    sessionId, role, content, timestamp: Date.now(),
    ...(images && images.length > 0 ? { images } : {}),
  });
}

export async function updateLastAssistantMessage(
  sessionId: string,
  content: string,
): Promise<void> {
  const msgs = await db.messages
    .where("sessionId")
    .equals(sessionId)
    .reverse()
    .limit(1)
    .toArray();
  const last = msgs[0];
  if (last?.id !== undefined && last.role === "assistant") {
    await db.messages.update(last.id, { content });
  }
}

export async function clearSession(sessionId: string): Promise<void> {
  await db.messages.where("sessionId").equals(sessionId).delete();
  await db.sessions.delete(sessionId);
}

// Session metadata

export async function saveSessionMeta(
  sessionId: string,
  data: {
    batches?: unknown[];
    selection?: string[];
    logos?: unknown[];
    referenceImages?: string[];
    selectedImages?: string[];
  },
): Promise<void> {
  const existing = await db.sessions.get(sessionId);
  const now = Date.now();
  await db.sessions.put({
    id: sessionId,
    createdAt: existing?.createdAt ?? now,
    lastActivityAt: now,
    // legacy `prompts` field is preserved verbatim — we don't write to it
    // anymore but old sessions' data lives there until first save below
    prompts: existing?.prompts ?? "[]",
    batches: data.batches ? JSON.stringify(data.batches) : existing?.batches ?? "[]",
    selection: data.selection ? JSON.stringify(data.selection) : existing?.selection ?? "[]",
    logos: data.logos ? JSON.stringify(data.logos) : existing?.logos ?? "[]",
    referenceImages: data.referenceImages ? JSON.stringify(data.referenceImages) : existing?.referenceImages ?? "[]",
    selectedImages: data.selectedImages ? JSON.stringify(data.selectedImages) : existing?.selectedImages ?? "[]",
  });
}

export async function touchSession(sessionId: string): Promise<void> {
  const existing = await db.sessions.get(sessionId);
  const now = Date.now();
  await db.sessions.put({
    id: sessionId,
    createdAt: existing?.createdAt ?? now,
    lastActivityAt: now,
    prompts: existing?.prompts ?? "[]",
    batches: existing?.batches ?? "[]",
    selection: existing?.selection ?? "[]",
    logos: existing?.logos ?? "[]",
    referenceImages: existing?.referenceImages ?? "[]",
    selectedImages: existing?.selectedImages ?? "[]",
  });
}

export async function listSessions(): Promise<SessionSummary[]> {
  const sessions = await db.sessions
    .orderBy("lastActivityAt")
    .reverse()
    .toArray();

  const summaries: SessionSummary[] = [];
  for (const s of sessions) {
    if (!s.id) continue;
    const firstUser = await db.messages
      .where("sessionId").equals(s.id)
      .filter((m) => m.role === "user")
      .first();
    const count = await db.messages.where("sessionId").equals(s.id).count();
    if (count === 0) continue; // hide sessions with no activity
    const raw = (firstUser?.content || "").replace(/\s+/g, " ").trim();
    const title = raw ? raw.slice(0, 60) + (raw.length > 60 ? "…" : "") : "New chat";
    summaries.push({
      id: s.id,
      createdAt: s.createdAt,
      lastActivityAt: s.lastActivityAt || s.createdAt,
      title,
      messageCount: count,
    });
  }
  return summaries;
}

export async function deleteSession(sessionId: string): Promise<void> {
  await db.messages.where("sessionId").equals(sessionId).delete();
  await db.sessions.delete(sessionId);
}

export interface SessionMeta {
  batches: unknown[];
  selection: string[];
  logos: unknown[];
  referenceImages: string[];
  selectedImages: string[];
}

export async function loadSessionMeta(sessionId: string): Promise<SessionMeta | null> {
  const session = await db.sessions.get(sessionId);
  if (!session) return null;

  // Backward compat: if no batches yet but legacy flat prompts exist,
  // synthesize one orphan batch (messageIdx=-1) so the prompts still
  // appear when a v3 session is opened in v4.
  let batches = JSON.parse(session.batches || "[]") as unknown[];
  if (batches.length === 0) {
    const legacy = JSON.parse(session.prompts || "[]") as unknown[];
    if (legacy.length > 0) {
      batches = [{
        id: crypto.randomUUID(),
        messageIdx: -1,
        prompts: legacy,
        createdAt: session.createdAt || Date.now(),
      }];
    }
  }

  return {
    batches,
    selection: JSON.parse(session.selection || "[]"),
    logos: JSON.parse(session.logos),
    referenceImages: JSON.parse(session.referenceImages || "[]"),
    selectedImages: JSON.parse(session.selectedImages || "[]"),
  };
}

// Settings persistence

export async function loadLocalSettings(): Promise<StoredSettings | undefined> {
  return db.settings.get("user-settings");
}

export async function saveLocalSettings(
  s: Omit<StoredSettings, "key">,
): Promise<void> {
  await db.settings.put({ key: "user-settings", ...s });
}
