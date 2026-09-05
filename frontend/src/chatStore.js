const STORAGE_KEY = "rgukt_academic_chats_v1";
const SCHEMA_VERSION = 1;
const MAX_CONVERSATIONS = 40;
const MAX_MESSAGES_PER_CHAT = 200;
const MAX_CONTENT_CHARS = 12000;

function uid() {
  return crypto.randomUUID();
}

export function createEmptyConversation() {
  const now = Date.now();
  return {
    id: uid(),
    title: "New chat",
    createdAt: now,
    updatedAt: now,
    messages: [],
  };
}

function sanitizeMessage(m) {
  if (!m || typeof m !== "object") return null;
  const role = m.role === "user" || m.role === "assistant" ? m.role : null;
  if (!role) return null;
  const content = String(m.content ?? "").slice(0, MAX_CONTENT_CHARS);
  const out = {
    id: typeof m.id === "string" ? m.id : uid(),
    role,
    content,
  };
  if (m.isError) out.isError = true;
  if (Array.isArray(m.sources)) out.sources = m.sources.slice(0, 20);
  if (m.metrics && typeof m.metrics === "object") out.metrics = m.metrics;
  return out;
}

function sanitizeConversation(c) {
  if (!c || typeof c !== "object") return null;
  const id = typeof c.id === "string" ? c.id : uid();
  const title = String(c.title || "New chat").slice(0, 80) || "New chat";
  const messages = Array.isArray(c.messages)
    ? c.messages.map(sanitizeMessage).filter(Boolean).slice(-MAX_MESSAGES_PER_CHAT)
    : [];
  return {
    id,
    title,
    createdAt: Number(c.createdAt) || Date.now(),
    updatedAt: Number(c.updatedAt) || Date.now(),
    messages,
  };
}

export function loadChatStore() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw);
    if (!parsed || typeof parsed !== "object") return null;

    // Future schema migrations land here
    const version = Number(parsed.schemaVersion) || 0;
    if (version > SCHEMA_VERSION) {
      // Newer client wrote this; reset safely rather than crash
      return null;
    }

    if (!Array.isArray(parsed.conversations)) return null;
    const conversations = parsed.conversations
      .map(sanitizeConversation)
      .filter(Boolean)
      .slice(0, MAX_CONVERSATIONS);

    if (!conversations.length) return null;
    return {
      schemaVersion: SCHEMA_VERSION,
      conversations,
      activeId: parsed.activeId,
    };
  } catch {
    return null;
  }
}

export function saveChatStore(store) {
  try {
    const conversations = (store.conversations || [])
      .map(sanitizeConversation)
      .filter(Boolean)
      .sort((a, b) => (b.updatedAt || 0) - (a.updatedAt || 0))
      .slice(0, MAX_CONVERSATIONS);

    localStorage.setItem(
      STORAGE_KEY,
      JSON.stringify({
        schemaVersion: SCHEMA_VERSION,
        conversations,
        activeId: store.activeId,
      })
    );
  } catch {
    // ignore quota / private mode
  }
}

export function titleFromMessage(text) {
  const cleaned = String(text || "")
    .replace(/\s+/g, " ")
    .trim();
  if (!cleaned) return "New chat";
  return cleaned.length > 48 ? `${cleaned.slice(0, 47)}…` : cleaned;
}

export function ensureStore() {
  const existing = loadChatStore();
  if (existing?.conversations?.length) {
    const activeId =
      existing.activeId && existing.conversations.some((c) => c.id === existing.activeId)
        ? existing.activeId
        : existing.conversations[0].id;
    return { conversations: existing.conversations, activeId };
  }
  const first = createEmptyConversation();
  return { conversations: [first], activeId: first.id };
}
