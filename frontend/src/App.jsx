import { useCallback, useEffect, useRef, useState } from "react";
import { MarkdownMessage } from "./components/MarkdownMessage.jsx";

const SUGGESTIONS = [
  "Summarise the main admission requirements.",
  "What are the attendance rules?",
  "Branches available in Engineering."
];

function IconDoc() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden>
      <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
      <path d="M14 2v6h6M16 13H8M16 17H8M10 9H8" />
    </svg>
  );
}

function IconSend() {
  return (
    <svg viewBox="0 0 24 24" fill="currentColor" aria-hidden>
      <path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z" />
    </svg>
  );
}

function IconBook() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden>
      <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20" />
      <path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z" />
    </svg>
  );
}

function SourcesSection({ sources }) {
  if (!sources || sources.length === 0) return null;
  return (
    <div className="sources-block">
      <div className="sources-title">
        <IconBook />
        Retrieved context
      </div>
      <div className="source-cards">
        {sources.map((s) => (
          <details key={s.id} className="source-card">
            <summary>
              <span className="source-badge">Source {s.id}</span>
              <span>Page {s.page}</span>
            </summary>
            <p className="source-snippet">{s.snippet}</p>
          </details>
        ))}
      </div>
    </div>
  );
}

export default function App() {
  const [appTitle, setAppTitle] = useState("Document Assistant");
  const [appSubtitle, setAppSubtitle] = useState(
    "Loading document title…"
  );

  useEffect(() => {
    fetch("/api/config")
      .then((r) => (r.ok ? r.json() : null))
      .then((d) => {
        if (!d) return;
        if (d.app_title) {
          setAppTitle(d.app_title);
          document.title = d.app_title;
        }
        if (d.app_subtitle) setAppSubtitle(d.app_subtitle);
      })
      .catch(() => {
        setAppSubtitle("Answers from your indexed PDFs · with citations");
      });
  }, []);

  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const scrollRef = useRef(null);
  const textareaRef = useRef(null);

  const scrollToBottom = useCallback(() => {
    const el = scrollRef.current;
    if (el) el.scrollTo({ top: el.scrollHeight, behavior: "smooth" });
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages, loading, scrollToBottom]);

  const send = async (text) => {
    const trimmed = text.trim();
    if (!trimmed || loading) return;

    const historyPayload = messages
      .filter((m) => (m.role === "user" || m.role === "assistant") && !m.isError)
      .map(({ role, content }) => ({
        role,
        content: String(content ?? "").slice(0, 12000),
      }));

    const userMsg = { id: crypto.randomUUID(), role: "user", content: trimmed };
    setMessages((m) => [...m, userMsg]);
    setInput("");
    setLoading(true);

    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: trimmed,
          top_k: 4,
          history: historyPayload,
        }),
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) {
        const detail = data.detail;
        const msg =
          typeof detail === "string"
            ? detail
            : Array.isArray(detail)
              ? detail.map((d) => d.msg).join(" ")
              : "Request failed. Is the API running on port 8000?";
        throw new Error(msg);
      }
      setMessages((m) => [
        ...m,
        {
          id: crypto.randomUUID(),
          role: "assistant",
          content: data.answer || "",
          sources: data.sources || [],
        },
      ]);
    } catch (e) {
      const errText = e.message || "Something went wrong.";
      setMessages((m) => [
        ...m,
        {
          id: crypto.randomUUID(),
          role: "assistant",
          content: errText,
          isError: true,
        },
      ]);
    } finally {
      setLoading(false);
      textareaRef.current?.focus();
    }
  };

  const onSubmit = (e) => {
    e.preventDefault();
    send(input);
  };

  const onKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      send(input);
    }
  };

  const clearChat = () => {
    setMessages([]);
    textareaRef.current?.focus();
  };

  useEffect(() => {
    const ta = textareaRef.current;
    if (!ta) return;
    ta.style.height = "auto";
    ta.style.height = `${Math.min(ta.scrollHeight, 160)}px`;
  }, [input]);

  return (
    <div className="app-shell">
      <header className="app-header">
        <div className="app-brand">
          <div className="app-logo" aria-hidden>
            <IconDoc />
          </div>
          <div className="app-title-wrap">
            <h1>{appTitle}</h1>
            <p>{appSubtitle}</p>
          </div>
        </div>
        <button type="button" className="btn-ghost" onClick={clearChat}>
          New chat
        </button>
      </header>

      <div className="messages-scroll" ref={scrollRef}>
        {messages.length === 0 && !loading && (
          <div className="empty-state">
            <h2>Ask your organisation&apos;s documents</h2>
            <p>
              Questions are sent to your local API, which retrieves relevant passages and
              composes a grounded answer.
            </p>
            <div className="suggestion-chips">
              {SUGGESTIONS.map((s) => (
                <button key={s} type="button" className="chip" onClick={() => send(s)}>
                  {s}
                </button>
              ))}
            </div>
          </div>
        )}

        {messages.map((msg) => (
          <div key={msg.id} className={`msg-row ${msg.role}`}>
            <div className="msg-avatar" aria-hidden>
              {msg.role === "user" ? "You" : "AI"}
            </div>
            <div className="msg-body">
              {msg.role === "user" ? (
                <p className="msg-user-text">{msg.content}</p>
              ) : msg.isError ? (
                <p className="msg-user-text" style={{ color: "var(--danger)" }}>
                  {msg.content}
                </p>
              ) : (
                <>
                  <MarkdownMessage content={msg.content} />
                  <SourcesSection sources={msg.sources} />
                </>
              )}
            </div>
          </div>
        ))}

        {loading && (
          <div className="msg-row assistant">
            <div className="msg-avatar" aria-hidden>
              AI
            </div>
            <div className="msg-body">
              <div className="typing-indicator" aria-label="Thinking">
                <span />
                <span />
                <span />
              </div>
            </div>
          </div>
        )}
      </div>

      <div className="composer-wrap">
        <form className="composer" onSubmit={onSubmit}>
          <textarea
            ref={textareaRef}
            rows={1}
            placeholder="Message your documents…"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={onKeyDown}
            disabled={loading}
            aria-label="Message input"
          />
          <button type="submit" className="btn-send" disabled={loading || !input.trim()} aria-label="Send">
            <IconSend />
          </button>
        </form>
      </div>
    </div>
  );
}
