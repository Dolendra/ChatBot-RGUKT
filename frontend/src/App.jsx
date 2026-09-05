import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { MarkdownMessage } from "./components/MarkdownMessage.jsx";
import {
  createEmptyConversation,
  ensureStore,
  saveChatStore,
  titleFromMessage,
} from "./chatStore.js";

const API_BASE = import.meta.env.VITE_API_URL ?? "";

const SUGGESTIONS = [
  "What are the attendance rules?",
  "What is the fee for PUC and B.Tech?",
  "What are the rules of promotion for PUC?",
  "What grading system is used for PUC?",
];

const PIPELINE_STAGES = [
  { id: "search", label: "Searching university documents…", doneLabel: "Evidence found" },
  { id: "generate", label: "Generating answer…", doneLabel: "Draft generated" },
  { id: "verify", label: "Verifying answer…", doneLabel: "Claims & citations checked" },
];

function IconDoc() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true">
      <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
      <path d="M14 2v6h6M16 13H8M16 17H8M10 9H8" />
    </svg>
  );
}

function IconSend() {
  return (
    <svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
      <path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z" />
    </svg>
  );
}

function IconBook() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true">
      <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20" />
      <path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z" />
    </svg>
  );
}

function IconPlay() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true">
      <polygon points="5 3 19 12 5 21 5 3" fill="currentColor" />
    </svg>
  );
}

function IconGauge() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true">
      <path d="M12 2a10 10 0 0 0-10 10c0 5.523 4.477 10 10 10s10-4.477 10-10a10 10 0 0 0-10-10z" />
      <polyline points="12 6 12 12 16 14" />
    </svg>
  );
}

function IconPlus() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true">
      <path d="M12 5v14M5 12h14" />
    </svg>
  );
}

function IconTrash() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true">
      <path d="M3 6h18M8 6V4h8v2M19 6l-1 14H6L5 6" />
    </svg>
  );
}

function IconMenu() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true">
      <path d="M4 6h16M4 12h16M4 18h16" />
    </svg>
  );
}

function IconPencil() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true">
      <path d="M12 20h9" />
      <path d="M16.5 3.5a2.1 2.1 0 0 1 3 3L7 19l-4 1 1-4Z" />
    </svg>
  );
}

function shortDocName(name) {
  if (!name) return "Handbook";
  return String(name)
    .replace(/\.pdf$/i, "")
    .replace(/_/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function SourcesSection({ sources, activeSourceId, onOpenSource, sourceRefs }) {
  if (!sources || sources.length === 0) return null;

  return (
    <div className="sources-block">
      <div className="sources-title">
        <IconBook />
        Sources ({sources.length})
      </div>
      <div className="source-cards">
        {sources.map((s) => {
          const highlighted = activeSourceId === s.id;
          return (
            <details
              key={s.id}
              className={`source-card${highlighted ? " source-card--active" : ""}`}
              ref={(el) => {
                if (sourceRefs) sourceRefs.current[s.id] = el;
              }}
              onToggle={(e) => {
                if (e.currentTarget.open) onOpenSource?.(s.id);
                else if (highlighted) onOpenSource?.(null);
              }}
            >
              <summary>
                <div className="source-header-left">
                  <span className="source-badge">Source {s.id}</span>
                  <span className="source-doc-name">{shortDocName(s.source)}</span>
                </div>
                <span className="source-page">Page {s.page}</span>
              </summary>
              <div className="source-meta-row">
                {s.section && (
                  <div className="source-meta-item">
                    <strong>Section:</strong> {s.section}
                  </div>
                )}
                {s.version && s.version !== "—" && (
                  <div className="source-meta-item">
                    <strong>Version:</strong> {s.version}
                  </div>
                )}
              </div>
              <p className="source-snippet">{s.snippet}</p>
            </details>
          );
        })}
      </div>
    </div>
  );
}

function PipelineProgress({ stageIndex }) {
  return (
    <div className="pipeline-progress" aria-live="polite" aria-label="Answer progress">
      {PIPELINE_STAGES.map((stage, i) => {
        const done = i < stageIndex;
        const active = i === stageIndex;
        return (
          <div
            key={stage.id}
            className={`pipeline-step${done ? " is-done" : ""}${active ? " is-active" : ""}`}
          >
            <span className="pipeline-mark" aria-hidden="true">
              {done ? "✓" : active ? "●" : "○"}
            </span>
            <span className="pipeline-label">
              {done ? stage.doneLabel : stage.label}
              {active && stage.id === "verify" && (
                <span className="pipeline-sub">Checking claims · numbers · citations</span>
              )}
            </span>
          </div>
        );
      })}
    </div>
  );
}

function EvaluationDashboard() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [expandedCase, setExpandedCase] = useState(null);

  const fetchResults = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/api/evaluate`);
      if (!res.ok) throw new Error("Failed to fetch evaluation metrics");
      const json = await res.json();
      if (json && !json.has_run) setData(null);
      else setData(json);
    } catch (e) {
      setError(e.message);
    }
  }, []);

  useEffect(() => {
    fetchResults();
  }, [fetchResults]);

  const runEvaluation = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API_BASE}/api/evaluate/run`, { method: "POST" });
      if (!res.ok) throw new Error("Failed to execute RAG evaluation suite");
      setData(await res.json());
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="dashboard-container dashboard-center">
        <div className="typing-indicator" style={{ transform: "scale(1.5)", marginBottom: 20 }} aria-hidden="true">
          <span />
          <span />
          <span />
        </div>
        <h3>Running evaluation suite</h3>
        <p className="dashboard-hint">
          Measuring Recall@K, faithfulness, correctness, and verification decisions on the 19-question regression set.
        </p>
      </div>
    );
  }

  if (!data) {
    return (
      <div className="dashboard-container dashboard-center">
        <div className="app-logo" style={{ marginBottom: 20 }} aria-hidden="true">
          <IconGauge />
        </div>
        <h2>Evaluate Your RAG Engine</h2>
        <p className="dashboard-hint">
          Measure retrieval recall, answer quality, citation grounding, and component latency.
        </p>
        {error && <div className="error-banner" style={{ marginBottom: 16 }}>{error}</div>}
        <button type="button" className="btn-primary" onClick={runEvaluation}>
          <IconPlay />
          Run RAG Evaluation Suite
        </button>
      </div>
    );
  }

  const { recall_at_1, recall_at_3, recall_at_5, faithfulness, correctness, avg_latencies, results } = data;
  const total_latency = avg_latencies?.total_ms || 1.0;

  return (
    <div className="dashboard-container">
      <div className="dashboard-header-row">
        <div>
          <h2>Evaluation Dashboard</h2>
          <p className="dashboard-meta">Last run: {data.timestamp || "Just now"}</p>
        </div>
        <button type="button" className="btn-primary" onClick={runEvaluation}>
          <IconPlay />
          Run Re-Evaluation
        </button>
      </div>

      {error && <div className="error-banner">{error}</div>}

      <div className="metrics-grid">
        <div className="metric-card">
          <span className="metric-value">{recall_at_1}%</span>
          <span className="metric-label">Recall@1</span>
        </div>
        <div className="metric-card">
          <span className="metric-value">{recall_at_3}%</span>
          <span className="metric-label">Recall@3</span>
        </div>
        <div className="metric-card">
          <span className="metric-value">{recall_at_5}%</span>
          <span className="metric-label">Recall@5</span>
        </div>
        <div className="metric-card">
          <span className="metric-value">{faithfulness}%</span>
          <span className="metric-label">Faithfulness</span>
        </div>
        <div className="metric-card">
          <span className="metric-value">{correctness}%</span>
          <span className="metric-label">Correctness</span>
        </div>
      </div>

      <div className="dashboard-split-row">
        <div className="card-panel">
          <h3>Test Query Results</h3>
          <div style={{ overflowX: "auto" }}>
            <table className="test-cases-table">
              <thead>
                <tr>
                  <th>Category</th>
                  <th>Question</th>
                  <th style={{ textAlign: "center" }}>Recall@5</th>
                  <th style={{ textAlign: "center" }}>F</th>
                  <th style={{ textAlign: "center" }}>C</th>
                </tr>
              </thead>
              <tbody>
                {results.map((res, i) => (
                  <tr key={i}>
                    <td style={{ color: "var(--text-muted)", fontWeight: 600 }}>{res.category}</td>
                    <td>
                      <button
                        type="button"
                        className="linkish"
                        onClick={() => setExpandedCase(expandedCase === i ? null : i)}
                      >
                        {res.question}
                        <span className="expand-caret">{expandedCase === i ? "▲" : "▼"}</span>
                      </button>
                      {expandedCase === i && (
                        <div className="test-case-details">
                          <div>
                            <strong>Answer:</strong>
                            <p>{res.generated_answer}</p>
                          </div>
                          {res.category !== "Out-of-Domain" && (
                            <div>
                              <strong>Pages:</strong> {(res.retrieved_pages || []).join(", ")}
                            </div>
                          )}
                          <div>
                            <strong>Latency:</strong>{" "}
                            <span className="mono-soft">
                              Retrieve {res.latency?.total_retrieve_ms}ms · Gen {res.latency?.generation_ms}ms
                              {res.latency?.verify_ms != null ? ` · Verify ${res.latency.verify_ms}ms` : ""}
                            </span>
                          </div>
                        </div>
                      )}
                    </td>
                    <td style={{ textAlign: "center" }}>
                      <span className={`badge-status ${res.recall_5 ? "pass" : "fail"}`}>
                        {res.recall_5 ? "MATCH" : "FAIL"}
                      </span>
                    </td>
                    <td style={{ textAlign: "center", fontFamily: "var(--mono)" }}>{res.faithfulness}/5</td>
                    <td style={{ textAlign: "center", fontFamily: "var(--mono)" }}>{res.correctness}/5</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        <div className="card-panel">
          <h3>Latency Profile</h3>
          <div className="latency-bars">
            {[
              ["Embedding", avg_latencies.embedding_ms],
              ["FAISS", avg_latencies.faiss_ms],
              ["BM25", avg_latencies.bm25_ms],
              ["RRF", avg_latencies.rrf_ms],
              ["Rerank", avg_latencies.rerank_ms],
              ["Generation", avg_latencies.generation_ms],
              ["Verify", avg_latencies.verify_ms || 0],
            ].map(([label, ms]) => (
              <div className="latency-bar-item" key={label}>
                <div className="latency-info">
                  <span className="latency-lbl">{label}</span>
                  <span className="latency-val">{ms ?? 0} ms</span>
                </div>
                <div className="latency-track">
                  <div
                    className="latency-fill"
                    style={{ width: `${((ms || 0) / total_latency) * 100}%` }}
                  />
                </div>
              </div>
            ))}
            <div className="latency-total">
              <span>Total</span>
              <span>{avg_latencies.total_ms} ms</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default function App() {
  const initial = useMemo(() => ensureStore(), []);
  const [conversations, setConversations] = useState(initial.conversations);
  const [activeId, setActiveId] = useState(initial.activeId);
  const [appTitle, setAppTitle] = useState("RGUKT Academic Assistant");
  const [appSubtitle, setAppSubtitle] = useState("Grounded answers from university documents");
  const [view, setView] = useState("chat");
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [pipelineStage, setPipelineStage] = useState(0);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [renamingId, setRenamingId] = useState(null);
  const [renameValue, setRenameValue] = useState("");
  const [activeSourceByMsg, setActiveSourceByMsg] = useState({});

  const scrollRef = useRef(null);
  const textareaRef = useRef(null);
  const sourceRefsByMsg = useRef({});
  const stageTimers = useRef([]);

  const activeConversation = conversations.find((c) => c.id === activeId) || conversations[0];
  const messages = activeConversation?.messages || [];

  useEffect(() => {
    fetch(`${API_BASE}/api/config`)
      .then((r) => (r.ok ? r.json() : null))
      .then((d) => {
        if (!d) return;
        if (d.app_title) {
          setAppTitle(d.app_title);
          document.title = d.app_title;
        }
        if (d.app_subtitle) setAppSubtitle(d.app_subtitle);
      })
      .catch(() => {});
  }, []);

  useEffect(() => {
    saveChatStore({ conversations, activeId });
  }, [conversations, activeId]);

  const scrollToBottom = useCallback(() => {
    const el = scrollRef.current;
    if (el) el.scrollTo({ top: el.scrollHeight, behavior: "smooth" });
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages, loading, pipelineStage, scrollToBottom]);

  const clearStageTimers = () => {
    stageTimers.current.forEach(clearTimeout);
    stageTimers.current = [];
  };

  const startPipelineStages = () => {
    clearStageTimers();
    setPipelineStage(0);
    stageTimers.current.push(setTimeout(() => setPipelineStage(1), 1600));
    stageTimers.current.push(setTimeout(() => setPipelineStage(2), 7000));
  };

  const updateActiveMessages = (updater) => {
    setConversations((prev) =>
      prev.map((c) => {
        if (c.id !== activeId) return c;
        const nextMessages = typeof updater === "function" ? updater(c.messages) : updater;
        return { ...c, messages: nextMessages, updatedAt: Date.now() };
      })
    );
  };

  const newChat = () => {
    const conv = createEmptyConversation();
    setConversations((prev) => [conv, ...prev]);
    setActiveId(conv.id);
    setInput("");
    setSidebarOpen(false);
    setView("chat");
    textareaRef.current?.focus();
  };

  const selectChat = (id) => {
    setActiveId(id);
    setSidebarOpen(false);
    setView("chat");
  };

  const deleteChat = (id, e) => {
    e?.stopPropagation();
    setConversations((prev) => {
      const next = prev.filter((c) => c.id !== id);
      if (!next.length) {
        const fresh = createEmptyConversation();
        setActiveId(fresh.id);
        return [fresh];
      }
      if (id === activeId) setActiveId(next[0].id);
      return next;
    });
  };

  const beginRename = (conv, e) => {
    e?.stopPropagation();
    setRenamingId(conv.id);
    setRenameValue(conv.title);
  };

  const commitRename = () => {
    if (!renamingId) return;
    const title = renameValue.trim() || "New chat";
    setConversations((prev) =>
      prev.map((c) => (c.id === renamingId ? { ...c, title, updatedAt: Date.now() } : c))
    );
    setRenamingId(null);
  };

  const openSource = (messageId, sourceId) => {
    setActiveSourceByMsg((prev) => ({ ...prev, [messageId]: sourceId }));
    requestAnimationFrame(() => {
      const el = sourceRefsByMsg.current[messageId]?.[sourceId];
      if (el) {
        el.open = true;
        el.scrollIntoView({ behavior: "smooth", block: "nearest" });
      }
    });
  };

  const send = async (text) => {
    const trimmed = text.trim();
    if (!trimmed || loading || !activeConversation) return;

    const historyPayload = messages
      .filter((m) => (m.role === "user" || m.role === "assistant") && !m.isError)
      .map(({ role, content }) => ({
        role,
        content: String(content ?? "").slice(0, 12000),
      }));

    const userMsg = { id: crypto.randomUUID(), role: "user", content: trimmed };
    const isFirst = messages.length === 0;

    setConversations((prev) =>
      prev.map((c) => {
        if (c.id !== activeId) return c;
        return {
          ...c,
          title: isFirst || c.title === "New chat" ? titleFromMessage(trimmed) : c.title,
          messages: [...c.messages, userMsg],
          updatedAt: Date.now(),
        };
      })
    );
    setInput("");
    setLoading(true);
    startPipelineStages();

    try {
      const res = await fetch(`${API_BASE}/api/chat`, {
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
              : "The assistant is temporarily unavailable. Please try again.";
        throw new Error(msg);
      }
      setPipelineStage(3);
      updateActiveMessages((m) => [
        ...m,
        {
          id: crypto.randomUUID(),
          role: "assistant",
          content: data.answer || "",
          sources: data.sources || [],
          metrics: data.metrics || {},
        },
      ]);
    } catch (e) {
      updateActiveMessages((m) => [
        ...m,
        {
          id: crypto.randomUUID(),
          role: "assistant",
          content: e.message || "Something went wrong.",
          isError: true,
        },
      ]);
    } finally {
      clearStageTimers();
      setLoading(false);
      setPipelineStage(0);
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

  useEffect(() => {
    const ta = textareaRef.current;
    if (!ta) return;
    ta.style.height = "auto";
    ta.style.height = `${Math.min(ta.scrollHeight, 160)}px`;
  }, [input]);

  const sortedConversations = useMemo(
    () => [...conversations].sort((a, b) => (b.updatedAt || 0) - (a.updatedAt || 0)),
    [conversations]
  );

  return (
    <div className={`app-layout${sidebarOpen ? " sidebar-open" : ""}`}>
      <div
        className="sidebar-backdrop"
        onClick={() => setSidebarOpen(false)}
        aria-hidden={!sidebarOpen}
      />

      <aside className="chat-sidebar" aria-label="Conversations">
        <div className="sidebar-top">
          <button type="button" className="btn-new-chat" onClick={newChat}>
            <IconPlus />
            New chat
          </button>
        </div>

        <div className="sidebar-section-label">Recent</div>
        <div className="sidebar-list">
          {sortedConversations.map((conv) => (
            <div
              key={conv.id}
              className={`sidebar-item${conv.id === activeId ? " is-active" : ""}`}
              onClick={() => selectChat(conv.id)}
              role="button"
              tabIndex={0}
              onKeyDown={(e) => {
                if (e.key === "Enter" || e.key === " ") selectChat(conv.id);
              }}
            >
              {renamingId === conv.id ? (
                <input
                  className="sidebar-rename-input"
                  value={renameValue}
                  autoFocus
                  onClick={(e) => e.stopPropagation()}
                  onChange={(e) => setRenameValue(e.target.value)}
                  onBlur={commitRename}
                  onKeyDown={(e) => {
                    if (e.key === "Enter") commitRename();
                    if (e.key === "Escape") setRenamingId(null);
                  }}
                />
              ) : (
                <span className="sidebar-item-title">{conv.title}</span>
              )}
              <div className="sidebar-item-actions">
                <button
                  type="button"
                  className="icon-btn"
                  title="Rename"
                  aria-label="Rename conversation"
                  onClick={(e) => beginRename(conv, e)}
                >
                  <IconPencil />
                </button>
                <button
                  type="button"
                  className="icon-btn"
                  title="Delete"
                  aria-label="Delete conversation"
                  onClick={(e) => deleteChat(conv.id, e)}
                >
                  <IconTrash />
                </button>
              </div>
            </div>
          ))}
        </div>
      </aside>

      <div className="app-shell">
        <header className="app-header">
          <div className="app-brand">
            <button
              type="button"
              className="btn-icon-menu"
              aria-label="Open conversations"
              onClick={() => setSidebarOpen((v) => !v)}
            >
              <IconMenu />
            </button>
            <div className="app-logo" aria-hidden="true">
              <IconDoc />
            </div>
            <div className="app-title-wrap">
              <h1>{appTitle}</h1>
              <p>{appSubtitle}</p>
            </div>
          </div>
          <div className="header-actions">
            <div className="view-toggle">
              <button
                type="button"
                className={`btn-tab ${view === "chat" ? "active" : ""}`}
                onClick={() => setView("chat")}
              >
                Chat
              </button>
              <button
                type="button"
                className={`btn-tab ${view === "evaluation" ? "active" : ""}`}
                onClick={() => setView("evaluation")}
              >
                Evaluation
              </button>
            </div>
            {view === "chat" && (
              <button type="button" className="btn-ghost" onClick={newChat}>
                New chat
              </button>
            )}
          </div>
        </header>

        {view === "chat" ? (
          <>
            <div className="messages-scroll" ref={scrollRef}>
              {messages.length === 0 && !loading && (
                <div className="empty-state">
                  <h2>RGUKT Academic Assistant</h2>
                  <p>
                    Ask about academic regulations, attendance, examinations, fees, grading, and
                    programmes — answers are grounded in the indexed handbook with page-level sources.
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

              {messages.map((msg) => {
                if (!sourceRefsByMsg.current[msg.id]) sourceRefsByMsg.current[msg.id] = {};
                return (
                  <div key={msg.id} className={`msg-row ${msg.role}`}>
                    <div className="msg-avatar" aria-hidden="true">
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
                          <MarkdownMessage
                            content={msg.content}
                            onSourceClick={(n) => openSource(msg.id, n)}
                          />
                          <SourcesSection
                            sources={msg.sources}
                            activeSourceId={activeSourceByMsg[msg.id]}
                            onOpenSource={(id) =>
                              setActiveSourceByMsg((prev) => ({ ...prev, [msg.id]: id }))
                            }
                            sourceRefs={{ current: sourceRefsByMsg.current[msg.id] }}
                          />
                          {msg.metrics?.total_ms > 0 && (
                            <div className="msg-latency">
                              <span>Retrieve {msg.metrics.total_retrieve_ms}ms</span>
                              <span>Gen {msg.metrics.generation_ms}ms</span>
                              {msg.metrics.verify_ms != null && (
                                <span>Verify {msg.metrics.verify_ms}ms</span>
                              )}
                              <span>Total {msg.metrics.total_ms}ms</span>
                              {msg.metrics.verification?.decision && (
                                <span className="verify-pill">
                                  {msg.metrics.verification.decision}
                                </span>
                              )}
                            </div>
                          )}
                        </>
                      )}
                    </div>
                  </div>
                );
              })}

              {loading && (
                <div className="msg-row assistant">
                  <div className="msg-avatar" aria-hidden="true">
                    AI
                  </div>
                  <div className="msg-body">
                    <PipelineProgress stageIndex={Math.min(pipelineStage, 2)} />
                  </div>
                </div>
              )}
            </div>

            <div className="composer-wrap">
              <form className="composer" onSubmit={onSubmit}>
                <textarea
                  ref={textareaRef}
                  rows={1}
                  placeholder="Ask about attendance, exams, fees…"
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={onKeyDown}
                  disabled={loading}
                  aria-label="Message input"
                />
                <button
                  type="submit"
                  className="btn-send"
                  disabled={loading || !input.trim()}
                  aria-label="Send"
                >
                  <IconSend />
                </button>
              </form>
            </div>
          </>
        ) : (
          <EvaluationDashboard />
        )}
      </div>
    </div>
  );
}
