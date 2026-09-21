"use client";

import Link from "next/link";
import { useParams, useRouter } from "next/navigation";
import { useCallback, useEffect, useRef, useState } from "react";
import { apiClient, ApiError } from "@/lib/api-client";
import { useSession } from "@/lib/auth-store";
import MarkdownContent from "@/components/markdown-content";
import type {
  ChatMessage,
  Citation,
  ConversationPublic,
  Rag,
} from "@contextvault/shared";

interface UIMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  citations: Citation[];
  /** Set while the assistant is still streaming this turn. */
  streaming?: boolean;
}

export default function ChatPage() {
  const params = useParams<{ ragId: string }>();
  const ragId = params?.ragId ?? "";
  const router = useRouter();
  const { hydrated, token } = useSession();

  const [rag, setRag] = useState<Rag | null>(null);
  const [conversations, setConversations] = useState<ConversationPublic[]>([]);
  const [activeConvId, setActiveConvId] = useState<string | null>(null);
  const [messages, setMessages] = useState<UIMessage[]>([]);
  const [input, setInput] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [streaming, setStreaming] = useState(false);
  const [notFound, setNotFound] = useState(false);

  const messagesRef = useRef<HTMLDivElement>(null);

  // --- bootstrap ---------------------------------------------------------
  useEffect(() => {
    if (!hydrated) return;
    if (!token) {
      router.replace("/login");
      return;
    }
    (async () => {
      try {
        const r = await apiClient.getRag(ragId);
        setRag(r);
        setConversations(await apiClient.listConversations(ragId));
      } catch (err) {
        if (err instanceof ApiError && err.status === 404) {
          setNotFound(true);
          return;
        }
        setError(err instanceof ApiError ? err.message : "failed to load");
      }
    })();
  }, [hydrated, token, ragId, router]);

  // Auto-scroll on new tokens
  useEffect(() => {
    const el = messagesRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [messages]);

  // --- load history when conversation selected ---------------------------
  const selectConversation = useCallback(
    async (convId: string) => {
      setActiveConvId(convId);
      setError(null);
      try {
        const conv = await apiClient.getConversation(convId);
        setMessages(
          conv.messages
            .filter((m) => m.role === "user" || m.role === "assistant")
            .map((m: ChatMessage) => ({
              id: m.id,
              role: m.role as "user" | "assistant",
              content: m.content,
              citations: m.citations ?? [],
            })),
        );
      } catch (err) {
        setError(err instanceof ApiError ? err.message : "load failed");
      }
    },
    [],
  );

  // --- stream one turn ---------------------------------------------------
  const onSend = useCallback(async () => {
    const text = input.trim();
    if (!text || !rag || streaming) return;
    setInput("");
    setError(null);
    setStreaming(true);

    // Optimistic user bubble
    const tempUserId = `tmp-user-${Date.now()}`;
    setMessages((prev) => [
      ...prev,
      { id: tempUserId, role: "user", content: text, citations: [] },
    ]);

    const tempAsstId = `tmp-asst-${Date.now()}`;
    setMessages((prev) => [
      ...prev,
      {
        id: tempAsstId,
        role: "assistant",
        content: "",
        citations: [],
        streaming: true,
      },
    ]);

    let streamedCitations: Citation[] = [];
    let accumulated = "";
    try {
      for await (const event of apiClient.streamChat(ragId, {
        message: text,
        conversation_id: activeConvId,
      })) {
        if (event.type === "citation") {
          streamedCitations = event.citations;
          setMessages((prev) =>
            prev.map((m) =>
              m.id === tempAsstId ? { ...m, citations: event.citations } : m,
            ),
          );
        } else if (event.type === "token") {
          accumulated += event.delta;
          const snapshot = accumulated;
          setMessages((prev) =>
            prev.map((m) =>
              m.id === tempAsstId ? { ...m, content: snapshot } : m,
            ),
          );
        } else if (event.type === "done") {
          setActiveConvId(event.conversation_id);
          setMessages((prev) =>
            prev.map((m) =>
              m.id === tempAsstId
                ? { ...m, id: event.message_id, streaming: false, citations: event.citations }
                : m,
            ),
          );
        } else if (event.type === "error") {
          setError(event.message);
          setMessages((prev) => prev.filter((m) => m.id !== tempAsstId));
        }
      }
    } catch (err) {
      setMessages((prev) => prev.filter((m) => m.id !== tempAsstId));
      setError(err instanceof ApiError ? err.message : "chat failed");
    } finally {
      setStreaming(false);
      // Refresh conversation list so the new one shows up
      apiClient.listConversations(ragId).then(setConversations).catch(() => {});
    }
  }, [input, rag, ragId, streaming, activeConvId]);

  if (!hydrated || !token) {
    return <p className="text-sm text-slate-500">Redirecting to login…</p>;
  }
  if (notFound) {
    return (
      <section className="space-y-4">
        <h1 className="text-2xl font-semibold">RAG not found</h1>
        <p className="text-sm text-slate-500">
          This knowledge base may have been deleted, or it does not belong
          to your account.
        </p>
        <Link
          href="/dashboard"
          className="inline-block rounded border border-slate-300 px-4 py-2 text-sm hover:bg-slate-50 dark:border-slate-700 dark:hover:bg-slate-800"
        >
          ← Back to dashboard
        </Link>
      </section>
    );
  }
  if (!rag) {
    return <p className="text-sm text-slate-500">Loading…</p>;
  }

  return (
    <section className="grid gap-4 lg:grid-cols-[16rem_minmax(0,1fr)]">
      {/* Sidebar: conversation list */}
      <aside className="space-y-3 rounded-lg border border-slate-200 bg-white p-4 dark:border-slate-800 dark:bg-slate-900">
        <div>
          <h2 className="text-sm font-semibold uppercase tracking-wide text-slate-500">
            RAG
          </h2>
          <p className="mt-1 text-base font-semibold">{rag.name}</p>
          <Link
            href={`/rags/${ragId}`}
            className="text-xs text-brand-600 hover:underline"
          >
            ← back to RAG details
          </Link>
        </div>
        <div>
          <h3 className="mt-4 text-sm font-semibold uppercase tracking-wide text-slate-500">
            Conversations
          </h3>
          <button
            type="button"
            onClick={() => {
              setActiveConvId(null);
              setMessages([]);
            }}
            className="mt-2 w-full rounded border border-slate-300 px-3 py-2 text-left text-sm hover:bg-slate-50 dark:border-slate-700 dark:hover:bg-slate-800"
          >
            + New conversation
          </button>
          <ul className="mt-2 space-y-1">
            {conversations.map((c) => (
              <li key={c.id}>
                <button
                  type="button"
                  onClick={() => selectConversation(c.id)}
                  className={`w-full truncate rounded px-3 py-2 text-left text-xs hover:bg-slate-50 dark:hover:bg-slate-800 ${
                    c.id === activeConvId
                      ? "bg-brand-50 text-brand-800 dark:bg-brand-900/30 dark:text-brand-100"
                      : ""
                  }`}
                >
                  {c.title || "Untitled"}
                </button>
              </li>
            ))}
            {conversations.length === 0 && (
              <li className="text-xs text-slate-400">No conversations yet.</li>
            )}
          </ul>
        </div>
      </aside>

      {/* Main: messages + composer */}
      <div className="flex h-[calc(100vh-12rem)] flex-col rounded-lg border border-slate-200 bg-white dark:border-slate-800 dark:bg-slate-900">
        <div
          ref={messagesRef}
          className="flex-1 space-y-4 overflow-y-auto p-4"
        >
          {messages.length === 0 && (
            <div className="rounded border border-dashed border-slate-300 p-4 text-sm text-slate-500 dark:border-slate-700">
              Ask a question about <strong>{rag.name}</strong>. The answer
              will cite the documents that informed it.
            </div>
          )}
          {messages.map((m) => (
            <MessageBubble key={m.id} msg={m} />
          ))}
          {streaming && (
            <p className="text-xs text-slate-400">assistant is typing…</p>
          )}
          {error && (
            <p className="rounded border border-rose-300 bg-rose-50 px-3 py-2 text-sm text-rose-700 dark:bg-rose-950/40 dark:text-rose-200">
              {error}
            </p>
          )}
        </div>

        <form
          className="flex gap-2 border-t border-slate-200 p-3 dark:border-slate-800"
          onSubmit={(e) => {
            e.preventDefault();
            onSend();
          }}
        >
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            disabled={streaming}
            placeholder="Ask the knowledge base…"
            className="flex-1 rounded border border-slate-300 bg-white p-2 text-sm dark:border-slate-700 dark:bg-slate-900"
          />
          <button
            type="submit"
            disabled={streaming || !input.trim()}
            className="rounded bg-brand-600 px-4 py-2 text-sm font-medium text-white hover:bg-brand-700 disabled:opacity-60"
          >
            {streaming ? "Sending…" : "Send"}
          </button>
        </form>
      </div>
    </section>
  );
}

function MessageBubble({ msg }: { msg: UIMessage }) {
  const isUser = msg.role === "user";
  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"}`}>
      <div
        className={`max-w-[85%] rounded-lg px-4 py-3 text-sm shadow-sm ${
          isUser
            ? "whitespace-pre-wrap bg-brand-600 text-white"
            : "overflow-x-auto bg-slate-100 text-slate-900 dark:bg-slate-800 dark:text-slate-100"
        }`}
      >
        {isUser
          ? msg.content
          : msg.content
            ? <MarkdownContent content={msg.content} />
            : msg.streaming
              ? "…"
              : null}
        {msg.citations && msg.citations.length > 0 && (
          <div
            className={`mt-2 flex flex-wrap gap-1 ${
              isUser ? "text-brand-50" : "text-slate-500"
            }`}
          >
            {msg.citations.map((c, i) => (
              <span
                key={c.chunk_id}
                title={`${c.filename} · page ${c.page_number} · score ${c.retrieval_score.toFixed(2)}`}
                className="inline-flex items-center gap-1 rounded bg-black/10 px-2 py-0.5 text-[10px] font-mono"
              >
                [{i + 1}] {c.filename} p.{c.page_number}
              </span>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}