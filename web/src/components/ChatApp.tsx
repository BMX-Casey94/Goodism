"use client";

import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  AlertTriangle,
  ArrowUp,
  BookMarked,
  Moon,
  Plus,
  SlidersHorizontal,
  Sparkles,
  Square,
  Sun,
  WifiOff,
} from "lucide-react";
import type { ChatTurn, ConflictRecord, MetaPayload, Passage } from "@/lib/types";
import { FeedbackBar } from "./FeedbackBar";
import { LibraryDrawer } from "./LibraryDrawer";
import { MarkdownContent } from "./MarkdownContent";
import { PasscodeGate } from "./PasscodeGate";
import { SourcesDrawer } from "./SourcesDrawer";

const STARTER_PROMPTS = [
  {
    title: "Cestui Que Vie Act 1666",
    prompt: "What is the Cestui Que Vie Act 1666 and how do the trust documents interpret it?",
  },
  {
    title: "Good faith under the UCC",
    prompt: "How does the Uniform Commercial Code define good faith and honesty in fact?",
  },
  {
    title: "Birth registration & passports",
    prompt: "What does the knowledge base say about birth certificates and applying for a passport for an unregistered baby?",
  },
  {
    title: "Source disagreements",
    prompt: "Where do the documents in the knowledge base disagree with each other?",
  },
];

function shortModelName(model: string): string {
  if (!model) return "Evidence only";
  const parts = model.split("/");
  return parts[parts.length - 1] || model;
}

export function ChatApp() {
  const [authState, setAuthState] = useState<"checking" | "locked" | "ready">("checking");
  const [meta, setMeta] = useState<MetaPayload | null>(null);
  const [turns, setTurns] = useState<ChatTurn[]>([]);
  const [input, setInput] = useState("");
  const [domain, setDomain] = useState<string>("");
  const [contextDepth, setContextDepth] = useState<number>(1);
  const [showFilters, setShowFilters] = useState(false);
  const [libraryOpen, setLibraryOpen] = useState(false);
  const [darkMode, setDarkMode] = useState(false);
  const [isStreaming, setIsStreaming] = useState(false);

  const abortRef = useRef<AbortController | null>(null);
  const bottomRef = useRef<HTMLDivElement | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);

  // Sync theme with system preference on first mount.
  useEffect(() => {
    const prefersDark = window.matchMedia?.("(prefers-color-scheme: dark)").matches;
    setDarkMode(prefersDark);
  }, []);

  useEffect(() => {
    document.documentElement.classList.toggle("dark", darkMode);
  }, [darkMode]);

  const loadMeta = useCallback(async () => {
    try {
      const res = await fetch("/api/meta", { cache: "no-store" });
      if (res.status === 401) {
        setAuthState("locked");
        return;
      }
      if (res.ok) {
        const data = (await res.json()) as MetaPayload;
        setMeta(data);
      }
    } catch {
      // Handled by backendReachable state.
    }
  }, []);

  useEffect(() => {
    (async () => {
      try {
        const res = await fetch("/api/auth", { cache: "no-store" });
        const data = (await res.json()) as { required: boolean; authenticated: boolean };
        if (data.required && !data.authenticated) {
          setAuthState("locked");
        } else {
          setAuthState("ready");
          await loadMeta();
        }
      } catch {
        setAuthState("ready");
        await loadMeta();
      }
    })();
  }, [loadMeta]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [turns]);

  // Auto-grow textarea up to 160px.
  useEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 160)}px`;
  }, [input]);

  const domains = useMemo(() => {
    const set = new Set<string>();
    for (const doc of meta?.documents ?? []) {
      if (doc.domain) set.add(doc.domain);
    }
    return Array.from(set).sort();
  }, [meta]);

  function stopStreaming() {
    abortRef.current?.abort();
    abortRef.current = null;
    setIsStreaming(false);
    setTurns((prev) =>
      prev.map((t) => (t.streaming ? { ...t, streaming: false } : t))
    );
  }

  function startNewChat() {
    stopStreaming();
    setTurns([]);
    setInput("");
    textareaRef.current?.focus();
  }

  async function sendMessage(rawText?: string) {
    const text = (rawText ?? input).trim();
    if (!text || isStreaming) return;

    const userId = `u-${Date.now()}`;
    const assistantId = `a-${Date.now() + 1}`;

    const priorHistory = turns
      .filter((t) => !t.streaming && t.content.trim())
      .slice(-6)
      .map((t) => ({ role: t.role, content: t.content }));

    setInput("");
    setIsStreaming(true);
    setTurns((prev) => [
      ...prev,
      { id: userId, role: "user", content: text },
      {
        id: assistantId,
        role: "assistant",
        content: "",
        streaming: true,
        userQuestion: text,
      },
    ]);

    const controller = new AbortController();
    abortRef.current = controller;

    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: text,
          history: priorHistory,
          domain: domain || null,
          context: contextDepth,
        }),
        signal: controller.signal,
      });

      if (res.status === 401) {
        setAuthState("locked");
        setIsStreaming(false);
        return;
      }

      if (!res.ok || !res.body) {
        const errData = (await res.json().catch(() => ({}))) as { error?: string };
        setTurns((prev) =>
          prev.map((t) =>
            t.id === assistantId
              ? {
                  ...t,
                  streaming: false,
                  error: errData.error || `Request failed (HTTP ${res.status})`,
                }
              : t
          )
        );
        setIsStreaming(false);
        return;
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let currentEvent = "message";

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        const blocks = buffer.split("\n\n");
        buffer = blocks.pop() ?? "";

        for (const block of blocks) {
          let eventName = currentEvent;
          const dataLines: string[] = [];
          for (const line of block.split("\n")) {
            if (line.startsWith("event:")) {
              eventName = line.slice(6).trim();
            } else if (line.startsWith("data:")) {
              dataLines.push(line.slice(5).trim());
            }
          }
          currentEvent = "message";
          if (!dataLines.length) continue;

          let payload: Record<string, unknown>;
          try {
            payload = JSON.parse(dataLines.join("\n")) as Record<string, unknown>;
          } catch {
            continue;
          }

          if (eventName === "context") {
            setTurns((prev) =>
              prev.map((t) =>
                t.id === assistantId
                  ? {
                      ...t,
                      mode: payload.mode === "evidence" ? "evidence" : "qa",
                      model: typeof payload.model === "string" ? payload.model : "",
                      passages: Array.isArray(payload.passages)
                        ? (payload.passages as Passage[])
                        : [],
                      conflicts: Array.isArray(payload.conflicts)
                        ? (payload.conflicts as ConflictRecord[])
                        : [],
                      weakMatch: Boolean(payload.weak_match),
                    }
                  : t
              )
            );
          } else if (eventName === "delta" && typeof payload.text === "string") {
            const chunk = payload.text;
            setTurns((prev) =>
              prev.map((t) =>
                t.id === assistantId ? { ...t, content: t.content + chunk } : t
              )
            );
          } else if (eventName === "error" && typeof payload.error === "string") {
            const errMsg = payload.error;
            setTurns((prev) =>
              prev.map((t) =>
                t.id === assistantId ? { ...t, error: errMsg } : t
              )
            );
          } else if (eventName === "done") {
            setTurns((prev) =>
              prev.map((t) =>
                t.id === assistantId ? { ...t, streaming: false } : t
              )
            );
          }
        }
      }
    } catch (err) {
      if (!(err instanceof DOMException && err.name === "AbortError")) {
        setTurns((prev) =>
          prev.map((t) =>
            t.id === assistantId
              ? {
                  ...t,
                  streaming: false,
                  error: "Connection interrupted. Please try again.",
                }
              : t
          )
        );
      }
    } finally {
      abortRef.current = null;
      setIsStreaming(false);
      setTurns((prev) =>
        prev.map((t) => (t.id === assistantId ? { ...t, streaming: false } : t))
      );
    }
  }

  if (authState === "checking") {
    return (
      <div className="flex min-h-dvh items-center justify-center bg-[#faf9f6] text-sm text-stone-500 dark:bg-[#0d0e11] dark:text-zinc-400">
        Loading Goodism Knowledge Base…
      </div>
    );
  }

  if (authState === "locked") {
    return (
      <PasscodeGate
        onSuccess={() => {
          setAuthState("ready");
          void loadMeta();
        }}
      />
    );
  }

  return (
    <div className="flex h-dvh flex-col bg-[#faf9f6] text-stone-900 dark:bg-[#0d0e11] dark:text-zinc-100">
      {/* Header */}
      <header className="sticky top-0 z-30 border-b border-stone-200/80 bg-[#faf9f6]/85 backdrop-blur-md dark:border-zinc-800/80 dark:bg-[#0d0e11]/85">
        <div className="mx-auto flex h-14 max-w-4xl items-center justify-between gap-3 px-4">
          <div className="flex items-center gap-2.5">
            <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-stone-900 text-amber-300 dark:bg-zinc-100 dark:text-stone-900">
              <Sparkles className="h-4 w-4" />
            </div>
            <div>
              <h1 className="font-serif text-[0.98rem] font-semibold leading-tight tracking-tight text-stone-950 dark:text-zinc-50">
                Goodism Knowledge Base
              </h1>
              <div className="flex items-center gap-2 text-[11px] text-stone-500 dark:text-zinc-400">
                {meta?.status ? (
                  <span>
                    {meta.status.documents ?? 0} documents ·{" "}
                    {(meta.status.chunks ?? 0).toLocaleString("en-GB")} sections
                  </span>
                ) : (
                  <span>Evidence-first assistant</span>
                )}
                {meta?.llmConfigured && (
                  <>
                    <span aria-hidden="true">·</span>
                    <span className="font-mono text-[10px] text-amber-800 dark:text-amber-400">
                      {shortModelName(meta.model)}
                    </span>
                  </>
                )}
              </div>
            </div>
          </div>

          <div className="flex items-center gap-1.5">
            {turns.length > 0 && (
              <button
                type="button"
                onClick={startNewChat}
                className="inline-flex items-center gap-1.5 rounded-lg border border-stone-200 bg-white px-2.5 py-1.5 text-xs font-medium text-stone-700 transition hover:bg-stone-50 dark:border-zinc-800 dark:bg-zinc-900 dark:text-zinc-200 dark:hover:bg-zinc-800"
              >
                <Plus className="h-3.5 w-3.5" />
                <span className="hidden sm:inline">New chat</span>
              </button>
            )}
            <button
              type="button"
              onClick={() => setLibraryOpen(true)}
              className="inline-flex items-center gap-1.5 rounded-lg border border-stone-200 bg-white px-2.5 py-1.5 text-xs font-medium text-stone-700 transition hover:bg-stone-50 dark:border-zinc-800 dark:bg-zinc-900 dark:text-zinc-200 dark:hover:bg-zinc-800"
            >
              <BookMarked className="h-3.5 w-3.5 text-amber-700 dark:text-amber-500" />
              <span className="hidden sm:inline">Library</span>
            </button>
            <button
              type="button"
              onClick={() => setDarkMode((v) => !v)}
              aria-label="Toggle theme"
              className="rounded-lg border border-stone-200 bg-white p-1.5 text-stone-600 transition hover:bg-stone-50 hover:text-stone-900 dark:border-zinc-800 dark:bg-zinc-900 dark:text-zinc-400 dark:hover:bg-zinc-800 dark:hover:text-zinc-100"
            >
              {darkMode ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
            </button>
          </div>
        </div>
      </header>

      {/* Backend connectivity notice */}
      {meta && !meta.backendReachable && (
        <div className="border-b border-amber-200 bg-amber-50 px-4 py-2 text-center text-xs text-amber-900 dark:border-amber-900/60 dark:bg-amber-950/40 dark:text-amber-200">
          <span className="inline-flex items-center gap-1.5 font-medium">
            <WifiOff className="h-3.5 w-3.5" />
            Knowledge base retrieval server is currently offline. Check{" "}
            <code className="font-mono">KB_BACKEND_URL</code> or start{" "}
            <code className="font-mono">python -m server.web_app</code>.
          </span>
        </div>
      )}

      {/* Main conversation stream */}
      <main className="flex-1 overflow-y-auto">
        <div className="mx-auto max-w-3xl px-4 pb-8 pt-6">
          {turns.length === 0 ? (
            <div className="mx-auto flex max-w-2xl flex-col items-center pt-10 text-center sm:pt-16">
              <div className="mb-4 flex h-12 w-12 items-center justify-center rounded-2xl bg-stone-900 text-amber-300 shadow-sm dark:bg-zinc-100 dark:text-stone-900">
                <Sparkles className="h-6 w-6" />
              </div>
              <h2 className="font-serif text-2xl font-semibold tracking-tight text-stone-950 sm:text-3xl dark:text-zinc-50">
                Ask the Goodism Knowledge Base
              </h2>
              <p className="mt-2.5 max-w-lg text-sm leading-relaxed text-stone-600 dark:text-zinc-400">
                Every reply is grounded directly in the source corpus, with verbatim
                quotations, section citations and expandable surrounding context.
              </p>

              <div className="mt-8 grid w-full grid-cols-1 gap-2.5 text-left sm:grid-cols-2">
                {STARTER_PROMPTS.map((item) => (
                  <button
                    key={item.title}
                    type="button"
                    onClick={() => void sendMessage(item.prompt)}
                    className="group rounded-xl border border-stone-200/90 bg-white p-3.5 transition hover:border-amber-700/50 hover:shadow-sm dark:border-zinc-800 dark:bg-zinc-900/70 dark:hover:border-amber-500/40"
                  >
                    <div className="text-xs font-semibold text-stone-900 group-hover:text-amber-800 dark:text-zinc-100 dark:group-hover:text-amber-400">
                      {item.title}
                    </div>
                    <p className="mt-1 line-clamp-2 text-xs leading-relaxed text-stone-500 dark:text-zinc-400">
                      {item.prompt}
                    </p>
                  </button>
                ))}
              </div>
            </div>
          ) : (
            <div className="space-y-6">
              {turns.map((turn) =>
                turn.role === "user" ? (
                  <div key={turn.id} className="flex justify-end">
                    <div className="max-w-[85%] rounded-2xl rounded-br-md bg-stone-900 px-4 py-2.5 text-sm leading-relaxed text-white shadow-sm dark:bg-zinc-100 dark:text-zinc-950">
                      {turn.content}
                    </div>
                  </div>
                ) : (
                  <div key={turn.id} className="flex items-start gap-3">
                    <div className="mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-lg bg-amber-100 text-amber-800 dark:bg-amber-950/70 dark:text-amber-400">
                      <Sparkles className="h-3.5 w-3.5" />
                    </div>

                    <div className="min-w-0 flex-1">
                      {turn.weakMatch && (
                        <div className="mb-2.5 flex items-center gap-2 rounded-lg border border-amber-200 bg-amber-50/80 px-3 py-1.5 text-xs text-amber-900 dark:border-amber-900/60 dark:bg-amber-950/30 dark:text-amber-300">
                          <AlertTriangle className="h-3.5 w-3.5 shrink-0" />
                          <span>
                            Low-confidence match — the corpus may not directly cover this question.
                          </span>
                        </div>
                      )}

                      {turn.conflicts && turn.conflicts.length > 0 && (
                        <div className="mb-2.5 rounded-lg border border-amber-200/80 bg-amber-50/50 px-3 py-2 text-xs text-amber-950 dark:border-amber-900/50 dark:bg-amber-950/20 dark:text-amber-200">
                          <span className="font-semibold">Cross-document conflict noted: </span>
                          {turn.conflicts
                            .map((c) => `${c.doc_id} vs ${c.conflicts_with}`)
                            .join("; ")}
                        </div>
                      )}

                      {turn.content ? (
                        <MarkdownContent content={turn.content} />
                      ) : turn.streaming ? (
                        <div className="flex items-center gap-2 py-1 text-xs text-stone-500 dark:text-zinc-400">
                          <span className="inline-block h-2 w-2 animate-pulse rounded-full bg-amber-600" />
                          {turn.passages
                            ? `Found ${turn.passages.length} passages · Drafting answer…`
                            : "Searching knowledge base…"}
                        </div>
                      ) : null}

                      {turn.streaming && turn.content && (
                        <span className="ml-0.5 inline-block h-3.5 w-1.5 animate-pulse rounded-sm bg-amber-700 align-middle dark:bg-amber-400" />
                      )}

                      {turn.error && (
                        <div className="mt-2 rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-xs text-red-800 dark:border-red-900/60 dark:bg-red-950/40 dark:text-red-300">
                          {turn.error}
                        </div>
                      )}

                      {turn.passages && turn.passages.length > 0 && (
                        <SourcesDrawer
                          passages={turn.passages}
                          defaultOpen={turn.mode === "evidence"}
                        />
                      )}

                      {!turn.streaming && turn.userQuestion && (
                        <FeedbackBar
                          question={turn.userQuestion}
                          mode={turn.mode === "evidence" ? "evidence" : "qa"}
                          docIds={Array.from(
                            new Set((turn.passages ?? []).map((p) => p.doc_id))
                          )}
                        />
                      )}
                    </div>
                  </div>
                )
              )}
              <div ref={bottomRef} />
            </div>
          )}
        </div>
      </main>

      {/* Sticky Composer */}
      <div className="border-t border-stone-200/80 bg-[#faf9f6]/95 px-4 pb-4 pt-3 backdrop-blur-md dark:border-zinc-800/80 dark:bg-[#0d0e11]/95">
        <div className="mx-auto max-w-3xl">
          {showFilters && (
            <div className="mb-2.5 flex flex-wrap items-center gap-3 rounded-xl border border-stone-200 bg-white px-3.5 py-2 text-xs dark:border-zinc-800 dark:bg-zinc-900">
              <label className="flex items-center gap-2 text-stone-600 dark:text-zinc-400">
                <span>Domain:</span>
                <select
                  value={domain}
                  onChange={(e) => setDomain(e.target.value)}
                  className="rounded-md border border-stone-200 bg-stone-50 px-2 py-1 text-xs text-stone-900 focus:border-amber-700 focus:outline-none dark:border-zinc-700 dark:bg-zinc-950 dark:text-zinc-100"
                >
                  <option value="">All domains</option>
                  {domains.map((d) => (
                    <option key={d} value={d}>
                      {d}
                    </option>
                  ))}
                </select>
              </label>

              <label className="flex items-center gap-2 text-stone-600 dark:text-zinc-400">
                <span>Surrounding chunks:</span>
                <input
                  type="range"
                  min={0}
                  max={3}
                  value={contextDepth}
                  onChange={(e) => setContextDepth(Number(e.target.value))}
                  className="accent-amber-700 dark:accent-amber-500"
                />
                <span className="w-3 font-mono text-stone-900 dark:text-zinc-100">
                  {contextDepth}
                </span>
              </label>
            </div>
          )}

          <form
            onSubmit={(e) => {
              e.preventDefault();
              void sendMessage();
            }}
            className="relative flex items-end gap-2 rounded-2xl border border-stone-300/90 bg-white p-2 shadow-sm transition focus-within:border-amber-700/70 focus-within:ring-2 focus-within:ring-amber-700/10 dark:border-zinc-700/90 dark:bg-zinc-900 dark:focus-within:border-amber-500/60"
          >
            <button
              type="button"
              onClick={() => setShowFilters((v) => !v)}
              title="Search options"
              aria-label="Search options"
              className={`mb-0.5 rounded-xl p-2 transition ${
                showFilters || domain
                  ? "bg-amber-100 text-amber-900 dark:bg-amber-950/70 dark:text-amber-300"
                  : "text-stone-400 hover:bg-stone-100 hover:text-stone-700 dark:text-zinc-500 dark:hover:bg-zinc-800 dark:hover:text-zinc-200"
              }`}
            >
              <SlidersHorizontal className="h-4 w-4" />
            </button>

            <textarea
              ref={textareaRef}
              rows={1}
              maxLength={512}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  void sendMessage();
                }
              }}
              placeholder="Ask a question about the corpus…"
              className="max-h-40 flex-1 resize-none bg-transparent py-2 text-sm text-stone-900 placeholder:text-stone-400 focus:outline-none dark:text-zinc-100 dark:placeholder:text-zinc-500"
            />

            {isStreaming ? (
              <button
                type="button"
                onClick={stopStreaming}
                aria-label="Stop generating"
                className="mb-0.5 flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-stone-900 text-white transition hover:bg-stone-800 dark:bg-zinc-100 dark:text-zinc-900 dark:hover:bg-white"
              >
                <Square className="h-3.5 w-3.5 fill-current" />
              </button>
            ) : (
              <button
                type="submit"
                disabled={!input.trim()}
                aria-label="Send message"
                className="mb-0.5 flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-stone-900 text-white transition hover:bg-stone-800 disabled:opacity-35 dark:bg-zinc-100 dark:text-zinc-900 dark:hover:bg-white"
              >
                <ArrowUp className="h-4 w-4" />
              </button>
            )}
          </form>

          <p className="mt-2 text-center text-[11px] text-stone-400 dark:text-zinc-500">
            Answers cite the indexed corpus verbatim. Nothing here constitutes legal advice.
          </p>
        </div>
      </div>

      <LibraryDrawer
        open={libraryOpen}
        onClose={() => setLibraryOpen(false)}
        documents={meta?.documents ?? []}
        conflicts={meta?.conflicts ?? []}
      />
    </div>
  );
}
