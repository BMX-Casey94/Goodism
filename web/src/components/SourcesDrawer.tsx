"use client";

import React, { useState } from "react";
import { BookOpen, ChevronDown, ChevronRight, Expand, ShieldCheck, AlertTriangle } from "lucide-react";
import type { NeighbourChunk, Passage } from "@/lib/types";
import { MarkdownContent } from "./MarkdownContent";

interface SourcesDrawerProps {
  passages: Passage[];
  defaultOpen?: boolean;
}

export function SourcesDrawer({ passages, defaultOpen = false }: SourcesDrawerProps) {
  const [open, setOpen] = useState(defaultOpen);
  const [expanded, setExpanded] = useState<Record<number, NeighbourChunk[]>>({});
  const [loadingIdx, setLoadingIdx] = useState<number | null>(null);
  const [errorIdx, setErrorIdx] = useState<Record<number, string>>({});

  if (!passages.length) return null;

  async function loadSurrounding(idx: number, passage: Passage) {
    if (expanded[idx]) {
      setExpanded((prev) => {
        const next = { ...prev };
        delete next[idx];
        return next;
      });
      return;
    }
    setLoadingIdx(idx);
    setErrorIdx((prev) => {
      const next = { ...prev };
      delete next[idx];
      return next;
    });
    try {
      const res = await fetch("/api/read-around", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          doc_id: passage.doc_id,
          chunk: passage.start,
          before: 2,
          after: 2,
        }),
      });
      const data = (await res.json()) as { chunks?: NeighbourChunk[]; error?: string };
      if (!res.ok || !data.chunks) {
        setErrorIdx((prev) => ({
          ...prev,
          [idx]: data.error || "Could not load surrounding chunks",
        }));
        return;
      }
      setExpanded((prev) => ({ ...prev, [idx]: data.chunks! }));
    } catch {
      setErrorIdx((prev) => ({ ...prev, [idx]: "Network error loading context" }));
    } finally {
      setLoadingIdx(null);
    }
  }

  return (
    <div className="mt-3.5 rounded-xl border border-stone-200/90 bg-stone-50/70 dark:border-zinc-800 dark:bg-zinc-900/50">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="flex w-full items-center justify-between px-3.5 py-2.5 text-left text-xs font-medium text-stone-600 transition hover:text-stone-900 dark:text-zinc-400 dark:hover:text-zinc-100"
      >
        <span className="flex items-center gap-2">
          <BookOpen className="h-3.5 w-3.5 text-amber-700 dark:text-amber-500" />
          <span>
            Sources ({passages.length} {passages.length === 1 ? "passage" : "passages"})
          </span>
        </span>
        {open ? (
          <ChevronDown className="h-3.5 w-3.5" />
        ) : (
          <ChevronRight className="h-3.5 w-3.5" />
        )}
      </button>

      {open && (
        <div className="divide-y divide-stone-200/80 border-t border-stone-200/80 dark:divide-zinc-800 dark:border-zinc-800">
          {passages.map((p, idx) => (
            <div key={`${p.doc_id}-${p.start}-${idx}`} className="px-3.5 py-3">
              <div className="flex flex-wrap items-start justify-between gap-2">
                <div>
                  <div className="text-xs font-semibold text-stone-900 dark:text-zinc-100">
                    {p.title}
                  </div>
                  {p.sections.length > 0 && (
                    <div className="mt-0.5 text-[11px] text-stone-500 dark:text-zinc-400">
                      {p.sections.join(" · ")}
                    </div>
                  )}
                </div>

                <div className="flex flex-wrap items-center gap-1.5">
                  {p.weak_match && (
                    <span className="inline-flex items-center gap-1 rounded-full bg-amber-100 px-2 py-0.5 text-[10px] font-medium text-amber-900 dark:bg-amber-950/70 dark:text-amber-300">
                      <AlertTriangle className="h-2.5 w-2.5" />
                      Weak match
                    </span>
                  )}
                  <span className="rounded-full bg-stone-200/80 px-2 py-0.5 text-[10px] font-medium text-stone-700 dark:bg-zinc-800 dark:text-zinc-300">
                    {p.authority}
                  </span>
                  <span className="rounded-full border border-stone-200 px-2 py-0.5 font-mono text-[10px] text-stone-500 dark:border-zinc-700 dark:text-zinc-400">
                    chunks {p.start}–{p.end}
                  </span>
                </div>
              </div>

              <div className="mt-2 rounded-lg bg-white/90 p-3 text-stone-700 ring-1 ring-stone-200/70 dark:bg-zinc-950/70 dark:text-zinc-300 dark:ring-zinc-800">
                <MarkdownContent content={p.text.trim()} compact />
              </div>

              <div className="mt-2 flex flex-wrap items-center justify-between gap-2">
                <button
                  type="button"
                  onClick={() => loadSurrounding(idx, p)}
                  disabled={loadingIdx === idx}
                  className="inline-flex items-center gap-1.5 text-[11px] font-medium text-amber-800 hover:text-amber-950 disabled:opacity-50 dark:text-amber-400 dark:hover:text-amber-300"
                >
                  <Expand className="h-3 w-3" />
                  {loadingIdx === idx
                    ? "Loading context…"
                    : expanded[idx]
                    ? "Hide surrounding context"
                    : "Read surrounding context"}
                </button>

                {p.provenance.length > 0 && (
                  <span className="inline-flex items-center gap-1 font-mono text-[10px] text-emerald-700 dark:text-emerald-400">
                    <ShieldCheck className="h-3 w-3" />
                    {p.provenance[0].verified ? "Verified TXID: " : "TXID: "}
                    {p.provenance[0].txid.slice(0, 12)}…
                  </span>
                )}
              </div>

              {errorIdx[idx] && (
                <p className="mt-1.5 text-[11px] text-red-600 dark:text-red-400">
                  {errorIdx[idx]}
                </p>
              )}

              {expanded[idx] && (
                <div className="mt-2.5 space-y-2 rounded-lg border border-stone-200 bg-stone-100/70 p-2.5 dark:border-zinc-800 dark:bg-zinc-900">
                  {expanded[idx].map((ch) => (
                    <div key={ch.ordinal} className="text-xs">
                      <div className="font-mono text-[10px] text-stone-500 dark:text-zinc-400">
                        [{ch.ordinal}] {ch.heading_path}
                      </div>
                      <div className="mt-0.5 text-stone-700 dark:text-zinc-300">
                        <MarkdownContent content={ch.text.trim()} compact />
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
