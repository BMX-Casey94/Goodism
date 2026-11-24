"use client";

import React, { useState } from "react";
import { AlertCircle, BookMarked, X } from "lucide-react";
import type { ConflictRecord, DocumentSummary } from "@/lib/types";

interface LibraryDrawerProps {
  open: boolean;
  onClose: () => void;
  documents: DocumentSummary[];
  conflicts: ConflictRecord[];
}

export function LibraryDrawer({
  open,
  onClose,
  documents,
  conflicts,
}: LibraryDrawerProps) {
  const [tab, setTab] = useState<"documents" | "conflicts">("documents");

  if (!open) return null;

  return (
    <div className="fixed inset-0 z-50 flex justify-end bg-black/35 backdrop-blur-[2px]">
      <div
        className="flex h-full w-full max-w-md flex-col border-l border-stone-200 bg-white shadow-2xl dark:border-zinc-800 dark:bg-zinc-950"
        role="dialog"
        aria-label="Knowledge Base Library"
      >
        <div className="flex items-center justify-between border-b border-stone-200 px-5 py-4 dark:border-zinc-800">
          <div className="flex items-center gap-2">
            <BookMarked className="h-4 w-4 text-amber-700 dark:text-amber-500" />
            <h2 className="text-sm font-semibold text-stone-900 dark:text-zinc-100">
              Corpus Library
            </h2>
          </div>
          <button
            type="button"
            onClick={onClose}
            aria-label="Close library"
            className="rounded-lg p-1.5 text-stone-500 transition hover:bg-stone-100 hover:text-stone-900 dark:text-zinc-400 dark:hover:bg-zinc-800 dark:hover:text-zinc-100"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        <div className="flex gap-1 border-b border-stone-200 px-5 py-2.5 dark:border-zinc-800">
          <button
            type="button"
            onClick={() => setTab("documents")}
            className={`rounded-lg px-3 py-1.5 text-xs font-medium transition ${
              tab === "documents"
                ? "bg-stone-900 text-white dark:bg-zinc-100 dark:text-zinc-900"
                : "text-stone-600 hover:bg-stone-100 dark:text-zinc-400 dark:hover:bg-zinc-900"
            }`}
          >
            Documents ({documents.length})
          </button>
          <button
            type="button"
            onClick={() => setTab("conflicts")}
            className={`rounded-lg px-3 py-1.5 text-xs font-medium transition ${
              tab === "conflicts"
                ? "bg-stone-900 text-white dark:bg-zinc-100 dark:text-zinc-900"
                : "text-stone-600 hover:bg-stone-100 dark:text-zinc-400 dark:hover:bg-zinc-900"
            }`}
          >
            Conflicts ({conflicts.length})
          </button>
        </div>

        <div className="flex-1 overflow-y-auto px-5 py-4">
          {tab === "documents" ? (
            <div className="space-y-3">
              {documents.map((doc) => (
                <div
                  key={doc.doc_id}
                  className="rounded-xl border border-stone-200/90 bg-stone-50/70 p-3.5 dark:border-zinc-800 dark:bg-zinc-900/60"
                >
                  <div className="text-xs font-semibold text-stone-900 dark:text-zinc-100">
                    {doc.title}
                  </div>
                  <div className="mt-1.5 flex flex-wrap gap-1.5">
                    <span className="rounded-full bg-amber-100/80 px-2 py-0.5 text-[10px] font-medium text-amber-900 dark:bg-amber-950/70 dark:text-amber-300">
                      {doc.domain}
                    </span>
                    <span className="rounded-full bg-stone-200/80 px-2 py-0.5 text-[10px] font-medium text-stone-700 dark:bg-zinc-800 dark:text-zinc-300">
                      {doc.authority}
                    </span>
                    <span className="rounded-full border border-stone-200 px-2 py-0.5 font-mono text-[10px] text-stone-500 dark:border-zinc-700 dark:text-zinc-400">
                      {doc.sections ?? 0} sections
                    </span>
                  </div>
                  {doc.summary && (
                    <p className="mt-2 text-xs leading-relaxed text-stone-600 dark:text-zinc-400">
                      {doc.summary}
                    </p>
                  )}
                </div>
              ))}
              {documents.length === 0 && (
                <p className="text-xs text-stone-500 dark:text-zinc-400">
                  No indexed documents returned by the backend.
                </p>
              )}
            </div>
          ) : (
            <div className="space-y-3">
              {conflicts.map((c, idx) => (
                <div
                  key={`${c.doc_id}-${c.conflicts_with}-${idx}`}
                  className="rounded-xl border border-amber-200/80 bg-amber-50/60 p-3.5 dark:border-amber-900/60 dark:bg-amber-950/20"
                >
                  <div className="flex items-start gap-2">
                    <AlertCircle className="mt-0.5 h-3.5 w-3.5 shrink-0 text-amber-700 dark:text-amber-400" />
                    <div className="text-xs font-semibold text-stone-900 dark:text-zinc-100">
                      {c.doc_id} vs {c.conflicts_with}
                    </div>
                  </div>
                  {c.prevails && (
                    <div className="mt-1.5 text-[11px] font-medium text-amber-900 dark:text-amber-300">
                      Prevailing: {c.prevails}
                    </div>
                  )}
                  {(c.notes || c.reason) && (
                    <p className="mt-1.5 text-xs leading-relaxed text-stone-700 dark:text-zinc-300">
                      {c.notes || c.reason}
                    </p>
                  )}
                </div>
              ))}
              {conflicts.length === 0 && (
                <p className="text-xs text-stone-500 dark:text-zinc-400">
                  No recorded cross-document conflicts.
                </p>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
