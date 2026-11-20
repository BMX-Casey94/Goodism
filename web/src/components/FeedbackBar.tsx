"use client";

import React, { useState } from "react";
import { Check, Flag, ThumbsDown, ThumbsUp } from "lucide-react";
import type { FeedbackRating } from "@/lib/types";

interface FeedbackBarProps {
  question: string;
  mode: "qa" | "evidence";
  docIds: string[];
}

export function FeedbackBar({ question, mode, docIds }: FeedbackBarProps) {
  const [selected, setSelected] = useState<FeedbackRating | null>(null);
  const [comment, setComment] = useState("");
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function submit(rating: FeedbackRating, note: string) {
    setSaving(true);
    setError(null);
    try {
      const res = await fetch("/api/feedback", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question,
          mode,
          rating,
          comment: note,
          doc_ids: docIds,
        }),
      });
      if (!res.ok) {
        const data = (await res.json().catch(() => ({}))) as { error?: string };
        setError(data.error || "Could not save feedback");
        return;
      }
      setSaved(true);
    } catch {
      setError("Network error saving feedback");
    } finally {
      setSaving(false);
    }
  }

  if (saved) {
    return (
      <div className="mt-2.5 flex items-center gap-1.5 text-[11px] font-medium text-emerald-700 dark:text-emerald-400">
        <Check className="h-3.5 w-3.5" />
        Saved to accuracy review log
      </div>
    );
  }

  const buttonBase =
    "inline-flex items-center gap-1 rounded-full border px-2.5 py-1 text-[11px] font-medium transition";

  return (
    <div className="mt-2.5">
      <div className="flex flex-wrap items-center gap-1.5">
        <span className="mr-1 text-[11px] text-stone-400 dark:text-zinc-500">
          Accuracy review:
        </span>
        <button
          type="button"
          onClick={() => setSelected("accurate")}
          className={`${buttonBase} ${
            selected === "accurate"
              ? "border-emerald-600 bg-emerald-50 text-emerald-800 dark:border-emerald-500 dark:bg-emerald-950/60 dark:text-emerald-300"
              : "border-stone-200 text-stone-600 hover:border-stone-300 hover:text-stone-900 dark:border-zinc-800 dark:text-zinc-400 dark:hover:border-zinc-700 dark:hover:text-zinc-200"
          }`}
        >
          <ThumbsUp className="h-3 w-3" />
          Accurate
        </button>
        <button
          type="button"
          onClick={() => setSelected("needs_review")}
          className={`${buttonBase} ${
            selected === "needs_review"
              ? "border-amber-600 bg-amber-50 text-amber-800 dark:border-amber-500 dark:bg-amber-950/60 dark:text-amber-300"
              : "border-stone-200 text-stone-600 hover:border-stone-300 hover:text-stone-900 dark:border-zinc-800 dark:text-zinc-400 dark:hover:border-zinc-700 dark:hover:text-zinc-200"
          }`}
        >
          <Flag className="h-3 w-3" />
          Needs review
        </button>
        <button
          type="button"
          onClick={() => setSelected("inaccurate")}
          className={`${buttonBase} ${
            selected === "inaccurate"
              ? "border-red-600 bg-red-50 text-red-800 dark:border-red-500 dark:bg-red-950/60 dark:text-red-300"
              : "border-stone-200 text-stone-600 hover:border-stone-300 hover:text-stone-900 dark:border-zinc-800 dark:text-zinc-400 dark:hover:border-zinc-700 dark:hover:text-zinc-200"
          }`}
        >
          <ThumbsDown className="h-3 w-3" />
          Inaccurate
        </button>
      </div>

      {selected && (
        <form
          onSubmit={(e) => {
            e.preventDefault();
            submit(selected, comment);
          }}
          className="mt-2 flex flex-col gap-2 sm:flex-row"
        >
          <input
            type="text"
            value={comment}
            onChange={(e) => setComment(e.target.value)}
            maxLength={500}
            placeholder="Optional note (e.g. missing citation, wording nuance)…"
            className="flex-1 rounded-lg border border-stone-200 bg-white px-3 py-1.5 text-xs text-stone-800 placeholder:text-stone-400 focus:border-amber-700 focus:outline-none dark:border-zinc-800 dark:bg-zinc-950 dark:text-zinc-100 dark:placeholder:text-zinc-500"
          />
          <button
            type="submit"
            disabled={saving}
            className="rounded-lg bg-stone-900 px-3 py-1.5 text-xs font-medium text-white transition hover:bg-stone-800 disabled:opacity-50 dark:bg-zinc-100 dark:text-zinc-900 dark:hover:bg-white"
          >
            {saving ? "Saving…" : "Save note"}
          </button>
        </form>
      )}

      {error && (
        <p className="mt-1.5 text-[11px] text-red-600 dark:text-red-400">{error}</p>
      )}
    </div>
  );
}
