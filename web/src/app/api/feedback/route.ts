import { NextResponse } from "next/server";
import { isAuthenticated } from "@/lib/auth";
import { BackendError, kbFetch } from "@/lib/kb-client";

const VALID_RATINGS = new Set(["accurate", "needs_review", "inaccurate"]);
const VALID_MODES = new Set(["qa", "evidence"]);

export async function POST(request: Request) {
  if (!(await isAuthenticated())) {
    return NextResponse.json({ error: "Authentication required" }, { status: 401 });
  }

  let body: {
    question?: unknown;
    mode?: unknown;
    rating?: unknown;
    comment?: unknown;
    doc_ids?: unknown;
  };
  try {
    body = (await request.json()) as typeof body;
  } catch {
    return NextResponse.json({ error: "Invalid JSON body" }, { status: 400 });
  }

  const question = typeof body.question === "string" ? body.question.trim() : "";
  const mode = typeof body.mode === "string" && VALID_MODES.has(body.mode) ? body.mode : "qa";
  const rating = typeof body.rating === "string" ? body.rating : "";
  const comment = typeof body.comment === "string" ? body.comment.trim().slice(0, 2000) : "";
  const docIds = Array.isArray(body.doc_ids)
    ? body.doc_ids
        .filter((d): d is string => typeof d === "string" && d.trim().length > 0)
        .slice(0, 20)
    : [];

  if (!question || question.length > 512) {
    return NextResponse.json({ error: "Question is required" }, { status: 400 });
  }
  if (!VALID_RATINGS.has(rating)) {
    return NextResponse.json({ error: "Invalid rating" }, { status: 400 });
  }

  try {
    await kbFetch("/api/feedback", {
      method: "POST",
      body: { question, mode, rating, comment, doc_ids: docIds },
    });
    return NextResponse.json({ ok: true });
  } catch (err) {
    const status = err instanceof BackendError ? err.status : 503;
    const msg = err instanceof BackendError ? err.message : "Could not save review feedback";
    return NextResponse.json({ error: msg }, { status });
  }
}
