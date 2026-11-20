import { NextResponse } from "next/server";
import { isAuthenticated } from "@/lib/auth";
import { kbFetch } from "@/lib/kb-client";
import type { ConflictRecord, DocumentSummary, IndexStatus, MetaPayload } from "@/lib/types";

export async function GET() {
  if (!(await isAuthenticated())) {
    return NextResponse.json({ error: "Authentication required" }, { status: 401 });
  }

  const model =
    (process.env.OPENROUTER_MODEL || "google/gemini-3.1-flash-lite-preview").trim();
  const llmConfigured = Boolean((process.env.OPENROUTER_API_KEY || "").trim());

  const [statusRes, docsRes, conflictsRes] = await Promise.allSettled([
    kbFetch<IndexStatus>("/api/status", { timeoutMs: 8_000 }),
    kbFetch<{ documents: DocumentSummary[] }>("/api/documents", { timeoutMs: 8_000 }),
    kbFetch<{ conflicts: ConflictRecord[] }>("/api/conflicts", { timeoutMs: 8_000 }),
  ]);

  const backendReachable = statusRes.status === "fulfilled";
  const payload: MetaPayload = {
    status: statusRes.status === "fulfilled" ? statusRes.value : null,
    documents: docsRes.status === "fulfilled" ? docsRes.value.documents ?? [] : [],
    conflicts:
      conflictsRes.status === "fulfilled" ? conflictsRes.value.conflicts ?? [] : [],
    llmConfigured,
    model,
    backendReachable,
  };

  return NextResponse.json(payload, {
    headers: { "Cache-Control": "no-store" },
  });
}
