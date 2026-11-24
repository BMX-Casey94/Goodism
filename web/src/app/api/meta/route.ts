import { NextResponse } from "next/server";
import { isAuthenticated } from "@/lib/auth";
import { BackendError, kbFetch } from "@/lib/kb-client";
import type { ConflictRecord, DocumentSummary, IndexStatus, MetaPayload } from "@/lib/types";

export async function GET() {
  if (!(await isAuthenticated())) {
    return NextResponse.json({ error: "Authentication required" }, { status: 401 });
  }

  const model =
    (process.env.OPENROUTER_MODEL || "google/gemini-3.1-flash-lite-preview").trim();
  const llmConfigured = Boolean((process.env.OPENROUTER_API_KEY || "").trim());

  const [statusRes, docsRes, conflictsRes] = await Promise.allSettled([
    kbFetch<IndexStatus>("/api/status", { timeoutMs: 20_000 }),
    kbFetch<{ documents: DocumentSummary[] }>("/api/documents", { timeoutMs: 20_000 }),
    kbFetch<{ conflicts: ConflictRecord[] }>("/api/conflicts", { timeoutMs: 20_000 }),
  ]);

  const backendReachable = statusRes.status === "fulfilled";
  let backendError: string | null = null;
  if (!backendReachable) {
    const reason = statusRes.reason;
    backendError =
      reason instanceof BackendError
        ? reason.message
        : "The retrieval service did not respond.";
  }
  const payload: MetaPayload = {
    status: statusRes.status === "fulfilled" ? statusRes.value : null,
    documents: docsRes.status === "fulfilled" ? docsRes.value.documents ?? [] : [],
    conflicts:
      conflictsRes.status === "fulfilled" ? conflictsRes.value.conflicts ?? [] : [],
    llmConfigured,
    model,
    backendReachable,
    backendError,
  };

  return NextResponse.json(payload, {
    headers: { "Cache-Control": "no-store" },
  });
}
