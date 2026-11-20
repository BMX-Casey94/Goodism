import { NextResponse } from "next/server";
import { isAuthenticated } from "@/lib/auth";
import { BackendError, kbFetch } from "@/lib/kb-client";
import type { NeighbourChunk } from "@/lib/types";

export async function POST(request: Request) {
  if (!(await isAuthenticated())) {
    return NextResponse.json({ error: "Authentication required" }, { status: 401 });
  }

  let body: { doc_id?: unknown; chunk?: unknown; before?: unknown; after?: unknown };
  try {
    body = (await request.json()) as typeof body;
  } catch {
    return NextResponse.json({ error: "Invalid JSON body" }, { status: 400 });
  }

  const docId = typeof body.doc_id === "string" ? body.doc_id.trim() : "";
  const chunk = typeof body.chunk === "number" ? Math.trunc(body.chunk) : -1;
  const before = typeof body.before === "number" ? Math.max(0, Math.min(Math.trunc(body.before), 10)) : 2;
  const after = typeof body.after === "number" ? Math.max(0, Math.min(Math.trunc(body.after), 10)) : 2;

  if (!docId || chunk < 0 || chunk > 1_000_000) {
    return NextResponse.json({ error: "Invalid doc_id or chunk" }, { status: 400 });
  }

  try {
    const result = await kbFetch<{ doc_id: string; chunks: NeighbourChunk[] }>(
      "/api/read-around",
      {
        method: "POST",
        body: { doc_id: docId, chunk, before, after },
      }
    );
    return NextResponse.json(result, { headers: { "Cache-Control": "no-store" } });
  } catch (err) {
    const status = err instanceof BackendError ? err.status : 503;
    const msg = err instanceof BackendError ? err.message : "Failed to load surrounding context";
    return NextResponse.json({ error: msg }, { status });
  }
}
