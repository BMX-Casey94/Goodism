const DEFAULT_TIMEOUT_MS = 25_000;

export class BackendError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

function getBackendUrl(): string {
  const raw = (process.env.KB_BACKEND_URL || "http://127.0.0.1:8080").trim();
  return raw.replace(/\/+$/, "");
}

function getBackendHeaders(extra?: Record<string, string>): Record<string, string> {
  const headers: Record<string, string> = {
    Accept: "application/json",
    ...extra,
  };
  const token = (process.env.KB_BACKEND_TOKEN || "").trim();
  if (token) {
    headers["Authorization"] = `Bearer ${token}`;
  }
  return headers;
}

export async function kbFetch<T>(
  path: string,
  options?: {
    method?: "GET" | "POST";
    body?: unknown;
    timeoutMs?: number;
  }
): Promise<T> {
  const controller = new AbortController();
  const timer = setTimeout(
    () => controller.abort(),
    options?.timeoutMs ?? DEFAULT_TIMEOUT_MS
  );
  const method = options?.method ?? "GET";
  const url = `${getBackendUrl()}${path}`;

  try {
    const res = await fetch(url, {
      method,
      headers: getBackendHeaders(
        options?.body !== undefined ? { "Content-Type": "application/json" } : undefined
      ),
      body: options?.body !== undefined ? JSON.stringify(options.body) : undefined,
      signal: controller.signal,
      cache: "no-store",
    });

    const text = await res.text();
    let parsed: Record<string, unknown> = {};
    try {
      parsed = text ? (JSON.parse(text) as Record<string, unknown>) : {};
    } catch {
      parsed = {};
    }

    if (!res.ok) {
      const msg =
        typeof parsed.error === "string"
          ? parsed.error
          : `Knowledge base backend returned HTTP ${res.status}`;
      throw new BackendError(res.status, msg);
    }

    return parsed as T;
  } catch (err) {
    if (err instanceof BackendError) throw err;
    if (err instanceof Error && err.name === "AbortError") {
      throw new BackendError(504, "Knowledge base backend timed out");
    }
    throw new BackendError(503, "Knowledge base backend is currently unreachable");
  } finally {
    clearTimeout(timer);
  }
}
