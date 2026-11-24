const DEFAULT_TIMEOUT_MS = 25_000;

export class BackendError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

function runtimeEnv(name: string): string {
  // Computed access so Next.js does not inline the value at build time.
  // Vercel service bindings are injected only when the function runs.
  return (process.env[name] ?? "").trim();
}

function getBackendBaseUrl(): string {
  const bound = runtimeEnv("KB_BACKEND_URL");
  const raw = bound || (process.env.VERCEL ? "" : "http://127.0.0.1:8080");
  if (!raw) {
    throw new BackendError(
      503,
      "The retrieval service address was not provided to this deployment."
    );
  }
  return raw.endsWith("/") ? raw : `${raw}/`;
}

function getBackendHeaders(extra?: Record<string, string>): Record<string, string> {
  const headers: Record<string, string> = {
    Accept: "application/json",
    ...extra,
  };
  const token = runtimeEnv("KB_BACKEND_TOKEN");
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
  const relativePath = path.replace(/^\/+/, "");
  const url = new URL(relativePath, getBackendBaseUrl()).toString();

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
