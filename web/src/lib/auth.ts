import crypto from "crypto";
import { cookies } from "next/headers";

export const SESSION_COOKIE_NAME = "kb_client_session";
const SESSION_TTL_MS = 12 * 60 * 60 * 1000; // 12 hours

export function isAuthRequired(): boolean {
  return Boolean((process.env.APP_ACCESS_PASSCODE || "").trim());
}

function getSigningSecret(): string {
  const secret = (process.env.APP_SESSION_SECRET || "").trim();
  if (secret.length >= 32) return secret;
  // Derive a deterministic fallback from the passcode if APP_SESSION_SECRET is not set.
  const passcode = (process.env.APP_ACCESS_PASSCODE || "").trim();
  return crypto
    .createHash("sha256")
    .update(`goodism-kb-session:${passcode}`)
    .digest("hex");
}

export function constantTimeEqual(a: string, b: string): boolean {
  const bufA = Buffer.from(a, "utf8");
  const bufB = Buffer.from(b, "utf8");
  if (bufA.length !== bufB.length) {
    // Compare against self to avoid early length timing leak on equal-cost path.
    crypto.timingSafeEqual(bufA, bufA);
    return false;
  }
  return crypto.timingSafeEqual(bufA, bufB);
}

export function verifyPasscode(candidate: string): boolean {
  const expected = (process.env.APP_ACCESS_PASSCODE || "").trim();
  if (!expected) return true;
  return constantTimeEqual(candidate.trim(), expected);
}

export function createSessionToken(): string {
  const expiresAt = Date.now() + SESSION_TTL_MS;
  const payload = String(expiresAt);
  const sig = crypto
    .createHmac("sha256", getSigningSecret())
    .update(payload)
    .digest("hex");
  return `${payload}.${sig}`;
}

export function verifySessionToken(token: string | undefined): boolean {
  if (!token) return false;
  const parts = token.split(".");
  if (parts.length !== 2) return false;
  const [payload, sig] = parts;
  const expiresAt = Number(payload);
  if (!Number.isFinite(expiresAt) || expiresAt < Date.now()) return false;
  const expectedSig = crypto
    .createHmac("sha256", getSigningSecret())
    .update(payload)
    .digest("hex");
  return constantTimeEqual(sig, expectedSig);
}

export async function isAuthenticated(): Promise<boolean> {
  if (!isAuthRequired()) return true;
  const store = await cookies();
  const token = store.get(SESSION_COOKIE_NAME)?.value;
  return verifySessionToken(token);
}
