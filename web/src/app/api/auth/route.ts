import { NextResponse } from "next/server";
import {
  createSessionToken,
  isAuthenticated,
  isAuthRequired,
  SESSION_COOKIE_NAME,
  verifyPasscode,
} from "@/lib/auth";

export async function GET() {
  const required = isAuthRequired();
  const authenticated = await isAuthenticated();
  return NextResponse.json(
    { required, authenticated },
    { headers: { "Cache-Control": "no-store" } }
  );
}

export async function POST(request: Request) {
  if (!isAuthRequired()) {
    return NextResponse.json({ ok: true });
  }

  let body: { passcode?: unknown };
  try {
    body = (await request.json()) as { passcode?: unknown };
  } catch {
    return NextResponse.json({ error: "Invalid request body" }, { status: 400 });
  }

  const passcode = typeof body.passcode === "string" ? body.passcode : "";
  if (!passcode.trim() || passcode.length > 256) {
    return NextResponse.json({ error: "Passcode is required" }, { status: 400 });
  }

  if (!verifyPasscode(passcode)) {
    return NextResponse.json({ error: "Incorrect passcode" }, { status: 401 });
  }

  const response = NextResponse.json({ ok: true });
  response.cookies.set({
    name: SESSION_COOKIE_NAME,
    value: createSessionToken(),
    httpOnly: true,
    sameSite: "strict",
    secure: process.env.NODE_ENV === "production",
    path: "/",
    maxAge: 60 * 60 * 12,
  });
  return response;
}

export async function DELETE() {
  const response = NextResponse.json({ ok: true });
  response.cookies.set({
    name: SESSION_COOKIE_NAME,
    value: "",
    httpOnly: true,
    sameSite: "strict",
    secure: process.env.NODE_ENV === "production",
    path: "/",
    maxAge: 0,
  });
  return response;
}
