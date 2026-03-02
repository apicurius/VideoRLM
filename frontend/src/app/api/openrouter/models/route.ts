import { NextRequest, NextResponse } from "next/server";

const OPENROUTER_BASE = "https://openrouter.ai/api/v1";

export async function GET(request: NextRequest) {
  const apiKey =
    request.headers.get("x-api-key") ||
    process.env.OPENROUTER_API_KEY ||
    "";

  if (!apiKey) {
    return NextResponse.json(
      { error: "No OpenRouter API key provided. Pass it via x-api-key header or set OPENROUTER_API_KEY env variable." },
      { status: 401 },
    );
  }

  const res = await fetch(`${OPENROUTER_BASE}/models`, {
    headers: {
      Authorization: `Bearer ${apiKey}`,
      "Content-Type": "application/json",
    },
  });

  if (!res.ok) {
    const text = await res.text();
    return NextResponse.json(
      { error: `OpenRouter error: ${text}` },
      { status: res.status },
    );
  }

  const data = await res.json();
  return NextResponse.json(data);
}
