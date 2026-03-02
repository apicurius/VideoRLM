import { NextRequest, NextResponse } from "next/server";

const OPENROUTER_BASE = "https://openrouter.ai/api/v1";

/**
 * POST /api/openrouter
 *
 * Proxy for OpenRouter chat completions. Accepts the same body as the
 * OpenAI chat completions API. API key is read from the x-api-key header
 * or the OPENROUTER_API_KEY env variable.
 */
export async function POST(request: NextRequest) {
  const apiKey =
    request.headers.get("x-api-key") ||
    process.env.OPENROUTER_API_KEY ||
    "";

  if (!apiKey) {
    return NextResponse.json(
      {
        error:
          "No OpenRouter API key provided. Pass it via x-api-key header or set OPENROUTER_API_KEY env variable.",
      },
      { status: 401 },
    );
  }

  let body: unknown;
  try {
    body = await request.json();
  } catch {
    return NextResponse.json({ error: "Invalid JSON body" }, { status: 400 });
  }

  const res = await fetch(`${OPENROUTER_BASE}/chat/completions`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${apiKey}`,
      "Content-Type": "application/json",
      "HTTP-Referer": request.headers.get("referer") || "",
      "X-Title": "VideoRLM",
    },
    body: JSON.stringify(body),
  });

  if (!res.ok) {
    const text = await res.text();
    return NextResponse.json(
      { error: `OpenRouter error: ${text}` },
      { status: res.status },
    );
  }

  // Stream the response directly if the caller requests streaming
  const contentType = res.headers.get("content-type") || "application/json";
  if (contentType.includes("text/event-stream")) {
    return new NextResponse(res.body, {
      headers: {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        Connection: "keep-alive",
      },
    });
  }

  const data = await res.json();
  return NextResponse.json(data);
}
