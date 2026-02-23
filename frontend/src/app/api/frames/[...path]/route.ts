import { NextRequest, NextResponse } from "next/server";
import { readFile } from "fs/promises";
import { join, extname } from "path";

const LOGS_DIR = join(process.cwd(), "..", "logs");

export async function GET(
  _request: NextRequest,
  { params }: { params: Promise<{ path: string[] }> },
) {
  const { path: segments } = await params;
  const relativePath = segments.join("/");

  // Security: no traversal
  if (relativePath.includes("..")) {
    return new NextResponse("Invalid path", { status: 400 });
  }

  const fullPath = join(LOGS_DIR, relativePath);

  // Ensure within logs dir
  if (!fullPath.startsWith(LOGS_DIR)) {
    return new NextResponse("Invalid path", { status: 403 });
  }

  try {
    const data = await readFile(fullPath);
    const ext = extname(fullPath).toLowerCase();
    const contentType =
      ext === ".jpg" || ext === ".jpeg"
        ? "image/jpeg"
        : ext === ".png"
          ? "image/png"
          : "application/octet-stream";

    return new NextResponse(data, {
      headers: {
        "Content-Type": contentType,
        "Cache-Control": "public, max-age=86400",
      },
    });
  } catch {
    return new NextResponse("Not found", { status: 404 });
  }
}
