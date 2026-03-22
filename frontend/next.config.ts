import type { NextConfig } from "next";

const BACKEND_URL = process.env.BACKEND_URL || "http://localhost:8000";

const nextConfig: NextConfig = {
  experimental: {
    serverActions: {
      bodySizeLimit: "500mb",
    },
  },

  async rewrites() {
    return [
      {
        // Proxy /backend/* → FastAPI server at localhost:7860
        // /api/logs, /api/openrouter, /api/frames are handled by local Next.js route handlers.
        source: "/backend/:path*",
        destination: `${BACKEND_URL}/:path*`,
      },
    ];
  },
};

export default nextConfig;
