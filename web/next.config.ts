import type { NextConfig } from "next";

const API_ORIGIN = process.env.DUMBOCRACY_API ?? "http://127.0.0.1:61008";

const nextConfig: NextConfig = {
  reactStrictMode: true,
  poweredByHeader: false,
  async rewrites() {
    return [{ source: "/api/:path*", destination: `${API_ORIGIN}/api/:path*` }];
  },
};

export default nextConfig;
