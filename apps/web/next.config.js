/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  output: "standalone",
  transpilePackages: ["@contextvault/shared"],
  experimental: {
    typedRoutes: false,
  },
};

module.exports = nextConfig;