/** @type {import('next').NextConfig} */
const nextConfig = {
  // Keep TypeScript type-checking on during build, but don't require an ESLint
  // config to be present (avoids an interactive prompt on a fresh checkout).
  eslint: { ignoreDuringBuilds: true },
};
export default nextConfig;
