import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

// The backend runs on 127.0.0.1:8000 (`make backend`). During development,
// Vite forwards every request that starts with /api to it, so the browser
// only talks to one origin (127.0.0.1:5173) and CORS never comes up.
const BACKEND_URL = process.env.BACKEND_URL ?? "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [react()],
  server: {
    host: "127.0.0.1",
    port: 5173,
    strictPort: true,
    proxy: {
      "/api": BACKEND_URL,
    },
  },
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: ["./src/test-setup.ts"],
    include: ["src/**/*.test.{ts,tsx}"],
  },
});
