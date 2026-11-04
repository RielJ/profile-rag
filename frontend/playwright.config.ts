import { defineConfig, devices } from "@playwright/test";

// End-to-end tests: `make e2e`. One-time setup: `make e2e-install`.
//
// Playwright starts two servers and stops them after the run:
//
// - the e2e API server (backend/tests/e2e_server.py): the real routes and
//   answer chain over a throwaway database filled with corpus-example/, fake
//   embeddings, and a fake Pi. It refuses to start without
//   PROFILE_RAG_TEST_MODE=1, which only this file sets.
// - Vite on its own port, proxying /api to that server.
//
// Neither reuses a server that is already running, so a test can never
// reach the development backend on port 8000 or a real model.

const API_PORT = 8765;
const WEB_PORT = 5174;

export default defineConfig({
  testDir: "./tests/e2e",
  fullyParallel: false,
  workers: 1,
  forbidOnly: true,
  retries: 0,
  reporter: [["list"]],
  use: {
    baseURL: `http://127.0.0.1:${WEB_PORT}`,
    trace: "off",
    screenshot: "off",
    video: "off",
  },
  outputDir: "node_modules/.playwright-results",
  projects: [
    { name: "desktop", use: { ...devices["Desktop Chrome"] } },
    {
      name: "phone",
      use: { ...devices["Desktop Chrome"], viewport: { width: 375, height: 740 }, hasTouch: true },
    },
  ],
  webServer: [
    {
      command: `uv --directory ../backend run python tests/e2e_server.py --port ${API_PORT}`,
      url: `http://127.0.0.1:${API_PORT}/api/health`,
      env: { PROFILE_RAG_TEST_MODE: "1" },
      reuseExistingServer: false,
      timeout: 60_000,
      // SIGTERM lets the server drop its throwaway database on the way out.
      gracefulShutdown: { signal: "SIGTERM", timeout: 10_000 },
      stdout: "ignore",
      stderr: "pipe",
    },
    {
      command: `pnpm exec vite --host 127.0.0.1 --port ${WEB_PORT} --strictPort`,
      url: `http://127.0.0.1:${WEB_PORT}`,
      env: { BACKEND_URL: `http://127.0.0.1:${API_PORT}` },
      reuseExistingServer: false,
      timeout: 60_000,
    },
  ],
});
