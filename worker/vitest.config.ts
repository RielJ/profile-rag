import { cloudflareTest } from "@cloudflare/vitest-plugin";
import { defineConfig } from "vitest/config";

export default defineConfig({
  plugins: [
    cloudflareTest({
      // Tests never call Workers AI: it costs neurons, and the remote
      // connection keeps Vitest from exiting. Tests fake the AI binding.
      remoteBindings: false,
      wrangler: { configPath: "./wrangler.jsonc" },
      miniflare: { bindings: { ACCESS_KEY: "test-key" } },
    }),
  ],
});
