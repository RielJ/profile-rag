import { env, exports } from "cloudflare:workers";
import { describe, expect, it } from "vitest";

import { cacheKey, questionCacheKey } from "../src/cache";
import { WORKERS_AI_MODEL } from "../src/model";
import { resultEvent } from "../src/stream";

const BASE = "https://api.example";

const ask = (headers: Record<string, string> = {}): Promise<Response> =>
  exports.default.fetch(
    new Request(`${BASE}/api/ask/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json", ...headers },
      body: JSON.stringify({ question: "What does he build?" }),
    }),
  );

describe("GET /api/health", () => {
  it("answers ok without a key and sets a request ID", async () => {
    const response = await exports.default.fetch(`${BASE}/api/health`);
    expect(response.status).toBe(200);
    expect(await response.json()).toEqual({ status: "ok" });
    expect(response.headers.get("X-Request-ID")).toMatch(/^[0-9a-f]{32}$/);
  });
});

describe("POST /api/ask/stream access", () => {
  it("refuses a request without the access key", async () => {
    const response = await ask();
    expect(response.status).toBe(401);
    expect(await response.json()).toEqual({ detail: "missing or wrong access key" });
  });

  it("refuses a wrong key, including one that only shares a prefix", async () => {
    expect((await ask({ "X-Access-Key": "wrong" })).status).toBe(401);
    expect((await ask({ "X-Access-Key": "test-key-and-more" })).status).toBe(401);
  });

  it("lets the right key through", async () => {
    const response = await ask({ "X-Access-Key": "test-key" });
    expect(response.status).not.toBe(401);
  });

  it.each([[{ question: "   " }], [{}], ["not an object"]])(
    "answers 422 for the body %j",
    async (body) => {
      const response = await exports.default.fetch(
        new Request(`${BASE}/api/ask/stream`, {
          method: "POST",
          headers: { "Content-Type": "application/json", "X-Access-Key": "test-key" },
          body: JSON.stringify(body),
        }),
      );
      expect(response.status).toBe(422);
      expect(await response.json()).toEqual({ detail: expect.any(String) });
      expect(response.headers.get("X-Request-ID")).toMatch(/^[0-9a-f]{32}$/);
    },
  );

  it("rejects GET", async () => {
    const response = await exports.default.fetch(`${BASE}/api/ask/stream`);
    expect(response.status).toBe(405);
  });
});

it("answers 404 for an unknown path", async () => {
  const response = await exports.default.fetch(`${BASE}/nope`);
  expect(response.status).toBe(404);
});

it("keys a question by the deploy's cache version, corpus version and configured model", async () => {
  expect(await questionCacheKey("q", env)).toBe(
    await cacheKey("q", {
      cacheVersion: env.CACHE_VERSION,
      corpusVersion: env.CORPUS_VERSION,
      model: WORKERS_AI_MODEL,
    }),
  );
});

describe("POST /api/ask/stream cache and limits", () => {
  const post = (question: string, visitor: string): Promise<Response> =>
    exports.default.fetch(
      new Request(`${BASE}/api/ask/stream`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-Access-Key": "test-key",
          "X-Visitor-IP": visitor,
        },
        body: JSON.stringify({ question }),
      }),
    );

  it("serves a stored answer for the same question, however it is written", async () => {
    const stored = { status: "unknown", reason: "not covered", model: WORKERS_AI_MODEL } as const;
    const key = await questionCacheKey("Is he open to remote work?", env);
    await env.ANSWERS.put(key, JSON.stringify(stored));

    const response = await post("  is HE open to remote work  ", "203.0.113.1");
    expect(response.status).toBe(200);
    expect(await response.text()).toBe(resultEvent({ ...stored, cached: true }));
  });

  it("refuses a visitor's fresh questions over the hourly limit, but not another visitor's", async () => {
    const perHour = Number(env.QUESTIONS_PER_VISITOR_PER_HOUR);
    // Tests have no database, so each fresh question fails after it is counted.
    for (let asked = 0; asked < perHour; asked++) {
      expect((await post(`fresh question ${asked}`, "203.0.113.2")).status).toBe(503);
    }
    const refused = await post("one more", "203.0.113.2");
    expect(refused.status).toBe(429);
    expect(await refused.json()).toMatchObject({ status: "failed", error: "rate_limited" });
    expect((await post("one more", "203.0.113.3")).status).toBe(503);
  });
});
