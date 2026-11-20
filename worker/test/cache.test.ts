import { env } from "cloudflare:workers";
import { describe, expect, it, vi } from "vitest";

import {
  CACHE_TTL_SECONDS,
  cacheKey,
  normalizeQuestion,
  readCached,
  writeCached,
  type CacheKeyParts,
} from "../src/cache";
import { failed, type AnsweredResponse, type UnknownResponse } from "../src/contract";

const PARTS: CacheKeyParts = { cacheVersion: "1", corpusVersion: "c1", model: "m1" };

const ANSWERED: AnsweredResponse = {
  status: "answered",
  answer: "He works remotely from UTC+8.",
  citations: [
    {
      chunk_id: "faq:v2:a",
      source_id: "faq",
      source_title: "Recruiter FAQ",
      section: "Availability",
      url: null,
      attribution: "self_report",
      excerpt: "Remote, UTC+8.",
    },
  ],
  model: "@cf/openai/gpt-oss-120b",
};

const UNKNOWN: UnknownResponse = {
  status: "unknown",
  reason: "No salary in the evidence.",
  model: "m",
};

describe("normalizeQuestion", () => {
  it.each([
    ["Is he open to remote work?", "is he open to remote work"],
    ["  is   he open\tto remote WORK  ", "is he open to remote work"],
    ['"Is he open to remote work?"', "is he open to remote work"],
    ["“Remote?!”...", "remote"],
    ["Ｒｅｍｏｔｅ？", "remote"], // fullwidth forms, NFKC
    ["What is Node.js?", "what is node.js"], // inner dots stay
    ["Which roles? Remote?", "which roles? remote"], // only trailing marks go
  ])("%j -> %j", (question, normalized) => {
    expect(normalizeQuestion(question)).toBe(normalized);
  });
});

describe("cacheKey", () => {
  it("gives equal questions after normalizing the same key", async () => {
    expect(await cacheKey("Is he remote?", PARTS)).toBe(await cacheKey("  is he REMOTE ", PARTS));
  });

  it("is a fixed-length, opaque key", async () => {
    expect(await cacheKey("Is he remote?", PARTS)).toMatch(/^answer:[0-9a-f]{64}$/u);
  });

  it.each<[string, Partial<CacheKeyParts>]>([
    ["cache version", { cacheVersion: "2" }],
    ["corpus version", { corpusVersion: "c2" }],
    ["model", { model: "m2" }],
  ])("changes with the %s", async (_, change) => {
    expect(await cacheKey("Is he remote?", { ...PARTS, ...change })).not.toBe(
      await cacheKey("Is he remote?", PARTS),
    );
  });

  it("keeps different questions apart", async () => {
    expect(await cacheKey("Did he work at Acme?", PARTS)).not.toBe(
      await cacheKey("Did he work at Initech?", PARTS),
    );
  });

  it("cannot be confused by a separator inside the question", async () => {
    const a = await cacheKey("x", { cacheVersion: "1", corpusVersion: "c1|m1", model: "m1" });
    const b = await cacheKey("m1|x", PARTS);
    expect(a).not.toBe(b);
  });
});

describe("readCached and writeCached", () => {
  it("round-trips an answered and an unknown response", async () => {
    await writeCached(env.ANSWERS, "k-answered", ANSWERED);
    await writeCached(env.ANSWERS, "k-unknown", UNKNOWN);
    expect(await readCached(env.ANSWERS, "k-answered")).toEqual(ANSWERED);
    expect(await readCached(env.ANSWERS, "k-unknown")).toEqual(UNKNOWN);
  });

  it("never caches a failure", async () => {
    await writeCached(env.ANSWERS, "k-failed", failed("pi_failed", "r1"));
    expect(await env.ANSWERS.get("k-failed")).toBeNull();
  });

  it("stores entries with the TTL", async () => {
    const kv = { put: vi.fn(async () => undefined) };
    await writeCached(kv as unknown as KVNamespace, "k", UNKNOWN);
    expect(kv.put).toHaveBeenCalledWith("k", JSON.stringify(UNKNOWN), {
      expirationTtl: CACHE_TTL_SECONDS,
    });
  });

  it("treats a missing key as a miss", async () => {
    expect(await readCached(env.ANSWERS, "k-missing")).toBeNull();
  });

  it.each([
    ["not JSON", "{"],
    ["a failure", JSON.stringify(failed("pi_failed", "r1"))],
    ["no citations", JSON.stringify({ ...ANSWERED, citations: [] })],
    ["a bad citation", JSON.stringify({ ...ANSWERED, citations: [{ chunk_id: "a" }] })],
    [
      "a bad attribution",
      JSON.stringify({
        ...ANSWERED,
        citations: [{ ...ANSWERED.citations[0], attribution: "rumour" }],
      }),
    ],
  ])("treats an entry with %s as a miss", async (_, stored) => {
    await env.ANSWERS.put("k-bad", stored);
    expect(await readCached(env.ANSWERS, "k-bad")).toBeNull();
  });

  it("treats a KV read error as a miss and a write error as a no-op", async () => {
    const broken = {
      get: async () => {
        throw new Error("KV GET failed: 429 Too Many Requests");
      },
      put: async () => {
        throw new Error("KV PUT failed: 429 Too Many Requests");
      },
    } as unknown as KVNamespace;
    const warn = vi.spyOn(console, "warn").mockImplementation(() => undefined);
    expect(await readCached(broken, "k")).toBeNull();
    await expect(writeCached(broken, "k", UNKNOWN)).resolves.toBeUndefined();
    expect(warn).toHaveBeenCalledTimes(2);
    warn.mockRestore();
  });
});
