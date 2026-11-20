import { describe, expect, it } from "vitest";

import {
  askStream,
  QUESTION_MAX_LENGTH,
  readQuestion,
  resultOf,
  type AskDeps,
  type QuestionCache,
} from "../src/ask";
import type { Chunk } from "../src/chunk";
import { failed, type AskResponse } from "../src/contract";
import { ModelFailed, type StartModel } from "../src/model";
import type { Message } from "../src/prompt";
import { resultEvent } from "../src/stream";
import { streamOf } from "./support/streams";

const chunk = (chunkId: string): Chunk => ({
  chunkId,
  sourceId: "resume",
  title: "Resume",
  attribution: "self_report",
  url: null,
  headingPath: ["Resume", "Soulbound"],
  text: "Soulbound\n\nBuilt the minting service.",
});
const CHUNKS = [chunk("resume:1"), chunk("resume:2")];

const sse = (content: string): string =>
  `data: ${JSON.stringify({ choices: [{ delta: { content } }] })}\n\ndata: [DONE]\n\n`;

const replying =
  (content: string, seen: Message[][] = []): StartModel =>
  async (messages) => {
    seen.push(messages);
    return { model: "test-model", body: streamOf(sse(content)) };
  };

/** Runs askStream with fakes; returns the response, its body, and every log line. */
async function ask(question: string, deps: Partial<AskDeps> = {}) {
  const pending: Promise<unknown>[] = [];
  const logs: Record<string, unknown>[] = [];
  const response = await askStream(question, {
    requestId: "req1",
    retrieve: async () => CHUNKS,
    startModel: replying('{"status":"unknown","reason":"not covered"}'),
    waitUntil: (promise) => pending.push(promise),
    log: (event) => logs.push(event),
    ...deps,
  });
  const text = await response.text();
  await Promise.all(pending);
  return { response, text, logs };
}

const resultIn = (text: string): unknown =>
  JSON.parse(/event: result\ndata: (.*)\n\n$/.exec(text)?.[1] ?? "null");

describe("readQuestion", () => {
  it("accepts a question", () => {
    expect(readQuestion({ question: "What did he build?" })).toEqual({
      question: "What did he build?",
    });
  });

  it.each([
    [null],
    ["What did he build?"],
    [{}],
    [{ question: 3 }],
    [{ question: "  \n" }],
    [{ question: "x".repeat(QUESTION_MAX_LENGTH + 1) }],
  ])("refuses %j", (body) => {
    expect(readQuestion(body)).toHaveProperty("problem");
  });

  it("counts characters, not UTF-16 units, as Python does", () => {
    expect(readQuestion({ question: "😀".repeat(QUESTION_MAX_LENGTH) })).toHaveProperty("question");
  });
});

describe("resultOf", () => {
  it("builds the cited response for a valid answer", () => {
    const result = resultOf(
      '{"status":"answered","answer":"He built the minting service.","citations":["resume:2"]}',
      CHUNKS,
      "test-model",
      "req1",
    );
    expect(result).toMatchObject({
      status: "answered",
      answer: "He built the minting service.",
      model: "test-model",
      citations: [{ chunk_id: "resume:2", source_id: "resume", section: "Soulbound" }],
    });
  });

  it("fails with invented_citation for a chunk that was not retrieved", () => {
    const reply = '{"status":"answered","answer":"Yes.","citations":["resume:9"]}';
    expect(resultOf(reply, CHUNKS, "m", "req1")).toEqual(failed("invented_citation", "req1"));
  });

  it.each([[""], ["I think he built things."], ['{"status":"answered","answer":"Yes."}']])(
    "fails with bad_reply for %j",
    (reply) => {
      expect(resultOf(reply, CHUNKS, "m", "req1")).toEqual(failed("bad_reply", "req1"));
    },
  );
});

describe("askStream", () => {
  it("streams the model's events and then the checked result", async () => {
    const seen: Message[][] = [];
    const reply =
      '{"status":"answered","answer":"He built the minting service.","citations":["resume:1"]}';
    const { response, text, logs } = await ask("What did he build?", {
      startModel: replying(reply, seen),
    });
    expect(response.status).toBe(200);
    expect(response.headers.get("Content-Type")).toBe("text/event-stream");
    expect(response.headers.get("X-Request-ID")).toBe("req1");
    expect(text.startsWith(sse(reply))).toBe(true);
    expect(resultIn(text)).toMatchObject({ status: "answered", model: "test-model" });
    expect(seen[0]?.[1]?.content).toContain("What did he build?");
    expect(seen[0]?.[1]?.content).toContain("resume:1");
    expect(logs).toContainEqual(
      expect.objectContaining({ event: "answer", status: "answered", model: "test-model" }),
    );
  });

  it("answers 503 retrieval_failed when search fails, without calling the model", async () => {
    const seen: Message[][] = [];
    const { response, text } = await ask("q", {
      retrieve: async () => {
        throw new Error("Neon is suspended");
      },
      startModel: replying("{}", seen),
    });
    expect(response.status).toBe(503);
    expect(JSON.parse(text)).toEqual(failed("retrieval_failed", "req1"));
    expect(seen).toEqual([]);
  });

  it("answers 502 pi_failed when no model starts", async () => {
    const { response, text, logs } = await ask("q", {
      startModel: async () => {
        throw new ModelFailed("Groq answered HTTP 429");
      },
    });
    expect(response.status).toBe(502);
    expect(JSON.parse(text)).toEqual(failed("pi_failed", "req1"));
    expect(logs).toContainEqual(
      expect.objectContaining({ cause: "ModelFailed: Groq answered HTTP 429" }),
    );
  });

  it("answers 504 pi_timeout when the model times out before streaming", async () => {
    const controller = new AbortController();
    const { response, text } = await ask("q", {
      signal: controller.signal,
      startModel: async () => {
        controller.abort();
        throw new Error("aborted");
      },
    });
    expect(response.status).toBe(504);
    expect(JSON.parse(text)).toEqual(failed("pi_timeout", "req1"));
  });

  it("ends a stream with bad_reply when the model's reply is not the JSON answer", async () => {
    const { response, text } = await ask("q", { startModel: replying("Sure! He is great.") });
    expect(response.status).toBe(200);
    expect(resultIn(text)).toEqual(failed("bad_reply", "req1"));
  });

  it("logs the start of a stream that holds an error instead of the answer", async () => {
    const error = 'data: {"errors":[{"message":"daily free allocation used up","code":4006}]}\n\n';
    const { text, logs } = await ask("q", {
      startModel: async () => ({ model: "test-model", body: streamOf(error) }),
    });
    expect(resultIn(text)).toEqual(failed("bad_reply", "req1"));
    const answer = logs.find((line) => line["event"] === "answer");
    expect(answer?.["events_head"]).toContain("4006");
  });

  it("never logs a model stream that has choices, which can echo the question", async () => {
    const { logs } = await ask("q", { startModel: replying("Sure! He is great.") });
    const answer = logs.find((line) => line["event"] === "answer");
    expect(answer?.["error"]).toBe("bad_reply");
    expect(answer).not.toHaveProperty("events_head");
  });
});

describe("askStream with the cache and the limits", () => {
  const UNKNOWN = { status: "unknown", reason: "not covered", model: "test-model" } as const;

  /** A cache in memory that records what was written. */
  const memoryCache = (stored: AskResponse | null = null) => {
    const writes: AskResponse[] = [];
    const cache: QuestionCache = {
      read: async () => (stored?.status === "failed" ? null : stored),
      write: async (response) => {
        writes.push(response);
      },
    };
    return { cache, writes };
  };

  it("answers a cache hit with the result event alone, before the limits and the model", async () => {
    const { cache, writes } = memoryCache(UNKNOWN);
    const calls: string[] = [];
    const { response, text } = await ask("q", {
      cache,
      takeQuestion: async () => {
        calls.push("limits");
        return { ok: true };
      },
      retrieve: async () => {
        calls.push("retrieve");
        return CHUNKS;
      },
    });
    expect(response.status).toBe(200);
    expect(response.headers.get("Content-Type")).toBe("text/event-stream");
    expect(text).toBe(resultEvent({ ...UNKNOWN, cached: true }));
    expect(calls).toEqual([]);
    expect(writes).toEqual([]);
  });

  it("stores the result of a fresh answer", async () => {
    const { cache, writes } = memoryCache();
    const { text } = await ask("q", { cache, takeQuestion: async () => ({ ok: true }) });
    expect(resultIn(text)).toEqual(UNKNOWN);
    expect(writes).toEqual([UNKNOWN]);
  });

  it("passes a failed result to the cache, which refuses to store it", async () => {
    // writeCached drops failures (test/cache.test.ts); askStream does not filter.
    const { cache, writes } = memoryCache();
    await ask("q", { cache, startModel: replying("not json") });
    expect(writes).toEqual([failed("bad_reply", "req1")]);
  });

  it("refuses a fresh question over a limit with 429 rate_limited, without searching", async () => {
    const calls: string[] = [];
    const { response, text } = await ask("q", {
      takeQuestion: async () => ({ ok: false, limit: "visitor" }),
      retrieve: async () => {
        calls.push("retrieve");
        return CHUNKS;
      },
    });
    expect(response.status).toBe(429);
    expect(JSON.parse(text)).toEqual(failed("rate_limited", "req1"));
    expect(calls).toEqual([]);
  });

  it("fails closed with 500 internal_error when the limiter fails", async () => {
    const calls: string[] = [];
    const { response, text } = await ask("q", {
      takeQuestion: async () => {
        throw new Error("Durable Object reset");
      },
      retrieve: async () => {
        calls.push("retrieve");
        return CHUNKS;
      },
    });
    expect(response.status).toBe(500);
    expect(JSON.parse(text)).toEqual(failed("internal_error", "req1"));
    expect(calls).toEqual([]);
  });
});
