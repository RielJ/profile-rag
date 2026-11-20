import { describe, expect, it } from "vitest";

import { failed, type AskResponse } from "../src/contract";
import { forwardWithResult, resultEvent } from "../src/stream";
import workersAiStream from "./fixtures/workers-ai-gpt-oss.sse.txt";
import { streamOf } from "./support/streams";

const UNKNOWN: AskResponse = { status: "unknown", reason: "not covered", model: "m" };

/** Runs a forward and returns what the visitor receives, with every waitUntil settled. */
async function forward(
  body: ReadableStream<Uint8Array>,
  options: {
    signal?: AbortSignal;
    finish?: (events: string) => AskResponse;
    graceMs?: number;
    stuckMs?: number;
  } = {},
): Promise<{ text: string; seen: string[]; errors: unknown[] }> {
  const pending: Promise<unknown>[] = [];
  const seen: string[] = [];
  const errors: unknown[] = [];
  const out = forwardWithResult({
    body,
    signal: options.signal ?? new AbortController().signal,
    finish:
      options.finish ??
      ((events) => {
        seen.push(events);
        return UNKNOWN;
      }),
    fail: (error) => {
      errors.push(error);
      return failed(options.signal?.aborted ? "pi_timeout" : "pi_failed", "req1");
    },
    waitUntil: (promise) => pending.push(promise),
    ...(options.graceMs === undefined ? {} : { graceMs: options.graceMs }),
    ...(options.stuckMs === undefined ? {} : { stuckMs: options.stuckMs }),
  });
  const text = await new Response(out).text();
  await Promise.all(pending);
  return { text, seen, errors };
}

/** `promise`, or a rejection after `ms`, so a stream that hangs fails the test quickly. */
function within<T>(ms: number, promise: Promise<T>): Promise<T> {
  return Promise.race([
    promise,
    new Promise<never>((_, reject) => setTimeout(() => reject(new Error("hung")), ms)),
  ]);
}

/** A model stream that ignores the deadline: one event, then `late` 60 ms after the abort. */
function ignoresDeadline(
  controller: AbortController,
  late: "close" | "error",
): ReadableStream<Uint8Array> {
  return new ReadableStream<Uint8Array>({
    start(stream) {
      stream.enqueue(new TextEncoder().encode("data: {}\n\n"));
      setTimeout(() => controller.abort(new Error("timed out")), 10);
      setTimeout(() => {
        if (late === "close") stream.close();
        else stream.error(new Error("gone"));
      }, 70);
    },
  });
}

/** The data of the stream's `result` event. */
const resultOf = (text: string): unknown => {
  const match = /\n\nevent: result\ndata: (.*)\n\n$/.exec(text);
  return match ? JSON.parse(match[1] ?? "") : undefined;
};

describe("forwardWithResult", () => {
  it("forwards the model's events untouched, then appends the result", async () => {
    const { text, seen } = await forward(streamOf(workersAiStream));
    expect(text).toBe(workersAiStream + resultEvent(UNKNOWN));
    expect(seen).toEqual([workersAiStream]);
    expect(resultOf(text)).toEqual(UNKNOWN);
  });

  it("forwards a stream that arrives in many pieces", async () => {
    const { text, seen } = await forward(streamOf(workersAiStream, 97));
    expect(text).toBe(workersAiStream + resultEvent(UNKNOWN));
    expect(seen).toEqual([workersAiStream]);
  });

  it("ends with a pi_failed result when the model's stream breaks", async () => {
    const encoder = new TextEncoder();
    const body = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(encoder.encode('data: {"choices":[{"delta":{"content":"{\\"sta'));
        controller.error(new Error("upstream reset"));
      },
    });
    const { text, seen, errors } = await forward(body);
    expect(seen).toEqual([]);
    expect(String(errors[0])).toContain("upstream reset");
    expect(resultOf(text)).toEqual(failed("pi_failed", "req1"));
  });

  it("ends with a pi_timeout result when the deadline errors the model's stream", async () => {
    // As fetch and AI.run do: their signal errors the body they return.
    const controller = new AbortController();
    const encoder = new TextEncoder();
    const body = new ReadableStream<Uint8Array>({
      start(stream) {
        stream.enqueue(encoder.encode("data: {}\n\n"));
        controller.signal.addEventListener("abort", () => stream.error(controller.signal.reason));
        setTimeout(() => controller.abort(new Error("timed out")), 10);
      },
    });
    const { text } = await forward(body, { signal: controller.signal });
    expect(text.startsWith("data: {}\n\n")).toBe(true);
    expect(resultOf(text)).toEqual(failed("pi_timeout", "req1"));
  });

  for (const late of ["close", "error"] as const) {
    it(`ends with a pi_timeout result once a stream that ignored the deadline ends (${late})`, async () => {
      const controller = new AbortController();
      const { text, errors } = await within(
        2_000,
        forward(ignoresDeadline(controller, late), { signal: controller.signal, graceMs: 10 }),
      );
      expect(String(errors[0])).toContain("timed out");
      expect(text.startsWith("data: {}\n\n")).toBe(true);
      expect(resultOf(text)).toEqual(failed("pi_timeout", "req1"));
    });
  }

  it("gives up on a stream that never ends", async () => {
    // MAX_TOKENS ends a real model stream; this one never ends, so the visitor's
    // stream stays open, but the background work still settles after stuckMs.
    const controller = new AbortController();
    const body = new ReadableStream<Uint8Array>({
      start(stream) {
        stream.enqueue(new TextEncoder().encode("data: {}\n\n"));
        setTimeout(() => controller.abort(new Error("timed out")), 10);
      },
    });
    const pending: Promise<unknown>[] = [];
    const errors: unknown[] = [];
    const out = forwardWithResult({
      body,
      signal: controller.signal,
      graceMs: 10,
      stuckMs: 20,
      finish: () => UNKNOWN,
      fail: (error) => {
        errors.push(error);
        return failed("pi_timeout", "req1");
      },
      waitUntil: (promise) => pending.push(promise),
    });
    await within(2_000, Promise.all(pending));
    expect(String(errors[0])).toContain("timed out");
    void out.cancel().catch(() => {});
  });

  it("ends with a failure when building the result throws", async () => {
    const { text, errors } = await forward(streamOf("data: [DONE]\n\n"), {
      finish: () => {
        throw new Error("a bug");
      },
    });
    expect(String(errors[0])).toContain("a bug");
    expect(resultOf(text)).toEqual(failed("pi_failed", "req1"));
  });
});
