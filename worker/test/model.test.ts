import { describe, expect, it } from "vitest";

import {
  GROQ_MODEL,
  GROQ_URL,
  hasModelOutput,
  MAX_TOKENS,
  ModelFailed,
  replyOf,
  WORKERS_AI_MODEL,
  groq,
  withFallback,
  workersAi,
  type StartModel,
} from "../src/model";
import type { Message } from "../src/prompt";
// A real Workers AI gpt-oss-120b stream from the spike, with most reasoning events cut.
import workersAiStream from "./fixtures/workers-ai-gpt-oss.sse.txt";

const MESSAGES: Message[] = [
  { role: "system", content: "rules" },
  { role: "user", content: "question" },
];
const signal = new AbortController().signal;
const streamOf = (text: string): ReadableStream<Uint8Array> => new Response(text).body!;

const event = (delta: Record<string, unknown>): string =>
  `data: ${JSON.stringify({ object: "chat.completion.chunk", choices: [{ index: 0, delta }] })}\n\n`;

describe("MAX_TOKENS", () => {
  // It bounds the stream, and so the Worker's CPU on the Free plan's 10 ms.
  // The longest dev-set answer measured took 268 events.
  it("leaves room for the longest measured answer but caps runaway streams", () => {
    expect(MAX_TOKENS).toBeGreaterThanOrEqual(2.5 * 268);
    expect(MAX_TOKENS).toBeLessThanOrEqual(1000);
  });
});

describe("hasModelOutput", () => {
  it("is true for a stream with choices", () => {
    expect(hasModelOutput(workersAiStream)).toBe(true);
  });

  it("is false for a stream that only holds an error", () => {
    const error = 'data: {"errors":[{"message":"daily free allocation used up","code":4006}]}\n\n';
    expect(hasModelOutput(error)).toBe(false);
  });
});

describe("replyOf", () => {
  it("joins the content of a recorded Workers AI stream, without its reasoning", () => {
    const reply = replyOf(workersAiStream);
    expect(JSON.parse(reply)).toEqual({
      status: "answered",
      answer: expect.stringMatching(/^Yes\. His resume lists numerous remote positions/),
      citations: ["c0", "c1", "c2"],
    });
    expect(workersAiStream).toContain('"reasoning_content":"We"');
    expect(reply).not.toContain("We need");
  });

  it("reads Groq's events, and skips reasoning, usage, [DONE] and broken lines", () => {
    const events = [
      event({ role: "assistant", content: "" }),
      event({ reasoning: "Think about content first." }),
      event({ content: '{"status":' }),
      'data: {"choices":[{"delta":{"content":"broken\n\n',
      event({ content: '"unknown","reason":"No \\"content\\": here."}' }),
      event({ content: null }),
      'data: {"choices":[],"x_groq":{"usage":{"completion_tokens":9}}}\n\n',
      "data: [DONE]\n\n",
    ].join("");
    expect(JSON.parse(replyOf(events))).toEqual({
      status: "unknown",
      reason: 'No "content": here.',
    });
  });

  it("is empty when the stream has no content", () => {
    expect(replyOf(event({ reasoning: "only thinking" }))).toBe("");
    expect(replyOf("")).toBe("");
  });
});

describe("workersAi", () => {
  it("streams gpt-oss-120b with low reasoning effort and passes the signal", async () => {
    const calls: unknown[][] = [];
    const ai = {
      run: async (...args: unknown[]) => {
        calls.push(args);
        return streamOf("data: [DONE]\n\n");
      },
    } as unknown as Ai;
    const started = await workersAi(ai)(MESSAGES, signal);
    expect(started.model).toBe(WORKERS_AI_MODEL);
    expect(await new Response(started.body).text()).toBe("data: [DONE]\n\n");
    expect(calls).toEqual([
      [
        WORKERS_AI_MODEL,
        expect.objectContaining({
          messages: MESSAGES,
          stream: true,
          max_tokens: MAX_TOKENS,
          temperature: 0,
          reasoning_effort: "low",
        }),
        { signal },
      ],
    ]);
  });

  it("fails when Workers AI does not return a stream", async () => {
    const ai = { run: async () => ({ response: "not a stream" }) } as unknown as Ai;
    await expect(workersAi(ai)(MESSAGES, signal)).rejects.toBeInstanceOf(ModelFailed);
  });
});

describe("groq", () => {
  it("posts the messages with the key and returns the body", async () => {
    const requests: Request[] = [];
    const fetcher = (async (input: RequestInfo | URL, init?: RequestInit) => {
      requests.push(new Request(input, init));
      return new Response("data: [DONE]\n\n");
    }) as typeof fetch;
    const started = await groq("gsk_test", fetcher)(MESSAGES, signal);
    expect(started.model).toBe(GROQ_MODEL);
    expect(await new Response(started.body).text()).toBe("data: [DONE]\n\n");

    const [request] = requests;
    expect(request?.url).toBe(GROQ_URL);
    expect(request?.headers.get("Authorization")).toBe("Bearer gsk_test");
    expect(await request?.json()).toMatchObject({
      model: GROQ_MODEL,
      messages: MESSAGES,
      stream: true,
      max_completion_tokens: MAX_TOKENS,
      temperature: 0,
      reasoning_effort: "low",
      include_reasoning: false,
    });
  });

  it("fails on an HTTP error such as Groq's 429", async () => {
    const fetcher = (async () => new Response("slow down", { status: 429 })) as typeof fetch;
    await expect(groq("gsk_test", fetcher)(MESSAGES, signal)).rejects.toThrow(
      "Groq answered HTTP 429",
    );
  });
});

describe("withFallback", () => {
  const ok =
    (model: string): StartModel =>
    async () => ({ model, body: streamOf("") });
  const broken: StartModel = async () => {
    throw new Error("4006: you have used up your daily free allocation of 10,000 neurons");
  };

  it("uses the primary model when it starts", async () => {
    const started = await withFallback(ok("primary"), ok("fallback"))(MESSAGES, signal);
    expect(started.model).toBe("primary");
  });

  it("falls back when the primary fails, and reports why", async () => {
    const reasons: unknown[] = [];
    const start = withFallback(broken, ok("fallback"), (error) => reasons.push(error));
    expect((await start(MESSAGES, signal)).model).toBe("fallback");
    expect(String(reasons[0])).toContain("neurons");
  });

  it("rethrows without a fallback", async () => {
    await expect(withFallback(broken, null)(MESSAGES, signal)).rejects.toThrow("neurons");
  });

  it("does not fall back once the answer has timed out", async () => {
    const controller = new AbortController();
    controller.abort();
    const start = withFallback(broken, ok("fallback"));
    await expect(start(MESSAGES, controller.signal)).rejects.toThrow("neurons");
  });
});
