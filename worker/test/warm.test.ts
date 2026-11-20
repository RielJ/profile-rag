import { describe, expect, it } from "vitest";

import { WARM_QUESTIONS, warmOne } from "../src/warm";

const streamed = (text: string) => {
  let drained = false;
  const body = new ReadableStream<Uint8Array>({
    start(controller) {
      controller.enqueue(new TextEncoder().encode(text));
      controller.close();
    },
  });
  const response = new Response(
    body.pipeThrough(
      new TransformStream({
        flush() {
          drained = true;
        },
      }),
    ),
  );
  return { response, drained: () => drained };
};

describe("WARM_QUESTIONS", () => {
  it("are the portfolio's suggested questions, exactly as the chips send them", () => {
    // web-portfollio app/routes/home/lib/ask-engine.ts, SUGGESTED (QA order).
    expect(WARM_QUESTIONS).toEqual([
      "Does he know Kubernetes?",
      "Has he shipped payments?",
      "What has he built with AI?",
      "Can he own deployments?",
      "Has he owned a product end to end?",
      "Does he write Rust?",
    ]);
  });
});

describe("warmOne", () => {
  it("answers only the first question that is not cached, and reads its answer to the end", async () => {
    const asked: string[] = [];
    const stream = streamed("event: result\ndata: {}\n\n");
    const warmed = await warmOne({
      questions: ["a", "b", "c"],
      isCached: async (q) => q === "a",
      ask: async (q) => {
        asked.push(q);
        return stream.response;
      },
    });
    expect(warmed).toEqual({ question: "b", status: 200 });
    expect(asked).toEqual(["b"]);
    expect(stream.drained()).toBe(true);
  });

  it("asks nothing when every question is cached", async () => {
    let asks = 0;
    const warmed = await warmOne({
      questions: ["a", "b"],
      isCached: async () => true,
      ask: async () => {
        asks += 1;
        return new Response("");
      },
    });
    expect(warmed).toBeNull();
    expect(asks).toBe(0);
  });

  it("reports a refused answer without throwing, so the next run tries again", async () => {
    const warmed = await warmOne({
      questions: ["a"],
      isCached: async () => false,
      ask: async () => Response.json({ status: "failed" }, { status: 503 }),
    });
    expect(warmed).toEqual({ question: "a", status: 503 });
  });
});
