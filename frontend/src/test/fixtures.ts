// Response bodies for the unit tests. Typed with the generated API types, so
// a schema change that breaks them fails `make typecheck`.

import type {
  AnsweredResponse,
  Citation,
  FailedCode,
  FailedResponse,
  UnknownResponse,
} from "../api/client.ts";

export function citation(overrides: Partial<Citation> = {}): Citation {
  return {
    chunk_id: "resume:v1:a1",
    source_id: "resume",
    source_title: "Resume 2026",
    section: "Experience > Staff Engineer, Acme",
    url: "https://example.com/resume",
    attribution: "self_report",
    excerpt: "Led the migration of the job queue to PostgreSQL.",
    ...overrides,
  };
}

export function answered(overrides: Partial<AnsweredResponse> = {}): AnsweredResponse {
  return {
    status: "answered",
    answer: "Dana led the job queue migration.",
    citations: [
      citation(),
      citation({
        chunk_id: "blog:v2:b2",
        source_id: "blog",
        source_title: "Why we moved our job queue",
        section: null,
        url: null,
        attribution: "self_authored",
        excerpt: "For three years our background jobs ran on Redis.",
      }),
    ],
    model: "provider/model-x",
    ...overrides,
  };
}

export function unknown(overrides: Partial<UnknownResponse> = {}): UnknownResponse {
  return {
    status: "unknown",
    reason: "The evidence does not mention a salary.",
    model: "provider/model-x",
    ...overrides,
  };
}

// The messages are the backend's (rag.py FAILED_MESSAGES); the UI shows
// whatever message the body carries.
export const FAILED_STATUS: Record<FailedCode, number> = {
  retrieval_failed: 503,
  pi_timeout: 504,
  pi_failed: 502,
  bad_reply: 502,
  invented_citation: 502,
  internal_error: 500,
};

export function failed(error: FailedCode): FailedResponse {
  return {
    status: "failed",
    error,
    message: `Safe message for ${error}.`,
    request_id: `req-${error}`,
  };
}

export function jsonResponse(
  status: number,
  body: unknown,
  headers: Record<string, string> = {},
): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json", ...headers },
  });
}

// A 200 response of POST /api/ask/stream: one JSON line per event (or the
// raw string as given), sent in pieces of `chunkBytes` bytes, so lines and
// multi-byte characters can be split across reads.
export function ndjsonResponse(
  events: unknown[],
  { chunkBytes = 5, headers = {} }: { chunkBytes?: number; headers?: Record<string, string> } = {},
): Response {
  const text = events
    .map((event) => (typeof event === "string" ? event : JSON.stringify(event)) + "\n")
    .join("");
  const bytes = new TextEncoder().encode(text);
  const body = new ReadableStream<Uint8Array>({
    start(controller) {
      for (let start = 0; start < bytes.length; start += chunkBytes) {
        controller.enqueue(bytes.slice(start, start + chunkBytes));
      }
      controller.close();
    },
  });
  return new Response(body, {
    status: 200,
    headers: { "Content-Type": "application/x-ndjson", ...headers },
  });
}
