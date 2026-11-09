// Client for POST /api/ask, POST /api/ask/stream, and GET /api/sources/{id}.
//
// The types come from schema.d.ts, which `make types` generates from the
// backend's OpenAPI schema. The response body is still checked at run time:
// the types describe what the backend promises, not what a proxy, an old
// server, or a broken deploy actually sends. Nothing here uses `as`.
//
// Neither function throws. Every outcome is a value of a union, so the
// caller's `switch` must handle each one:
//
// - `response`: a parsed AskResponse. HTTP 200 carries `answered` or
//   `unknown`; a 5xx carries `failed` (docs/ARCHITECTURE.md, "API").
// - `invalid`: HTTP 422, the server rejected the request body.
// - `client_error`: the request never produced a usable response: no
//   network, an aborted request, a body that is not JSON, or JSON of the
//   wrong shape or with an unexpected status code.
//
// `askStream` returns the same union. While the answer is written it also
// passes each draft piece to `onDraft`. A draft is unchecked text: the
// returned result replaces it, and can still be `unknown` or `failed`.

import type { components } from "./schema";

type Schemas = components["schemas"];
export type AskRequest = Schemas["AskRequest"];
export type AskResponse = Schemas["AskResponse"];
export type AnsweredResponse = Schemas["AnsweredResponse"];
export type UnknownResponse = Schemas["UnknownResponse"];
export type FailedResponse = Schemas["FailedResponse"];
export type Citation = Schemas["Citation"];
export type Attribution = Citation["attribution"];
export type FailedCode = FailedResponse["error"];
export type SearchMode = NonNullable<AskRequest["mode"]>;
export type SourceResponse = Schemas["SourceResponse"];
export type AskStreamEvent = Schemas["AskStreamEvent"];

// Mirrors backend/src/profile_rag/api/models.py QUESTION_MAX_LENGTH.
export const QUESTION_MAX_LENGTH = 1000;

export type ClientErrorReason =
  | "network" // fetch rejected (offline, DNS, refused), or a stream was cut off
  | "aborted" // the caller aborted the request
  | "not_json" // the body is not JSON (for example a proxy's HTML error page)
  | "bad_shape" // JSON, but not the shape this status code promises
  | "unexpected_status"; // a status code the API does not document here

export interface ClientError {
  kind: "client_error";
  reason: ClientErrorReason;
  httpStatus: number | null; // null when no response arrived
  requestId: string | null;
}

export type AskResult =
  | { kind: "response"; response: AskResponse; httpStatus: number; requestId: string | null }
  | { kind: "invalid"; messages: string[]; requestId: string | null }
  | ClientError;

export type SourceResult =
  | { kind: "source"; source: SourceResponse }
  | { kind: "not_found" }
  | ClientError;

// --- Runtime shape checks ---------------------------------------------------
// Each guard checks every field its type claims, so the narrowed value is
// safe to render. `in` checks narrow `unknown` without a type assertion.

const ATTRIBUTIONS: readonly Attribution[] = [
  "self_authored",
  "self_report",
  "project_doc",
  "third_party",
];
const FAILED_CODES: readonly FailedCode[] = [
  "retrieval_failed",
  "pi_timeout",
  "pi_failed",
  "bad_reply",
  "invented_citation",
  "internal_error",
];
const SOURCE_KINDS: readonly SourceResponse["kind"][] = [
  "resume",
  "github",
  "portfolio",
  "blog",
  "linkedin",
  "note",
];
const VISIBILITIES: readonly SourceResponse["visibility"][] = ["public", "sanitized"];

function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isOneOf<T extends string>(options: readonly T[], value: unknown): value is T {
  return options.some((option) => option === value);
}

function isStringOrNull(value: unknown): value is string | null {
  return value === null || typeof value === "string";
}

function isCitation(value: unknown): value is Citation {
  return (
    isObject(value) &&
    typeof value.chunk_id === "string" &&
    typeof value.source_id === "string" &&
    typeof value.source_title === "string" &&
    isStringOrNull(value.url) &&
    isOneOf(ATTRIBUTIONS, value.attribution) &&
    typeof value.excerpt === "string"
  );
}

function parseAskResponse(value: unknown): AskResponse | null {
  if (!isObject(value)) return null;
  switch (value.status) {
    case "answered": {
      const { answer, citations, model } = value;
      if (typeof answer !== "string" || !isStringOrNull(model)) return null;
      if (!Array.isArray(citations) || citations.length === 0) return null;
      if (!citations.every(isCitation)) return null;
      return { status: "answered", answer, citations, model };
    }
    case "unknown": {
      const { reason, model } = value;
      if (typeof reason !== "string" || !isStringOrNull(model)) return null;
      return { status: "unknown", reason, model };
    }
    case "failed": {
      const { error, message, request_id } = value;
      if (!isOneOf(FAILED_CODES, error)) return null;
      if (typeof message !== "string" || typeof request_id !== "string") return null;
      return { status: "failed", error, message, request_id };
    }
    default:
      return null;
  }
}

function parseStreamEvent(value: unknown): AskStreamEvent | null {
  if (!isObject(value)) return null;
  switch (value.type) {
    case "draft":
      return typeof value.text === "string" ? { type: "draft", text: value.text } : null;
    case "result": {
      const response = parseAskResponse(value.response);
      return response === null ? null : { type: "result", response };
    }
    default:
      return null;
  }
}

// FastAPI's 422 body: {"detail": [{"msg": "...", ...}, ...]}. Only `msg` is
// kept; `input` can echo the question back and is never shown.
function parseValidationMessages(value: unknown): string[] | null {
  if (!isObject(value) || !Array.isArray(value.detail)) return null;
  const messages: string[] = [];
  for (const item of value.detail) {
    if (!isObject(item) || typeof item.msg !== "string") return null;
    messages.push(item.msg);
  }
  return messages;
}

function parseSource(value: unknown): SourceResponse | null {
  if (!isObject(value)) return null;
  const { id, kind, title, url, attribution, visibility, approved_at } = value;
  if (typeof id !== "string" || typeof title !== "string") return null;
  if (!isOneOf(SOURCE_KINDS, kind) || !isOneOf(ATTRIBUTIONS, attribution)) return null;
  if (!isStringOrNull(url) || !isOneOf(VISIBILITIES, visibility)) return null;
  if (typeof approved_at !== "string") return null;
  return { id, kind, title, url, attribution, visibility, approved_at };
}

// --- Requests -----------------------------------------------------------------

type Received = { kind: "json"; response: Response; body: unknown } | ClientError;

function clientError(reason: ClientErrorReason, response: Response | null = null): ClientError {
  return {
    kind: "client_error",
    reason,
    httpStatus: response?.status ?? null,
    requestId: response?.headers.get("X-Request-ID") ?? null,
  };
}

async function request(url: string, init: RequestInit): Promise<Received> {
  let response: Response;
  try {
    response = await fetch(url, init);
  } catch (error: unknown) {
    const aborted = error instanceof DOMException && error.name === "AbortError";
    return clientError(aborted ? "aborted" : "network");
  }
  let body: unknown;
  try {
    body = await response.json();
  } catch (error: unknown) {
    const aborted = error instanceof DOMException && error.name === "AbortError";
    return clientError(aborted ? "aborted" : "not_json", response);
  }
  return { kind: "json", response, body };
}

export interface AskOptions {
  mode?: SearchMode;
  k?: number;
  signal?: AbortSignal;
}

function askInit(question: string, options: AskOptions, accept: string): RequestInit {
  const payload: AskRequest = { question: question.trim() };
  if (options.mode !== undefined) payload.mode = options.mode;
  if (options.k !== undefined) payload.k = options.k;
  const init: RequestInit = {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: accept },
    body: JSON.stringify(payload),
  };
  if (options.signal) init.signal = options.signal;
  return init;
}

function invalidResult(response: Response, body: unknown): AskResult {
  const messages = parseValidationMessages(body);
  return messages === null
    ? clientError("bad_shape", response)
    : { kind: "invalid", messages, requestId: response.headers.get("X-Request-ID") };
}

/** POST /api/ask. `question` is sent trimmed. */
export async function ask(question: string, options: AskOptions = {}): Promise<AskResult> {
  const received = await request("/api/ask", askInit(question, options, "application/json"));
  if (received.kind !== "json") return received;
  const { response, body } = received;
  const requestId = response.headers.get("X-Request-ID");

  if (response.status === 422) return invalidResult(response, body);
  if (response.status !== 200 && (response.status < 500 || response.status > 599)) {
    return clientError("unexpected_status", response);
  }
  const parsed = parseAskResponse(body);
  // 200 must be answered or unknown; 5xx must be failed.
  const expectFailed = response.status !== 200;
  if (parsed === null || (parsed.status === "failed") !== expectFailed) {
    return clientError("bad_shape", response);
  }
  return { kind: "response", response: parsed, httpStatus: response.status, requestId };
}

export interface AskStreamOptions extends AskOptions {
  /** Called with each new piece of the draft answer, in order. */
  onDraft?: (text: string) => void;
}

/**
 * POST /api/ask/stream. Resolves to the stream's `result` line, as an
 * `AskResult` with `httpStatus` 200: the stream reports a failure in that
 * line, not in its status code.
 */
export async function askStream(
  question: string,
  options: AskStreamOptions = {},
): Promise<AskResult> {
  const init = askInit(question, options, "application/x-ndjson");
  let response: Response;
  try {
    response = await fetch("/api/ask/stream", init);
  } catch (error: unknown) {
    return clientError(isAbort(error) ? "aborted" : "network");
  }
  if (response.status === 422) {
    try {
      return invalidResult(response, await response.json());
    } catch (error: unknown) {
      return clientError(isAbort(error) ? "aborted" : "not_json", response);
    }
  }
  if (response.status !== 200) return clientError("unexpected_status", response);
  if (response.body === null) return clientError("bad_shape", response);

  const requestId = response.headers.get("X-Request-ID");
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffered = "";
  try {
    for (;;) {
      const { done, value } = await reader.read();
      buffered += done ? decoder.decode() : decoder.decode(value, { stream: true });
      const lines = buffered.split("\n");
      buffered = done ? "" : (lines.pop() ?? "");
      for (const line of lines) {
        if (line.trim() === "") continue;
        let json: unknown;
        try {
          json = JSON.parse(line);
        } catch {
          return clientError("not_json", response);
        }
        const event = parseStreamEvent(json);
        if (event === null) return clientError("bad_shape", response);
        if (event.type === "result") {
          return { kind: "response", response: event.response, httpStatus: 200, requestId };
        }
        options.onDraft?.(event.text);
      }
      if (done) return clientError("network", response); // ended without a result
    }
  } catch (error: unknown) {
    return clientError(isAbort(error) ? "aborted" : "network", response);
  } finally {
    // Stops the download if we returned early; a no-op once the stream ended.
    reader.cancel().catch(() => {});
  }
}

function isAbort(error: unknown): boolean {
  return error instanceof DOMException && error.name === "AbortError";
}

/** GET /api/sources/{id}: the metadata of one approved source. */
export async function getSource(id: string, signal?: AbortSignal): Promise<SourceResult> {
  const init: RequestInit = { headers: { Accept: "application/json" } };
  if (signal) init.signal = signal;
  const received = await request(`/api/sources/${encodeURIComponent(id)}`, init);
  if (received.kind !== "json") return received;
  const { response, body } = received;
  if (response.status === 404) return { kind: "not_found" };
  if (response.status !== 200) return clientError("unexpected_status", response);
  const source = parseSource(body);
  return source === null ? clientError("bad_shape", response) : { kind: "source", source };
}
