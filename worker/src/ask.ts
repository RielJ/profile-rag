/**
 * `POST /api/ask/stream` after the access check, in this order:
 *
 * 1. The answer cache. A hit is the `result` event alone, with
 *    `"cached": true`: no limits, embedding, search, or model.
 * 2. The question limits. Over one: 429 `rate_limited`.
 * 3. Search, start the model, and stream its events followed by the checked
 *    result, which is then stored in the cache.
 *
 * Failures before the stream starts get a plain JSON `FailedResponse` with
 * the status codes in src/failure.ts (429 `rate_limited`, 503
 * `retrieval_failed`, 502 `pi_failed`, 504 `pi_timeout`, 500
 * `internal_error`). Once the stream has started, a failure is the `result`
 * event's `FailedResponse`, after a 200: `pi_failed` or `pi_timeout` if the
 * model's stream breaks, `bad_reply` or `invented_citation` if its reply is
 * wrong, and `internal_error` if building the result has a bug.
 */

import { BadReply, checkCitations, InventedCitation, parseAnswer } from "./answer";
import type { CachedResponse } from "./cache";
import type { Chunk } from "./chunk";
import { toResponse } from "./citations";
import { failed, type AskResponse, type FailedResponse } from "./contract";
import { FAILED_STATUS, modelFailure, type RefusedCode } from "./failure";
import { hasModelOutput, replyOf, type StartModel } from "./model";
import { buildMessages } from "./prompt";
import type { TakeResult } from "./limits";
import { forwardWithResult, resultEvent } from "./stream";

/**
 * The longest question the Worker accepts, in characters. The portfolio's
 * proxy (web-portfollio app/lib/profile-rag-proxy.server.ts) repeats it.
 */
export const QUESTION_MAX_LENGTH = 1000;

/** How long an answer may take, from the model call to its last event. */
export const ANSWER_TIMEOUT_MS = 60_000;

/**
 * How much of a stream with no model output to log. Such a stream holds an
 * error, such as Workers AI's used-up daily allocation, sent after the stream
 * started (src/model.ts, withFallback). A stream with `"choices"` is never
 * logged: its reasoning can repeat the visitor's question.
 */
const EVENTS_HEAD_LENGTH = 300;

/** `{ events_head }` for a `bad_reply` stream with no model output, else nothing. */
function errorStreamHead(result: AskResponse, events: string): { events_head?: string } {
  if (result.status !== "failed" || result.error !== "bad_reply") return {};
  if (hasModelOutput(events)) return {};
  return { events_head: events.slice(0, EVENTS_HEAD_LENGTH) };
}

/** The answer cache, already keyed to this question. */
export interface QuestionCache {
  /** The stored response, or null. Never throws (src/cache.ts). */
  read: () => Promise<CachedResponse | null>;
  /** Stores the response if it may be cached. Never throws. */
  write: (response: AskResponse) => Promise<void>;
}

export interface AskDeps {
  requestId: string;
  cache?: QuestionCache;
  /** Counts one fresh question against the limits. Throws if the limiter fails. */
  takeQuestion?: () => Promise<TakeResult>;
  /** The chunks to answer from, best first. Throws if search fails. */
  retrieve: (question: string) => Promise<Chunk[]>;
  startModel: StartModel;
  waitUntil: (promise: Promise<unknown>) => void;
  /**
   * The answer's deadline, started just before the model call, after the
   * cache, the limits and search. Defaults to ANSWER_TIMEOUT_MS from then.
   */
  answerDeadline?: () => AbortSignal;
  log?: (event: Record<string, unknown>) => void;
}

/** The question in a request body, or why it is not acceptable. */
export function readQuestion(body: unknown): { question: string } | { problem: string } {
  if (typeof body !== "object" || body === null || !("question" in body)) {
    return { problem: "the body must be a JSON object with a question" };
  }
  const { question } = body;
  if (typeof question !== "string" || !/\S/.test(question)) {
    return { problem: "the question must be text that is not blank" };
  }
  if ([...question].length > QUESTION_MAX_LENGTH) {
    return { problem: `the question is longer than ${QUESTION_MAX_LENGTH} characters` };
  }
  return { question };
}

export async function askStream(question: string, deps: AskDeps): Promise<Response> {
  const { requestId, waitUntil } = deps;
  const log = deps.log ?? ((event) => console.log(JSON.stringify(event)));
  const refuse = (error: RefusedCode, cause: unknown): Response => {
    log({ event: "ask_failed", request_id: requestId, error, cause: String(cause) });
    return Response.json(failed(error, requestId), {
      status: FAILED_STATUS[error],
      headers: { "X-Request-ID": requestId, "Cache-Control": "no-store" },
    });
  };

  const cached = await deps.cache?.read();
  if (cached) {
    log({ event: "answer", request_id: requestId, status: cached.status, cached: true });
    return eventStream(resultEvent({ ...cached, cached: true }), requestId);
  }

  if (deps.takeQuestion) {
    let taken: TakeResult;
    try {
      taken = await deps.takeQuestion();
    } catch (error) {
      // Without a working limiter no fresh question goes to the model.
      return refuse("internal_error", error);
    }
    if (!taken.ok) return refuse("rate_limited", `over the ${taken.limit} limit`);
  }

  let chunks: Chunk[];
  try {
    chunks = await deps.retrieve(question);
  } catch (error) {
    return refuse("retrieval_failed", error);
  }

  const signal = (deps.answerDeadline ?? (() => AbortSignal.timeout(ANSWER_TIMEOUT_MS)))();
  let started;
  try {
    started = await deps.startModel(buildMessages(question, chunks), signal);
  } catch (error) {
    return refuse(modelFailure(signal), error);
  }

  const body = forwardWithResult({
    body: started.body,
    signal,
    finish: (events) => {
      let result: AskResponse;
      try {
        result = resultOf(replyOf(events), chunks, started.model, requestId);
      } catch (error) {
        // A bug here, not the model: say so, and do not cache it.
        log({
          event: "ask_failed",
          request_id: requestId,
          model: started.model,
          error: "internal_error",
          cause: String(error),
        });
        return failed("internal_error", requestId);
      }
      log({
        event: "answer",
        request_id: requestId,
        model: started.model,
        status: result.status,
        ...(result.status === "failed" ? { error: result.error } : {}),
        ...errorStreamHead(result, events),
      });
      if (deps.cache) waitUntil(deps.cache.write(result));
      return result;
    },
    fail: (error): FailedResponse => {
      const code = modelFailure(signal);
      log({ event: "ask_failed", request_id: requestId, error: code, cause: String(error) });
      return failed(code, requestId);
    },
    waitUntil,
  });
  return eventStream(body, requestId);
}

function eventStream(body: BodyInit, requestId: string): Response {
  return new Response(body, {
    headers: {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-store",
      "X-Request-ID": requestId,
    },
  });
}

/** The checked response for the model's `reply`, citing `chunks`. */
export function resultOf(
  reply: string,
  chunks: readonly Chunk[],
  model: string,
  requestId: string,
): AskResponse {
  try {
    const answer = checkCitations(
      parseAnswer(reply),
      chunks.map((chunk) => chunk.chunkId),
    );
    return toResponse(answer, chunks, model);
  } catch (error) {
    if (error instanceof InventedCitation) return failed("invented_citation", requestId);
    if (error instanceof BadReply) return failed("bad_reply", requestId);
    throw error;
  }
}
