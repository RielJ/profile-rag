/**
 * The rules for a failed answer, apart from its wording: the HTTP status of a
 * failure before the stream starts, which code a failed model call gets, and
 * that a failure is never cached. The wording is FAILURES in src/contract.ts,
 * generated from the Python code.
 */

import type { AnsweredResponse, AskResponse, FailedCode, UnknownResponse } from "./contract";

/** The HTTP status of a failure before the stream starts. The Worker owns these values. */
export const FAILED_STATUS = {
  rate_limited: 429,
  internal_error: 500,
  retrieval_failed: 503,
  pi_timeout: 504,
  pi_failed: 502,
} as const satisfies Partial<Record<FailedCode, number>>;

/** A failure that can be refused with an HTTP status, before the stream starts. */
export type RefusedCode = keyof typeof FAILED_STATUS;

/** The code for a model call or model stream that failed: a timeout if the deadline passed. */
export function modelFailure(signal: AbortSignal): "pi_timeout" | "pi_failed" {
  return signal.aborted ? "pi_timeout" : "pi_failed";
}

/** True if `response` may be stored in the answer cache. A failure never is. */
export function mayCache(response: AskResponse): response is AnsweredResponse | UnknownResponse {
  return response.status !== "failed";
}
