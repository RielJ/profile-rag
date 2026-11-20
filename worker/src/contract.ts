/**
 * The response types of the API and the text shared with the Python backend.
 *
 * The Worker owns the wire shapes. The Python CLI's responses
 * (backend/src/profile_rag/answers.py) have the same shapes, so
 * `just worker-parity` can compare the two.
 *
 * `generated/backend-contract.json` holds the system prompt, the failure
 * messages, and the search constants, exported from the Python code so the
 * two cannot drift apart. Do not edit it; regenerate it with
 * `just worker-contract`. test/contract.test.ts checks that its failure codes
 * and attributions match the lists below.
 */

import contract from "./generated/backend-contract.json";

export const ATTRIBUTIONS = ["self_authored", "self_report", "project_doc", "third_party"] as const;
export type Attribution = (typeof ATTRIBUTIONS)[number];

export const FAILED_CODES = [
  "retrieval_failed",
  "pi_timeout",
  "pi_failed",
  "bad_reply",
  "invented_citation",
  "internal_error",
  "rate_limited",
] as const;
export type FailedCode = (typeof FAILED_CODES)[number];

export interface Citation {
  chunk_id: string;
  source_id: string;
  source_title: string;
  /** The headings above the chunk without the top one, joined by " > ", or null. */
  section: string | null;
  /** The source's public http(s) link, or null. */
  url: string | null;
  attribution: Attribution;
  /** The start of the chunk's text after its headings, one paragraph or list item per line. */
  excerpt: string;
}

export interface AnsweredResponse {
  status: "answered";
  answer: string;
  citations: Citation[];
  model: string | null;
}

export interface UnknownResponse {
  status: "unknown";
  reason: string;
  model: string | null;
}

export interface FailedResponse {
  status: "failed";
  error: FailedCode;
  message: string;
  request_id: string;
}

export type AskResponse = AnsweredResponse | UnknownResponse | FailedResponse;

/** The system message: the rules plus the answer's JSON schema. */
export const SYSTEM_PROMPT: string = contract.system;
/** The human message template, with `{context}` and `{question}`. */
export const HUMAN_TEMPLATE: string = contract.human;
export const EXCERPT_CHARS: number = contract.excerpt_chars;
export const HEADING_SEPARATOR: string = contract.heading_separator;

/** What a visitor sees for each failure. Fixed text, never the error's own. */
export const FAILURES: Readonly<Record<FailedCode, string>> = contract.failures;

export function failed(error: FailedCode, requestId: string): FailedResponse {
  return { status: "failed", error, message: FAILURES[error], request_id: requestId };
}
