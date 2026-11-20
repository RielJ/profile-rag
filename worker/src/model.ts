/**
 * The answer model: `gpt-oss-120b` on Workers AI, or the same model on Groq
 * when Workers AI fails before it starts streaming (for example, after the
 * free 10,000 neurons a day are used) and a Groq key is set.
 *
 * Both send OpenAI `chat.completion.chunk` server-sent events. The Worker
 * forwards them untouched (src/stream.ts) and reads the reply from them only
 * once, at the end (`replyOf`).
 */

import type { Message } from "./prompt";

export const WORKERS_AI_MODEL = "@cf/openai/gpt-oss-120b";
export const GROQ_MODEL = "openai/gpt-oss-120b";
export const GROQ_URL = "https://api.groq.com/openai/v1/chat/completions";

/**
 * Room for low-effort reasoning plus the JSON answer, and a cap on the
 * Worker's CPU: forwarding costs about 1 ms per 20 events (one per token),
 * and the Free plan allows 10 ms. On five dev-set questions, answers took
 * 42-268 events. A reply cut off at the cap fails as `bad_reply`.
 */
export const MAX_TOKENS = 800;
/** Low effort keeps the stream, and so the Worker's CPU and the neurons, small. */
const REASONING_EFFORT = "low";

/** A started answer: the model that writes it, and its server-sent events. */
export interface ModelStream {
  model: string;
  body: ReadableStream<Uint8Array>;
}

/** The model could not start an answer. */
export class ModelFailed extends Error {
  override name = "ModelFailed";
}

/** Starts streaming an answer to `messages`. `signal` aborts it, before or during the stream. */
export type StartModel = (messages: Message[], signal: AbortSignal) => Promise<ModelStream>;

export function workersAi(ai: Ai): StartModel {
  return async (messages, signal) => {
    const body: unknown = await ai.run(
      WORKERS_AI_MODEL,
      {
        messages,
        stream: true,
        max_tokens: MAX_TOKENS,
        temperature: 0,
        reasoning_effort: REASONING_EFFORT,
      },
      { signal },
    );
    if (!(body instanceof ReadableStream)) throw new ModelFailed("Workers AI did not stream");
    return { model: WORKERS_AI_MODEL, body };
  };
}

export function groq(apiKey: string, fetcher: typeof fetch = fetch): StartModel {
  return async (messages, signal) => {
    const response = await fetcher(GROQ_URL, {
      method: "POST",
      headers: { Authorization: `Bearer ${apiKey}`, "Content-Type": "application/json" },
      body: JSON.stringify({
        model: GROQ_MODEL,
        messages,
        stream: true,
        max_completion_tokens: MAX_TOKENS,
        temperature: 0,
        reasoning_effort: REASONING_EFFORT,
        // The Worker never shows reasoning; leaving it out shortens the stream.
        include_reasoning: false,
      }),
      signal,
    });
    if (!response.ok || response.body === null) {
      await response.body?.cancel();
      throw new ModelFailed(`Groq answered HTTP ${response.status}`);
    }
    return { model: GROQ_MODEL, body: response.body };
  };
}

/**
 * `primary`, or `fallback` if `primary` fails before it streams.
 *
 * Any failure falls back, not only a used-up quota: Workers AI's errors do
 * not say reliably which limit was hit, and Groq costs nothing either.
 * A timeout does not fall back; there is no time left.
 *
 * Only a failure before the stream exists falls back. If Workers AI ever
 * sends an error inside a started stream instead, the answer ends as
 * `bad_reply` and Groq is not tried: catching that would mean reading the
 * stream in JavaScript, which the Free plan's 10 ms of CPU does not allow.
 * `ask.ts` logs the start of such a stream (`events_head`). If the logs show
 * one, read only the first chunk here and pass it on before piping the rest.
 */
export function withFallback(
  primary: StartModel,
  fallback: StartModel | null,
  onFallback: (error: unknown) => void = () => {},
): StartModel {
  return async (messages, signal) => {
    try {
      return await primary(messages, signal);
    } catch (error) {
      if (fallback === null || signal.aborted) throw error;
      onFallback(error);
      return fallback(messages, signal);
    }
  };
}

/**
 * True if a server-sent event stream holds model output (`choices`). Only such
 * a stream can repeat the visitor's question, so only such a stream is never
 * logged (src/ask.ts).
 */
export function hasModelOutput(events: string): boolean {
  return events.includes('"choices"');
}

/**
 * The model's reply in a server-sent event stream: every `delta.content`,
 * in order. Reasoning (`delta.reasoning`, `delta.reasoning_content`) is not
 * part of it.
 *
 * Only lines with a `"content":` key are parsed, so the reasoning events,
 * about two thirds of a Workers AI stream, cost a substring test each.
 */
export function replyOf(events: string): string {
  let reply = "";
  for (const line of events.split("\n")) {
    if (!line.startsWith("data:") || !line.includes('"content":')) continue;
    let event: unknown;
    try {
      event = JSON.parse(line.slice(5));
    } catch {
      continue;
    }
    reply += contentOf(event);
  }
  return reply;
}

function contentOf(event: unknown): string {
  if (typeof event !== "object" || event === null || !("choices" in event)) return "";
  const { choices } = event;
  if (!Array.isArray(choices)) return "";
  const choice: unknown = choices[0];
  if (typeof choice !== "object" || choice === null || !("delta" in choice)) return "";
  const { delta } = choice;
  if (typeof delta !== "object" || delta === null || !("content" in delta)) return "";
  return typeof delta.content === "string" ? delta.content : "";
}
