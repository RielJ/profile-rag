/**
 * Send the model's server-sent events to the visitor untouched, then one
 * last event written by the Worker:
 *
 *     event: result
 *     data: {"status": "answered", ...}
 *
 * Reading the events in JavaScript one by one costs over 100 ms CPU per
 * answer (the plan's spike); the Free plan allows 10 ms. So the stream is
 * teed. One branch is piped natively into the response, the other is read
 * with one native `Response.text()`, and the reply is parsed once at the end.
 */

import type { AskResponse, FailedResponse } from "./contract";

export interface ForwardOptions {
  /** The model's server-sent events. */
  body: ReadableStream<Uint8Array>;
  /**
   * The answer's deadline. The model call gets the same signal and errors
   * `body` when it aborts, which ends the forward with `fail`'s result. If
   * `body` ignores it, the forward builds `fail`'s result `graceMs` later and
   * writes it once `body` ends (MAX_TOKENS ends a real model stream).
   */
  signal: AbortSignal;
  graceMs?: number;
  /**
   * How long to wait for a `body` that ignored the deadline to end. Until it
   * ends, the pipe holds the visitor's stream and nothing else can write to
   * it; past this, the forward gives up and the stream stays open.
   */
  stuckMs?: number;
  /** The result for the whole event stream, once it has ended. */
  finish: (events: string) => AskResponse;
  /** The result when the stream fails or `signal` aborts it. */
  fail: (error: unknown) => FailedResponse;
  /** Keeps the Worker alive until the result is written (ExecutionContext.waitUntil). */
  waitUntil: (promise: Promise<unknown>) => void;
}

/** The model's events, then the `result` event. */
export function forwardWithResult({
  body,
  signal,
  finish,
  fail,
  waitUntil,
  graceMs = 5_000,
  stuckMs = 30_000,
}: ForwardOptions): ReadableStream<Uint8Array> {
  const [toVisitor, toRead] = body.tee();
  const out = new IdentityTransformStream();
  // preventAbort keeps `out` open after a failure, so the result can follow.
  // No `signal` here: in workerd an aborted pipe still waits for the model's
  // next read, so it releases `out` no sooner than waiting for the pipe does.
  const piped = toVisitor.pipeTo(out.writable, { preventClose: true, preventAbort: true });
  const read = new Response(toRead).text();
  // After an abort the copy may never finish; nothing waits for it then.
  read.catch(() => {});

  waitUntil(
    (async () => {
      let result: AskResponse;
      try {
        // Once the pipe has finished, the model's stream has ended, so the copy ends too.
        await Promise.race([piped, abortedAfter(signal, graceMs)]);
        result = finish(await read);
      } catch (error) {
        result = fail(error);
      }
      // A model that ignored the abort holds `out` through the pipe until its
      // stream ends or fails. Wait for that, but not forever.
      if (!(await settlesWithin(piped, stuckMs))) return;
      try {
        const writer = out.writable.getWriter();
        await writer.write(new TextEncoder().encode(resultEvent(result)));
        await writer.close();
      } catch {
        // The visitor has gone; there is no one to tell.
      }
    })(),
  );
  return out.readable;
}

/** True once `promise` settles either way, false if `ms` passes first. */
async function settlesWithin(promise: Promise<unknown>, ms: number): Promise<boolean> {
  let timer: ReturnType<typeof setTimeout> | undefined;
  const late = new Promise<false>((resolve) => {
    timer = setTimeout(() => resolve(false), ms);
  });
  try {
    return await Promise.race([
      promise.then(
        () => true,
        () => true,
      ),
      late,
    ]);
  } finally {
    clearTimeout(timer);
  }
}

/** Rejects `graceMs` after `signal` aborts; never settles otherwise. */
function abortedAfter(signal: AbortSignal, graceMs: number): Promise<never> {
  return new Promise((_, reject) => {
    const later = () => setTimeout(() => reject(signal.reason), graceMs);
    if (signal.aborted) later();
    else signal.addEventListener("abort", later, { once: true });
  });
}

/**
 * The `result` event. It starts with a blank line, so it stays a separate
 * event even after a stream that broke off in the middle of a line.
 * `cached` marks a response served from the answer cache (src/cache.ts).
 */
export function resultEvent(response: AskResponse & { cached?: boolean }): string {
  return `\n\nevent: result\ndata: ${JSON.stringify(response)}\n\n`;
}
