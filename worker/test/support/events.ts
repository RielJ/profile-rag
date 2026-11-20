import type { AskResponse } from "../../src/contract";

/**
 * The data of the `result` event that ends `text`, or null if there is none.
 * The read side of `resultEvent` in src/stream.ts; test/stream.test.ts checks
 * that the two agree.
 */
export function readResultEvent(text: string): (AskResponse & { cached?: true }) | null {
  const match = /(?:^|\n\n)event: result\ndata: (.*)\n\n$/.exec(text);
  return match?.[1] === undefined ? null : JSON.parse(match[1]);
}
