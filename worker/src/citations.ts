/**
 * The response for a checked answer: the citations with their sources,
 * sections, and excerpts. Port of `to_response` and its helpers in
 * backend/src/profile_rag/rag.py.
 *
 * Lengths count code points, as Python's `len` does, so an excerpt is cut
 * at the same place in both.
 */

import type { Answer } from "./answer";
import type { Chunk } from "./chunk";
import {
  EXCERPT_CHARS,
  HEADING_SEPARATOR,
  type AnsweredResponse,
  type Citation,
  type UnknownResponse,
} from "./contract";

/** The response for `answer`, citing `chunks` by ID. Every citation must be in `chunks`. */
export function toResponse(
  answer: Answer,
  chunks: readonly Chunk[],
  model: string | null,
): AnsweredResponse | UnknownResponse {
  if (answer.status === "unknown") return { status: "unknown", reason: answer.reason, model };
  const byId = new Map(chunks.map((chunk) => [chunk.chunkId, chunk]));
  const citations = [...new Set(answer.citations)].map((id) => {
    const chunk = byId.get(id);
    if (!chunk) throw new Error(`cited chunk ${id} is not among the chunks`);
    return toCitation(chunk);
  });
  return { status: "answered", answer: answer.answer, citations, model };
}

export function toCitation(chunk: Chunk): Citation {
  return {
    chunk_id: chunk.chunkId,
    source_id: chunk.sourceId,
    source_title: chunk.title,
    section: section(chunk.headingPath),
    url: publicUrl(chunk.url),
    attribution: chunk.attribution,
    excerpt: excerpt(body(chunk.text, chunk.headingPath)),
  };
}

/** The headings below the top one, joined by " > ", or null if there are none. */
export function section(headingPath: readonly string[]): string | null {
  return headingPath.slice(1).join(HEADING_SEPARATOR) || null;
}

/** `text` without the heading line the splitter put first; other text whole. */
export function body(text: string, headingPath: readonly string[]): string {
  const line = headingPath.join(HEADING_SEPARATOR);
  const prefix = `${line}\n\n`;
  return line && text.startsWith(prefix) ? text.slice(prefix.length) : text;
}

/** `value` if it is an http(s) URL, else null. */
export function publicUrl(value: string | null): string | null {
  return value !== null && /^https?:\/\//u.test(value) ? value : null;
}

/** `text` as plain lines, cut at a word boundary to at most `limit` code points. */
export function excerpt(text: string, limit: number = EXCERPT_CHARS): string {
  const flat = Array.from(lines(text).join("\n"));
  if (flat.length <= limit) return flat.join("");
  let cut = flat.slice(0, limit - 1).join("");
  const space = Math.max(cut.lastIndexOf(" "), cut.lastIndexOf("\n"));
  if (space > 0) cut = cut.slice(0, space);
  return `${cut.trimEnd()}…`;
}

const LIST_ITEM = /^([-*+]|\d+[.)])\s/u;
const FENCE = /^(```|~~~)/u;
// The line breaks of Python's str.splitlines ("\r\n" counts as one).
const LINE_BREAKS = new Set(
  [0x0a, 0x0b, 0x0c, 0x0d, 0x1c, 0x1d, 0x1e, 0x85, 0x2028, 0x2029].map((code) =>
    String.fromCodePoint(code),
  ),
);

/** `text` split like Python's str.splitlines, keeping a trailing empty piece. */
function splitLines(text: string): string[] {
  const out: string[] = [];
  let current = "";
  for (let i = 0; i < text.length; i++) {
    const char = text.charAt(i);
    if (!LINE_BREAKS.has(char)) {
      current += char;
      continue;
    }
    if (char === "\r" && text.charAt(i + 1) === "\n") i++;
    out.push(current);
    current = "";
  }
  out.push(current);
  return out;
}

/**
 * The paragraphs, list items, and code lines of Markdown `text`, one per line.
 * Spaces collapse to one; a wrapped paragraph's lines join with a space; fence
 * markers are dropped and each line inside a fence stays its own line.
 */
export function lines(text: string): string[] {
  const out: string[] = [];
  let joinable = false;
  let inFence = false;
  for (const raw of splitLines(text)) {
    const line = raw.split(/\s+/u).filter(Boolean).join(" ");
    if (FENCE.test(line)) {
      inFence = !inFence;
      joinable = false;
    } else if (!line) {
      joinable = false;
    } else if (inFence) {
      out.push(line);
    } else if (joinable && !LIST_ITEM.test(line)) {
      out[out.length - 1] += ` ${line}`;
    } else {
      out.push(line);
      joinable = true;
    }
  }
  return out;
}
