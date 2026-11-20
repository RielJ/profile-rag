/**
 * The answer prompt's messages. Port of backend/src/profile_rag/generation/prompt.py.
 *
 * The system message comes from the Python code (see contract.ts). The human
 * message holds one delimited evidence block per chunk, then the question.
 */

import type { Chunk } from "./chunk";
import { HUMAN_TEMPLATE, SYSTEM_PROMPT } from "./contract";

export const EVIDENCE_OPEN = "<evidence>";
export const EVIDENCE_CLOSE = "</evidence>";

// An `<evidence` or `</evidence` inside a value would let chunk text close its
// block and pose as instructions. It becomes `[evidence` / `[/evidence`.
const DELIMITER = /<(\/?)(evidence)/giu;

const neutralize = (text: string): string => text.replace(DELIMITER, "[$1$2");

/** A metadata value on one line, with no delimiter. */
const field = (value: string): string => neutralize(value).split(/\s+/u).filter(Boolean).join(" ");

/** One delimited block per chunk, in the order given. */
export function formatEvidence(chunks: readonly Chunk[]): string {
  return chunks
    .map((chunk) =>
      [
        EVIDENCE_OPEN,
        `chunk_id: ${field(chunk.chunkId)}`,
        `source: ${field(chunk.title)}`,
        `attribution: ${chunk.attribution}`,
        "text:",
        neutralize(chunk.text.trim()),
        EVIDENCE_CLOSE,
      ].join("\n"),
    )
    .join("\n\n");
}

export interface Message {
  role: "system" | "user";
  content: string;
}

/** The two messages for `question` over `chunks`. */
export function buildMessages(question: string, chunks: readonly Chunk[]): Message[] {
  const user = HUMAN_TEMPLATE.replace("{context}", () => formatEvidence(chunks)).replace(
    "{question}",
    () => question,
  );
  return [
    { role: "system", content: SYSTEM_PROMPT },
    { role: "user", content: user },
  ];
}
