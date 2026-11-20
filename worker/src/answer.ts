/**
 * The answer the model must return, and the check of its citations.
 * Port of backend/src/profile_rag/generation/answer.py.
 *
 * The model's reply is untrusted input. `parseAnswer` accepts exactly one of
 *
 *     {"status": "answered", "answer": "...", "citations": ["chunk_id", ...]}
 *     {"status": "unknown", "reason": "..."}
 *
 * with no other keys and no blank text, optionally inside a ```json fence.
 * `checkCitations` then checks that every cited chunk_id is one of the chunks
 * the model was shown.
 *
 * Unlike the Python parser, a reply cut off mid-JSON is rejected, not
 * completed: a cut reply is a bad reply.
 */

export type Answer =
  | { status: "answered"; answer: string; citations: string[] }
  | { status: "unknown"; reason: string };

/** The reply is not a valid answer. The message describes the shape, never quotes the reply. */
export class BadReply extends Error {
  override name = "BadReply";
}

/** The answer cites chunk IDs that are not among the retrieved chunks. */
export class InventedCitation extends Error {
  override name = "InventedCitation";
  constructor(readonly invented: string[]) {
    super(`the answer cites chunks that were not retrieved: ${invented.join(", ")}`);
  }
}

const isNonBlank = (value: unknown): value is string =>
  typeof value === "string" && /\S/u.test(value);

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value);

const hasExactly = (record: Record<string, unknown>, keys: readonly string[]): boolean => {
  const own = Object.keys(record);
  return own.length === keys.length && keys.every((key) => Object.hasOwn(record, key));
};

/** `text` without whitespace and backticks at either end, as LangChain strips it. */
const stripJson = (text: string): string => text.replace(/^[\s`]+|[\s`]+$/gu, "");

/**
 * The JSON object in `reply`, read as LangChain's `parse_json_markdown` reads
 * it: the whole reply first, then everything after the first ``` (and an
 * optional `json`). The fence match is greedy, so a ``` inside a string
 * does not cut the reply short.
 */
function readObject(reply: string): Record<string, unknown> {
  const parse = (text: string): Record<string, unknown> | undefined => {
    try {
      const value: unknown = JSON.parse(escapeControlCharsInStrings(stripJson(text)));
      if (!isRecord(value)) throw new BadReply("the reply is not a JSON object");
      return value;
    } catch (error) {
      if (error instanceof BadReply) throw error;
      return undefined;
    }
  };
  const whole = parse(reply);
  if (whole) return whole;
  const fenced = /```(?:json)?(.*)/su.exec(reply);
  const inner = fenced ? parse(fenced[1] ?? "") : undefined;
  if (!inner) throw new BadReply("the reply is not JSON");
  return inner;
}

/**
 * `text` with raw newlines, carriage returns, and tabs inside strings escaped.
 * Python's `json.loads(strict=False)`, which the backend uses, accepts them;
 * `JSON.parse` does not, and models do write them.
 */
function escapeControlCharsInStrings(text: string): string {
  let out = "";
  let inString = false;
  let escaped = false;
  for (const char of text) {
    if (inString) {
      if (escaped) escaped = false;
      else if (char === "\\") escaped = true;
      else if (char === '"') inString = false;
      else if (char === "\n") {
        out += "\\n";
        continue;
      } else if (char === "\r") {
        out += "\\r";
        continue;
      } else if (char === "\t") {
        out += "\\t";
        continue;
      }
    } else if (char === '"') inString = true;
    out += char;
  }
  return out;
}

/** The answer in `reply`. Raises BadReply if it is not one of the two shapes. */
export function parseAnswer(reply: string): Answer {
  const value = readObject(reply);
  if (value["status"] === "answered") {
    const { answer, citations } = value;
    if (
      hasExactly(value, ["status", "answer", "citations"]) &&
      isNonBlank(answer) &&
      Array.isArray(citations) &&
      citations.length > 0 &&
      citations.every(isNonBlank)
    ) {
      return { status: "answered", answer, citations };
    }
    throw new BadReply("the reply is not a valid answered object");
  }
  if (value["status"] === "unknown") {
    const { reason } = value;
    if (hasExactly(value, ["status", "reason"]) && isNonBlank(reason)) {
      return { status: "unknown", reason };
    }
    throw new BadReply("the reply is not a valid unknown object");
  }
  throw new BadReply("the reply has no valid status");
}

/**
 * `answer` unchanged if every citation is one of `retrieved`.
 * Raises InventedCitation, listing each unknown ID once, otherwise.
 */
export function checkCitations(answer: Answer, retrieved: Iterable<string>): Answer {
  if (answer.status === "unknown") return answer;
  const known = new Set(retrieved);
  const invented = [...new Set(answer.citations)].filter((id) => !known.has(id));
  if (invented.length > 0) throw new InventedCitation(invented);
  return answer;
}
