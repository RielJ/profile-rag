/**
 * The answer cache: a repeated question is answered from Workers KV, without
 * the rate limits, the embedding, the database, or the model.
 *
 * Asking the same question twice gives the same response, so the model runs
 * once per distinct question for the life of an entry. Two questions share an
 * entry when they are equal after `normalizeQuestion`: case, spacing, Unicode
 * forms, surrounding quotes, and trailing ?!. do not matter. Paraphrases do
 * not share an entry: "Did he work at X?" and "Did he work at Y?" look alike
 * to an embedding, so a similarity match could return a wrong answer.
 *
 * The key also holds:
 * - `cacheVersion`: bumped when the prompt or the answer format changes.
 * - `corpusVersion`: a hash of the indexed chunks, set at deploy. Re-indexing
 *   changes every key, so no answer outlives the evidence it cites.
 * - `model`: the configured answer model, so a model change starts afresh.
 *
 * Only `answered` and `unknown` responses are stored. A failure is never
 * cached: the next ask must try again.
 *
 * The cache never breaks an answer. A KV error on read is a miss; a KV error
 * on write is logged and dropped. KV Free errors at its daily limits instead
 * of billing, so at the limit the Worker answers without the cache.
 */

import {
  ATTRIBUTIONS,
  type AnsweredResponse,
  type AskResponse,
  type Attribution,
  type UnknownResponse,
} from "./contract";
import { mayCache } from "./failure";
import { WORKERS_AI_MODEL } from "./model";

export type CachedResponse = AnsweredResponse | UnknownResponse;

/** How long an entry lives. A new corpus or prompt version retires it sooner. */
export const CACHE_TTL_SECONDS = 30 * 24 * 60 * 60;

export interface CacheKeyParts {
  cacheVersion: string;
  corpusVersion: string;
  model: string;
}

const QUOTES = /^["'“”‘’`]+|["'“”‘’`]+$/gu;
const TRAILING_PUNCTUATION = /[\s?!.。？！]+$/u;

/**
 * `question` in the form the cache compares: NFKC, lowercase, one space
 * between words, no surrounding quotes, no trailing ?!. marks.
 */
export function normalizeQuestion(question: string): string {
  let text = question.normalize("NFKC").toLowerCase().split(/\s+/u).filter(Boolean).join(" ");
  // Strip quotes and trailing marks until neither changes the text, so
  // '"Remote?"' and 'remote' meet.
  for (let previous = ""; previous !== text;) {
    previous = text;
    text = text.replace(QUOTES, "").replace(TRAILING_PUNCTUATION, "").trim();
  }
  return text;
}

const hex = (buffer: ArrayBuffer): string =>
  Array.from(new Uint8Array(buffer), (byte) => byte.toString(16).padStart(2, "0")).join("");

/** The KV key for `question`: `answer:` and a SHA-256 of the parts and the normalized question. */
/**
 * The key of `question`'s entry for this deploy. The model part is the
 * configured model, also when Groq runs the same weights.
 */
export function questionCacheKey(
  question: string,
  env: { CACHE_VERSION: string; CORPUS_VERSION: string },
): Promise<string> {
  return cacheKey(question, {
    cacheVersion: env.CACHE_VERSION,
    corpusVersion: env.CORPUS_VERSION,
    model: WORKERS_AI_MODEL,
  });
}

export async function cacheKey(question: string, parts: CacheKeyParts): Promise<string> {
  const material = [
    parts.cacheVersion,
    parts.corpusVersion,
    parts.model,
    normalizeQuestion(question),
  ].join("\u0000");
  const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(material));
  return `answer:${hex(digest)}`;
}

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value);

const isStringOrNull = (value: unknown): value is string | null =>
  value === null || typeof value === "string";

/** `value` if it is a stored response this version can serve, else null. */
function asCached(value: unknown): CachedResponse | null {
  if (!isRecord(value) || !isStringOrNull(value["model"])) return null;
  const { status, model } = value;
  if (status === "unknown" && typeof value["reason"] === "string") {
    return { status, reason: value["reason"], model };
  }
  if (status === "answered" && typeof value["answer"] === "string") {
    const citations = value["citations"];
    // Citations were built by this Worker before storing; check only that they are there.
    if (Array.isArray(citations) && citations.length > 0 && citations.every(isRecord)) {
      try {
        return { status, answer: value["answer"], citations: citations.map(asCitation), model };
      } catch {
        return null; // a bad citation makes the whole entry a miss
      }
    }
  }
  return null;
}

function asCitation(value: Record<string, unknown>): AnsweredResponse["citations"][number] {
  const text = (key: string): string => {
    const field = value[key];
    if (typeof field !== "string") throw new TypeError(`cached citation has no ${key}`);
    return field;
  };
  const nullable = (key: string): string | null => {
    const field = value[key];
    return typeof field === "string" ? field : null;
  };
  const attribution = text("attribution");
  if (!isAttribution(attribution)) throw new TypeError("cached citation has a bad attribution");
  return {
    chunk_id: text("chunk_id"),
    source_id: text("source_id"),
    source_title: text("source_title"),
    section: nullable("section"),
    url: nullable("url"),
    attribution,
    excerpt: text("excerpt"),
  };
}

const ATTRIBUTION_SET: ReadonlySet<string> = new Set(ATTRIBUTIONS);

const isAttribution = (value: string): value is Attribution => ATTRIBUTION_SET.has(value);

/** The cached response under `key`, or null on a miss, a bad entry, or a KV error. */
export async function readCached(kv: KVNamespace, key: string): Promise<CachedResponse | null> {
  try {
    return asCached(await kv.get(key, "json"));
  } catch (error) {
    console.warn("answer cache read failed", { error: String(error) });
    return null;
  }
}

/**
 * Store `response` under `key` if it may be cached. Never throws. Run it in
 * `ctx.waitUntil` so the visitor does not wait for the write.
 */
export async function writeCached(
  kv: KVNamespace,
  key: string,
  response: AskResponse,
): Promise<void> {
  if (!mayCache(response)) return;
  try {
    await kv.put(key, JSON.stringify(response), { expirationTtl: CACHE_TTL_SECONDS });
  } catch (error) {
    console.warn("answer cache write failed", { error: String(error) });
  }
}
