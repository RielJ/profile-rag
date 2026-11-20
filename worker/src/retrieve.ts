/**
 * Hybrid search: dense search and keyword search, fused by rank. Port of
 * backend/src/profile_rag/retrievers/{hybrid,keyword}.py for the Worker.
 *
 * Dense search finds chunks that mean something close to the question, even
 * with other words. Keyword search finds chunks that contain any word of the
 * question, such as a project name the embedding model has never seen.
 *
 * Both halves run in one SQL statement, so a question costs one database
 * round trip (over Neon's HTTP driver in production). The statement returns
 * up to `candidates` rows per half; reciprocal rank fusion runs here.
 *
 * The keyword half builds the same query as the Python code: Postgres parses
 * the question with the table's text search configuration (dropping
 * stopwords, stemming the rest), quotes each lexeme for the tsquery format,
 * and joins them with OR. A question of stopwords only gives no keyword hits.
 *
 * The settings (table, model, k, candidates, RRF constant, visibility
 * values) come from the Python code through generated/backend-contract.json.
 *
 * Every value in the SQL is a parameter. The table name cannot be one, so it
 * is written into the statement, and the module refuses to load if it
 * differs from the contract's table.
 *
 * Tests: worker/test/retrieve.test.ts. Parity with the Python retriever:
 * worker/scripts/retrieval-parity.ts.
 */

import { ATTRIBUTIONS, type Attribution } from "./contract";
import type { Chunk } from "./chunk";
import contract from "./generated/backend-contract.json";

export const SEARCH = contract.search;

/** The embedding model, typed for the AI binding. A test checks it equals the contract's. */
export const EMBEDDING_MODEL = "@cf/baai/bge-m3";

/** The chunk table HYBRID_SQL reads. */
export const TABLE = "chunks_bge_m3";

if (SEARCH.table !== TABLE || SEARCH.embedding_model !== EMBEDDING_MODEL) {
  throw new Error(
    `the contract searches ${SEARCH.table} with ${SEARCH.embedding_model}; ` +
      `update TABLE and EMBEDDING_MODEL in worker/src/retrieve.ts`,
  );
}

/**
 * The search statement. Parameters: $1 the question's vector as pgvector
 * text, $2 the question, $3 the text search configuration, $4 the visible
 * values, $5 the candidates per half.
 *
 * Each row carries its list and an `order_key` (cosine distance for dense,
 * the negated `ts_rank_cd` for keyword), so the rows come back in rank order
 * within each list. Keyword ties go by `chunk_id`, as in Python.
 */
export const HYBRID_SQL = String.raw`
WITH query AS (
  SELECT string_agg('''' || replace(replace(lexeme, '\', '\\'), '''', '''''') || '''', ' | ' ORDER BY lexeme)::tsquery AS q
  FROM unnest(tsvector_to_array(to_tsvector($3::regconfig, $2::text))) AS lexeme
),
dense AS (
  SELECT 'dense' AS list, chunk_id, source_id, attribution, content, langchain_metadata AS metadata,
         embedding <=> $1::vector AS order_key
  FROM chunks_bge_m3
  WHERE visibility = ANY($4::text[])
  ORDER BY embedding <=> $1::vector
  LIMIT $5
),
keyword AS (
  SELECT 'keyword' AS list, chunk_id, source_id, attribution, content, langchain_metadata AS metadata,
         -ts_rank_cd(content_tsv, query.q) AS order_key
  FROM chunks_bge_m3, query
  WHERE content_tsv @@ query.q AND visibility = ANY($4::text[])
  ORDER BY ts_rank_cd(content_tsv, query.q) DESC, chunk_id
  LIMIT $5
)
SELECT * FROM dense
UNION ALL
SELECT * FROM keyword
ORDER BY list, order_key, chunk_id`;

/** Runs a parameterized statement and returns its rows (Neon's `sql.query`, or `pg` in scripts). */
export type Query = (text: string, params: unknown[]) => Promise<Record<string, unknown>[]>;

/** Embeds the question; returns one vector. */
export type Embed = (text: string) => Promise<number[]>;

/** Search could not run: the embedding or the database failed, or returned something unexpected. */
export class RetrievalError extends Error {
  override name = "RetrievalError";
}

/** An embedder over the AI binding, checking the vector's size. */
export function workersAiEmbed(ai: Ai): Embed {
  return async (text) => {
    const output = await ai.run(EMBEDDING_MODEL, { text: [SEARCH.query_prefix + text] });
    const vector = "data" in output ? output.data?.[0] : undefined;
    if (!Array.isArray(vector) || vector.length !== SEARCH.dimensions) {
      throw new RetrievalError(`the embedding is not ${SEARCH.dimensions} numbers`);
    }
    return vector;
  };
}

export interface RetrieveOptions {
  query: Query;
  embed: Embed;
  question: string;
  k?: number;
}

/** At most `k` visible chunks for `question`, best fused score first. */
export async function retrieve({
  query,
  embed,
  question,
  k = SEARCH.k,
}: RetrieveOptions): Promise<Chunk[]> {
  let rows: Record<string, unknown>[];
  try {
    const vector = await embed(question);
    rows = await query(HYBRID_SQL, [
      `[${vector.join(",")}]`,
      question,
      SEARCH.ts_config,
      SEARCH.visible,
      SEARCH.candidates,
    ]);
  } catch (error) {
    if (error instanceof RetrievalError) throw error;
    throw new RetrievalError("search failed", { cause: error });
  }
  const dense: Chunk[] = [];
  const keyword: Chunk[] = [];
  for (const row of rows) (row["list"] === "dense" ? dense : keyword).push(chunkOf(row));
  return reciprocalRankFusion([dense, keyword], k);
}

/**
 * The `k` best chunks of all lists by summed `1 / (rank + rrfK)`, rank from 0.
 *
 * Chunks are the same when their `chunkId` is. A chunk in only one list
 * still counts. Equal scores keep the order in which the chunks first
 * appear, going through the lists in turn.
 */
export function reciprocalRankFusion(
  lists: Chunk[][],
  k: number,
  rrfK: number = SEARCH.rrf_k,
): Chunk[] {
  const scores = new Map<string, { chunk: Chunk; score: number }>();
  for (const list of lists) {
    list.forEach((chunk, rank) => {
      const entry = scores.get(chunk.chunkId) ?? { chunk, score: 0 };
      entry.score += 1 / (rank + rrfK);
      scores.set(chunk.chunkId, entry);
    });
  }
  // Array.prototype.sort is stable, and a Map iterates in insertion order.
  return [...scores.values()]
    .sort((a, b) => b.score - a.score)
    .slice(0, k)
    .map((entry) => entry.chunk);
}

const ATTRIBUTION_SET: ReadonlySet<string> = new Set(ATTRIBUTIONS);

/** A database row as a Chunk. Raises RetrievalError for a row of the wrong shape. */
export function chunkOf(row: Record<string, unknown>): Chunk {
  const { chunk_id, source_id, attribution, content, metadata } = row;
  if (
    typeof chunk_id !== "string" ||
    typeof source_id !== "string" ||
    typeof content !== "string" ||
    typeof attribution !== "string" ||
    !ATTRIBUTION_SET.has(attribution) ||
    typeof metadata !== "object" ||
    metadata === null
  ) {
    throw new RetrievalError("a search row has the wrong shape");
  }
  const meta = metadata as Record<string, unknown>;
  const headingPath = meta["heading_path"];
  return {
    chunkId: chunk_id,
    sourceId: source_id,
    title: typeof meta["title"] === "string" ? meta["title"] : source_id,
    attribution: attribution as Attribution,
    url: typeof meta["url"] === "string" ? meta["url"] : null,
    headingPath: Array.isArray(headingPath)
      ? headingPath.filter((h): h is string => typeof h === "string")
      : [],
    text: content,
  };
}
