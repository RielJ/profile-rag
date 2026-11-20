/**
 * Prints the corpus version: a hash of every chunk the Worker can cite.
 *
 * `just worker-deploy` passes it as the Worker's CORPUS_VERSION, which is part
 * of every answer-cache key (src/cache.ts). Re-indexing changes the hash, so
 * no cached answer outlives the excerpts it cites. The hash covers everything
 * an answer reads from a chunk: its ID, source, attribution, text and
 * metadata.
 *
 * Usage: DATABASE_URL=<Neon connection string> tsx scripts/corpus-version.ts
 */
import pg from "pg";

import { TABLE } from "../src/retrieve";

const databaseUrl = process.env["DATABASE_URL"];
if (!databaseUrl) {
  console.error("Set DATABASE_URL to the database the Worker searches (Neon).");
  process.exit(2);
}

const client = new pg.Client({ connectionString: databaseUrl });
await client.connect();
try {
  const { rows } = await client.query<{ chunks: string; version: string | null }>(
    `SELECT count(*)::text AS chunks,
            md5(string_agg(
              md5(concat_ws(E'\\x1f', chunk_id::text, source_id::text, attribution, content,
                            langchain_metadata::text)),
              '' ORDER BY chunk_id)) AS version
     FROM ${TABLE}`,
  );
  const row = rows[0];
  if (!row?.version) {
    console.error(`${TABLE} is empty. Index the corpus first (just worker-index-neon).`);
    process.exit(1);
  }
  console.error(`${row.chunks} chunks in ${TABLE}`);
  console.log(row.version.slice(0, 16));
} finally {
  await client.end();
}
