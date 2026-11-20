/**
 * Checks that the Worker's search returns the same chunks as the Python
 * hybrid retriever, on the dev set, against the local database.
 *
 * Input: the fixture from backend/scripts/retrieval_fixture.py (question,
 * bge-m3 vector, and Python's chunk IDs per dev question). This script runs
 * src/retrieve.ts's `retrieve` with the same vectors over `pg` and compares
 * the chunk IDs in order. It exits 1 on any difference.
 *
 *   DATABASE_URL=postgresql://... pnpm exec tsx scripts/retrieval-parity.ts FIXTURE
 *
 * `just worker-parity` runs both steps. Local database only: a host other
 * than 127.0.0.1 or localhost is refused.
 */

import { readFileSync } from "node:fs";

import pg from "pg";

import { retrieve, type Query } from "../src/retrieve";

interface Case {
  id: string;
  question: string;
  vector: number[];
  chunk_ids: string[];
}

const [fixturePath] = process.argv.slice(2);
const databaseUrl = process.env["DATABASE_URL"]?.replace("postgresql+psycopg://", "postgresql://");
if (!fixturePath || !databaseUrl) {
  console.error("usage: DATABASE_URL=... tsx scripts/retrieval-parity.ts FIXTURE");
  process.exit(2);
}
const host = new URL(databaseUrl).hostname;
if (host !== "127.0.0.1" && host !== "localhost") {
  console.error(`refusing to run against ${host}: local database only`);
  process.exit(2);
}

const fixture = JSON.parse(readFileSync(fixturePath, "utf8")) as { k: number; cases: Case[] };
const client = new pg.Client({ connectionString: databaseUrl });
await client.connect();
const query: Query = async (text, params) => (await client.query(text, params)).rows;

let mismatches = 0;
try {
  for (const testCase of fixture.cases) {
    const found = await retrieve({
      query,
      embed: async () => testCase.vector,
      question: testCase.question,
      k: fixture.k,
    });
    const ids = found.map((chunk) => chunk.chunkId);
    if (JSON.stringify(ids) === JSON.stringify(testCase.chunk_ids)) continue;
    mismatches++;
    const python = new Set(testCase.chunk_ids);
    const shared = ids.filter((id) => python.has(id)).length;
    console.log(`${testCase.id}: differs (${shared}/${testCase.chunk_ids.length} shared)`);
    console.log(`  python: ${testCase.chunk_ids.join(", ")}`);
    console.log(`  worker: ${ids.join(", ")}`);
  }
} finally {
  await client.end();
}
console.log(`${fixture.cases.length - mismatches}/${fixture.cases.length} questions identical`);
process.exit(mismatches === 0 ? 0 : 1);
