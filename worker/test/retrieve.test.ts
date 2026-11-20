import { describe, expect, it } from "vitest";

import type { Chunk } from "../src/chunk";
import {
  HYBRID_SQL,
  KEYWORD_SQL,
  RetrievalError,
  SEARCH,
  chunkOf,
  reciprocalRankFusion,
  retrieve,
  workersAiEmbed,
  type Query,
} from "../src/retrieve";

const chunk = (chunkId: string): Chunk => ({
  chunkId,
  sourceId: "src",
  title: "Title",
  attribution: "self_authored",
  url: null,
  headingPath: [],
  text: chunkId,
});

const ids = (chunks: Chunk[]) => chunks.map((c) => c.chunkId);

const row = (list: "dense" | "keyword", chunkId: string): Record<string, unknown> => ({
  list,
  chunk_id: chunkId,
  source_id: "proj-x",
  attribution: "project_doc",
  content: `Heading\n\n${chunkId} body`,
  metadata: {
    title: "Project X",
    url: "https://example.com/x",
    heading_path: ["Heading"],
    ordinal: 0,
  },
});

describe("the search settings", () => {
  it("match the Python API's hybrid defaults", () => {
    expect(SEARCH).toMatchObject({
      embedding_model: "@cf/baai/bge-m3",
      dimensions: 1024,
      table: "chunks_bge_m3",
      ts_config: "pg_catalog.english",
      visible: ["public", "sanitized"],
      k: 8,
      candidates: 20,
      rrf_k: 60,
    });
  });

  it("put no value into the SQL text", () => {
    expect(HYBRID_SQL).not.toMatch(/'public'|'sanitized'|LIMIT \d/u);
    for (const n of [1, 2, 3, 4, 5]) expect(HYBRID_SQL).toContain(`$${n}`);
    expect(KEYWORD_SQL).not.toMatch(/'public'|'sanitized'|LIMIT \d/u);
    for (const n of [1, 2, 3, 4]) expect(KEYWORD_SQL).toContain(`$${n}`);
    expect(KEYWORD_SQL).not.toContain("$5");
  });

  it("keep the keyword search the same in both statements", () => {
    // KEYWORD_SQL takes the question as $1; HYBRID_SQL takes it as $2.
    const shifted = KEYWORD_SQL.replace(/\$(\d)/gu, (_: string, n: string) => `$${Number(n) + 1}`);
    const keywordPart = (sql: string) =>
      sql.slice(sql.indexOf("keyword AS ("), sql.indexOf("\n)", sql.indexOf("keyword AS (")));
    const queryPart = (sql: string) =>
      sql.slice(sql.indexOf("query AS ("), sql.indexOf("\n)", sql.indexOf("query AS (")));
    expect(keywordPart(shifted)).toBe(keywordPart(HYBRID_SQL));
    expect(queryPart(shifted)).toBe(queryPart(HYBRID_SQL));
  });
});

describe("reciprocalRankFusion", () => {
  it("ranks a chunk found by both lists first", () => {
    const fused = reciprocalRankFusion(
      [
        [chunk("a"), chunk("b")],
        [chunk("c"), chunk("b")],
      ],
      8,
    );
    expect(ids(fused)).toEqual(["b", "a", "c"]);
  });

  it("breaks ties by first appearance, going through the lists in turn", () => {
    // a and c both score 1/60; a appears first.
    expect(ids(reciprocalRankFusion([[chunk("a")], [chunk("c")]], 8))).toEqual(["a", "c"]);
    expect(ids(reciprocalRankFusion([[chunk("c")], [chunk("a")]], 8))).toEqual(["c", "a"]);
  });

  it("keeps the dense order when the keyword list is empty", () => {
    expect(ids(reciprocalRankFusion([[chunk("x"), chunk("y"), chunk("z")], []], 8))).toEqual([
      "x",
      "y",
      "z",
    ]);
  });

  it("returns at most k chunks", () => {
    const list = ["a", "b", "c", "d"].map(chunk);
    expect(ids(reciprocalRankFusion([list, []], 2))).toEqual(["a", "b"]);
  });

  it("matches Python's scores for a mixed case", () => {
    // Python: reciprocal_rank_fusion([[a,b,c],[c,d,a]], k=4)
    // a: 1/60 + 1/62, c: 1/62 + 1/60 (tie; a first), b: 1/61, d: 1/61 (tie; b first).
    const fused = reciprocalRankFusion(
      [
        [chunk("a"), chunk("b"), chunk("c")],
        [chunk("c"), chunk("d"), chunk("a")],
      ],
      4,
    );
    expect(ids(fused)).toEqual(["a", "c", "b", "d"]);
  });
});

describe("chunkOf", () => {
  it("maps a row to a chunk", () => {
    expect(chunkOf(row("dense", "proj-x:v1:abc"))).toEqual({
      chunkId: "proj-x:v1:abc",
      sourceId: "proj-x",
      title: "Project X",
      attribution: "project_doc",
      url: "https://example.com/x",
      headingPath: ["Heading"],
      text: "Heading\n\nproj-x:v1:abc body",
    });
  });

  it("falls back for missing optional metadata", () => {
    const parsed = chunkOf({ ...row("dense", "c"), metadata: {} });
    expect(parsed).toMatchObject({ title: "proj-x", url: null, headingPath: [] });
  });

  it.each([
    ["no chunk_id", { chunk_id: undefined }],
    ["an unknown attribution", { attribution: "rumor" }],
    ["no metadata", { metadata: null }],
    ["non-text content", { content: 3 }],
  ])("rejects a row with %s", (_, change) => {
    expect(() => chunkOf({ ...row("dense", "c"), ...change })).toThrow(RetrievalError);
  });
});

describe("retrieve", () => {
  const embed = async () => Array.from({ length: SEARCH.dimensions }, (_, i) => i / 1000);

  it("sends one statement with the parameters, then fuses the lists", async () => {
    const calls: unknown[][] = [];
    const query: Query = async (text, params) => {
      calls.push([text, params]);
      return [row("dense", "a"), row("dense", "b"), row("keyword", "b"), row("keyword", "c")];
    };
    const found = await retrieve({ query, embed, question: "What did he build?" });
    expect(ids(found)).toEqual(["b", "a", "c"]);
    expect(calls).toHaveLength(1);
    const [text, params] = calls[0] as [string, unknown[]];
    expect(text).toBe(HYBRID_SQL);
    expect(params[0]).toMatch(/^\[0,0\.001,0\.002,/u);
    expect(params.slice(1)).toEqual([
      "What did he build?",
      "pg_catalog.english",
      ["public", "sanitized"],
      20,
    ]);
  });

  it("returns no chunks when the database finds none", async () => {
    expect(await retrieve({ query: async () => [], embed, question: "the and of" })).toEqual([]);
  });

  it("wraps a database error as a RetrievalError", async () => {
    const query: Query = async () => {
      throw new Error("connection refused: postgres://user:secret@host");
    };
    const error = await retrieve({ query, embed, question: "q" }).catch((e: unknown) => e);
    expect(error).toBeInstanceOf(RetrievalError);
    expect((error as Error).message).not.toContain("secret");
  });

  const failingEmbed = async (): Promise<number[]> => {
    throw new Error("AI quota");
  };

  it("searches by keyword alone when the embedding fails", async () => {
    const calls: unknown[][] = [];
    const query: Query = async (text, params) => {
      calls.push([text, params]);
      return [row("keyword", "c"), row("keyword", "a")];
    };
    const fallbacks: unknown[] = [];
    const found = await retrieve({
      query,
      embed: failingEmbed,
      question: "What did he build?",
      onFallback: (error) => fallbacks.push(error),
    });
    expect(ids(found)).toEqual(["c", "a"]);
    expect(calls).toEqual([
      [KEYWORD_SQL, ["What did he build?", "pg_catalog.english", ["public", "sanitized"], 20]],
    ]);
    expect(fallbacks).toHaveLength(1);
    expect(String(fallbacks[0])).toContain("AI quota");
  });

  it("keeps at most k chunks from the keyword search", async () => {
    const rows = Array.from({ length: 12 }, (_, i) => row("keyword", `k${i}`));
    const found = await retrieve({
      query: async () => rows,
      embed: failingEmbed,
      question: "q",
      k: 3,
    });
    expect(ids(found)).toEqual(["k0", "k1", "k2"]);
  });

  it("does not fall back when the database fails after embedding", async () => {
    let calls = 0;
    const query: Query = async () => {
      calls += 1;
      throw new Error("connection refused");
    };
    await expect(retrieve({ query, embed, question: "q" })).rejects.toThrow(RetrievalError);
    expect(calls).toBe(1);
  });

  it("throws a RetrievalError when the embedding and the keyword search both fail", async () => {
    const query: Query = async () => {
      throw new Error("connection refused: postgres://user:secret@host");
    };
    const error = await retrieve({ query, embed: failingEmbed, question: "q" }).catch(
      (e: unknown) => e,
    );
    expect(error).toBeInstanceOf(RetrievalError);
    expect((error as Error).message).not.toContain("secret");
  });
});

describe("workersAiEmbed", () => {
  const fakeAi = (output: unknown) => ({ run: async () => output }) as unknown as Ai;

  it("returns the first vector", async () => {
    const vector = Array.from({ length: 1024 }, () => 0.5);
    expect(await workersAiEmbed(fakeAi({ shape: [1, 1024], data: [vector] }))("q")).toEqual(vector);
  });

  it.each([
    ["the wrong size", { data: [[1, 2, 3]] }],
    ["no data", { shape: [0] }],
    ["an async response", { request_id: "r1" }],
  ])("rejects %s", async (_, output) => {
    await expect(workersAiEmbed(fakeAi(output))("q")).rejects.toThrow(RetrievalError);
  });
});
