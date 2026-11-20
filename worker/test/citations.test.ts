// Port of the citation cases in backend/tests/test_api_contract.py.
import { describe, expect, it } from "vitest";

import type { Chunk } from "../src/chunk";
import { toResponse } from "../src/citations";
import { EXCERPT_CHARS } from "../src/contract";

const LONG_TEXT = "Dana maintains lantern-queue, a Postgres-backed job queue. ".repeat(12);

const chunk = (overrides: Partial<Chunk> & Pick<Chunk, "chunkId">): Chunk => ({
  sourceId: overrides.chunkId.split(":")[0] ?? "",
  title: "Migration notes",
  attribution: "project_doc",
  url: null,
  headingPath: [],
  text: "Dana moved the queue to Postgres.",
  ...overrides,
});

const CHUNKS = [
  chunk({
    chunkId: "lantern:v1:a",
    title: "lantern-queue README",
    url: "https://git.example.com/lantern",
    text: LONG_TEXT,
  }),
  chunk({ chunkId: "notes:v1:a" }),
  chunk({
    chunkId: "local:v1:a",
    title: "Local copy",
    url: "notes/local.md",
    text: "Dana writes Rust.",
  }),
];

const cite = (chunks: Chunk[], ...citations: string[]) =>
  toResponse(
    { status: "answered", answer: "Dana maintains lantern-queue.", citations },
    chunks,
    null,
  );

describe("toResponse", () => {
  it("gives one citation per cited chunk with its source", () => {
    const response = cite(CHUNKS, "lantern:v1:a", "notes:v1:a", "local:v1:a", "lantern:v1:a");
    if (response.status !== "answered") throw new Error("expected answered");
    expect(response.citations).toHaveLength(3);
    expect(response.citations.slice(1)).toEqual([
      {
        chunk_id: "notes:v1:a",
        source_id: "notes",
        source_title: "Migration notes",
        section: null,
        url: null,
        attribution: "project_doc",
        excerpt: "Dana moved the queue to Postgres.",
      },
      {
        chunk_id: "local:v1:a",
        source_id: "local",
        source_title: "Local copy",
        section: null,
        url: null,
        attribution: "project_doc",
        excerpt: "Dana writes Rust.",
      },
    ]);
    const [first] = response.citations;
    expect(first?.url).toBe("https://git.example.com/lantern");
    expect(Array.from(first?.excerpt ?? "").length).toBeLessThanOrEqual(EXCERPT_CHARS);
    expect(first?.excerpt.endsWith("…")).toBe(true);
    expect(LONG_TEXT.startsWith(first?.excerpt.slice(0, -1) ?? "x")).toBe(true);
  });

  it("names the section and starts the excerpt after the headings", () => {
    const headingPath = ["Dana Whitlock - Resume", "Experience", "Staff Engineer, Acme"];
    const body =
      "Moved four services\nto Kubernetes.\n\n" +
      "- Led the queue migration.\n" +
      "- Cut deploy time\n  from 40 to 6 minutes.\n" +
      "1. First item\n\n" +
      "Run a worker:\n```bash\nlantern-queue worker\n  --concurrency 4\n```";
    const text = `${headingPath.join(" > ")}\n\n${body}`;
    const response = cite([chunk({ chunkId: "resume:v2:a", headingPath, text })], "resume:v2:a");
    if (response.status !== "answered") throw new Error("expected answered");
    expect(response.citations[0]?.section).toBe("Experience > Staff Engineer, Acme");
    expect(response.citations[0]?.excerpt).toBe(
      [
        "Moved four services to Kubernetes.",
        "- Led the queue migration.",
        "- Cut deploy time from 40 to 6 minutes.",
        "1. First item",
        "Run a worker:",
        "lantern-queue worker",
        "--concurrency 4",
      ].join("\n"),
    );
  });

  it("has no section under only the top heading", () => {
    const headingPath = ["Dana Whitlock - Resume"];
    const text = "Dana Whitlock - Resume\n\nDana writes Rust.";
    const response = cite([chunk({ chunkId: "resume:v2:b", headingPath, text })], "resume:v2:b");
    if (response.status !== "answered") throw new Error("expected answered");
    expect(response.citations[0]?.section).toBeNull();
    expect(response.citations[0]?.excerpt).toBe("Dana writes Rust.");
  });

  it("passes an unknown answer through with its reason", () => {
    expect(toResponse({ status: "unknown", reason: "No evidence." }, [], "m")).toEqual({
      status: "unknown",
      reason: "No evidence.",
      model: "m",
    });
  });
});
