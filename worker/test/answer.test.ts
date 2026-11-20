// Port of backend/tests/test_answer.py (draft_text moves to the portfolio proxy).
import { describe, expect, it } from "vitest";

import {
  BadReply,
  InventedCitation,
  checkCitations,
  parseAnswer,
  type Answer,
} from "../src/answer";

describe("parseAnswer", () => {
  it("parses an answered reply", () => {
    expect(parseAnswer('{"status": "answered", "answer": "Rust.", "citations": ["a:1"]}')).toEqual({
      status: "answered",
      answer: "Rust.",
      citations: ["a:1"],
    });
  });

  it("parses an unknown reply", () => {
    expect(parseAnswer('{"status": "unknown", "reason": "Not in the evidence."}')).toEqual({
      status: "unknown",
      reason: "Not in the evidence.",
    });
  });

  it("accepts a json fence", () => {
    expect(
      parseAnswer('```json\n{"status": "unknown", "reason": "No evidence."}\n```').status,
    ).toBe("unknown");
  });

  it("does not cut a reply at a fence inside a string", () => {
    const reply = '```json\n{"status": "unknown", "reason": "Run ```make``` first."}\n```';
    expect(parseAnswer(reply)).toEqual({ status: "unknown", reason: "Run ```make``` first." });
  });

  it("accepts raw newlines inside strings, as the Python parser does", () => {
    expect(parseAnswer('{"status": "unknown", "reason": "line one\nline two"}')).toEqual({
      status: "unknown",
      reason: "line one\nline two",
    });
  });

  it.each([
    ["prose", "The answer is Rust."],
    ["unknown status", '{"status": "maybe", "reason": "x"}'],
    ["no status", '{"answer": "Rust.", "citations": ["a:1"]}'],
    [
      "extra field",
      '{"status": "answered", "answer": "Rust.", "citations": ["a:1"], "confidence": 1}',
    ],
    ["no cites", '{"status": "answered", "answer": "Rust.", "citations": []}'],
    ["blank", '{"status": "answered", "answer": "  ", "citations": ["a:1"]}'],
    ["blank id", '{"status": "answered", "answer": "R", "citations": [" "]}'],
    ["empty reason", '{"status": "unknown", "reason": ""}'],
    ["whitespace reason", '{"status": "unknown", "reason": "\\n\\t"}'],
    ["unknown with cites", '{"status": "unknown", "reason": "x", "citations": ["a:1"]}'],
    ["array", '[{"status": "unknown", "reason": "x"}]'],
    ["cut off", '{"status": "unknown", "reason": "Not in'],
  ])("rejects a reply of the wrong shape: %s", (_, reply) => {
    expect(() => parseAnswer(reply)).toThrow(BadReply);
  });

  it("never quotes the reply in its error", () => {
    expect(() => parseAnswer("secret prose")).toThrow(/^the reply is not JSON$/u);
  });
});

describe("checkCitations", () => {
  const answered = (...citations: string[]): Answer => ({
    status: "answered",
    answer: "Rust.",
    citations,
  });

  it("returns an answer citing retrieved chunks unchanged", () => {
    const answer = answered("a:1", "b:2");
    expect(checkCitations(answer, ["a:1", "b:2", "c:3"])).toBe(answer);
  });

  it("raises on invented citations, listing each unknown ID once", () => {
    let caught: unknown;
    try {
      checkCitations(answered("a:1", "x:9", "x:9", "y:8"), ["a:1"]);
    } catch (error) {
      caught = error;
    }
    expect(caught).toBeInstanceOf(InventedCitation);
    expect(caught instanceof InventedCitation && caught.invented).toEqual(["x:9", "y:8"]);
  });

  it("has nothing to check in an unknown answer", () => {
    const answer: Answer = { status: "unknown", reason: "No evidence." };
    expect(checkCitations(answer, [])).toBe(answer);
  });
});
