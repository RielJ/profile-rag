import { describe, expect, it } from "vitest";

import { failed, FAILED_CODES } from "../src/contract";
import { FAILED_STATUS, mayCache, modelFailure } from "../src/failure";

describe("modelFailure", () => {
  it("is pi_timeout once the deadline has passed", () => {
    const controller = new AbortController();
    controller.abort();
    expect(modelFailure(controller.signal)).toBe("pi_timeout");
  });

  it("is pi_failed before the deadline", () => {
    expect(modelFailure(new AbortController().signal)).toBe("pi_failed");
  });
});

describe("mayCache", () => {
  it("never stores a failure", () => {
    expect(mayCache(failed("pi_failed", "r"))).toBe(false);
  });

  it("stores an answer and an unknown", () => {
    expect(mayCache({ status: "unknown", reason: "not covered", model: "m" })).toBe(true);
    expect(mayCache({ status: "answered", answer: "Yes.", citations: [], model: "m" })).toBe(true);
  });
});

describe("FAILED_STATUS", () => {
  it("gives every failure before the stream starts its HTTP status", () => {
    expect(FAILED_STATUS).toEqual({
      rate_limited: 429,
      internal_error: 500,
      retrieval_failed: 503,
      pi_timeout: 504,
      pi_failed: 502,
    });
  });

  it("only names known failure codes", () => {
    for (const code of Object.keys(FAILED_STATUS)) expect(FAILED_CODES).toContain(code);
  });
});
