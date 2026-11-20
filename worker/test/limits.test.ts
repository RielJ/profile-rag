import { env } from "cloudflare:workers";
import { describe, expect, it } from "vitest";

import { limitsFrom, visitorAddress, visitorId, type Limits } from "../src/limits";

const LIMITS: Limits = { perVisitorPerHour: 2, perDay: 3 };
const HOUR = 3_600_000;
// 2026-11-20T10:00:00Z, the start of an hour.
const T0 = Date.UTC(2026, 10, 20, 10);

let objects = 0;
/** A fresh limiter object per test, so counts never leak between tests. */
const limiter = () => env.LIMITER.getByName(`test-${objects++}`);

describe("QuestionLimiter", () => {
  it("allows a visitor up to the hourly limit, then refuses", async () => {
    const stub = limiter();
    expect(await stub.take("a", T0, LIMITS)).toEqual({ ok: true });
    expect(await stub.take("a", T0 + 1, LIMITS)).toEqual({ ok: true });
    expect(await stub.take("a", T0 + 2, LIMITS)).toEqual({ ok: false, limit: "visitor" });
  });

  it("counts visitors separately", async () => {
    const stub = limiter();
    await stub.take("a", T0, LIMITS);
    await stub.take("a", T0, LIMITS);
    expect(await stub.take("b", T0, LIMITS)).toEqual({ ok: true });
  });

  it("resets a visitor's count at the next clock hour", async () => {
    const stub = limiter();
    await stub.take("a", T0, LIMITS);
    await stub.take("a", T0, LIMITS);
    expect(await stub.take("a", T0 + HOUR - 1, LIMITS)).toEqual({ ok: false, limit: "visitor" });
    expect(await stub.take("a", T0 + HOUR, LIMITS)).toEqual({ ok: true });
  });

  it("refuses everyone once the daily total is reached", async () => {
    const stub = limiter();
    await stub.take("a", T0, LIMITS);
    await stub.take("b", T0, LIMITS);
    await stub.take("c", T0, LIMITS);
    expect(await stub.take("d", T0, LIMITS)).toEqual({ ok: false, limit: "day" });
  });

  it("resets the daily total at midnight UTC", async () => {
    const stub = limiter();
    const lastHour = Date.UTC(2026, 10, 20, 23);
    for (const visitor of ["a", "b", "c"]) await stub.take(visitor, lastHour, LIMITS);
    expect(await stub.take("d", lastHour, LIMITS)).toEqual({ ok: false, limit: "day" });
    expect(await stub.take("d", Date.UTC(2026, 10, 21), LIMITS)).toEqual({ ok: true });
  });

  it("does not count a refused question", async () => {
    const stub = limiter();
    await stub.take("a", T0, LIMITS);
    await stub.take("a", T0, LIMITS);
    await stub.take("a", T0, LIMITS); // refused
    await stub.take("b", T0, LIMITS);
    // Day total is 3 (a, a, b), not 4: the refused question did not count.
    expect(await stub.take("c", T0, LIMITS)).toEqual({ ok: false, limit: "day" });
  });

  it("refuses everything when a limit is 0", async () => {
    const stub = limiter();
    expect(await stub.take("a", T0, { perVisitorPerHour: 5, perDay: 0 })).toEqual({
      ok: false,
      limit: "day",
    });
  });

  it("keeps counts across stubs for the same object", async () => {
    const name = `test-${objects++}`;
    await env.LIMITER.getByName(name).take("a", T0, LIMITS);
    await env.LIMITER.getByName(name).take("a", T0, LIMITS);
    expect(await env.LIMITER.getByName(name).take("a", T0, LIMITS)).toEqual({
      ok: false,
      limit: "visitor",
    });
  });
});

describe("visitorAddress", () => {
  const request = (headers: Record<string, string>) =>
    new Request("https://worker.test/api/ask/stream", { headers });

  it("uses the proxy's header when trusted", () => {
    const r = request({ "X-Visitor-IP": " 203.0.113.7 ", "CF-Connecting-IP": "198.51.100.1" });
    expect(visitorAddress(r, true)).toBe("203.0.113.7");
  });

  it("ignores the proxy's header when not trusted", () => {
    const r = request({ "X-Visitor-IP": "203.0.113.7", "CF-Connecting-IP": "198.51.100.1" });
    expect(visitorAddress(r, false)).toBe("198.51.100.1");
  });

  it("falls back to the connecting address, then unknown", () => {
    expect(visitorAddress(request({ "CF-Connecting-IP": "198.51.100.1" }), true)).toBe(
      "198.51.100.1",
    );
    expect(visitorAddress(request({}), true)).toBe("unknown");
  });

  it("caps the length", () => {
    expect(visitorAddress(request({ "X-Visitor-IP": "x".repeat(500) }), true)).toHaveLength(100);
  });
});

describe("visitorId", () => {
  it("is a stable hash that does not contain the address", async () => {
    const id = await visitorId("203.0.113.7");
    expect(id).toMatch(/^[0-9a-f]{32}$/u);
    expect(id).toBe(await visitorId("203.0.113.7"));
    expect(id).not.toBe(await visitorId("203.0.113.8"));
    expect(id).not.toContain("203");
  });
});

describe("limitsFrom", () => {
  it("reads whole numbers", () => {
    expect(limitsFrom({ QUESTIONS_PER_VISITOR_PER_HOUR: "5", QUESTIONS_PER_DAY: "60" })).toEqual({
      perVisitorPerHour: 5,
      perDay: 60,
    });
  });

  it.each(["", " ", "-1", "2.5", "1e3", "lots"])("rejects %j", (value) => {
    expect(() =>
      limitsFrom({ QUESTIONS_PER_VISITOR_PER_HOUR: value, QUESTIONS_PER_DAY: "60" }),
    ).toThrow("QUESTIONS_PER_VISITOR_PER_HOUR");
  });
});
