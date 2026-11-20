/**
 * How often visitors may ask: per visitor per clock hour, and for everyone
 * together per UTC day. Port of backend/src/profile_rag/api/access.py's
 * QuestionLimiter, with the counts in a Durable Object instead of memory.
 *
 * Only fresh questions count. A cache hit is answered before the limits
 * (src/cache.ts) and costs nothing, so it is never refused.
 *
 * One Durable Object instance ("global") holds every count, so the counts
 * are exact across all Worker isolates and survive restarts. Its storage is
 * SQLite, which the Workers Free plan includes. A question goes through it
 * one at a time; at a few questions a minute that is no bottleneck.
 *
 * A visitor is the `X-Visitor-IP` header, which the portfolio's proxy sets
 * and which the Worker trusts only on requests with the access key, else
 * the connecting address. The object stores a hash of it, never the address.
 *
 * The caller passes the time and the limits, so tests can pass their own.
 */

import { DurableObject } from "cloudflare:workers";

const MS_PER_HOUR = 3_600_000;
const MS_PER_DAY = 86_400_000;

export interface Limits {
  perVisitorPerHour: number;
  perDay: number;
}

export type TakeResult = { ok: true } | { ok: false; limit: "visitor" | "day" };

export class QuestionLimiter extends DurableObject<Env> {
  private readonly sql: SqlStorage;

  constructor(ctx: DurableObjectState, env: Env) {
    super(ctx, env);
    this.sql = ctx.storage.sql;
    this.sql.exec(`
      CREATE TABLE IF NOT EXISTS visitor_hour (
        visitor TEXT NOT NULL,
        hour INTEGER NOT NULL,
        count INTEGER NOT NULL,
        PRIMARY KEY (visitor, hour)
      );
      CREATE TABLE IF NOT EXISTS day_total (
        day INTEGER PRIMARY KEY,
        count INTEGER NOT NULL
      );
    `);
  }

  /**
   * Count one question from `visitor` at `nowMs`. Over a limit, it counts
   * nothing and says which limit refused it. Old windows are deleted as it goes.
   */
  take(visitor: string, nowMs: number, limits: Limits): TakeResult {
    const hour = Math.floor(nowMs / MS_PER_HOUR);
    const day = Math.floor(nowMs / MS_PER_DAY);
    this.sql.exec("DELETE FROM visitor_hour WHERE hour < ?", hour);
    this.sql.exec("DELETE FROM day_total WHERE day < ?", day);

    const visitorCount = this.count(
      "SELECT count FROM visitor_hour WHERE visitor = ? AND hour = ?",
      visitor,
      hour,
    );
    if (visitorCount >= limits.perVisitorPerHour) return { ok: false, limit: "visitor" };
    const dayCount = this.count("SELECT count FROM day_total WHERE day = ?", day);
    if (dayCount >= limits.perDay) return { ok: false, limit: "day" };

    this.sql.exec(
      `INSERT INTO visitor_hour (visitor, hour, count) VALUES (?, ?, 1)
       ON CONFLICT (visitor, hour) DO UPDATE SET count = count + 1`,
      visitor,
      hour,
    );
    this.sql.exec(
      `INSERT INTO day_total (day, count) VALUES (?, 1)
       ON CONFLICT (day) DO UPDATE SET count = count + 1`,
      day,
    );
    return { ok: true };
  }

  private count(query: string, ...bindings: (string | number)[]): number {
    const row = this.sql.exec<{ count: number }>(query, ...bindings).toArray()[0];
    return row?.count ?? 0;
  }
}

/** The header the portfolio's proxy sets to the visitor's address. */
export const VISITOR_IP_HEADER = "X-Visitor-IP";

/**
 * The visitor's address: the proxy's header if `trusted` (the request had
 * the access key), else Cloudflare's connecting address, else "unknown".
 */
export function visitorAddress(request: Request, trusted: boolean): string {
  const fromProxy = trusted ? request.headers.get(VISITOR_IP_HEADER)?.trim() : undefined;
  return (fromProxy || request.headers.get("CF-Connecting-IP") || "unknown").slice(0, 100);
}

/** An opaque visitor ID: a SHA-256 of the address, so no address is stored. */
export async function visitorId(address: string): Promise<string> {
  const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(address));
  return Array.from(new Uint8Array(digest).slice(0, 16), (byte) =>
    byte.toString(16).padStart(2, "0"),
  ).join("");
}

/**
 * The limits from the Worker's variables. A missing or bad value fails every
 * fresh question with 500 `internal_error` (askStream logs the cause);
 * cached answers are still served.
 */
export function limitsFrom(env: {
  QUESTIONS_PER_VISITOR_PER_HOUR: string;
  QUESTIONS_PER_DAY: string;
}): Limits {
  const parse = (name: string, value: string): number => {
    const number = Number(value);
    if (!/^\d+$/u.test(value.trim()) || !Number.isSafeInteger(number)) {
      throw new Error(`${name} must be a whole number`);
    }
    return number;
  };
  return {
    perVisitorPerHour: parse("QUESTIONS_PER_VISITOR_PER_HOUR", env.QUESTIONS_PER_VISITOR_PER_HOUR),
    perDay: parse("QUESTIONS_PER_DAY", env.QUESTIONS_PER_DAY),
  };
}
