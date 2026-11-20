/**
 * The profile-rag API on Cloudflare Workers.
 *
 * Routes:
 * - `GET /api/health`: `{"status": "ok"}`, open to anyone.
 * - `POST /api/ask/stream`: needs `X-Access-Key`. Answers the question as a
 *   server-sent event stream (src/ask.ts, and the plan's "Wire format").
 *
 * Every response carries `X-Request-ID`.
 *
 * The hourly cron trigger keeps the portfolio's suggested questions cached
 * (src/warm.ts).
 */

import { neon } from "@neondatabase/serverless";

import { hasAccessKey } from "./access";
import { askStream, readQuestion, type AskDeps } from "./ask";
import { questionCacheKey, readCached, writeCached } from "./cache";
import { limitsFrom, visitorAddress, visitorId } from "./limits";
import { groq, withFallback, workersAi } from "./model";
import { retrieve, workersAiEmbed, type Query } from "./retrieve";
import { WARM_QUESTIONS, warmOne } from "./warm";

export { QuestionLimiter } from "./limits";

/** The one limiter object that holds every count (src/limits.ts). */
const LIMITER_NAME = "global";

const json = (body: unknown, status: number, requestId: string): Response =>
  Response.json(body, {
    status,
    headers: { "X-Request-ID": requestId, "Cache-Control": "no-store" },
  });

const newRequestId = (): string => crypto.randomUUID().replaceAll("-", "");

const warn = (event: string, requestId: string, error: unknown): void =>
  console.warn(JSON.stringify({ event, request_id: requestId, cause: String(error) }));

/**
 * Everything `askStream` needs except the question limits, for one question.
 *
 * An answer found by keyword search alone (the embedding failed) is not
 * cached: it may be weaker than a hybrid-search answer, and a cached answer
 * is served to everyone for up to 30 days.
 */
async function answerDeps(
  question: string,
  env: Env,
  ctx: ExecutionContext,
  requestId: string,
): Promise<Omit<AskDeps, "takeQuestion">> {
  // One HTTP round trip per query; neon() throws without a URL, inside retrieve's error handling.
  const query: Query = (text, params) => neon(env.DATABASE_URL).query(text, params);
  const embed = workersAiEmbed(env.AI);
  const key = await questionCacheKey(question, env);
  let keywordOnly = false;
  return {
    requestId,
    cache: {
      read: () => readCached(env.ANSWERS, key),
      write: async (response) => {
        if (!keywordOnly) await writeCached(env.ANSWERS, key, response);
      },
    },
    retrieve: (q) =>
      retrieve({
        query,
        embed,
        question: q,
        onFallback: (error) => {
          keywordOnly = true;
          warn("retrieval_fallback", requestId, error);
        },
      }),
    startModel: withFallback(
      workersAi(env.AI),
      env.GROQ_API_KEY ? groq(env.GROQ_API_KEY) : null,
      (error) => warn("model_fallback", requestId, error),
    ),
    waitUntil: (promise) => ctx.waitUntil(promise),
  };
}

export default {
  async fetch(request, env, ctx): Promise<Response> {
    const requestId = newRequestId();
    const { pathname } = new URL(request.url);

    if (pathname === "/api/health") {
      if (request.method !== "GET") return json({ detail: "Method Not Allowed" }, 405, requestId);
      return json({ status: "ok" }, 200, requestId);
    }

    if (pathname === "/api/ask/stream") {
      if (request.method !== "POST") return json({ detail: "Method Not Allowed" }, 405, requestId);
      if (!(await hasAccessKey(request, env.ACCESS_KEY))) {
        return json({ detail: "missing or wrong access key" }, 401, requestId);
      }
      const body: unknown = await request.json().catch(() => null);
      const read = readQuestion(body);
      if ("problem" in read) return json({ detail: read.problem }, 422, requestId);

      return askStream(read.question, {
        ...(await answerDeps(read.question, env, ctx, requestId)),
        // The access key was checked above, so the proxy's visitor header is trusted.
        takeQuestion: async () =>
          env.LIMITER.getByName(LIMITER_NAME).take(
            await visitorId(visitorAddress(request, true)),
            Date.now(),
            limitsFrom(env),
          ),
      });
    }

    return json({ detail: "Not Found" }, 404, requestId);
  },

  async scheduled(_controller, env, ctx): Promise<void> {
    const requestId = newRequestId();
    const warmed = await warmOne({
      questions: WARM_QUESTIONS,
      isCached: async (question) =>
        (await readCached(env.ANSWERS, await questionCacheKey(question, env))) !== null,
      // No question limits: the cron is not a visitor, and asks at most once an hour.
      ask: async (question) => askStream(question, await answerDeps(question, env, ctx, requestId)),
    });
    console.log(JSON.stringify({ event: "warm", request_id: requestId, ...warmed }));
  },
} satisfies ExportedHandler<Env>;
