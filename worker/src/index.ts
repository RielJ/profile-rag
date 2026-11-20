/**
 * The profile-rag API on Cloudflare Workers.
 *
 * Routes:
 * - `GET /api/health`: `{"status": "ok"}`, open to anyone.
 * - `POST /api/ask/stream`: needs `X-Access-Key`. Answers the question as a
 *   server-sent event stream (src/ask.ts, and the plan's "Wire format").
 *
 * Every response carries `X-Request-ID`.
 */

import { neon } from "@neondatabase/serverless";

import { hasAccessKey } from "./access";
import { askStream, readQuestion } from "./ask";
import { cacheKey, readCached, writeCached } from "./cache";
import { limitsFrom, visitorAddress, visitorId } from "./limits";
import { groq, withFallback, WORKERS_AI_MODEL, workersAi } from "./model";
import { retrieve, workersAiEmbed, type Query } from "./retrieve";

export { QuestionLimiter } from "./limits";

/** The one limiter object that holds every count (src/limits.ts). */
const LIMITER_NAME = "global";

const json = (body: unknown, status: number, requestId: string): Response =>
  Response.json(body, {
    status,
    headers: { "X-Request-ID": requestId, "Cache-Control": "no-store" },
  });

export default {
  async fetch(request, env, ctx): Promise<Response> {
    const requestId = crypto.randomUUID().replaceAll("-", "");
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

      // One HTTP round trip per query; neon() throws without a URL, inside retrieve's error handling.
      const query: Query = (text, params) => neon(env.DATABASE_URL).query(text, params);
      const embed = workersAiEmbed(env.AI);
      const key = await cacheKey(read.question, {
        cacheVersion: env.CACHE_VERSION,
        corpusVersion: env.CORPUS_VERSION,
        // The configured model, also when Groq runs the same weights.
        model: WORKERS_AI_MODEL,
      });
      return askStream(read.question, {
        requestId,
        cache: {
          read: () => readCached(env.ANSWERS, key),
          write: (response) => writeCached(env.ANSWERS, key, response),
        },
        // The access key was checked above, so the proxy's visitor header is trusted.
        takeQuestion: async () =>
          env.LIMITER.getByName(LIMITER_NAME).take(
            await visitorId(visitorAddress(request, true)),
            Date.now(),
            limitsFrom(env),
          ),
        retrieve: (question) => retrieve({ query, embed, question }),
        startModel: withFallback(
          workersAi(env.AI),
          env.GROQ_API_KEY ? groq(env.GROQ_API_KEY) : null,
          (error) =>
            console.warn(
              JSON.stringify({
                event: "model_fallback",
                request_id: requestId,
                cause: String(error),
              }),
            ),
        ),
        waitUntil: (promise) => ctx.waitUntil(promise),
      });
    }

    return json({ detail: "Not Found" }, 404, requestId);
  },
} satisfies ExportedHandler<Env>;
