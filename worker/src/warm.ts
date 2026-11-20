/**
 * Keeps the portfolio's suggested questions in the answer cache.
 *
 * Most visitors tap a suggested question instead of typing one, and a cached
 * answer skips the question limits, Workers AI and the database. An entry
 * goes away when it expires (CACHE_TTL_SECONDS) or when a deploy changes
 * CORPUS_VERSION or CACHE_VERSION, which changes every key.
 *
 * The cron trigger in wrangler.jsonc runs `warmOne` every hour. It answers at
 * most one question per run: one answer is what a request's CPU limit on the
 * Free plan is sized for, and it spreads the Workers AI neurons over the
 * day. After a deploy, all six are cached within six hours. A failed answer
 * is not cached, so the next run tries the same question again.
 */

/**
 * The suggested questions, exactly as the portfolio's chips send them
 * (web-portfollio app/routes/home/lib/ask-engine.ts, SUGGESTED). The cache
 * key ignores case, spacing and trailing ?!. but not other wording.
 */
export const WARM_QUESTIONS: readonly string[] = [
  "Does he know Kubernetes?",
  "Has he shipped payments?",
  "What has he built with AI?",
  "Can he own deployments?",
  "Has he owned a product end to end?",
  "Does he write Rust?",
];

export interface WarmDeps {
  questions: readonly string[];
  isCached: (question: string) => Promise<boolean>;
  /** Asks the question the way a visitor would, without the question limits. */
  ask: (question: string) => Promise<Response>;
}

/** The question this run answered and the answer's HTTP status, or null if all were cached. */
export async function warmOne(
  deps: WarmDeps,
): Promise<{ question: string; status: number } | null> {
  for (const question of deps.questions) {
    if (await deps.isCached(question)) continue;
    const response = await deps.ask(question);
    // Read the stream to the end: the answer is checked and cached as it finishes.
    await response.body?.pipeTo(new WritableStream());
    return { question, status: response.status };
  }
  return null;
}
