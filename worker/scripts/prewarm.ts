/**
 * Asks the deployed Worker the portfolio's suggested questions, so the answer
 * cache holds them and visitors who click a suggestion never wait on the model.
 *
 * Run after each deploy (`just worker-prewarm`). A question already cached
 * costs nothing; a fresh one counts against the daily limit like any other.
 * Each question is sent as its own visitor, so the per-visitor hourly limit
 * does not stop the run.
 *
 * Usage: WORKER_URL=https://... ACCESS_KEY=... tsx scripts/prewarm.ts
 */
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

// A path, not a URL: this tsconfig's global URL is the Workers one, which node:fs rejects.
const QUESTIONS_FILE = join(dirname(fileURLToPath(import.meta.url)), "../prewarm-questions.txt");
const TIMEOUT_MS = 90_000;

const workerUrl = process.env["WORKER_URL"];
const accessKey = process.env["ACCESS_KEY"];
if (!workerUrl || !accessKey) {
  console.error("Set WORKER_URL and ACCESS_KEY.");
  process.exit(2);
}
// Fail with the value, not with a fetch error per question.
if (!URL.canParse(workerUrl)) {
  console.error(`WORKER_URL is not a URL: ${JSON.stringify(workerUrl)}`);
  process.exit(2);
}

const questions = readFileSync(QUESTIONS_FILE, "utf8")
  .split("\n")
  .map((line) => line.trim())
  .filter((line) => line !== "" && !line.startsWith("#"));

/** The JSON of the stream's `event: result`, or null if it never came. */
function resultOf(stream: string): { status?: string; error?: string; cached?: boolean } | null {
  const data = stream.split("\nevent: result\ndata: ")[1]?.split("\n")[0];
  return data ? JSON.parse(data) : null;
}

let failures = 0;
for (const [i, question] of questions.entries()) {
  const started = Date.now();
  let outcome: string;
  try {
    const response = await fetch(new URL("/api/ask/stream", workerUrl), {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-Access-Key": accessKey,
        "X-Visitor-IP": `prewarm-${i}`,
      },
      body: JSON.stringify({ question }),
      signal: AbortSignal.timeout(TIMEOUT_MS),
    });
    const body = await response.text();
    const result = response.ok ? resultOf(body) : null;
    if (!response.ok) outcome = `HTTP ${response.status} ${body.slice(0, 120)}`;
    else if (!result) outcome = "stream ended without a result";
    else if (result.status === "failed") outcome = `failed: ${result.error}`;
    else outcome = `${result.status}${result.cached ? " (already cached)" : ""}`;
    if (!result || result.status === "failed") failures += 1;
  } catch (error) {
    failures += 1;
    outcome = `error: ${error instanceof Error ? error.message : String(error)}`;
  }
  console.log(`${question}  →  ${outcome}  ${Date.now() - started} ms`);
}
process.exit(failures === 0 ? 0 : 1);
