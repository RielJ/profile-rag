# Deploy the API

This guide puts the profile-rag API on the internet for free, so a website (here, the portfolio's hero terminal) can send it questions. The API is the Cloudflare Worker in `worker/`, written in TypeScript. The Python backend has no server: it stays on your machine, indexes the corpus, runs the evaluation, and answers from the command line (`profile-rag ask`).

| Part | Service | Free limits that matter |
| --- | --- | --- |
| API | Cloudflare Workers Free | 100,000 requests a day and 10 ms of CPU per request. At a limit, requests fail; nothing is billed |
| Answer model and question embeddings | Workers AI (`@cf/openai/gpt-oss-120b`, `@cf/baai/bge-m3`) | 10,000 neurons a day, about 25 to 40 answers. At the limit, calls fail until 00:00 UTC |
| Fallback answer model (optional) | Groq Free, same `gpt-oss-120b` weights | A daily request and token quota. Used only when Workers AI fails |
| Answer cache | Workers KV | 100,000 reads and 1,000 writes a day |
| Question limits | One Durable Object with SQLite storage | Far above what the limits let through |
| Postgres with pgvector | Neon Free | 1 GB storage and 100 compute hours a month per project. Suspends after 5 minutes idle and wakes in under a second |

The Worker differs from the local `profile-rag ask` in three ways:

- **Answers come from `gpt-oss-120b` on Workers AI, not Pi.** Pi uses your own logins, so it cannot run on a public server. If Workers AI fails before the answer starts, and `GROQ_API_KEY` is set, Groq answers instead.
- **Search uses `bge-m3` embeddings**, in the `chunks_bge_m3` table, because Workers AI does not offer `nomic-embed-text`. The evaluation in `docs/eval/dev-comparison.md` checked that `bge-m3` retrieves as well, and `just worker-parity` checks that the Worker's search returns the same chunks as Python's.
- **Repeated questions are answered from a cache.** The Worker stores each `answered` or `unknown` result for 30 days. A question that matches a stored one after normalizing case, spaces, and the final punctuation gets the stored result at once, without the model, and does not count against the limits. Failures are never stored.

The Worker holds code only. The corpus never goes into it: the chunks and vectors live in Neon, and you fill Neon from your machine.

## Before you start

You need a working local setup (after `just demo corpus`, `profile-rag ask` answers questions over your corpus) and these accounts, none of which asks for a card:

- [Cloudflare](https://dash.cloudflare.com/sign-up) for the Worker, Workers AI, and KV.
- [Neon](https://neon.tech) for Postgres.
- [Groq](https://console.groq.com), optional, for the fallback model.

Log in to Cloudflare once. The `just` recipes below use this login, also to embed the corpus with Workers AI:

```sh
cd worker && pnpm exec wrangler login
```

## 1. Fill the database

1. In Neon, create a project. Pick the region closest to most of your visitors.
2. Copy both connection strings from the **Connect** dialog:
   - The **direct** one, for indexing from your machine. It looks like `postgresql://user:password@ep-example-123.eu-central-1.aws.neon.tech/neondb?sslmode=require`.
   - The **pooled** one, with `-pooler` in the host name, for the Worker.
3. Create the tables and load your corpus:

   ```sh
   export NEON_URL='postgresql://...'   # the direct connection string
   just worker-index-neon corpus
   ```

   This applies the migrations, then embeds the approved chunks with `bge-m3` on Workers AI (about 75 neurons for the whole corpus) and writes them to `chunks_bge_m3`. Ollama is not needed.

Run `just worker-index-neon corpus` again whenever the corpus changes, then deploy again (step 3), so that the answer cache starts over.

## 2. Set the secrets

```sh
just worker-secret ACCESS_KEY     # a long random string, for example from `openssl rand -hex 32`
just worker-secret DATABASE_URL   # Neon's pooled connection string
just worker-secret GROQ_API_KEY   # optional: a Groq API key, starting with gsk_
```

Each command prompts for the value, so it never appears in your shell history. The first command also creates the Worker if it does not exist yet. The website's proxy sends `ACCESS_KEY` with every question.

The limits are plain variables in `worker/wrangler.jsonc`:

| Variable | Default | Meaning |
| --- | --- | --- |
| `QUESTIONS_PER_VISITOR_PER_HOUR` | `5` | Fresh questions one visitor may ask per clock hour |
| `QUESTIONS_PER_DAY` | `60` | Fresh questions all visitors together may ask per UTC day |
| `CACHE_VERSION` | `1` | Part of every cache key. Raise it after changing the prompt or the answer format, so no stored answer has the old format |

A fresh question is one the cache does not hold. Keep `QUESTIONS_PER_DAY` near what Workers AI and Groq can answer in a day together.

## 3. Deploy

```sh
NEON_URL='postgresql://...' just worker-deploy   # the direct connection string
```

`just worker-deploy`:

1. Checks that the Worker's copy of the prompt and answer rules matches the Python code, and runs the Worker's type check, lint, and tests. A failure stops the deploy.
2. Hashes the chunks in Neon into `CORPUS_VERSION`. This is part of every cache key, so after a re-index no stored answer outlives the excerpts it cites.
3. Deploys with `wrangler deploy` and prints the Worker's address, `https://profile-rag-api.<your-subdomain>.workers.dev`.
4. Asks the questions in `worker/prewarm-questions.txt`, so the website's suggested questions are answered from the cache. Keep that file in step with the website. Pre-warming needs the access key in `ACCESS_KEY` or in `worker/.dev.vars`; without it, this step is skipped. The recipe reads the Worker's `workers.dev` address from wrangler's output. On a custom domain, set `WORKER_URL=https://...` to use that address instead. If no address is found, the deploy still succeeds and the recipe prints the `just worker-prewarm` command to run by hand.

Deploy only with `just worker-deploy`. A plain `wrangler deploy` sets `CORPUS_VERSION` back to `dev`.

## 4. Check it

The health check needs no key:

```sh
curl https://profile-rag-api.<your-subdomain>.workers.dev/api/health
# {"status":"ok"}
```

A question needs the access key in the `X-Access-Key` header:

```sh
curl -N -X POST https://profile-rag-api.<your-subdomain>.workers.dev/api/ask/stream \
  -H 'content-type: application/json' \
  -H "X-Access-Key: $ACCESS_KEY" \
  -d '{"question":"What projects has Riel built?"}'
```

The answer is a server-sent event stream: the model's events as Workers AI sends them, then a final `event: result` whose data is the checked answer: the same JSON that `rag.to_response` builds in the Python code (`backend/src/profile_rag/answers.py` defines its shape). A cached answer is the `result` event alone, with `"cached": true`.

| Status | Meaning |
| --- | --- |
| 200 | The stream. Its `result` is `answered`, `unknown`, or `failed`, as a failure after the stream starts (`bad_reply`, `invented_citation`, `pi_failed`, `pi_timeout`) cannot change the status |
| 401 | The `X-Access-Key` header is missing or wrong |
| 422 | The question is empty or too long |
| 429 | Over a question limit (`rate_limited`). The counts reset each clock hour and each UTC day |
| 500 | The limit check failed (`internal_error`). The Worker refuses rather than answer without counting |
| 502 | Workers AI failed before the answer started, and so did Groq if `GROQ_API_KEY` is set (`pi_failed`). The error code names stay `pi_*` for every backend |
| 503 | Neon or the embedding cannot be reached (`retrieval_failed`) |

## 5. Connect the website

The browser must never see the access key, so the website calls the Worker from its own server. In the portfolio, `app/lib/profile-rag-proxy.server.ts` does this. Its server needs two variables:

| Variable | Value |
| --- | --- |
| `PROFILE_RAG_API_URL` | The Worker's address from step 3 |
| `PROFILE_RAG_ACCESS_KEY` | The `ACCESS_KEY` from step 2 |

The proxy:

1. Receives the visitor's question.
2. Sends it to `POST /api/ask/stream` with `X-Access-Key` and `X-Visitor-IP`, the visitor's address.
3. Turns the event stream into the NDJSON lines the page reads: `draft` lines with the answer text as it arrives, then the `result`.

The Worker counts questions per `X-Visitor-IP` only on requests with the right key; without the key, anyone could claim any address. It stores a hash of the address, never the address itself.

The website should show its prepared answers when the API answers anything but 200, or takes too long.

## Watch it

The Worker's logs are in the Cloudflare dashboard under **Workers & Pages → profile-rag-api → Logs**, or live with `cd worker && pnpm exec wrangler tail profile-rag-api`. Each question logs one line with its request ID, the model, the status, and an error code. Logs never hold questions or answers.

Check two things now and then:

- **Requests stopped for CPU.** On the Free plan a request gets 10 ms of CPU. Forwarding the model's stream untouched uses 3 to 15 ms, and an answer is capped at 800 tokens, so most requests stay under the limit and the rest have so far been let through. A stopped request shows the outcome `exceededCpu` in the logs. If that happens often, move to Workers Paid ($5 a month) and set a spending limit.
- **Workers AI quota.** The dashboard's **AI → Workers AI** page shows neurons used today. If the quota runs out most days, set `GROQ_API_KEY` or lower `QUESTIONS_PER_DAY`.

## Limits of this setup

- **The model providers see each question and its excerpts.** Cloudflare does not keep Workers AI inputs after the request or train on them ([data usage](https://developers.cloudflare.com/workers-ai/platform/data-usage/)); Groq's free tier has its own terms. Only deploy a corpus you are willing to publish.
- **The access key is the only gate.** Anyone with the key can spend the daily quota, so keep it in the website's server environment only, and replace it (`just worker-secret ACCESS_KEY`) if it leaks.
- **The cache matches exact questions only.** "Does he know Go?" and "Does he use Go?" are separate entries. Matching by meaning could return the answer to a different question, such as one about another company.
- **A burst of the same new question** before the first answer is stored calls the model once for each request.
