# Deploy the API

This guide puts the profile-rag API on the internet for free, with no credit card, so a website (here, the portfolio's hero terminal) can send it questions. The React page in `frontend/` is not deployed; the website is the client.

| Part | Service | Free limits that matter |
| --- | --- | --- |
| API container | Hugging Face Spaces, Docker, CPU basic | 2 vCPU, 16 GB RAM. Sleeps after 48 hours with no requests |
| Postgres with pgvector | Neon Free | 1 GB storage and 100 compute hours a month per project. Suspends after 5 minutes idle and wakes in under a second |
| Answer model | Groq Free (OpenAI-compatible API) | A daily request and token quota per model. Your console's Limits page shows the numbers |
| Question embeddings | `fastembed`, inside the container | No account and no quota |

The deployed server differs from the local one in two ways:

- **Answers come from an OpenAI-compatible API, not Pi.** Pi uses your own logins, so it cannot run on a public server. `PROFILE_RAG_LLM=openai` switches the answer model to any OpenAI-compatible endpoint: Groq by default, OpenRouter, or another provider.
- **Questions are embedded with `fastembed`, not Ollama.** `PROFILE_RAG_EMBEDDER=fastembed` runs the same `nomic-embed-text-v1.5` weights inside the container. The corpus is still embedded on your machine with Ollama; `backend/tests/test_fastembed_embeddings.py --live` checks that both give the same vectors.

The image holds code and embedding weights only. The corpus never goes into it: the chunks and vectors live in Postgres, and you fill the database from your machine.

## Before you start

You need a working local setup (`just demo corpus` answers questions over your corpus) and these accounts, none of which asks for a card:

- [Neon](https://neon.tech) for Postgres.
- [Groq](https://console.groq.com) for the answer model.
- [Hugging Face](https://huggingface.co) for the container, and the `hf` command-line tool (`uv tool install huggingface_hub`).

## 1. Fill the database

1. In Neon, create a project in the region closest to the Space (Hugging Face runs Spaces in the US and Europe).
2. Copy the connection string. Use the direct connection, not the pooled one. It looks like `postgresql://user:password@ep-example-123.eu-central-1.aws.neon.tech/neondb?sslmode=require`.
3. Create the tables and load your corpus. A `DATABASE_URL` in the shell overrides the one in `.env`:

   ```sh
   export NEON_URL='postgresql://...'
   DATABASE_URL="$NEON_URL" just migrate
   DATABASE_URL="$NEON_URL" just index corpus
   ```

   `just index` embeds the approved chunks with Ollama on your machine, so Ollama must be running.

Run the `just index` line again whenever the corpus changes. The deployed server reads the new chunks on its next question.

## 2. Get a model key

In the Groq console, create an API key. The default model is `openai/gpt-oss-120b`. To use another model or provider, set `PROFILE_RAG_LLM_MODEL` and `PROFILE_RAG_LLM_BASE_URL` in step 3. For OpenRouter, the base URL is `https://openrouter.ai/api/v1` and free model IDs end in `:free`.

Free models and their quotas change without notice. Check the model is still listed before you deploy.

## 3. Create the Space

1. On Hugging Face, create a Space: SDK **Docker**, template **Blank**, hardware **CPU basic**, visibility **Public**. A private Space would need a Hugging Face token on every request; the access key already keeps strangers out.
2. In the Space's **Settings**, add these **secrets**:

   | Secret | Value |
   | --- | --- |
   | `DATABASE_URL` | The Neon connection string from step 1 |
   | `PROFILE_RAG_LLM_API_KEY` | The Groq key from step 2 |
   | `PROFILE_RAG_ACCESS_KEY` | A long random string, for example from `openssl rand -hex 32`. The website's proxy sends it on every question |

3. Add these **variables**. They are not secret:

   | Variable | Suggested value | Meaning |
   | --- | --- | --- |
   | `PROFILE_RAG_QUESTIONS_PER_DAY` | A little below the model's daily free quota, for example `90` | Questions all visitors together may ask per UTC day |
   | `PROFILE_RAG_CLIENT_QUESTIONS_PER_HOUR` | `10` | Questions one visitor may ask per UTC clock hour |
   | `PROFILE_RAG_LLM_MODEL` | Leave unset for `openai/gpt-oss-120b` | The answer model |
   | `PROFILE_RAG_LLM_BASE_URL` | Leave unset for Groq | The provider's OpenAI-compatible base URL |

   The image already sets `PROFILE_RAG_LLM=openai` and `PROFILE_RAG_EMBEDDER=fastembed`.

## 4. Push the API

Commit your changes first: the push sends the committed `HEAD`, not your working tree.

```sh
hf auth login            # once; answer yes to "Add token as git credential?"
just space-push <user>/<space>
```

`scripts/space-push.sh` copies only what the image needs (`Dockerfile`, `.dockerignore`, and `backend/` without its tests) into the Space's own git repository, adds the README front matter that Spaces require, and pushes one commit. Nothing else from this repository goes to the Space: no `.env`, `corpus/`, `eval/`, or `frontend/`. Run the same command after each code change.

The Space then builds the image, which takes a few minutes the first time. The **Logs** tab shows the build and the server. To build and run the same image on your machine first:

```sh
just image
docker run --rm -p 7860:7860 --env-file deploy.env profile-rag-api
```

`deploy.env` holds the same names as the Space's secrets and variables. Git ignores it.

## 5. Check it

The Space's API is at `https://<user>-<space>.hf.space`. The health check needs no key:

```sh
curl https://<user>-<space>.hf.space/api/health
# {"status":"ok","database":"ok"}
```

A question needs the access key in the `X-Access-Key` header:

```sh
curl -X POST https://<user>-<space>.hf.space/api/ask \
  -H 'content-type: application/json' \
  -H "X-Access-Key: $PROFILE_RAG_ACCESS_KEY" \
  -d '{"question":"What projects has Riel built?"}'
```

| Status | Meaning |
| --- | --- |
| 200 | An answer (`answered` or `unknown`) |
| 401 | The `X-Access-Key` header is missing or wrong |
| 429 | Over a question limit (`rate_limited`). The counts reset each UTC hour and day, and when the Space restarts |
| 502 | The model provider failed (`pi_failed`, `bad_reply`, `invented_citation`). The error code names stay `pi_*` for every backend |
| 503 | The database cannot be reached (`retrieval_failed`) |
| 504 | The model took longer than `PROFILE_RAG_LLM_TIMEOUT_SECONDS` (`pi_timeout`) |

`POST /api/ask/stream` streams the same answer as NDJSON. It answers 401 and 429 before the stream starts, like `POST /api/ask`.

## 6. Connect the website

The browser must never see the access key, so the website calls the API from its own server, never from the browser. The website's server route:

1. Receives the visitor's question.
2. Sends it to `POST /api/ask/stream` with two headers: `X-Access-Key` with the key, and `X-Visitor-IP` with the visitor's address.
3. Passes the NDJSON stream back to the browser.

The API counts `PROFILE_RAG_CLIENT_QUESTIONS_PER_HOUR` per `X-Visitor-IP` value, but only on requests with the right key; without the key, anyone could claim any address. Without the header, every visitor counts as the proxy's one address.

The website should show its prepared answers when the API answers anything but 200, or takes too long: the Space can take 30 seconds or more to wake after 48 hours with no requests.

## Keep it awake (optional)

A free Space sleeps after 48 hours with no requests. A daily `GET /api/health` keeps it awake and costs no model quota. A scheduled GitHub Actions workflow or an UptimeRobot monitor can send it; neither needs a card.

## Limits of this setup

- **The counts live in memory.** One container counts every question, and a restart resets the counts. Several replicas would each count on their own.
- **The model provider sees each question and its excerpts.** Free tiers may log prompts or use them for training; check the provider's terms. Only deploy a corpus you are willing to publish.
- **The access key is the only gate.** Anyone with the key can spend the daily quota, so keep it in the website's server environment only, and replace it if it leaks.
- **Logs** go to the Space's log tab only. They hold request IDs and error codes, never questions or answers.
