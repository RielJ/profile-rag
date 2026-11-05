# profile-rag

profile-rag answers questions about one person's professional experience, and it answers only from sources that a person has reviewed and approved. Every answer cites the sources it uses. If the approved sources do not answer a question, the app says so and gives no answer. The backend indexes the approved sources into Postgres with pgvector, using local Ollama embeddings. It retrieves chunks with dense, MMR, or hybrid search, and gives them to an answer model through [Pi](https://pi.dev). A React page and a command-line tool ask the questions. The repository ships a fictional example corpus about Dana Whitlock, so you can run the whole app without private data. [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) describes the pipeline, the modules, the tables, and the corpus policy in detail.

## Prerequisites

| Tool | Version | Install | Used for |
| --- | --- | --- | --- |
| Docker with Compose v2 | Docker Desktop or OrbStack (tested: Docker 28.0, Compose 2.34) | [Docker](https://docs.docker.com/get-docker/), [OrbStack](https://orbstack.dev) | Postgres with pgvector |
| uv | 0.13.6 or later | [uv](https://docs.astral.sh/uv/getting-started/installation/) | Python 3.14 and the backend dependencies. uv downloads Python itself |
| Node | 26 or later (`.node-version` pins 26) | [fnm](https://github.com/Schniz/fnm) (`fnm install 26`) or [nodejs.org](https://nodejs.org) | The frontend. The Makefile uses fnm when it is installed |
| pnpm | 12.16.1 | `corepack enable pnpm`, or [pnpm](https://pnpm.io/installation) | The frontend dependencies |
| Ollama | Recent (tested: 0.40) | [Ollama](https://ollama.com/download), then `ollama pull nomic-embed-text:v1.5` | Embeddings, on the host |
| Pi | 1.5.0 (tested) | [Pi](https://pi.dev), then `pi` and `/login` | The answer model. Pi needs credentials for the provider of `PROFILE_RAG_PI_MODEL` |

`make demo` checks each of these before it changes anything, and prints one fix for each missing item.

## Quick start

```sh
git clone <this repository> profile-rag-app
cd profile-rag-app
cp .env.example .env
make demo
```

`make demo` runs these steps:

1. It checks the tools in the table above. Then it installs the backend dependencies (`uv sync`) and checks the ports, Ollama and the embedding model, Pi, and the corpus. If anything is missing, it lists every problem and stops before it starts the database.
2. It installs the frontend dependencies (`pnpm install`).
3. It starts Postgres in Docker and waits until the database is healthy.
4. It applies the migrations in `db/migrations/`.
5. It indexes `corpus-example/` with the default embedding model, `nomic-v15`.
6. It starts the API and the frontend, waits until both answer, and prints the address:

```text
The demo is running. Corpus: corpus-example. Embedding model: nomic-v15.

  Open http://127.0.0.1:5173
  API: http://127.0.0.1:8000/docs
```

Press Ctrl-C to stop the API and the frontend. The database keeps running. `make db-down` stops it, and `make db-reset` also deletes its data. You can run `make demo` again at any time: the database stays up, applied migrations are skipped, and unchanged chunks are skipped.

On a machine with warm uv and pnpm caches, the first `make demo` takes about 40 seconds and a second run about 5 seconds. A cold first run also downloads Python 3.14, the Python and Node packages, and the Postgres image.

The demo uses the database in `.env`. `make demo` replaces the chunks of that database's table with the corpus it indexes.

### Run a second checkout next to the first

docker compose finds a container by its project name only. Two checkouts with the same `COMPOSE_PROJECT_NAME` would share one database container and its data. `make demo` refuses to start when another checkout owns the project name. To run a second checkout, give it its own name and ports in its `.env`:

```sh
COMPOSE_PROJECT_NAME=profile-rag-second
DB_PORT=5453
DATABASE_URL=postgresql://profile_rag:profile_rag@127.0.0.1:5453/profile_rag
TEST_DATABASE_URL=postgresql://profile_rag:profile_rag@127.0.0.1:5453/profile_rag_test
API_PORT=8010
FRONTEND_PORT=5183
```

Set `API_PORT` or `FRONTEND_PORT` the same way when another program already uses port 8000 or 5173. `make backend` and `make frontend` use these ports too.

## Use the app

### The web page

Open the address that `make demo` prints and type a question, for example "What web framework did Dana use for the shipment-tracking API?". Enter asks the question. Shift+Enter adds a new line. Each question starts one Pi process, so an answer takes a few seconds.

The page shows one of these results:

- **Answer**: the answer, with numbered markers such as `[1]` that jump to the evidence they cite. Each evidence item gives the source title, who wrote the source, and the excerpt. The title is a link when the source has a public `http(s)` URL. A small line names the model that answered.
- **Not in the evidence**: the approved sources do not answer the question. The page gives the model's reason.
- **No answer this time**: the server failed, for example because Pi timed out. The page shows a request ID. The API's log has a line with the same request ID, the error class, and the error code.
- **The question was not accepted** (HTTP 422), or **Connection problem** (no network, or a response that the page cannot read).

To run the API and the frontend in separate terminals, with auto-reload for the API, run `make backend` and `make frontend` after `make demo` has indexed once. Vite forwards every `/api` request to the API, so the browser talks to one origin.

### The command line

Run the `profile-rag` command from `backend/`:

```sh
cd backend
uv run profile-rag search "Which databases has Dana used?" --mode hybrid -k 5
uv run profile-rag ask "What web framework did Dana use for the shipment-tracking API?"
uv run profile-rag agent "What web framework did Dana use for the shipment-tracking API?"
```

- `search` prints the retrieved chunks and calls no model. `--mode` is `dense` (the default for `search`), `mmr`, or `hybrid`. Hybrid search fuses dense results with a keyword search. `--model` selects the embedding model: `nomic-v15` (the default) or `qwen3-06b`.
- `ask` retrieves chunks, gives them to the answer model, and prints the answer and its citations. It makes one Pi call. If the evidence does not answer the question, it prints `Unknown.` and the reason.
- `agent` first has the model grade the retrieved chunks, and rewrites the question once if they are not relevant. It makes 2 Pi calls, or 4 with a rewrite. The API and `ask` use the plain chain. [`docs/eval/agent-decision.md`](docs/eval/agent-decision.md) gives the reason.

The exit codes of `ask` and `agent` are 0 for an answer or `Unknown.`, 2 for a setup error (no table, Ollama not running) or a failed evidence search, 3 when Pi fails, and 4 when the model's reply is rejected (not the answer format, or a citation of a chunk it was not given). `agent` also exits with 5 if the graph exceeds its step limit, which means a bug.

Run `uv run profile-rag --help` for every command, including `index`, `eval`, `compare-agent`, `hnsw`, and `corpus`. "Index and search", "Generation", and "Agent graph" in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) explain them.

### The API

`POST /api/ask` answers a question with the answer chain, as `ask` does:

```sh
curl -s -X POST http://127.0.0.1:8000/api/ask \
  -H 'content-type: application/json' \
  -d '{"question": "What is lantern-queue and what is it built on?"}'
```

The body takes `question` (1 to 1000 characters, not blank), and optionally `mode` (`dense`, `hybrid`, or `mmr`) and `k` (1 to 20). The response's `status` is one of these:

| `status` | HTTP | Fields |
| --- | --- | --- |
| `answered` | 200 | `answer`, `citations`, `model` |
| `unknown` | 200 | `reason`, `model` (null when nothing was retrieved) |
| `failed` | 500, 502, 503, or 504 | `error`, `message`, `request_id` |

`GET /api/sources/{id}` returns the metadata of one approved, visible source, never its text. `GET /api/health` reports the database. The OpenAPI page is at `/docs` on the API's port. "API" in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) lists the error codes.

The API logs one warning line for each failed answer, with the request ID, the exception class, and the error code. `PROFILE_RAG_LOG_LEVEL` in `.env` sets the level (default `INFO`). The log never holds the question, the reply, or Pi's error output.

## What the corpus covers

`corpus-example/` is a fictional profile of Dana Whitlock. Every name, company, URL, and address in it is invented. It has 3 approved sources and 15 chunks: a resume, a project README (`lantern-queue`), and a blog post. It also registers a LinkedIn export and a note that are not approved, so the index skips them. The tests, the browser tests, the demo, and the example evaluation set all use this corpus.

A real corpus lives in `corpus/`, which Git ignores except for its README. Each source has a kind: `resume`, `github`, `portfolio`, `blog`, `linkedin`, or `note`. Each source also has an attribution, which tells the model whose claim a sentence is:

| Attribution | Meaning |
| --- | --- |
| `self_authored` | The person wrote or built it, and the source shows that. |
| `self_report` | The person's own description of their work, such as a resume. |
| `project_doc` | Documentation of a project that the person took part in. |
| `third_party` | Someone else's statement about the person. |

The answer names the attribution ("According to Dana's resume, ..."), so the app does not turn a project's documentation into a claim about what the person did. Visibility is `public`, or `sanitized` for a reviewed copy of private material with confidential details removed. Only approved sources with one of these two visibilities can appear in a search result.

## Use your own corpus

1. Register, copy, and review each source with `profile-rag corpus add`, `corpus snapshot`, and `corpus review`. [`corpus/README.md`](corpus/README.md) gives the steps. A source moves from `candidate` to `in_review` to `approved` (or `rejected`), and only approved sources are indexed. Approval records the file's SHA-256 hash. If the file changes later, the approval is stale, and `index` and `make demo` stop until you review the source again.

   `corpus review` with no ID starts a terminal session over every source in review and every stale approval. Each source prints its contents in the terminal (findings marked, secrets masked), then the session asks the checklist questions and then offers to approve. A `n` anywhere brings up a menu to view, edit, reject, or skip. Approval needs a `y` to every checklist item for the file's current content, and the answers go into the audit record in `reviews/`. Quitting keeps the decisions already made. For example (inputs after each prompt; abridged):

   ```text
   $ uv run profile-rag corpus --root ../corpus-example review
   [1/1] lantern-queue-readme (in_review)
     kind: github  attribution: project_doc  visibility: public  revision: 3f9a1c2e7b4d…
     file: approved/lantern-queue-readme.md  sha256: b046b1f5507e…
     findings: none. That does not make the file safe; read it in full.

   --- approved/lantern-queue-readme.md ---
   (the file, with line numbers)
   --- end of approved/lantern-queue-readme.md (36 lines) ---

   Checklist for lantern-queue-readme:
   1/5 Is the content public, or sanitized into a version any visitor could see? [y/n]: y
   ...
   5/5 Would you be comfortable with an answer quoting any sentence of it verbatim? [y/n]: y
   Every item answered yes. [a]pprove is available.
   Approve lantern-queue-readme? [y/n]: y
   lantern-queue-readme is approved.

   Approved 1, rejected 0, skipped 0. Remaining: in_review 0, stale 0.
   ```

2. Run the demo on that corpus:

   ```sh
   make demo CORPUS_DIR=corpus
   ```

   `CORPUS_DIR` is the folder that holds `sources.yaml` and `approved/`, relative to the repository root. If the folder has no approved source, `make demo` stops before it touches the database, with a message that gives the number of registered sources and how to approve one. It does not print source titles or text.

To index without starting the servers, run `make index CORPUS_DIR=corpus`. Add `MODEL=qwen3-06b` to index the second embedding model (pull `qwen3-embedding:0.6b` first). Each model has its own table.

A new splitter version (`SPLITTER_VERSION` in `backend/src/profile_rag/corpus/splitting.py`) changes every chunk ID. After that change, run `make index` again for each model and each corpus. The command replaces every row and reports them as added and deleted.

## Evaluation

`profile-rag eval` runs a labeled question set through one retriever and reports recall@k and MRR:

```sh
cd backend
uv run profile-rag eval --set example --model nomic-v15 --mode hybrid -k 3
```

`--set example` uses the fictional questions in `eval-example/dev.jsonl` over `corpus-example/`. `--set dev` and `--set heldout` use private questions in `eval/` over `corpus/`. Index the set's corpus with the same model first. The command writes per-question detail to `eval/results/` (Git ignores it), a report to `docs/eval/`, and the comparison table [`docs/eval/<set>-comparison.md`](docs/eval/example-comparison.md). Committed reports hold only numbers and question IDs.

Results on the example set (14 questions: 12 answerable, 2 unanswerable; 15 chunks), from [`docs/eval/example-comparison.md`](docs/eval/example-comparison.md):

| Model | Mode | k | Recall@k | MRR |
| --- | --- | --- | --- | --- |
| nomic-v15 | dense | 3 | 0.750 | 0.750 |
| nomic-v15 | dense | 8 | 1.000 | 0.783 |
| nomic-v15 | hybrid | 3 | 0.833 | 0.778 |
| nomic-v15 | hybrid | 8 | 1.000 | 0.794 |
| nomic-v15 | mmr | 3 | 0.750 | 0.750 |
| nomic-v15 | mmr | 8 | 1.000 | 0.794 |
| qwen3-06b | dense | 3 | 0.792 | 0.792 |
| qwen3-06b | dense | 8 | 0.958 | 0.816 |
| qwen3-06b | hybrid | 3 | 0.833 | 0.736 |
| qwen3-06b | hybrid | 8 | 1.000 | 0.748 |
| qwen3-06b | mmr | 3 | 0.750 | 0.778 |
| qwen3-06b | mmr | 8 | 0.917 | 0.812 |

One question moves recall by 0.083, so these numbers test the pipeline and do not rank the models. At k = 8, each question gets more than half of the corpus. On the same set, with hybrid k = 3, the agent graph and the chain both answered 10 of the 12 answerable questions and both answered the 2 unanswerable ones with `Unknown.`, but the agent used 36 Pi calls and the chain 14 ([`docs/eval/agent-vs-chain-example-nomic-v15.md`](docs/eval/agent-vs-chain-example-nomic-v15.md)). HNSW and exact search return the same results on this corpus ([`docs/eval/hnsw-example-nomic-v15.md`](docs/eval/hnsw-example-nomic-v15.md)).

The dev set is not evaluated yet. Its questions are written and their labels resolve to chunks, but the real sources still await approval, so there is no index to run it against. The choice of the default embedding model waits for the dev set, followed by one check on the held-out set. "Evaluation" in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) gives the question format, the label rule, and the metrics.

## Testing

The tests need the database. Start it with `make demo` or `make db-up`, then run:

```sh
make lint        # ruff and oxlint, formatting, and the API types check
make typecheck   # pyrefly and tsc
make test        # pytest and Vitest
```

The tests use fake embeddings, a fake chat model, and a fake Pi script. They need no Ollama and no Pi, and they call no model. To also run the tests that call a real Ollama and a real Pi, run `uv --directory backend run pytest --live`. The Pi tests check the installed version and its flags, and make one model call.

The browser tests run in Playwright's Chromium, at desktop size and at 375 pixels wide. Install the browser once per machine (about 170 MB), then run them:

```sh
make e2e-install
make e2e
```

`make e2e` starts its own API server on port 8765 and Vite on port 5174, and stops both at the end. The API server runs the real routes over `corpus-example/` in a throwaway database, with fake embeddings and a fake Pi. It never uses `.env`'s database, `corpus/`, or the Pi settings.

The frontend's API types, `frontend/src/api/schema.d.ts`, are generated from the OpenAPI schema. After you change a route or a model in `backend/src/profile_rag/api/`, run `make types`. `make lint` fails while the file is out of date.

Run `make` with no arguments to list every command.

## Privacy model

- **What stays local.** The corpus, the database, the embeddings (Ollama runs on your machine), the evaluation questions, and the per-question results. `corpus/` and `eval/` are in `.gitignore` except for their READMEs, and committed reports in `docs/eval/` hold only numbers and question IDs.
- **What leaves the machine.** For each question, Pi sends the question and the retrieved excerpts to the provider of `PROFILE_RAG_PI_MODEL`. Only approved chunks with visibility `public` or `sanitized` are retrieved, so the provider sees only text that any visitor may see.
- **Pi isolation.** Each question starts a new `pi --mode rpc` process in an empty temporary folder, with no session file, no tools, no extensions, skills, or context files, and no project `.pi/` settings. The prompt goes in on stdin, so it is not in the process list. "Pi isolation" in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) lists every flag and the reason for it.
- **Tracing guard.** Tracing to LangSmith is off unless `LANGSMITH_TRACING=true`. With tracing on, the API and every command refuse to start unless `CORPUS_DIR=corpus-example` is set in the environment and `LANGSMITH_API_KEY` is set. Commands that read the database also check that the table holds exactly `corpus-example/`. Traces then hold run names, timings, and error class names, without inputs, outputs, metadata, or error messages. "Tracing" in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) explains the rules.
- **Logs and errors.** API logs and error messages hold request IDs, error classes, and error codes. They never hold the question, the model's reply, or Pi's error output.

## Troubleshooting

| Problem | Fix |
| --- | --- |
| `Port 8000 (the API) is in use`, or the same for 5173 or 5443 | Stop the program that uses the port. `lsof -nP -iTCP:<port> -sTCP:LISTEN` names it. Or set `API_PORT`, `FRONTEND_PORT`, or `DB_PORT` in `.env`. Use the same port in `DATABASE_URL` and `TEST_DATABASE_URL` as in `DB_PORT`. |
| `Another checkout (...) uses the same Compose project name` | Set `COMPOSE_PROJECT_NAME` and a free `DB_PORT` in this checkout's `.env`, as in [Run a second checkout next to the first](#run-a-second-checkout-next-to-the-first). |
| `Ollama does not answer at http://127.0.0.1:11434` | Start Ollama (`ollama serve`, or open the Ollama app), or set `OLLAMA_URL` in `.env`. |
| `Ollama has no nomic-embed-text:v1.5` | Run `ollama pull nomic-embed-text:v1.5`. |
| `Ollama's nomic-embed-text:v1.5 has digest ..., but vectorstore/models.py expects ...` | The tag now points at other weights. Vectors from the two versions do not match. `backend/src/profile_rag/vectorstore/models.py` explains how to move to new weights. |
| `Pi ('pi') is not on PATH` | Install Pi and log in (`pi`, then `/login`), or set `PROFILE_RAG_PI_BIN` in `.env` to its path. |
| The page shows **No answer this time**, and the API log has `code=pi_failed` | Pi started but the model call failed, usually because Pi has no credentials for the provider of `PROFILE_RAG_PI_MODEL`. Run `pi` and `/login`. `make demo` cannot check credentials without a model call. |
| `code=pi_timeout` | The model took longer than `PROFILE_RAG_PI_TIMEOUT_SECONDS` (default 120). Raise it in `.env`. |
| `has no approved source` | Approve a source (see [Use your own corpus](#use-your-own-corpus)), or run `make demo` without `CORPUS_DIR`. |
| `make lint` fails with `frontend/src/api/schema.d.ts is out of date` | Run `make types`, then commit the regenerated file with the API change. |
| `make test` fails to connect to the database | Run `make db-up`. Check that `TEST_DATABASE_URL` uses `DB_PORT`. |
| `fnm has no Node 26` | Run `fnm install 26`. |

## Not covered

This app runs on one machine, for its owner. Public deployment is out of scope and needs its own review before anyone does it. That review has to cover at least these points:

- Provider access: every visitor's question would spend the owner's Pi credentials and provider account.
- Cost: each question is at least one paid model call, and nothing limits the total.
- Rate limits: the API has no authentication, no rate limit, and no limit on concurrent Pi processes.
- Logging: a public server needs a log and retention policy. Today the API logs to stderr only.
- Network exposure: every server binds to 127.0.0.1, and the API has no TLS or CORS setup.

## Layout

| Path | Contents |
| --- | --- |
| `backend/` | The FastAPI service and the `profile-rag` command-line tool (Python) |
| `frontend/` | The React page (TypeScript, Vite). Browser tests are in `frontend/tests/e2e/` |
| `db/` | Database initialization and migrations |
| `scripts/` | `demo.sh`, which `make demo` runs |
| `corpus/` | The real corpus. Git ignores everything except its README |
| `corpus-example/` | The fictional corpus for tests and the demo |
| `eval/` | Private evaluation questions and results. Git ignores everything except its README |
| `eval-example/` | Fictional evaluation questions over `corpus-example/` |
| `docs/` | [`ARCHITECTURE.md`](docs/ARCHITECTURE.md), and evaluation reports in `docs/eval/` |
