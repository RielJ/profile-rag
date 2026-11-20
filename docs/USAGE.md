# Using profile-rag

This guide covers setup, the command line, the HTTP API, the corpus, evaluation, testing, privacy, and troubleshooting. For a short overview, read the [README](../README.md). For the internals, read [`ARCHITECTURE.md`](ARCHITECTURE.md).

## Prerequisites

| Tool | Version | Install | Used for |
| --- | --- | --- | --- |
| Docker with Compose v2 | Docker Desktop or OrbStack (tested: Docker 28.0, Compose 2.34) | [Docker](https://docs.docker.com/get-docker/), [OrbStack](https://orbstack.dev) | Postgres with pgvector |
| just | 1.39 or later (tested: 1.58) | `brew install just`, or [just](https://just.systems/man/en/packages.html) | Runs the project's commands (`justfile`) |
| uv | 0.13.6 or later | [uv](https://docs.astral.sh/uv/getting-started/installation/) | Python 3.14 and the backend dependencies. uv downloads Python itself |
| Node | 26 or later (`.node-version` pins 26) | [fnm](https://github.com/Schniz/fnm) (`fnm install 26`) or [nodejs.org](https://nodejs.org) | The Worker in `worker/` only. `just demo` does not need it. The justfile uses fnm when it is installed |
| pnpm | 12.16.1 | `corepack enable pnpm`, or [pnpm](https://pnpm.io/installation) | The Worker's dependencies |
| Ollama | Recent (tested: 0.40) | [Ollama](https://ollama.com/download), then `ollama pull nomic-embed-text:v1.5` | Embeddings, on the host |
| Pi | 1.5.0 (tested) | [Pi](https://pi.dev), then `pi` and `/login` | The answer model. Pi needs credentials for the provider of `PROFILE_RAG_PI_MODEL` |

`just demo` checks Docker, uv, Ollama, and Pi before it changes anything, and prints one fix for each missing item.

## Quick start

```sh
git clone <this repository> profile-rag-app
cd profile-rag-app
cp .env.example .env
just demo
```

`just demo` runs these steps:

1. It checks Docker and uv. Then it installs the backend dependencies (`uv sync`) and checks the database port, Ollama and the embedding model, Pi, and the corpus. If anything is missing, it lists every problem and stops before it starts the database.
2. It starts Postgres in Docker and waits until the database is healthy.
3. It applies the migrations in `db/migrations/`.
4. It indexes `corpus-example/` with the default embedding model, `nomic-v15`.
5. It prints how to ask a question:

```text
The corpus is indexed. Corpus: corpus-example. Embedding model: nomic-v15.

Ask a question from the terminal:

  cd backend && uv run profile-rag ask --model nomic-v15 "Your question?"
```

The database keeps running. `just db-down` stops it, and `just db-reset` also deletes its data. You can run `just demo` again at any time: the database stays up, applied migrations are skipped, and unchanged chunks are skipped.

A cold first run also downloads Python 3.14, the Python packages, and the Postgres image.

The demo uses the database in `.env`. `just demo` replaces the chunks of that database's table with the corpus it indexes.

### Run a second checkout next to the first

docker compose finds a container by its project name only. Two checkouts with the same `COMPOSE_PROJECT_NAME` would share one database container and its data. `just demo` refuses to start when another checkout owns the project name. To run a second checkout, give it its own name and ports in its `.env`:

```sh
COMPOSE_PROJECT_NAME=profile-rag-second
DB_PORT=5453
DATABASE_URL=postgresql://profile_rag:profile_rag@127.0.0.1:5453/profile_rag
TEST_DATABASE_URL=postgresql://profile_rag:profile_rag@127.0.0.1:5453/profile_rag_test
```

## Use the app

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
- `agent` first has the model grade the retrieved chunks, and rewrites the question once if they are not relevant. It makes 2 Pi calls, or 4 with a rewrite. `ask` and the Worker use the plain chain. [`docs/eval/agent-decision.md`](eval/agent-decision.md) gives the reason.

The exit codes of `ask` and `agent` are 0 for an answer or `Unknown.`, 2 for a setup error (no table, Ollama not running) or a failed evidence search, 3 when Pi fails, and 4 when the model's reply is rejected (not the answer format, or a citation of a chunk it was not given). `agent` also exits with 5 if the graph exceeds its step limit, which means a bug.

Run `uv run profile-rag --help` for every command, including `index`, `eval`, `compare-agent`, `hnsw`, and `corpus`. "Index and search", "Generation", and "Agent graph" in [`docs/ARCHITECTURE.md`](ARCHITECTURE.md) explain them.

### The HTTP API

The only HTTP API is the Cloudflare Worker in `worker/`. It answers with Workers AI, not Pi, and searches the `bge-m3` chunks in Neon. `just worker-dev` runs it locally with `wrangler dev`; copy `worker/.dev.vars.example` to `worker/.dev.vars` first. [`DEPLOY.md`](DEPLOY.md) covers its routes, error codes, and deployment.

## What the corpus covers

`corpus-example/` is a fictional profile of Dana Whitlock. Every name, company, URL, and address in it is invented. It has 3 approved sources and 15 chunks: a resume, a project README (`lantern-queue`), and a blog post. It also registers a LinkedIn export and a note that are not approved, so the index skips them. The tests, the demo, and the example evaluation set all use this corpus.

A real corpus lives in `corpus/`, which Git ignores except for its README. Each source has a kind: `resume`, `github`, `portfolio`, `blog`, `linkedin`, or `note`. Each source also has an attribution, which tells the model whose claim a sentence is:

| Attribution | Meaning |
| --- | --- |
| `self_authored` | The person wrote or built it, and the source shows that. |
| `self_report` | The person's own description of their work, such as a resume. |
| `project_doc` | Documentation of a project that the person took part in. |
| `third_party` | Someone else's statement about the person. |

The answer names the attribution ("According to Dana's resume, ..."), so the app does not turn a project's documentation into a claim about what the person did. Visibility is `public`, or `sanitized` for a reviewed copy of private material with confidential details removed. Only approved sources with one of these two visibilities can appear in a search result.

## Use your own corpus

1. Register, copy, and review each source with `profile-rag corpus add`, `corpus snapshot`, and `corpus review`. [`corpus/README.md`](../corpus/README.md) gives the steps. A source moves from `candidate` to `in_review` to `approved` (or `rejected`), and only approved sources are indexed. Approval records the file's SHA-256 hash. If the file changes later, the approval is stale, and `index` and `just demo` stop until you review the source again.

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
   just demo corpus
   ```

   `corpus` is the folder that holds `sources.yaml` and `approved/`, relative to the repository root. If the folder has no approved source, `just demo` stops before it touches the database, with a message that gives the number of registered sources and how to approve one. It does not print source titles or text.

To index without the demo's checks, run `just index corpus`. Add the model name, as in `just index corpus qwen3-06b`, to index the second embedding model (pull `qwen3-embedding:0.6b` first). Each model has its own table.

A new splitter version (`SPLITTER_VERSION` in `backend/src/profile_rag/corpus/splitting.py`) changes every chunk ID. After that change, run `just index` again for each model and each corpus. The command replaces every row and reports them as added and deleted.

## Evaluation

`profile-rag eval` runs a labeled question set through one retriever and reports recall@k and MRR:

```sh
cd backend
uv run profile-rag eval --set example --model nomic-v15 --mode hybrid -k 3
```

`--set example` uses the fictional questions in `eval-example/dev.jsonl` over `corpus-example/`. `--set dev` and `--set heldout` use private questions in `eval/` over `corpus/`. Index the set's corpus with the same model first. The command writes per-question detail to `eval/results/` (Git ignores it), a report to `docs/eval/`, and the comparison table [`docs/eval/<set>-comparison.md`](eval/example-comparison.md). Committed reports hold only numbers and question IDs.

Results on the example set (14 questions: 12 answerable, 2 unanswerable; 15 chunks), from [`docs/eval/example-comparison.md`](eval/example-comparison.md):

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

One question moves recall by 0.083, so these numbers test the pipeline and do not rank the models. At k = 8, each question gets more than half of the corpus. On the same set, with hybrid k = 3, the agent graph and the chain both answered 10 of the 12 answerable questions and both answered the 2 unanswerable ones with `Unknown.`, but the agent used 36 Pi calls and the chain 14 ([`docs/eval/agent-vs-chain-example-nomic-v15.md`](eval/agent-vs-chain-example-nomic-v15.md)). HNSW and exact search return the same results on this corpus ([`docs/eval/hnsw-example-nomic-v15.md`](eval/hnsw-example-nomic-v15.md)).

The dev set has 39 questions (34 answerable, 5 unanswerable) over the real corpus, so its questions stay private in `eval/`. At k = 8 with `nomic-v15`, dense and hybrid search both reach a recall of 0.779, and hybrid ranks the right chunk higher (MRR 0.611, against 0.512 for dense). MMR trails on both (0.632 and 0.474). [`docs/eval/dev-comparison.md`](eval/dev-comparison.md) lists the questions that each mode misses. `qwen3-06b` is not yet evaluated on the dev set, so the choice of the default embedding model is still open, and it ends with one check on the held-out set. "Evaluation" in [`docs/ARCHITECTURE.md`](ARCHITECTURE.md) gives the question format, the label rule, and the metrics.

## Testing

The tests need the database. Start it with `just demo` or `just db-up`, then run:

```sh
just lint        # ruff and oxlint, formatting, and the Worker contract check
just typecheck   # pyrefly and tsc
just test        # pytest and the Worker's Vitest tests
```

The tests use fake embeddings, a fake chat model, and a fake Pi script. They need no Ollama and no Pi, and they call no model. To also run the tests that call a real Ollama and a real Pi, run `uv --directory backend run pytest --live`. The Pi tests check the installed version and its flags, and make one model call.

The Worker reads the prompt, the failure messages, and the excerpt rules from `worker/src/generated/backend-contract.json`, which `just worker-contract` writes from the Python code. `just lint` fails while that file is out of date.

Run `just` with no arguments to list every command.

## Privacy model

- **What stays local.** The corpus, the database, the embeddings (Ollama runs on your machine), the evaluation questions, and the per-question results. `corpus/` and `eval/` are in `.gitignore` except for their READMEs, and committed reports in `docs/eval/` hold only numbers and question IDs.
- **What leaves the machine.** For each question, Pi sends the question and the retrieved excerpts to the provider of `PROFILE_RAG_PI_MODEL`. Only approved chunks with visibility `public` or `sanitized` are retrieved, so the provider sees only text that any visitor may see.
- **Pi isolation.** Each question starts a new `pi --mode rpc` process in an empty temporary folder, with no session file, no tools, no extensions, skills, or context files, and no project `.pi/` settings. The prompt goes in on stdin, so it is not in the process list. "Pi isolation" in [`docs/ARCHITECTURE.md`](ARCHITECTURE.md) lists every flag and the reason for it.
- **Tracing guard.** Tracing to LangSmith is off unless `LANGSMITH_TRACING=true`. With tracing on, every command refuses to start unless `CORPUS_DIR=corpus-example` is set in the environment and `LANGSMITH_API_KEY` is set. Commands that read the database also check that the table holds exactly `corpus-example/`. Traces then hold run names, timings, and error class names, without inputs, outputs, metadata, or error messages. "Tracing" in [`docs/ARCHITECTURE.md`](ARCHITECTURE.md) explains the rules.
- **Logs and errors.** Error messages hold error classes and error codes. They never hold the question, the model's reply, or Pi's error output.

## Troubleshooting

| Problem | Fix |
| --- | --- |
| `Port 5443 (the database) is in use` | Stop the program that uses the port. `lsof -nP -iTCP:<port> -sTCP:LISTEN` names it. Or set `DB_PORT` in `.env`. Use the same port in `DATABASE_URL` and `TEST_DATABASE_URL` as in `DB_PORT`. |
| `Another checkout (...) uses the same Compose project name` | Set `COMPOSE_PROJECT_NAME` and a free `DB_PORT` in this checkout's `.env`, as in [Run a second checkout next to the first](#run-a-second-checkout-next-to-the-first). |
| `Ollama does not answer at http://127.0.0.1:11434` | Start Ollama (`ollama serve`, or open the Ollama app), or set `OLLAMA_URL` in `.env`. |
| `Ollama has no nomic-embed-text:v1.5` | Run `ollama pull nomic-embed-text:v1.5`. |
| `Ollama's nomic-embed-text:v1.5 has digest ..., but vectorstore/models.py expects ...` | The tag now points at other weights. Vectors from the two versions do not match. `backend/src/profile_rag/vectorstore/models.py` explains how to move to new weights. |
| `Pi ('pi') is not on PATH` | Install Pi and log in (`pi`, then `/login`), or set `PROFILE_RAG_PI_BIN` in `.env` to its path. |
| `ask` exits with 3 (Pi failed) | Pi started but the model call failed, usually because Pi has no credentials for the provider of `PROFILE_RAG_PI_MODEL`. Run `pi` and `/login`. `just demo` cannot check credentials without a model call. |
| `ask` reports a Pi timeout | The model took longer than `PROFILE_RAG_PI_TIMEOUT_SECONDS` (default 120). Raise it in `.env`. |
| `has no approved source` | Approve a source (see [Use your own corpus](#use-your-own-corpus)), or run `just demo` with no argument. |
| `just lint` fails because `backend-contract.json` is out of date | Run `just worker-contract`, then commit the regenerated file with the Python change. |
| `just test` fails to connect to the database | Run `just db-up`. Check that `TEST_DATABASE_URL` uses `DB_PORT`. |
| `fnm has no Node 26` | Run `fnm install 26`. |

## Not covered

By default this app runs on one machine, for its owner. [`DEPLOY.md`](DEPLOY.md) covers the public setup: a Cloudflare Worker (`worker/`) behind a website's server-side proxy, with Workers AI instead of Pi, an answer cache, an access key, and hourly and daily question limits. These points stay open:

- Provider access: run locally, every question spends the owner's Pi credentials. The Python code has no server, so Pi stays on your machine.
- Concurrency: the Worker's limits count questions, not concurrent requests.
- Logging: the Worker's logs stay in Cloudflare's Workers Logs.
- Network exposure: the Worker sets up no CORS. The deployed setup relies on the browser never calling it directly.

## Layout

| Path | Contents |
| --- | --- |
| `backend/` | The `profile-rag` command-line tool: corpus, index, search, answers, evaluation (Python) |
| `worker/` | The HTTP API, a Cloudflare Worker (TypeScript). [`DEPLOY.md`](DEPLOY.md) |
| `db/` | Database initialization and migrations |
| `scripts/` | `demo.sh`, which `just demo` runs |
| `corpus/` | The real corpus. Git ignores everything except its README |
| `corpus-example/` | The fictional corpus for tests and the demo |
| `eval/` | Private evaluation questions and results. Git ignores everything except its README |
| `eval-example/` | Fictional evaluation questions over `corpus-example/` |
| `docs/` | [`ARCHITECTURE.md`](ARCHITECTURE.md), and evaluation reports in `docs/eval/` |
