# profile-rag

profile-rag answers questions about one person's professional experience. It answers only from sources that a person has reviewed and approved, and every answer cites the sources it uses. If the approved sources do not answer a question, the app says so instead of guessing.

The repository ships a fictional corpus about Dana Whitlock, so you can run the whole app without private data.

## How it works

1. **Review.** Each source (a resume, a project README, a blog post) is registered in `sources.yaml` and copied into the corpus. A person reads it, answers a five-item privacy checklist, and approves or rejects it. Approval records the file's SHA-256 hash. If the file changes later, the approval becomes stale and indexing stops until someone reviews the file again.
2. **Index.** The approved sources are split at headings and then by size. Ollama embeds the chunks on your machine, and Postgres stores them with pgvector.
3. **Retrieve.** A question finds chunks by dense search, MMR, or hybrid search (dense results fused with a keyword search). Only chunks from approved sources with visibility `public` or `sanitized` can come back.
4. **Answer.** The answer model gets the question and the retrieved chunks through [Pi](https://pi.dev). It must cite chunks it was given and say whose claim each sentence is ("According to Dana's resume, ..."). If the reply does not match the answer format, or it cites a chunk that was not retrieved, the app rejects the reply.

A React page, a FastAPI service, and the `profile-rag` command-line tool all ask questions through the same answer chain. An experimental agent graph (`profile-rag agent`) grades the retrieved chunks and can rewrite the question once.

## Quick start

You need Docker with Compose v2, uv, Node 26 with pnpm, Ollama with `nomic-embed-text:v1.5`, and Pi logged in to a model provider. [`docs/USAGE.md`](docs/USAGE.md) lists the versions and how to install each one.

```sh
git clone <this repository> profile-rag-app
cd profile-rag-app
cp .env.example .env
make demo
```

`make demo` checks the tools, starts Postgres in Docker, applies the migrations, indexes `corpus-example/`, and starts the API and the frontend. It then prints the address to open, `http://127.0.0.1:5173`. If a tool is missing, `make demo` stops before it changes anything and prints the fix.

Ask a question such as "What web framework did Dana use for the shipment-tracking API?". To use your own sources, review them with `profile-rag corpus review` and run `make demo CORPUS_DIR=corpus`. [`corpus/README.md`](corpus/README.md) gives the steps.

## Privacy

The corpus, the database, the embeddings, and the evaluation questions stay on your machine. Git ignores `corpus/` and `eval/`, except for their READMEs. For each question, Pi sends the question and the retrieved excerpts to the model provider. Each question runs in a new Pi process with no tools, no session file, and no project settings. Logs hold request IDs and error codes, never questions or answers.

The app runs on one machine, for its owner. It has no authentication and no rate limit, so do not deploy it publicly without a separate review. "Not covered" in [`docs/USAGE.md`](docs/USAGE.md#not-covered) lists what that review has to cover.

## Development

```sh
make lint        # ruff, oxlint, formatting, and the API types check
make typecheck   # pyrefly and tsc
make test        # pytest and Vitest. Needs the database (make db-up)
make e2e         # Playwright browser tests. Run make e2e-install once first
```

The tests use fake embeddings and a fake Pi, so they need neither Ollama nor Pi. Run `make` with no arguments to list every command.

## Documentation

| Document | Contents |
| --- | --- |
| [`docs/USAGE.md`](docs/USAGE.md) | Prerequisites, the web page, the command line, the API, your own corpus, evaluation results, testing, and troubleshooting |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | The pipeline, the modules, the database tables, Pi isolation, tracing, and the corpus policy |
| [`corpus/README.md`](corpus/README.md) | How to add, copy, and review a source |
| [`docs/eval/`](docs/eval/) | Retrieval and agent evaluation reports |

## Layout

| Path | Contents |
| --- | --- |
| `backend/` | The FastAPI service and the `profile-rag` command-line tool (Python 3.14) |
| `frontend/` | The React page (TypeScript, Vite) and its browser tests |
| `db/` | Database initialization and migrations |
| `corpus/`, `eval/` | The real corpus and the private evaluation questions. Git ignores both, except for their READMEs |
| `corpus-example/`, `eval-example/` | The fictional corpus and its evaluation questions, for the demo and the tests |
| `scripts/` | `demo.sh`, which `make demo` runs |
