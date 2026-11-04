# profile-rag

A question-answering application about Riel's professional experience. It answers only from a reviewed corpus of sources, and each answer cites its sources.

The backend indexes approved sources into Postgres with pgvector, through LangChain and local Ollama embeddings. It retrieves chunks with dense, MMR, or hybrid (dense plus full-text) search, and measures retrieval with recall@k and MRR on labeled question sets. [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) describes the pipeline, the modules, the tables, and the corpus policy.

## Requirements

| Tool | Version | Use |
| --- | --- | --- |
| Docker | Recent Docker Desktop or OrbStack | Runs Postgres with pgvector |
| uv | 0.13.6 or later | Installs Python 3.14 and the backend dependencies |
| Node | 26 LTS | Runs the frontend. `.node-version` pins it (`fnm install 26`) |
| pnpm | 12.16.1 | Installs the frontend dependencies |
| Ollama | Recent | Computes embeddings on the host |

## Set up

1. Install the dependencies and create `.env` from `.env.example`:

   ```sh
   make setup
   ```

2. Start Postgres and apply the migrations:

   ```sh
   make db-up      # Postgres + pgvector on 127.0.0.1:5443
   make migrate
   ```

   The Compose project is named `profile-rag-app`, so its container and volume do not clash with other projects. On first start, `db/init/` also creates the test database, `profile_rag_test`.

3. Pull the embedding model:

   ```sh
   ollama pull nomic-embed-text:v1.5
   ```

   To compare models, also pull `qwen3-embedding:0.6b` and index with `MODEL=qwen3-06b`.

## Index and search

To index the real corpus in `corpus/`, run `make index`. To try the app without a corpus, index the fictional example:

```sh
make index CORPUS_DIR=corpus-example
```

The command prints the counts of chunks added, updated, skipped, and deleted. A second run with no changes skips every chunk.

Then ask a question:

```sh
uv --directory backend run profile-rag search "Which databases has Dana used?" --mode hybrid -k 5
```

`--mode` is `dense` (the default), `mmr`, or `hybrid`. `--model` selects the table: `nomic-v15` (the default) or `qwen3-06b`.

## Evaluate retrieval

`profile-rag eval` runs a labeled question set through a retriever and reports recall@k and MRR:

```sh
uv --directory backend run profile-rag eval --set example --model nomic-v15 --mode dense -k 8
```

`--set` is `example` (fictional questions in `eval-example/dev.jsonl`, over `corpus-example/`), `dev`, or `heldout` (private questions in `eval/`, over `corpus/`). `--model`, `--mode`, and `-k` work as in `search`. Index the set's corpus with the same model first.

The command prints the numbers and writes three files: per-question detail in `eval/results/` (gitignored), a report for the run in `docs/eval/`, and the comparison table for the set, `docs/eval/<set>-comparison.md`. Committed reports contain only numbers and question IDs. "Evaluation" in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) gives the question format, the label rule, and the metrics.

To add a source to the corpus, follow [`corpus/README.md`](corpus/README.md). Each source is registered, copied into `corpus/approved/`, and reviewed before it can be indexed.

## Run the app

Run the API and the UI in two terminals:

```sh
make backend    # API on http://127.0.0.1:8000 (OpenAPI UI at /docs)
make frontend   # UI on http://127.0.0.1:5173
```

## Check the code

```sh
make lint
make typecheck
make test       # needs make db-up
```

The tests use fake embeddings and need no Ollama. To also run the tests that call a real Ollama, run `uv --directory backend run pytest --live`.

Run `make` with no arguments to list every command.

## Layout

| Path | Contents |
| --- | --- |
| `backend/` | FastAPI service and the `profile-rag` command-line tool (Python) |
| `frontend/` | React UI (TypeScript, Vite) |
| `db/` | Database initialization and migrations |
| `corpus/` | The real corpus. Git ignores everything except its README |
| `corpus-example/` | A fictional corpus for tests and demos |
| `eval/` | Evaluation questions and results. Git ignores everything except its README |
| `eval-example/` | Fictional evaluation questions over `corpus-example/` |
| `docs/` | Architecture, and evaluation reports in `docs/eval/` |

## Privacy

This repository is public. The real corpus and evaluation data are never committed. See "Corpus policy" in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).
