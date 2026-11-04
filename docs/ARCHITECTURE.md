# Architecture

profile-rag answers questions about Riel's professional experience from a reviewed corpus. This page describes the indexing pipeline, the retrieval step, the database tables, and the corpus policy.

## Pipeline

Indexing runs offline, from the command line. Search runs on every question.

```text
INDEXING (profile-rag index)

corpus/sources.yaml ──► approved entries only ──► one Document per source
                                                   (metadata: source_id, title,
                                                    attribution, url, visibility)
                                                         │
                                                         ▼
                             MarkdownHeaderTextSplitter (headings into metadata)
                                                         │
                                                         ▼
                          RecursiveCharacterTextSplitter (size and overlap in words)
                                                         │  chunk_id into metadata
                                                         ▼
        index(docs, SQLRecordManager, PGVectorStore, cleanup="full", source_id_key="source_id")
                                                         │
                        PrefixedEmbeddings ──► OllamaEmbeddings ("search_document: ...")
                                                         │
                                                         ▼
        Postgres: sources, chunks_<model>, upsertion_record

SEARCH (profile-rag search)

question ──► PrefixedEmbeddings ("search_query: ...")
         ──► retriever on chunks_<model>, filtered to visibility public or sanitized
               dense:  cosine distance (<=>), closest first
               mmr:    fetch_k closest, then k picked for closeness and variety
               hybrid: dense list + full-text list (plainto_tsquery), fused with RRF
         ──► chunks with their metadata
```

## Modules

All paths are under `backend/src/profile_rag/`.

| Module | Does |
| --- | --- |
| `corpus/registry.py` | Reads, validates, and writes `sources.yaml`. Enforces the review states and detects stale approvals by SHA-256. |
| `corpus/snapshot.py` | Copies a local file or URL into `approved/` (or `private/`) and records its revision. |
| `corpus/scan.py` | Flags secrets, email addresses, phone numbers, and internal hosts for the reviewer. |
| `corpus/documents.py` | Loads each approved snapshot as a `Document`, after it checks the file hash. |
| `corpus/splitting.py` | Splits documents at headings, then by size in words. Gives each chunk a stable `chunk_id`. |
| `corpus/sources_table.py` | Keeps the `sources` table equal to the approved registry entries. |
| `vectorstore/models.py` | The embedding models: Ollama tag, digest, dimensions, table name, and prefixes. |
| `vectorstore/prefixed.py` | `PrefixedEmbeddings`: adds the model's document or query prefix and checks every vector. |
| `vectorstore/store.py` | Creates a model's chunk table and opens it for indexing or for search. |
| `vectorstore/indexing.py` | Runs LangChain `index()` with a record manager per model. |
| `retrievers/dense.py` | `dense_retriever`, `mmr_retriever`, and the `VISIBLE` filter. |
| `retrievers/hybrid.py` | `HybridRetriever`: dense and keyword search fused with reciprocal rank fusion. |
| `db/` | Connection pool and the migration runner for `db/migrations/`. |
| `api/` | FastAPI application. `GET /api/health` reports database status. |
| `cli.py` | The `profile-rag` command: `corpus`, `chunks show`, `index`, and `search`. |

## Tables

Migrations in `db/migrations/` own one table, `sources`. It holds one row per approved source: `id`, `kind`, `title`, `url`, `visibility`, `attribution`, `revision`, `content_sha256`, `approved_at`, and `approved_by`. Candidate and rejected sources never reach the database. `0002_drop_phase1_chunks.sql` drops three older chunk tables that `0001_initial.sql` created.

LangChain owns the other tables:

- `chunks_nomic_v15` (768 dimensions) and `chunks_qwen3_06b` (1024 dimensions) come from `PGEngine.init_vectorstore_table()`. One table per model keeps one vector size per table.
- `upsertion_record` comes from `SQLRecordManager`. It serves every model, with one namespace per model, such as `pgvector/chunks_nomic_v15`.

A chunk table has these columns, in this order:

| Column | Type | Set by |
| --- | --- | --- |
| `langchain_id` | `text`, primary key | `index()`: a SHA-256 of the content and metadata |
| `content` | `text` | The chunk text |
| `embedding` | `vector(N)` | `PrefixedEmbeddings` |
| `content_tsv` | `tsvector` | `to_tsvector('pg_catalog.english', content)` on every insert |
| `source_id`, `chunk_id`, `attribution`, `visibility` | `text NOT NULL` | The matching metadata keys |
| `langchain_metadata` | `json` | Every other metadata key: `title`, `url`, `heading_path`, `ordinal` |

`backend/tests/test_vectorstore_table.py` reads this shape from `information_schema.columns`. If a `langchain-postgres` upgrade changes it, that test fails.

These langchain-postgres 0.0.19 behaviors shape the code:

- `index()` ignores `Document.id`. Its own hash becomes the row ID, so `chunk_id` is a metadata column.
- A store opened without a `HybridSearchConfig` leaves `content_tsv` empty, and the insert fails. The index store always has one.
- A search writes the query into the `HybridSearchConfig` it gets. `HybridRetriever` builds a new config for every query and searches a store opened without one.
- MMR treats `lambda_mult=0` as the default 0.5. `mmr_retriever` refuses 0 and always passes `fetch_k`.

## Corpus policy

The repository is public. The real corpus and evaluation data never enter Git.

| Path | Committed | Contents |
| --- | --- | --- |
| `corpus/` | Only `README.md` | Real sources, review notes, and private material |
| `eval/` | Only `README.md` | Evaluation questions and results. They quote and reveal corpus text |
| `corpus-example/` | Yes | A fictional profile (Dana Whitlock) for tests and demos |

Every source in `sources.yaml` has one state:

```text
candidate ──► in_review ──► approved
                   │
                   └──────► rejected (with a reason)
```

Only `approved` sources are indexed. Approval records the snapshot's SHA-256. If the file changes, the approval is stale: `profile-rag corpus status` exits with code 1, and `profile-rag index` stops with an error that names the source. Review the source again to fix it.

Each source has an attribution, so an answer does not turn a project's features into a claim about what Riel did:

| Attribution | Meaning |
| --- | --- |
| `self_authored` | Riel wrote or built it, and the source shows that. |
| `self_report` | Riel's own description of his work, such as a resume. |
| `project_doc` | Documentation of a project Riel took part in. |
| `third_party` | Someone else's statement about Riel. |

Visibility is `public` or `sanitized`. A sanitized source is a reviewed copy of private material with confidential details removed. Raw private material stays in `corpus/private/` and is never indexed. The retrievers filter on visibility, so only these two values can reach a search result.

## Index and search

To index the corpus, start Postgres, apply the migrations, and run `make index`:

```sh
make db-up
make migrate
make index                                  # corpus/ with nomic-v15
make index CORPUS_DIR=corpus-example        # the fictional example corpus
make index MODEL=qwen3-06b                  # the second model's table
```

`profile-rag index` checks that Ollama has the model with the digest in `vectorstore/models.py`, writes the approved entries to `sources`, and indexes every chunk. A second run with nothing changed reports every chunk as skipped. A source that lost approval loses its rows.

To search, give a question and a mode:

```sh
uv --directory backend run profile-rag search "Which databases has Dana used?" --mode hybrid -k 5
```

Each output line shows the rank, the cosine distance between the question and the chunk, the `chunk_id`, and the heading path. The distance is the same measure in every mode. It shows where MMR and hybrid search leave the dense order, but it is not the score those modes rank by.
