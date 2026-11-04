# backend

FastAPI service and `profile-rag` command-line tool. Run commands from the repository root through the `Makefile`; see the root `README.md`.

Source layout:

| Path | Contents |
| --- | --- |
| `src/profile_rag/settings.py` | Typed settings read from the root `.env` |
| `src/profile_rag/db/` | Connection pool and the migration runner |
| `src/profile_rag/api/` | FastAPI application and routes |
| `src/profile_rag/corpus/` | Source registry, snapshots, review scan, loading, and splitting |
| `src/profile_rag/vectorstore/` | Embedding models, prefixed embeddings, chunk tables, and indexing |
| `src/profile_rag/retrievers/` | Dense, MMR, and hybrid retrievers |
| `src/profile_rag/cli.py` | `profile-rag` command-line entry point |
| `tests/` | pytest tests |
