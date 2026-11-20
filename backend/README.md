# backend

The `profile-rag` command-line tool: corpus review, indexing, search, answers, and evaluation. It has no HTTP server; the HTTP API is the Worker in `worker/`. Run commands from the repository root through the `justfile` (`just`); see the root `README.md`.

Source layout:

| Path | Contents |
| --- | --- |
| `src/profile_rag/settings.py` | Typed settings read from the root `.env` |
| `src/profile_rag/db/` | The migration runner |
| `src/profile_rag/corpus/` | Source registry, snapshots, review scan, loading, and splitting |
| `src/profile_rag/vectorstore/` | Embedding models, prefixed embeddings, chunk tables, and indexing |
| `src/profile_rag/retrievers/` | Dense, MMR, and hybrid retrievers |
| `src/profile_rag/rag.py`, `answers.py` | The answer chain's result or error as response JSON |
| `scripts/export_contract.py` | Writes the Worker's `backend-contract.json` |
| `src/profile_rag/cli.py` | `profile-rag` command-line entry point |
| `tests/` | pytest tests |
