-- 0002: drop the chunk and vector tables of 0001.
--
-- Specification: docs/ARCHITECTURE.md, section "Tables".
-- Tests:         backend/tests/test_migrations.py
-- Apply:         make migrate          (to the main database)
--                make test-backend     (tests apply it to throwaway databases)
--
-- PGVectorStore creates and owns one chunk table per embedding model (for
-- example chunks_nomic_v15), so these three tables are unused. Only
-- `sources` stays ours. 0001_initial.sql cannot be edited (the
-- runner stores its checksum), so this file drops them instead.
--
-- The vector extension stays: PGVectorStore tables need it.
--
-- Order matters: chunk_embeddings references both other tables.
-- The DROP of `chunks` also drops its index chunks_tsv_idx.

DROP TABLE chunk_embeddings;
DROP TABLE embedding_configs;
DROP TABLE chunks;
