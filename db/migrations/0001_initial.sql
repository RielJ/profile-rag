-- 0001: initial schema.
--
-- Specification: docs/ARCHITECTURE.md, section "Tables".
-- Tests:         backend/tests/test_migrations.py
-- Apply:         make migrate          (to the main database)
--                make test-backend     (tests apply it to throwaway databases)
--
-- The runner (backend/src/profile_rag/db/migrate.py) runs this whole file in
-- one transaction. If any statement fails, nothing in the file is kept.
--
-- Once this file has been applied to your main database, do not edit it:
-- the runner stores its checksum and refuses to continue if it changes.
-- While developing, `make db-reset && make db-up && make migrate` starts over.

CREATE EXTENSION IF NOT EXISTS vector;

-- One row per approved source document.
CREATE TABLE sources (
  id text PRIMARY KEY, -- stable, e.g. 'resume-2026'
  kind text NOT NULL CHECK (
    kind IN (
      'resume',
      'github',
      'portfolio',
      'blog',
      'linkedin',
      'note'
    )
  ),
  title text NOT NULL,
  url text, -- public link, NULL for private notes
  visibility text NOT NULL CHECK (visibility IN ('public', 'sanitized')),
  attribution text NOT NULL CHECK (
    attribution IN (
      'self_authored',
      'self_report',
      'project_doc',
      'third_party'
    )
  ),
  revision text NOT NULL, -- commit SHA, snapshot date, or file hash
  content_sha256 text NOT NULL,
  approved_at timestamptz NOT NULL,
  approved_by text NOT NULL
);

-- One row per chunk produced by a specific chunking configuration.
CREATE TABLE chunks (
  id text PRIMARY KEY, -- source_id + chunker version + content hash
  source_id text NOT NULL REFERENCES sources (id) ON DELETE CASCADE,
  chunker_version text NOT NULL,
  ordinal int NOT NULL,
  heading_path text[] NOT NULL DEFAULT '{}',
  text text NOT NULL,
  char_start int NOT NULL,
  char_end int NOT NULL,
  tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED,
  UNIQUE (source_id, chunker_version, ordinal)
);

CREATE INDEX chunks_tsv_idx ON chunks USING gin (tsv);

-- One row per embedding configuration you have tried.
CREATE TABLE embedding_configs (
  id text PRIMARY KEY, -- e.g. 'nomic-v1.5-768'
  model text NOT NULL, -- 'nomic-embed-text:v1.5'
  model_digest text NOT NULL, -- full digest from `ollama show`
  dimensions int NOT NULL,
  query_template text NOT NULL, -- 'search_query: {text}'
  doc_template text NOT NULL, -- 'search_document: {text}'
  created_at timestamptz NOT NULL DEFAULT now()
);

-- One row per (chunk, embedding config). The column has no fixed
-- dimension so that 768- and 1024-dimension models can share the table.
-- pgvector supports this, but each index must cover one dimension only,
-- using a partial expression index per config.
CREATE TABLE chunk_embeddings (
  chunk_id text NOT NULL REFERENCES chunks (id) ON DELETE CASCADE,
  config_id text NOT NULL REFERENCES embedding_configs (id),
  embedding vector NOT NULL,
  PRIMARY KEY (chunk_id, config_id)
);
