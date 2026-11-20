# One entry point for common tasks. Run `just` to list them.
#
# Backend commands run through uv inside backend/.
# Frontend commands run through pnpm inside frontend/, on the Node version
# pinned in .node-version (via fnm when it is installed).
#
# Every recipe runs from the repository root, wherever you call `just` from.

set shell := ["bash", "-euo", "pipefail", "-c"]

# A port comes from the environment, else from .env, else a default. The
# backend reads .env the same way (settings.py), and docker compose reads it
# for compose.yaml. `.env` is not loaded into the recipes' environment, so a
# recipe sees only what it sets itself.
DB_PORT := env("DB_PORT", `sed -n 's/^DB_PORT=//p' .env 2>/dev/null | tail -n 1 | grep . || echo 5443`)
API_PORT := env("API_PORT", `sed -n 's/^API_PORT=//p' .env 2>/dev/null | tail -n 1 | grep . || echo 8000`)
FRONTEND_PORT := env("FRONTEND_PORT", `sed -n 's/^FRONTEND_PORT=//p' .env 2>/dev/null | tail -n 1 | grep . || echo 5173`)

node_version := trim(read(".node-version"))
fnm := `command -v fnm || true`
node_exec := if fnm != "" { "fnm exec --using=" + node_version + " " } else { "" }

uv := "uv --directory backend"
pnpm := node_exec + "pnpm --dir frontend"
compose := "docker compose"
api_types := justfile_directory() / "frontend/src/api/schema.d.ts"

# List available commands
default:
    @{{ just_executable() }} --justfile {{ justfile() }} --list --unsorted

# Install backend and frontend dependencies, create .env if missing
setup:
    @test -f .env || (cp .env.example .env && echo "Created .env from .env.example")
    {{ uv }} sync
    {{ pnpm }} install

# --- Database ---------------------------------------------------------------

# Start Postgres + pgvector on 127.0.0.1:$DB_PORT (default 5443) and wait until healthy
[group('database')]
db-up:
    {{ compose }} up -d --wait db

# Stop Postgres (data is kept in the volume)
[group('database')]
db-down:
    {{ compose }} down

# Stop Postgres and DELETE all its data
[group('database')]
db-reset:
    {{ compose }} down --volumes

# Follow Postgres logs
[group('database')]
db-logs:
    {{ compose }} logs -f db

# Open psql on the main database
[group('database')]
psql:
    {{ compose }} exec db psql -U profile_rag -d profile_rag

# Open psql on the test database
[group('database')]
psql-test:
    {{ compose }} exec db psql -U profile_rag -d profile_rag_test

# Apply pending db/migrations to the main database
[group('database')]
migrate:
    {{ uv }} run python -m profile_rag.db.migrate

# CORPUS is relative to the repository root, for example
# `just index corpus-example qwen3-06b`.
[doc('Embed approved chunks into the vector store (needs db-up, migrate, Ollama)')]
[group('database')]
index corpus="corpus" model="nomic-v15":
    {{ uv }} run profile-rag index --model {{ model }} --root {{ absolute_path(corpus) }}

# --- Deploy -----------------------------------------------------------------

# Build the deploy image locally (docs/DEPLOY.md)
[group('deploy')]
image tag="profile-rag-api":
    docker build -t {{ tag }} .

# Push the committed API to a Hugging Face Space, e.g. `just space-push RielJ/profile-rag-api`
[group('deploy')]
space-push space:
    HF_SPACE={{ space }} scripts/space-push.sh

# --- Run --------------------------------------------------------------------

# Uses corpus-example/ by default; `just demo corpus` uses your own corpus.
[doc('Check prerequisites, start the database, index, and run the API and UI (Ctrl-C stops)')]
[group('run')]
demo corpus="corpus-example" model="nomic-v15":
    @CORPUS_DIR="{{ corpus }}" MODEL="{{ model }}" DB_PORT="{{ DB_PORT }}" \
        API_PORT="{{ API_PORT }}" FRONTEND_PORT="{{ FRONTEND_PORT }}" NODE_EXEC="{{ node_exec }}" \
        scripts/demo.sh

# Run the API on 127.0.0.1:$API_PORT (default 8000) with auto-reload
[group('run')]
backend:
    {{ uv }} run uvicorn profile_rag.api.app:app --reload --host 127.0.0.1 --port {{ API_PORT }}

# Run the Vite dev server on 127.0.0.1:$FRONTEND_PORT (default 5173)
[group('run')]
frontend:
    BACKEND_URL="${BACKEND_URL:-http://127.0.0.1:{{ API_PORT }}}" \
        {{ pnpm }} exec vite --host 127.0.0.1 --port {{ FRONTEND_PORT }} --strictPort

# --- Quality ----------------------------------------------------------------

# Run all tests
[group('quality')]
test: test-backend test-frontend

# Run backend tests (needs `just db-up`)
[group('quality')]
test-backend:
    {{ uv }} run pytest

# Run frontend unit tests
[group('quality')]
test-frontend:
    {{ pnpm }} test

# Browser tests are kept out of `just test` because they need a downloaded
# browser. They start their own API server (fake Pi, fake embeddings,
# throwaway database; backend/tests/e2e_server.py) and Vite. Needs db-up.

# One-time: download the Chromium build Playwright uses
[group('quality')]
e2e-install:
    {{ pnpm }} exec playwright install chromium

# Run the Playwright browser tests (needs db-up and e2e-install)
[group('quality')]
e2e:
    {{ pnpm }} exec playwright test

# Lint backend (ruff) and frontend (oxlint); check the API types
[group('quality')]
lint: types-check
    {{ uv }} run ruff check .
    {{ uv }} run ruff format --check .
    {{ pnpm }} lint
    {{ pnpm }} format:check

# Auto-format backend and frontend
[group('quality')]
format:
    {{ uv }} run ruff check --fix .
    {{ uv }} run ruff format .
    {{ pnpm }} format

# Type-check backend (pyrefly) and frontend (tsc)
[group('quality')]
typecheck:
    {{ uv }} run pyrefly check
    {{ pnpm }} typecheck

# Lint, type-check, and test everything
[group('quality')]
check: lint typecheck test

# --- API types --------------------------------------------------------------
# The backend writes its OpenAPI schema to a temporary file (no server runs),
# and openapi-typescript turns it into TypeScript types for the frontend.

# Regenerate frontend/src/api/schema.d.ts from the API's OpenAPI schema
[group('api types')]
types: _openapi-ts

# Fail if frontend/src/api/schema.d.ts is out of date (fix: just types)
[group('api types')]
types-check:
    @{{ just_executable() }} --justfile {{ justfile() }} _openapi-ts --check || \
        { echo "frontend/src/api/schema.d.ts is out of date. Run \`just types\`." >&2; exit 1; }

# Write the schema and run openapi-typescript with extra FLAGS
[private]
_openapi-ts *flags:
    #!/usr/bin/env bash
    set -euo pipefail
    tmp="$(mktemp "${TMPDIR:-/tmp}/profile-rag-openapi.XXXXXX")"
    trap 'rm -f "$tmp"' EXIT
    {{ uv }} run python -m profile_rag.api.openapi "$tmp"
    {{ pnpm }} exec openapi-typescript "$tmp" --output "{{ api_types }}" {{ flags }}
