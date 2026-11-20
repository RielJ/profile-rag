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
pnpm_worker := node_exec + "pnpm --dir worker"
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
    {{ pnpm_worker }} install

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
# The public API is the Cloudflare Worker in worker/ (docs/DEPLOY.md).

# Set one Worker secret; wrangler prompts for the value, e.g. `just worker-secret ACCESS_KEY`
[group('deploy')]
worker-secret name:
    {{ pnpm_worker }} exec wrangler secret put {{ name }}

# Checks, then deploys with CORPUS_VERSION set from the chunks in Neon (so a
# re-index empties the answer cache), then pre-warms the suggested questions.
# Set NEON_URL as for worker-index-neon. Pre-warming needs ACCESS_KEY, from
# the environment or worker/.dev.vars; without it, that step is skipped.
# WORKER_URL overrides the workers.dev address read from wrangler's output
# (for a custom domain).
[doc('Check, deploy the Worker, and pre-warm its answer cache (needs NEON_URL)')]
[group('deploy')]
worker-deploy: worker-contract-check
    #!/usr/bin/env bash
    set -euo pipefail
    : "${NEON_URL:?Set NEON_URL to the Neon connection string (direct, not pooled)}"
    {{ pnpm_worker }} typecheck
    {{ pnpm_worker }} lint
    {{ pnpm_worker }} test
    version="$(DATABASE_URL="$NEON_URL" {{ pnpm_worker }} exec tsx scripts/corpus-version.ts)"
    echo "==> Deploying with CORPUS_VERSION=$version"
    # pipefail: a failed deploy fails the recipe here.
    out="$({{ pnpm_worker }} exec wrangler deploy --var "CORPUS_VERSION:$version" 2>&1 | tee /dev/stderr)"
    # Not finding the address is no reason to fail a deploy that worked.
    url="${WORKER_URL:-$(grep -oE 'https://[a-z0-9.-]+\.workers\.dev' <<<"$out" | head -1 || true)}"
    if [ -z "$url" ]; then
        echo "==> Deployed, but wrangler's output has no workers.dev address, so the cache is not pre-warmed." >&2
        echo "    Pre-warm by hand: just worker-prewarm https://<worker>.<subdomain>.workers.dev" >&2
        echo "    Or deploy with WORKER_URL=https://... set." >&2
        exit 0
    fi
    just worker-prewarm "$url"

# Ask the deployed Worker the suggested questions in worker/prewarm-questions.txt
[group('deploy')]
worker-prewarm url:
    #!/usr/bin/env bash
    set -euo pipefail
    if [ -z "{{ url }}" ]; then echo "Give the Worker's address: just worker-prewarm https://..." >&2; exit 2; fi
    key="${ACCESS_KEY:-$(sed -n 's/^ACCESS_KEY=//p' worker/.dev.vars 2>/dev/null || true)}"
    if [ -z "$key" ]; then echo "No ACCESS_KEY (environment or worker/.dev.vars); skipping pre-warm." >&2; exit 0; fi
    WORKER_URL={{ url }} ACCESS_KEY="$key" {{ pnpm_worker }} exec tsx scripts/prewarm.ts

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
test: test-backend test-frontend test-worker

# Run backend tests (needs `just db-up`)
[group('quality')]
test-backend:
    {{ uv }} run pytest

# Run frontend unit tests
[group('quality')]
test-frontend:
    {{ pnpm }} test

# Run the Worker's tests (no network: the AI binding is faked)
[group('quality')]
test-worker:
    {{ pnpm_worker }} test

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

# Lint backend (ruff), frontend and Worker (oxlint); check generated files
[group('quality')]
lint: types-check worker-contract-check
    {{ uv }} run ruff check .
    {{ uv }} run ruff format --check .
    {{ pnpm }} lint
    {{ pnpm }} format:check
    {{ pnpm_worker }} lint
    {{ pnpm_worker }} format:check

# Auto-format backend and frontend
[group('quality')]
format:
    {{ uv }} run ruff check --fix .
    {{ uv }} run ruff format .
    {{ pnpm }} format
    {{ pnpm_worker }} format

# Type-check backend (pyrefly), frontend and Worker (tsc)
[group('quality')]
typecheck:
    {{ uv }} run pyrefly check
    {{ pnpm }} typecheck
    {{ pnpm_worker }} typecheck

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

# --- Worker contract --------------------------------------------------------
# The Worker reads the system prompt, failure messages, and excerpt rules from
# worker/src/generated/backend-contract.json, written from the Python code.

# Regenerate worker/src/generated/backend-contract.json from the backend
[group('worker')]
worker-contract:
    {{ uv }} run python scripts/export_contract.py ../worker/src/generated/backend-contract.json

# Fail if the Worker's contract is out of date (fix: just worker-contract)
[group('worker')]
worker-contract-check:
    @{{ uv }} run python scripts/export_contract.py ../worker/src/generated/backend-contract.json --check || \
        { echo "worker/src/generated/backend-contract.json is out of date. Run \`just worker-contract\`." >&2; exit 1; }

# Index the corpus with bge-m3 into Neon, the database the Worker searches.
# Set NEON_URL to Neon's direct (not pooled) connection string. Embeds with
# Workers AI through your wrangler login (about 75 neurons). Runs the
# migrations, syncs `sources`, and adds, updates, or deletes chunks in
# chunks_bge_m3 so it matches the corpus; other chunk tables are untouched.
[doc('Index the corpus with bge-m3 into Neon (needs NEON_URL)')]
[group('worker')]
worker-index-neon corpus="corpus":
    #!/usr/bin/env bash
    set -euo pipefail
    : "${NEON_URL:?Set NEON_URL to the Neon connection string (direct, not pooled)}"
    # whoami refreshes wrangler's OAuth token before `auth token` prints it.
    export PROFILE_RAG_CF_ACCOUNT_ID="$({{ pnpm_worker }} exec wrangler whoami --json | jq -r '.accounts[0].id')"
    export PROFILE_RAG_CF_API_TOKEN="$({{ pnpm_worker }} exec wrangler auth token --json | jq -r .token)"
    export DATABASE_URL="$NEON_URL"
    {{ uv }} run python -m profile_rag.db.migrate
    {{ uv }} run profile-rag index --model bge-m3 --root {{ absolute_path(corpus) }}

# Check the Worker's search returns the same chunks as Python's hybrid
# retriever on the dev set. Local database with bge-m3 indexed
# (`just index corpus bge-m3`); embeds the 39 questions once with Workers AI
# (PROFILE_RAG_CF_ACCOUNT_ID and PROFILE_RAG_CF_API_TOKEN).
[group('worker')]
worker-parity fixture="../eval/results/retrieval-parity-fixture.json":
    {{ uv }} run python scripts/retrieval_fixture.py {{ fixture }}
    DATABASE_URL="$({{ uv }} run python -c 'from profile_rag.settings import get_settings; print(get_settings().database_url)')" \
        {{ pnpm_worker }} exec tsx scripts/retrieval-parity.ts {{ fixture }}
