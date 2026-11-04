# One entry point for common tasks. Run `make` or `make help` to list them.
#
# Backend commands run through uv inside backend/.
# Frontend commands run through pnpm inside frontend/, on the Node version
# pinned in .node-version (via fnm when it is installed).

SHELL := /bin/bash
.DEFAULT_GOAL := help

ROOT := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
NODE_VERSION := $(shell cat $(ROOT).node-version)
FNM := $(shell command -v fnm 2>/dev/null)
NODE_EXEC := $(if $(FNM),fnm exec --using=$(NODE_VERSION) ,)

UV := uv --directory backend
PNPM := $(NODE_EXEC)pnpm --dir frontend
COMPOSE := docker compose

# $(call env_value,NAME,DEFAULT): NAME from the command line or the
# environment, else from .env, else DEFAULT. The backend reads .env the same
# way (settings.py), and docker compose reads it for compose.yaml.
env_value = $(or $($(1)),$(shell sed -n 's/^$(1)=//p' $(ROOT).env 2>/dev/null | tail -n 1),$(2))
DB_PORT := $(call env_value,DB_PORT,5443)
API_PORT := $(call env_value,API_PORT,8000)
FRONTEND_PORT := $(call env_value,FRONTEND_PORT,5173)

.PHONY: help setup db-up db-down db-reset db-logs psql psql-test migrate index \
        demo backend frontend test test-backend test-frontend \
        e2e e2e-install lint format typecheck check types types-check

help: ## List available commands
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

setup: ## Install backend and frontend dependencies, create .env if missing
	@test -f .env || (cp .env.example .env && echo "Created .env from .env.example")
	$(UV) sync
	$(PNPM) install

# --- Database -------------------------------------------------------------

db-up: ## Start Postgres + pgvector on 127.0.0.1:$DB_PORT (default 5443) and wait until healthy
	$(COMPOSE) up -d --wait db

db-down: ## Stop Postgres (data is kept in the volume)
	$(COMPOSE) down

db-reset: ## Stop Postgres and DELETE all its data
	$(COMPOSE) down --volumes

db-logs: ## Follow Postgres logs
	$(COMPOSE) logs -f db

psql: ## Open psql on the main database
	$(COMPOSE) exec db psql -U profile_rag -d profile_rag

psql-test: ## Open psql on the test database
	$(COMPOSE) exec db psql -U profile_rag -d profile_rag_test

migrate: ## Apply pending db/migrations to the main database
	$(UV) run python -m profile_rag.db.migrate

# Override on the command line, for example
# `make index CORPUS_DIR=corpus-example MODEL=qwen3-06b`. A relative
# CORPUS_DIR is relative to the repository root.
CORPUS_DIR ?= corpus
MODEL ?= nomic-v15
index: ## Embed approved chunks into the vector store (needs db-up, migrate, Ollama)
	$(UV) run profile-rag index --model $(MODEL) --root $(abspath $(CORPUS_DIR))

# --- Run ------------------------------------------------------------------

# `make demo` uses corpus-example/ unless CORPUS_DIR is given on the command
# line or in the environment: `make demo CORPUS_DIR=corpus`.
DEMO_CORPUS_DIR := $(if $(filter command line environment,$(origin CORPUS_DIR)),$(CORPUS_DIR),corpus-example)
demo: ## Check prerequisites, start the database, index, and run the API and UI (Ctrl-C stops)
	@CORPUS_DIR="$(DEMO_CORPUS_DIR)" MODEL="$(MODEL)" DB_PORT="$(DB_PORT)" \
		API_PORT="$(API_PORT)" FRONTEND_PORT="$(FRONTEND_PORT)" NODE_EXEC="$(NODE_EXEC)" \
		$(ROOT)scripts/demo.sh

backend: ## Run the API on 127.0.0.1:$API_PORT (default 8000) with auto-reload
	$(UV) run uvicorn profile_rag.api.app:app --reload --host 127.0.0.1 --port $(API_PORT)

frontend: ## Run the Vite dev server on 127.0.0.1:$FRONTEND_PORT (default 5173)
	BACKEND_URL="$${BACKEND_URL:-http://127.0.0.1:$(API_PORT)}" \
		$(PNPM) exec vite --host 127.0.0.1 --port $(FRONTEND_PORT) --strictPort

# --- Quality --------------------------------------------------------------

test: test-backend test-frontend ## Run all tests

test-backend: ## Run backend tests (needs `make db-up`)
	$(UV) run pytest

test-frontend: ## Run frontend unit tests
	$(PNPM) test

# Browser tests: kept out of `make test` because they need a downloaded
# browser. They start their own API server (fake Pi, fake embeddings,
# throwaway database; backend/tests/e2e_server.py) and Vite. Needs db-up.
e2e-install: ## One-time: download the Chromium build Playwright uses
	$(PNPM) exec playwright install chromium

e2e: ## Run the Playwright browser tests (needs db-up and e2e-install)
	$(PNPM) exec playwright test

lint: types-check ## Lint backend (ruff) and frontend (oxlint); check the API types
	$(UV) run ruff check .
	$(UV) run ruff format --check .
	$(PNPM) lint
	$(PNPM) format:check

format: ## Auto-format backend and frontend
	$(UV) run ruff check --fix .
	$(UV) run ruff format .
	$(PNPM) format

typecheck: ## Type-check backend (pyrefly) and frontend (tsc)
	$(UV) run pyrefly check
	$(PNPM) typecheck

check: lint typecheck test ## Lint, type-check, and test everything

# --- API types ------------------------------------------------------------
# The backend writes its OpenAPI schema to a temporary file (no server runs),
# and openapi-typescript turns it into TypeScript types for the frontend.

API_TYPES := $(ROOT)frontend/src/api/schema.d.ts

# $(1): extra openapi-typescript flags.
define openapi_ts
	@tmp="$$(mktemp "$${TMPDIR:-/tmp}/profile-rag-openapi.XXXXXX")"; \
	trap 'rm -f "$$tmp"' EXIT; \
	$(UV) run python -m profile_rag.api.openapi "$$tmp" && \
	$(PNPM) exec openapi-typescript "$$tmp" --output "$(API_TYPES)" $(1)
endef

types: ## Regenerate frontend/src/api/schema.d.ts from the API's OpenAPI schema
	$(call openapi_ts,)

types-check: ## Fail if frontend/src/api/schema.d.ts is out of date (fix: make types)
	$(call openapi_ts,--check) || \
	{ echo "frontend/src/api/schema.d.ts is out of date. Run \`make types\`." >&2; exit 1; }
