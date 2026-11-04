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

.PHONY: help setup db-up db-down db-reset db-logs psql psql-test migrate index \
        backend frontend test test-backend test-frontend \
        lint format typecheck check types types-check

help: ## List available commands
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

setup: ## Install backend and frontend dependencies, create .env if missing
	@test -f .env || (cp .env.example .env && echo "Created .env from .env.example")
	$(UV) sync
	$(PNPM) install

# --- Database -------------------------------------------------------------

db-up: ## Start Postgres + pgvector and wait until it is healthy
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

backend: ## Run the API on http://127.0.0.1:8000 with auto-reload
	$(UV) run uvicorn profile_rag.api.app:app --reload --host 127.0.0.1 --port 8000

frontend: ## Run the Vite dev server on http://127.0.0.1:5173
	$(PNPM) dev

# --- Quality --------------------------------------------------------------

test: test-backend test-frontend ## Run all tests

test-backend: ## Run backend tests (needs `make db-up`)
	$(UV) run pytest

test-frontend: ## Run frontend unit tests
	$(PNPM) test

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
