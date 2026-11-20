#!/usr/bin/env bash
# `just demo`: check the prerequisites, start Postgres, apply the migrations,
# and index a corpus. Then print how to ask a question.
#
# The justfile sets these variables (DB_PORT comes from the environment,
# then .env, then a default): CORPUS_DIR, MODEL, DB_PORT.
#
# Every step is safe to repeat: `docker compose up` leaves a running
# database alone, the migration runner skips applied files, and `index`
# skips unchanged chunks.

set -euo pipefail

cd "$(dirname "$0")/.."

: "${CORPUS_DIR:?}" "${MODEL:?}" "${DB_PORT:?}"
UV=(uv --directory backend)
CORPUS_ROOT=$(cd "$CORPUS_DIR" 2>/dev/null && pwd) || {
	echo "just demo: corpus $CORPUS_DIR is not a directory." >&2
	exit 1
}

step() { printf '\n==> %s\n' "$*"; }

# --- 1. Tools -------------------------------------------------------------
# These need no project dependencies, so they run first.

step "Checking tools"
problems=()
if [[ ! -f .env ]]; then
	problems+=(".env does not exist. Run \`cp .env.example .env\`.")
fi
if ! command -v docker >/dev/null; then
	problems+=("Docker is not installed. Install Docker Desktop (https://docs.docker.com/get-docker/) or OrbStack.")
elif ! docker compose version >/dev/null 2>&1; then
	problems+=("\`docker compose\` (Compose v2) is missing. Update Docker Desktop, or install the Compose plugin.")
elif ! docker info >/dev/null 2>&1; then
	problems+=("Docker is installed but not running. Start Docker Desktop or OrbStack.")
fi
if ! command -v uv >/dev/null; then
	problems+=("uv is not installed. Install it: https://docs.astral.sh/uv/getting-started/installation/")
fi
if ((${#problems[@]})); then
	printf '  - %s\n' "${problems[@]}" >&2
	echo "just demo cannot start. Fix the items above, then run it again." >&2
	exit 1
fi

# --- 2. Services, the database port, and the corpus -----------------------

step "Installing backend dependencies (uv sync)"
"${UV[@]}" sync --frozen

step "Checking the database container and port, Ollama, Pi, and $CORPUS_DIR"
# docker compose finds containers by project name only. If another checkout
# uses the same COMPOSE_PROJECT_NAME, `up` would take over its container
# and its data volume.
# The other checks run as well, so every problem is listed in one run.
container=$(docker compose ps --all --quiet db 2>/dev/null | head -n 1)
owner_problem=
db_running=()
if [[ -n $container ]]; then
	owner=$(docker inspect --format \
		'{{index .Config.Labels "com.docker.compose.project.working_dir"}}' "$container")
	owner_real=$(cd "$owner" 2>/dev/null && pwd -P || echo "$owner")
	if [[ $owner_real != "$(pwd -P)" ]]; then
		owner_problem="Another checkout ($owner) uses the same Compose project name."
		owner_problem+=" Set COMPOSE_PROJECT_NAME in .env to a new name, and DB_PORT to a free port."
	elif [[ -n $(docker compose ps --status running --quiet db 2>/dev/null) ]]; then
		db_running=(--db-running) # this checkout's database holds DB_PORT
	fi
fi
checks_ok=1
"${UV[@]}" run --frozen python -m profile_rag.demo_check \
	--model "$MODEL" --root "$CORPUS_ROOT" --db-port "$DB_PORT" \
	${db_running[@]+"${db_running[@]}"} ||
	checks_ok=0
if [[ -n $owner_problem ]]; then
	echo "  - $owner_problem" >&2
	checks_ok=0
fi
if ((!checks_ok)); then
	echo "just demo cannot start. Fix the items above, then run it again." >&2
	exit 1
fi

# --- 3. Database and index ------------------------------------------------

step "Starting Postgres on 127.0.0.1:$DB_PORT"
docker compose up --detach --wait db

step "Applying migrations"
"${UV[@]}" run --frozen python -m profile_rag.db.migrate

step "Indexing $CORPUS_DIR with $MODEL"
"${UV[@]}" run --frozen profile-rag index --model "$MODEL" --root "$CORPUS_ROOT"

cat <<EOF

The corpus is indexed. Corpus: $CORPUS_DIR. Embedding model: $MODEL.

Ask a question from the terminal:

  cd backend && uv run profile-rag ask --model $MODEL "Your question?"

Each question starts Pi and sends the question and the retrieved excerpts to
the model in PROFILE_RAG_PI_MODEL.

The HTTP API is the Cloudflare Worker in worker/. \`just worker-dev\` runs it
locally (docs/DEPLOY.md). The database keeps running; \`just db-down\` stops it.
EOF
