#!/usr/bin/env bash
# `make demo`: check the prerequisites, start Postgres, apply the migrations,
# index a corpus, then run the API and the frontend until Ctrl-C.
#
# The Makefile sets these variables (from the environment, then .env, then
# a default): CORPUS_DIR, MODEL, DB_PORT, API_PORT, FRONTEND_PORT, NODE_EXEC.
#
# Every step is safe to repeat: `docker compose up` leaves a running
# database alone, the migration runner skips applied files, and `index`
# skips unchanged chunks. The API and Vite run in their own process groups;
# Ctrl-C, or either server exiting, stops both groups.

set -euo pipefail

cd "$(dirname "$0")/.."

: "${CORPUS_DIR:?}" "${MODEL:?}" "${DB_PORT:?}" "${API_PORT:?}" "${FRONTEND_PORT:?}"
NODE_EXEC=${NODE_EXEC:-}
NODE_VERSION=$(cat .node-version)
UV=(uv --directory backend)
CORPUS_ROOT=$(cd "$CORPUS_DIR" 2>/dev/null && pwd) || {
	echo "make demo: CORPUS_DIR=$CORPUS_DIR is not a directory." >&2
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
if ! node_version=$($NODE_EXEC node --version 2>/dev/null); then
	if [[ -n $NODE_EXEC ]]; then
		problems+=("fnm has no Node $NODE_VERSION. Run \`fnm install $NODE_VERSION\`.")
	else
		problems+=("Node is not installed. Install Node $NODE_VERSION (https://nodejs.org, or \`fnm install $NODE_VERSION\`).")
	fi
else
	major=${node_version#v}
	major=${major%%.*}
	if ((major < NODE_VERSION)); then
		problems+=("Node is $node_version; this project needs Node $NODE_VERSION or later. Run \`fnm install $NODE_VERSION\`, or install it from https://nodejs.org.")
	elif ! $NODE_EXEC pnpm --version >/dev/null 2>&1; then
		problems+=("pnpm is not installed. Run \`corepack enable pnpm\`, or see https://pnpm.io/installation.")
	fi
fi
if ((${#problems[@]})); then
	printf '  - %s\n' "${problems[@]}" >&2
	echo "make demo cannot start. Fix the items above, then run it again." >&2
	exit 1
fi

# --- 2. Services, ports, and the corpus -----------------------------------

step "Installing backend dependencies (uv sync)"
"${UV[@]}" sync --frozen

step "Checking the database container, ports, Ollama, Pi, and $CORPUS_DIR"
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
	--api-port "$API_PORT" --frontend-port "$FRONTEND_PORT" ${db_running[@]+"${db_running[@]}"} ||
	checks_ok=0
if [[ -n $owner_problem ]]; then
	echo "  - $owner_problem" >&2
	checks_ok=0
fi
if ((!checks_ok)); then
	echo "make demo cannot start. Fix the items above, then run it again." >&2
	exit 1
fi

# --- 3. Database and index ------------------------------------------------

step "Installing frontend dependencies (pnpm install)"
$NODE_EXEC pnpm --dir frontend install --frozen-lockfile

step "Starting Postgres on 127.0.0.1:$DB_PORT"
docker compose up --detach --wait db

step "Applying migrations"
"${UV[@]}" run --frozen python -m profile_rag.db.migrate

step "Indexing $CORPUS_DIR with $MODEL"
"${UV[@]}" run --frozen profile-rag index --model "$MODEL" --root "$CORPUS_ROOT"

# --- 4. API and frontend --------------------------------------------------

api_pid=
web_pid=

# Print the PIDs of every descendant of a process.
descendants() {
	local child
	for child in $(pgrep -P "$1" 2>/dev/null); do
		echo "$child"
		descendants "$child"
	done
}

# Stop a server: SIGTERM to its process group and to every descendant (pnpm
# starts Vite in a process group of its own), then SIGKILL after 10 seconds.
stop_server() {
	local pid=$1 pids alive
	[[ -n $pid ]] || return 0
	pids="$pid $(descendants "$pid")"
	kill -TERM -- "-$pid" 2>/dev/null || true
	# shellcheck disable=SC2086 # one PID per word
	kill -TERM $pids 2>/dev/null || true
	for _ in {1..20}; do
		alive=0
		for p in $pids; do kill -0 "$p" 2>/dev/null && alive=1; done
		((alive)) || return 0
		sleep 0.5
	done
	kill -KILL -- "-$pid" 2>/dev/null || true
	# shellcheck disable=SC2086
	kill -KILL $pids 2>/dev/null || true
}

cleanup() {
	trap - INT TERM EXIT
	echo
	echo "Stopping the API and the frontend..."
	stop_server "$web_pid"
	stop_server "$api_pid"
	wait 2>/dev/null || true
	echo "Stopped. The database keeps running; \`make db-down\` stops it."
}
trap 'cleanup; exit 130' INT
trap 'cleanup; exit 143' TERM
trap cleanup EXIT

# Job control gives each background job its own process group, so
# stop_server reaches uvicorn under `uv run` and Vite under pnpm. stdin is
# /dev/null: a background job that reads the terminal would be stopped.
set -m

step "Starting the API on 127.0.0.1:$API_PORT"
CORPUS_DIR=$CORPUS_ROOT PROFILE_RAG_API_MODEL=$MODEL \
	"${UV[@]}" run --frozen uvicorn profile_rag.api.app:app \
	--host 127.0.0.1 --port "$API_PORT" </dev/null &
api_pid=$!

step "Starting the frontend on 127.0.0.1:$FRONTEND_PORT"
BACKEND_URL=http://127.0.0.1:$API_PORT \
	$NODE_EXEC pnpm --dir frontend exec vite --host 127.0.0.1 --port "$FRONTEND_PORT" --strictPort \
	</dev/null &
web_pid=$!

set +m

# Wait until a URL answers, while the process that serves it is alive.
wait_for() {
	local url=$1 pid=$2 name=$3
	for _ in {1..120}; do
		if ! kill -0 "$pid" 2>/dev/null; then
			echo "make demo: $name exited during startup. Its error is above." >&2
			return 1
		fi
		curl --silent --fail --output /dev/null "$url" && return 0
		sleep 0.5
	done
	echo "make demo: $name did not answer $url within 60 seconds." >&2
	return 1
}

wait_for "http://127.0.0.1:$API_PORT/api/health" "$api_pid" "The API" || exit 1
wait_for "http://127.0.0.1:$FRONTEND_PORT/" "$web_pid" "Vite" || exit 1

cat <<EOF

The demo is running. Corpus: $CORPUS_DIR. Embedding model: $MODEL.

  Open http://127.0.0.1:$FRONTEND_PORT
  API: http://127.0.0.1:$API_PORT/docs

Each question starts Pi and sends the question and the retrieved excerpts to
the model in PROFILE_RAG_PI_MODEL. Press Ctrl-C to stop both servers.
EOF

while kill -0 "$api_pid" 2>/dev/null && kill -0 "$web_pid" 2>/dev/null; do
	sleep 1
done
echo "make demo: a server exited. Its error is above." >&2
exit 1
