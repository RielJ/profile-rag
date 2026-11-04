"""The prerequisite checks of `make demo` that need Python.

scripts/demo.sh checks the tools first (Docker, uv, Node, pnpm), then runs
`python -m profile_rag.demo_check`, before it starts the database or changes
anything. This module runs every check, prints one line with a fix for each
one that fails, and exits 1 if any failed, so a new developer sees every
missing piece at once instead of one per run.

The checks:

- `.env` is readable and `DATABASE_URL` uses the port that compose.yaml
  publishes (`DB_PORT`).
- The ports for the database, the API, and the frontend are free. The
  database port may be taken by this project's own database container
  (`--db-running`), so a second `make demo` works.
- Ollama answers at `OLLAMA_URL` and has the embedding model, with the digest
  that vectorstore/models.py expects.
- Pi (`PROFILE_RAG_PI_BIN`) is on PATH or is an executable file. Whether Pi has
  credentials for the model's provider is only known at the first question.
- The corpus has `sources.yaml` with at least one approved source, and no
  approved file has changed since its review. `make index` with no approved
  source empties the table, so the demo stops before that can happen.

Output holds counts, ports, model tags, and paths that the user passed in,
never a source's title or text.

Tests: backend/tests/test_demo_check.py.
"""

import argparse
import shutil
import socket
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import httpx
import ollama
from pydantic import ValidationError

from profile_rag.corpus import registry
from profile_rag.corpus.registry import REGISTRY_FILENAME, RegistryError
from profile_rag.settings import Settings, get_settings
from profile_rag.vectorstore.models import MODELS, EmbeddingModel

PortProbe = Callable[[int], bool]
DigestLister = Callable[[str], dict[str, str]]


@dataclass(frozen=True)
class Ports:
    db: int
    api: int
    frontend: int
    db_running: bool  # this project's database container is up, so it holds `db`


def port_in_use(port: int) -> bool:
    """Whether something accepts connections on `port` at 127.0.0.1 or ::1.

    Both, because `localhost` can resolve to either, and a server bound to
    ::1 does not stop another one from binding 127.0.0.1 on the same port.
    """
    for family, host in ((socket.AF_INET, "127.0.0.1"), (socket.AF_INET6, "::1")):
        try:
            with socket.socket(family, socket.SOCK_STREAM) as probe:
                probe.settimeout(0.5)
                if probe.connect_ex((host, port)) == 0:
                    return True
        except OSError:
            continue  # no IPv6 on this machine
    return False


def ollama_digests(base_url: str) -> dict[str, str]:
    """Tag to digest of every model Ollama has, as cli.py's `_ollama_digests`.

    Raises ConnectionError if Ollama is down.
    """
    listed = ollama.Client(host=base_url, timeout=5).list()
    return {
        entry.model: (entry.digest or "").removeprefix("sha256:")
        for entry in listed.models
        if entry.model
    }


def check_database_url(settings: Settings, db_port: int) -> list[str]:
    url_port = settings.database_url.hosts()[0].get("port")
    if url_port != db_port:
        return [
            f"DATABASE_URL in .env uses port {url_port}, but the database container publishes"
            f" port {db_port} (DB_PORT). Make them the same in .env."
        ]
    return []


def check_ports(ports: Ports, in_use: PortProbe = port_in_use) -> list[str]:
    problems = []
    if not ports.db_running and in_use(ports.db):
        problems.append(
            f"Port {ports.db} (the database) is in use by another program. Set DB_PORT in .env"
            " to a free port, and use the same port in DATABASE_URL and TEST_DATABASE_URL."
        )
    for port, name, variable in (
        (ports.api, "the API", "API_PORT"),
        (ports.frontend, "the frontend", "FRONTEND_PORT"),
    ):
        if in_use(port):
            problems.append(
                f"Port {port} ({name}) is in use. Stop the program that uses it (`lsof -nP"
                f" -iTCP:{port} -sTCP:LISTEN` names it), or set {variable} in .env to a free port."
            )
    return problems


def check_ollama(
    model: EmbeddingModel, base_url: str, list_digests: DigestLister = ollama_digests
) -> list[str]:
    try:
        digests = list_digests(base_url)
    except OSError, ollama.ResponseError, httpx.HTTPError:  # ConnectionError is an OSError
        return [
            f"Ollama does not answer at {base_url}. Start it (`ollama serve`, or open the"
            " Ollama app), or set OLLAMA_URL in .env."
        ]
    installed = digests.get(model.ollama_tag)
    if installed is None:
        return [f"Ollama has no {model.ollama_tag}. Run `ollama pull {model.ollama_tag}`."]
    if installed != model.digest:
        return [
            f"Ollama's {model.ollama_tag} has digest {installed[:12]}, but"
            f" vectorstore/models.py expects {model.digest[:12]}. Run"
            f" `ollama pull {model.ollama_tag}`; if the digest still differs, the tag now"
            " points at other weights (see vectorstore/models.py)."
        ]
    return []


def check_pi(pi_bin: str) -> list[str]:
    if shutil.which(pi_bin) is None:
        return [
            f"Pi ({pi_bin!r}) is not on PATH. Install it from https://pi.dev and log in"
            " (`pi`, then `/login`), or set PROFILE_RAG_PI_BIN in .env to its path."
            " Answers need Pi."
        ]
    return []


def check_corpus(root: Path) -> list[str]:
    path = root / REGISTRY_FILENAME
    if not path.is_file():
        return [
            f"{path} does not exist. CORPUS_DIR must be a corpus root with registered sources:"
            " see corpus/README.md, or run `make demo` without CORPUS_DIR to use corpus-example/."
        ]
    try:
        loaded = registry.load_registry(path)
    except RegistryError:
        return [f"{path} is not a valid registry. Run `profile-rag corpus status` for details."]
    approved = sum(entry.state == "approved" for entry in loaded.sources)
    if approved == 0:
        return [
            f"{path} has no approved source ({len(loaded.sources)} registered). The demo indexes"
            " approved sources only. Review one with `profile-rag corpus review`"
            " (corpus/README.md), or run `make demo` without CORPUS_DIR to use corpus-example/."
        ]
    try:
        stale = registry.find_stale(loaded.sources, root)
    except OSError:
        stale = ["?"]
    if stale:
        return [
            f"{len(stale)} approved source(s) in {path} changed since review."
            " Run `profile-rag corpus status` and review them again."
        ]
    return []


def run_checks(
    settings: Settings,
    model: EmbeddingModel,
    root: Path,
    ports: Ports,
    *,
    in_use: PortProbe = port_in_use,
    list_digests: DigestLister = ollama_digests,
) -> list[str]:
    return [
        *check_database_url(settings, ports.db),
        *check_ports(ports, in_use),
        *check_ollama(model, settings.ollama_url, list_digests),
        *check_pi(settings.pi_bin),
        *check_corpus(root),
    ]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="The prerequisite checks of `make demo`.")
    parser.add_argument("--model", required=True, choices=sorted(MODELS))
    parser.add_argument("--root", required=True, type=Path, help="The corpus root.")
    parser.add_argument("--db-port", required=True, type=int)
    parser.add_argument("--api-port", required=True, type=int)
    parser.add_argument("--frontend-port", required=True, type=int)
    parser.add_argument("--db-running", action="store_true")
    args = parser.parse_args(argv)

    try:
        settings = get_settings()
    except ValidationError as error:
        names = ", ".join(str(item["loc"][0]) for item in error.errors())
        print(f"  - .env is missing or has invalid values for: {names}.", file=sys.stderr)
        return 1
    ports = Ports(args.db_port, args.api_port, args.frontend_port, args.db_running)
    problems = run_checks(settings, MODELS[args.model], args.root.resolve(), ports)
    for problem in problems:
        print(f"  - {problem}", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
