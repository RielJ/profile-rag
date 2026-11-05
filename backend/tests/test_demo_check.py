"""Tests for `make demo`'s prerequisite checks (profile_rag/demo_check.py).

Ports, Ollama, and Pi are faked, so the tests need no running service.
"""

import shutil
import socket
from pathlib import Path

import pytest

from profile_rag import demo_check
from profile_rag.demo_check import Ports, check_corpus, check_ports, run_checks
from profile_rag.settings import REPO_ROOT, Settings
from profile_rag.vectorstore import ollama_embeddings
from profile_rag.vectorstore.models import NOMIC_V15

EXAMPLE = REPO_ROOT / "corpus-example"
PORTS = Ports(db=5443, api=8000, frontend=5173, db_running=True)


def settings(**values: str) -> Settings:
    return Settings.model_validate(
        {
            "database_url": "postgresql://u:p@127.0.0.1:5443/db",
            "ollama_url": "http://127.0.0.1:11434",
            "PROFILE_RAG_PI_BIN": "sh",
            **values,
        }
    )


def ollama_with(tag: str, digest: str) -> demo_check.DigestLister:
    return lambda _url: {tag: digest}


def no_port_in_use(_port: int) -> bool:
    return False


def ollama_down(_url: str) -> dict[str, str]:
    raise ConnectionError("down")


def test_everything_present_passes() -> None:
    problems = run_checks(
        settings(),
        NOMIC_V15,
        EXAMPLE,
        PORTS,
        in_use=lambda _port: False,
        list_digests=ollama_with(NOMIC_V15.ollama_tag, NOMIC_V15.digest),
    )
    assert problems == []


def test_every_missing_prerequisite_is_reported_at_once(tmp_path: Path) -> None:
    problems = run_checks(
        settings(database_url="postgresql://u:p@127.0.0.1:5432/db", PROFILE_RAG_PI_BIN="no-pi-x"),
        NOMIC_V15,
        tmp_path,
        PORTS,
        in_use=lambda port: port == 8000,
        list_digests=ollama_down,
    )
    assert len(problems) == 5, problems
    assert "DATABASE_URL in .env uses port 5432" in problems[0]
    assert "Port 8000 (the API) is in use" in problems[1]
    assert "Ollama does not answer" in problems[2]
    assert "Pi ('no-pi-x') is not on PATH" in problems[3]
    assert "sources.yaml does not exist" in problems[4]


def test_a_missing_model_names_the_pull_command() -> None:
    problems = demo_check.check_ollama(
        NOMIC_V15, "http://x", ollama_with("other:latest", NOMIC_V15.digest)
    )
    assert problems == [
        "Ollama has no nomic-embed-text:v1.5. Run `ollama pull nomic-embed-text:v1.5`."
    ]


def test_a_different_digest_is_reported() -> None:
    problems = demo_check.check_ollama(
        NOMIC_V15, "http://x", ollama_with(NOMIC_V15.ollama_tag, "f" * 64)
    )
    assert "expects 0a109f422b47" in problems[0]


def test_the_database_port_may_be_held_by_the_projects_own_container() -> None:
    assert check_ports(PORTS, in_use=lambda _port: _port == 5443) == []
    taken = Ports(db=5443, api=8000, frontend=5173, db_running=False)
    problems = check_ports(taken, in_use=lambda _port: _port == 5443)
    assert len(problems) == 1
    assert "Set DB_PORT in .env" in problems[0]


def test_a_corpus_without_an_approved_source_is_refused(tmp_path: Path) -> None:
    (tmp_path / "sources.yaml").write_text(
        "sources:\n"
        "  - id: private-title-x\n"
        "    kind: resume\n"
        "    title: SECRET-TITLE\n"
        "    attribution: self_report\n"
        "    visibility: public\n"
        "    state: candidate\n"
    )
    problems = check_corpus(tmp_path)
    assert len(problems) == 1
    assert "has no approved source (1 registered)" in problems[0]
    assert "SECRET-TITLE" not in problems[0]
    assert "private-title-x" not in problems[0]


def test_an_approved_file_changed_since_review_is_refused(tmp_path: Path) -> None:
    shutil.copytree(EXAMPLE, tmp_path, dirs_exist_ok=True)
    first = next((tmp_path / "approved").iterdir())
    first.write_text(first.read_text() + "\nedited\n")
    problems = check_corpus(tmp_path)
    assert problems == [
        f"1 approved source(s) in {tmp_path / 'sources.yaml'} changed since review."
        " Run `profile-rag corpus status` and review them again."
    ]


def test_a_port_with_a_listener_is_in_use() -> None:
    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        server.listen()
        port = server.getsockname()[1]
        assert demo_check.port_in_use(port)
    assert not demo_check.port_in_use(port)


def test_main_prints_problems_and_exits_1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(demo_check, "port_in_use", no_port_in_use)
    monkeypatch.setattr(ollama_embeddings, "ollama_digests", ollama_down)
    args = ["--model", "nomic-v15", "--root", str(tmp_path)]
    ports = ["--db-port", "5443", "--api-port", "8000", "--frontend-port", "5173"]
    code = demo_check.main([*args, *ports, "--db-running"])
    assert code == 1
    assert "sources.yaml does not exist" in capsys.readouterr().err
