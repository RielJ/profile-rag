"""vectorstore/ollama_embeddings.py: the model check that the CLI, the API
and `make demo` share, and the embeddings it builds.

No test calls Ollama: each passes a fake digest lister.
"""

from collections.abc import Callable

import httpx
import ollama
import pytest

from profile_rag.vectorstore import ollama_embeddings
from profile_rag.vectorstore.models import NOMIC_V15
from profile_rag.vectorstore.prefixed import PrefixedEmbeddings

URL = "http://ollama.test"


def lister(digests: dict[str, str]) -> Callable[[str], dict[str, str]]:
    return lambda _url: digests


def raising(error: Exception) -> Callable[[str], dict[str, str]]:
    def list_digests(_url: str) -> dict[str, str]:
        raise error

    return list_digests


def test_no_problem_when_the_tag_has_the_expected_digest() -> None:
    digests = lister({NOMIC_V15.ollama_tag: NOMIC_V15.digest})
    assert ollama_embeddings.model_problem(NOMIC_V15, URL, digests) is None


@pytest.mark.parametrize(
    "error",
    [
        ConnectionError("refused"),
        TimeoutError("slow"),
        httpx.ConnectError("refused"),
        ollama.ResponseError("server error", 500),
    ],
)
def test_an_unreachable_ollama_is_a_problem_not_an_exception(error: Exception) -> None:
    problem = ollama_embeddings.model_problem(NOMIC_V15, URL, raising(error))
    assert problem is not None
    assert f"does not answer at {URL}" in problem
    assert "OLLAMA_URL" in problem


def test_a_missing_tag_names_the_pull_command() -> None:
    problem = ollama_embeddings.model_problem(NOMIC_V15, URL, lister({}))
    assert problem == (
        f"Ollama has no {NOMIC_V15.ollama_tag}. Run `ollama pull {NOMIC_V15.ollama_tag}`."
    )


def test_a_different_digest_names_both_digests() -> None:
    other = "f" * 64
    problem = ollama_embeddings.model_problem(NOMIC_V15, URL, lister({NOMIC_V15.ollama_tag: other}))
    assert problem is not None
    assert other[:12] in problem
    assert NOMIC_V15.digest[:12] in problem


def test_the_default_lister_is_looked_up_at_call_time(monkeypatch: pytest.MonkeyPatch) -> None:
    """Callers that pass no lister must see a replaced `ollama_digests`."""
    monkeypatch.setattr(ollama_embeddings, "ollama_digests", lister({}))
    assert ollama_embeddings.model_problem(NOMIC_V15, URL) is not None


def test_for_model_adds_the_model_prefixes_and_does_not_call_ollama() -> None:
    embeddings = ollama_embeddings.for_model(NOMIC_V15, URL)
    assert isinstance(embeddings, PrefixedEmbeddings)
    assert embeddings.model is NOMIC_V15
