"""The Ollama side of an embedding model: is it pulled, and its embeddings.

Three callers check the model before they use it: the CLI (`index`,
`search`, `ask`, `agent`, `eval`, ...), the API at startup
(api/search.py), and `make demo` (demo_check.py). Each turns the problem
into its own output: an exit code, a `SearchSetupError`, or a line in the
list of problems. This module is the one place that knows how to ask Ollama
and what the fix for each problem is.

- `model_problem(model, base_url)` is None when Ollama has `model.ollama_tag`
  with the digest vectorstore/models.py expects. Otherwise it is one
  sentence that says what is wrong and how to fix it.
- `for_model(model, base_url)` returns the model's embeddings, with the
  query and document prefixes the model needs (prefixed.py). It does not
  call Ollama, so commands that never embed (`hnsw apply`) can use it
  without the check.

Tests replace `ollama_digests` and `for_model` on this module, so no test
needs Ollama. Both are looked up when called, so a replaced function
reaches every caller.
"""

from collections.abc import Callable

import httpx
import ollama
from langchain_core.embeddings import Embeddings
from langchain_ollama import OllamaEmbeddings

from profile_rag.vectorstore.models import EmbeddingModel
from profile_rag.vectorstore.prefixed import PrefixedEmbeddings

DigestLister = Callable[[str], dict[str, str]]

# Listing the pulled models is fast, so a slow answer means Ollama is stuck.
LIST_TIMEOUT_SECONDS = 5


def ollama_digests(base_url: str) -> dict[str, str]:
    """Tag to digest (64 hex characters) of every model Ollama has pulled.

    Raises ConnectionError, httpx.HTTPError, or ollama.ResponseError if
    Ollama does not answer.
    """
    listed = ollama.Client(host=base_url, timeout=LIST_TIMEOUT_SECONDS).list()
    return {
        entry.model: (entry.digest or "").removeprefix("sha256:")
        for entry in listed.models
        if entry.model
    }


def model_problem(
    model: EmbeddingModel, base_url: str, list_digests: DigestLister | None = None
) -> str | None:
    """What stops `model` from being used through Ollama at `base_url`, or None."""
    lister = list_digests or ollama_digests
    try:
        digests = lister(base_url)
    except OSError, ollama.ResponseError, httpx.HTTPError:  # ConnectionError is an OSError
        return (
            f"Ollama does not answer at {base_url}. Start it (`ollama serve`, or open the"
            " Ollama app), or set OLLAMA_URL in .env."
        )
    installed = digests.get(model.ollama_tag)
    if installed is None:
        return f"Ollama has no {model.ollama_tag}. Run `ollama pull {model.ollama_tag}`."
    if installed != model.digest:
        return (
            f"Ollama's {model.ollama_tag} has digest {installed[:12]}, but"
            f" vectorstore/models.py expects {model.digest[:12]}. Run"
            f" `ollama pull {model.ollama_tag}`. If the digest still differs, the tag now"
            " points at other weights, so its vectors would not match the ones in the table."
            " To use the new weights on purpose, copy the full digest from"
            " `curl -s $OLLAMA_URL/api/tags` into models.py and re-index into an empty table."
        )
    return None


def for_model(model: EmbeddingModel, base_url: str) -> Embeddings:
    """The embeddings of `model` through Ollama at `base_url`, with its prefixes."""
    return PrefixedEmbeddings(OllamaEmbeddings(model=model.ollama_tag, base_url=base_url), model)
